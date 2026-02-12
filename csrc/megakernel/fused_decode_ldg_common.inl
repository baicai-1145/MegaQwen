/**
 * Fused Decode with __ldg() cached reads
 *
 * Strategy: Use __ldg() for read-only weight data to leverage texture cache
 * Based on minimal_sync but with cached loads
 */

#include "config.cuh"
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

// =============================================================================
// Debug: per-block/per-layer stage timing (no Nsight counters required)
// =============================================================================
//
// Records per-block stage durations (clock cycles) inside the cooperative decode
// megakernel so we can identify whether time is spent in compute vs. waiting at
// grid.sync() barriers, without relying on NVIDIA performance counters.
//
// Buffer layout: uint64 cycles [num_blocks, num_layers, 16].
__device__ unsigned long long* g_ldg_dbg_cycles = nullptr;
__device__ int g_ldg_dbg_num_blocks = 0;   // capacity (>= gridDim.x)
__device__ int g_ldg_dbg_num_layers = 0;
__device__ int g_ldg_dbg_nstages = 0;

extern "C" void launch_ldg_set_stage_debug(
    unsigned long long* buf,
    int num_blocks,
    int num_layers,
    int nstages,
    cudaStream_t stream
) {
    cudaMemcpyToSymbolAsync(g_ldg_dbg_cycles, &buf, sizeof(buf), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(g_ldg_dbg_num_blocks, &num_blocks, sizeof(num_blocks), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(g_ldg_dbg_num_layers, &num_layers, sizeof(num_layers), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(g_ldg_dbg_nstages, &nstages, sizeof(nstages), 0, cudaMemcpyHostToDevice, stream);
}

extern "C" void launch_ldg_disable_stage_debug(cudaStream_t stream) {
    unsigned long long* null_ptr = nullptr;
    int zero = 0;
    cudaMemcpyToSymbolAsync(g_ldg_dbg_cycles, &null_ptr, sizeof(null_ptr), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(g_ldg_dbg_num_blocks, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(g_ldg_dbg_num_layers, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(g_ldg_dbg_nstages, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice, stream);
}

// Per-layer stage indices (nstages=16).
enum : int {
    DBG_MATVEC_RMSNORM = 0,
    DBG_MATVEC_SYNC1 = 1,
    DBG_MATVEC_QKV = 2,
    DBG_MATVEC_SYNC2 = 3,
    DBG_QK_NORM_ROPE_CACHE = 4,
    DBG_QK_SYNC = 5,
    DBG_ATTN = 6,
    DBG_ATTN_SYNC = 7,
    DBG_OPROJ_RESID = 8,
    DBG_OPROJ_SYNC = 9,
    DBG_POSTNORM = 10,
    DBG_POSTNORM_SYNC = 11,
    DBG_GATE_UP = 12,
    DBG_GATE_UP_SYNC = 13,
    DBG_DOWNPROJ = 14,
    DBG_DOWNPROJ_SYNC = 15,
};

// Configuration defaults (tuned for RTX 3090 by default; override at runtime in launch wrapper)
constexpr int LDG_NUM_BLOCKS_DEFAULT = 82;
constexpr int LDG_BLOCK_SIZE = 256;
constexpr int LDG_NUM_WARPS = LDG_BLOCK_SIZE / WARP_SIZE;
constexpr float LDG_RMS_EPS = 1e-6f;

// Compile-time knobs (set via NVCC -D...; see csrc/megakernel/megakernel_decode.py).
#ifndef MEGAQWEN_LDG_UNROLL
#define MEGAQWEN_LDG_UNROLL 8
#endif

#ifndef MEGAQWEN_LDG_MIN_BLOCKS_PER_SM
#define MEGAQWEN_LDG_MIN_BLOCKS_PER_SM 1
#endif

// NVCC does not reliably macro-expand the numeric argument of `#pragma unroll N`.
// Use `_Pragma("unroll N")` so N can be controlled via a macro.
#define MEGAQWEN_STR_IMPL(x) #x
#define MEGAQWEN_STR(x) MEGAQWEN_STR_IMPL(x)
#define MEGAQWEN_PRAGMA_UNROLL_N(n) _Pragma(MEGAQWEN_STR(unroll n))

// LM head
constexpr int LDG_LM_NUM_BLOCKS_DEFAULT = 1184;
constexpr int LDG_LM_BLOCK_SIZE = 256;
constexpr int LDG_VOCAB_SIZE = VOCAB_SIZE;

struct LDGLayerWeights {
    const __nv_bfloat16* input_layernorm_weight;
    const __nv_bfloat16* q_proj_weight;
    const __nv_bfloat16* k_proj_weight;
    const __nv_bfloat16* v_proj_weight;
    const __nv_bfloat16* q_norm_weight;
    const __nv_bfloat16* k_norm_weight;
    const __nv_bfloat16* o_proj_weight;
    const __nv_bfloat16* post_attn_layernorm_weight;
    const __nv_bfloat16* gate_proj_weight;
    const __nv_bfloat16* up_proj_weight;
    const __nv_bfloat16* down_proj_weight;
};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float ldg_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float ldg_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =============================================================================
// Optimized matvec with __ldg and aggressive unrolling
// =============================================================================

__device__ void ldg_matvec_qkv(
    cg::grid_group& grid,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ v_weight,
    float* __restrict__ g_normalized,
    float* __restrict__ g_residual,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    unsigned long long* __restrict__ dbg_stage
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Block 0 does RMSNorm
    if (block_id == 0) {
        unsigned long long t0 = 0;
        if (dbg_stage != nullptr) t0 = clock64();
        __shared__ float smem[HIDDEN_SIZE];
        __shared__ float smem_reduce[LDG_NUM_WARPS];

        float local_sum_sq = 0.0f;

        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float v = __bfloat162float(__ldg(input + i));
            smem[i] = v;
            g_residual[i] = v;
            local_sum_sq += v * v;
        }

        local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float w = __bfloat162float(__ldg(norm_weight + i));
            g_normalized[i] = smem[i] * rstd * w;
        }
        if (dbg_stage != nullptr) {
            unsigned long long t1 = clock64();
            dbg_stage[DBG_MATVEC_RMSNORM] = t1 - t0;
        }
    }

    {
        unsigned long long t0 = 0;
        if (dbg_stage != nullptr) t0 = clock64();
        grid.sync();
        if (dbg_stage != nullptr) {
            unsigned long long t1 = clock64();
            dbg_stage[DBG_MATVEC_SYNC1] = t1 - t0;
        }
    }

    // QKV projection with vec4 and __ldg
    constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
    int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

    unsigned long long t0_qkv = 0;
    if (dbg_stage != nullptr) t0_qkv = clock64();
    for (int m_base = row_start; m_base < row_end; m_base += LDG_NUM_WARPS) {
        int m = m_base + warp_id;

        if (m < row_end) {
            const __nv_bfloat16* weight_row;
            float* output_ptr;

            if (m < Q_SIZE) {
                weight_row = q_weight + m * HIDDEN_SIZE;
                output_ptr = q_out + m;
            } else if (m < Q_SIZE + KV_SIZE) {
                weight_row = k_weight + (m - Q_SIZE) * HIDDEN_SIZE;
                output_ptr = k_out + (m - Q_SIZE);
            } else {
                weight_row = v_weight + (m - Q_SIZE - KV_SIZE) * HIDDEN_SIZE;
                output_ptr = v_out + (m - Q_SIZE - KV_SIZE);
            }

            // Use vec4 loads with __ldg through uint2
            float sum = 0.0f;
            MEGAQWEN_PRAGMA_UNROLL_N(MEGAQWEN_LDG_UNROLL)
            for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
                uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(weight_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

                sum += __bfloat162float(w_ptr[0]) * g_normalized[k] +
                       __bfloat162float(w_ptr[1]) * g_normalized[k+1] +
                       __bfloat162float(w_ptr[2]) * g_normalized[k+2] +
                       __bfloat162float(w_ptr[3]) * g_normalized[k+3];
            }

            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0) {
                *output_ptr = sum;
            }
        }
    }

    if (dbg_stage != nullptr) {
        unsigned long long t1 = clock64();
        dbg_stage[DBG_MATVEC_QKV] = t1 - t0_qkv;
    }
    {
        unsigned long long t0 = 0;
        if (dbg_stage != nullptr) t0 = clock64();
        grid.sync();
        if (dbg_stage != nullptr) {
            unsigned long long t1 = clock64();
            dbg_stage[DBG_MATVEC_SYNC2] = t1 - t0;
        }
    }
}

// =============================================================================
// QK Norm + RoPE + KV Cache
// =============================================================================

__device__ void ldg_qk_norm_rope_cache(
    cg::grid_group& grid,
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int position,
    int max_seq_len,
    unsigned long long* __restrict__ dbg_stage
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const __nv_bfloat16* cos_pos = cos_table + position * HEAD_DIM;
    const __nv_bfloat16* sin_pos = sin_table + position * HEAD_DIM;

    unsigned long long t0 = 0;
    if (dbg_stage != nullptr) t0 = clock64();

    // Process Q heads
    int q_heads_per_block = (NUM_Q_HEADS + num_blocks - 1) / num_blocks;
    int q_head_start = block_id * q_heads_per_block;
    int q_head_end = min(q_head_start + q_heads_per_block, NUM_Q_HEADS);

    for (int h = q_head_start + warp_id; h < q_head_end; h += LDG_NUM_WARPS) {
        float* q_head = q + h * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
            sum_sq += q_head[i] * q_head[i];
        }
        sum_sq = ldg_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float q_local[HEAD_DIM / WARP_SIZE];
        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            q_local[j] = q_head[i] * scale * __bfloat162float(__ldg(q_norm_weight + i));
        }

        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            float cos_v = __bfloat162float(__ldg(cos_pos + i));
            float sin_v = __bfloat162float(__ldg(sin_pos + i));

            int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
            int pair_idx = i + pair_offset;
            int pair_j = pair_idx / WARP_SIZE;
            float pair_v = __shfl_sync(0xffffffff, q_local[pair_j], pair_idx % WARP_SIZE);

            if (i < HEAD_DIM/2) {
                q_head[i] = q_local[j] * cos_v - pair_v * sin_v;
            } else {
                q_head[i] = pair_v * sin_v + q_local[j] * cos_v;
            }
        }
    }

    // Process K heads + cache
    int k_heads_per_block = (NUM_KV_HEADS + num_blocks - 1) / num_blocks;
    int k_head_start = block_id * k_heads_per_block;
    int k_head_end = min(k_head_start + k_heads_per_block, NUM_KV_HEADS);

    for (int h = k_head_start + warp_id; h < k_head_end; h += LDG_NUM_WARPS) {
        float* k_head = k + h * HEAD_DIM;
        const float* v_head = v + h * HEAD_DIM;
        __nv_bfloat16* k_cache_head = k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
        __nv_bfloat16* v_cache_head = v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
            sum_sq += k_head[i] * k_head[i];
        }
        sum_sq = ldg_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float k_local[HEAD_DIM / WARP_SIZE];
        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            k_local[j] = k_head[i] * scale * __bfloat162float(__ldg(k_norm_weight + i));
        }

        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            float cos_v = __bfloat162float(__ldg(cos_pos + i));
            float sin_v = __bfloat162float(__ldg(sin_pos + i));

            int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
            int pair_idx = i + pair_offset;
            int pair_j = pair_idx / WARP_SIZE;
            float pair_v = __shfl_sync(0xffffffff, k_local[pair_j], pair_idx % WARP_SIZE);

            float k_final;
            if (i < HEAD_DIM/2) {
                k_final = k_local[j] * cos_v - pair_v * sin_v;
            } else {
                k_final = pair_v * sin_v + k_local[j] * cos_v;
            }
            k_head[i] = k_final;
            k_cache_head[i] = __float2bfloat16(k_final);
            v_cache_head[i] = __float2bfloat16(v_head[i]);
        }
    }

    if (dbg_stage != nullptr) {
        unsigned long long t1 = clock64();
        dbg_stage[DBG_QK_NORM_ROPE_CACHE] = t1 - t0;
    }
    {
        unsigned long long t0s = 0;
        if (dbg_stage != nullptr) t0s = clock64();
        grid.sync();
        if (dbg_stage != nullptr) {
            unsigned long long t1 = clock64();
            dbg_stage[DBG_QK_SYNC] = t1 - t0s;
        }
    }
}

// =============================================================================
// Attention with __ldg for KV cache + block divergence for prefetching
// =============================================================================

// Prefetch weights into L2 cache using __ldg reads
__device__ void ldg_prefetch_weights_l2(
    const __nv_bfloat16* __restrict__ weights,
    int num_elements
) {
    // Each thread prefetches strided elements to warm L2 cache
    // Using __ldg ensures we go through texture/L2 path
    float dummy = 0.0f;
    for (int i = threadIdx.x; i < num_elements; i += LDG_BLOCK_SIZE * 4) {
        // Read but don't use - compiler won't optimize out due to volatile-like __ldg
        dummy += __bfloat162float(__ldg(weights + i));
    }
    // Prevent optimization (result stored to shared but never used)
    __shared__ float s_dummy;
    if (threadIdx.x == 0) s_dummy = dummy;
}

__device__ void ldg_attention(
    cg::grid_group& grid,
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    float* __restrict__ attn_out,
    float* __restrict__ attn_partial_max,
    float* __restrict__ attn_partial_sum,
    float* __restrict__ attn_partial_out,  // [gridDim.x, HEAD_DIM]
    int cache_len,
    int max_seq_len,
    float attn_scale,
    int attn_all_blocks,
    // Weights to prefetch during attention (for blocks not doing attention)
    const __nv_bfloat16* __restrict__ o_weight,
    const __nv_bfloat16* __restrict__ gate_weight,
    const __nv_bfloat16* __restrict__ up_weight,
    unsigned long long* __restrict__ dbg_stage
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    unsigned long long t0 = 0;
    if (dbg_stage != nullptr) t0 = clock64();

    // All-block attention requires at least one block per Q head.
    int use_all_blocks = (attn_all_blocks != 0) && (num_blocks >= NUM_Q_HEADS);

    if (!use_all_blocks) {
        // Only first NUM_Q_HEADS blocks do attention work
        // Remaining blocks prefetch MLP weights to warm L2 cache
        const int ATTN_BLOCKS = NUM_Q_HEADS;  // 16 blocks for 16 Q heads

        if (block_id >= ATTN_BLOCKS) {
        // This block prefetches weights while attention computes
        // Distribute prefetch work across non-attention blocks
        int prefetch_block_id = block_id - ATTN_BLOCKS;
        int num_prefetch_blocks = num_blocks - ATTN_BLOCKS;

        // O projection: Q_SIZE x HIDDEN_SIZE = 2048 x 1024 = 2M elements
        // Gate: HIDDEN_SIZE x INTERMEDIATE_SIZE = 1024 x 3072 = 3M elements
        // Up: same as gate

        // Divide O projection among first half of prefetch blocks
        if (prefetch_block_id < num_prefetch_blocks / 3) {
            int elems_per_block = (Q_SIZE * HIDDEN_SIZE) / (num_prefetch_blocks / 3 + 1);
            int start = prefetch_block_id * elems_per_block;
            ldg_prefetch_weights_l2(o_weight + start, elems_per_block);
        }
        // Gate projection
        else if (prefetch_block_id < 2 * num_prefetch_blocks / 3) {
            int adjusted_id = prefetch_block_id - num_prefetch_blocks / 3;
            int elems_per_block = (HIDDEN_SIZE * INTERMEDIATE_SIZE) / (num_prefetch_blocks / 3 + 1);
            int start = adjusted_id * elems_per_block;
            ldg_prefetch_weights_l2(gate_weight + start, elems_per_block);
        }
        // Up projection
        else {
            int adjusted_id = prefetch_block_id - 2 * num_prefetch_blocks / 3;
            int elems_per_block = (HIDDEN_SIZE * INTERMEDIATE_SIZE) / (num_prefetch_blocks / 3 + 1);
            int start = adjusted_id * elems_per_block;
            ldg_prefetch_weights_l2(up_weight + start, elems_per_block);
        }

        // Wait for all blocks at grid.sync() at the end
            if (dbg_stage != nullptr) {
                unsigned long long t1 = clock64();
                dbg_stage[DBG_ATTN] = t1 - t0;
            }
            unsigned long long t0s = 0;
            if (dbg_stage != nullptr) t0s = clock64();
            grid.sync();
            if (dbg_stage != nullptr) {
                unsigned long long t1 = clock64();
                dbg_stage[DBG_ATTN_SYNC] = t1 - t0s;
            }
            return;
        }

        // Shared memory for cross-warp reduction of online softmax
        __shared__ float s_max_score[LDG_NUM_WARPS];
        __shared__ float s_sum_exp[LDG_NUM_WARPS];
        __shared__ float s_out_acc[LDG_NUM_WARPS][HEAD_DIM];

        // Each of the 16 attention blocks handles one Q head
        int heads_per_block = (NUM_Q_HEADS + ATTN_BLOCKS - 1) / ATTN_BLOCKS;
        int head_start = block_id * heads_per_block;
        int head_end = min(head_start + heads_per_block, NUM_Q_HEADS);

        for (int qh = head_start; qh < head_end; qh++) {
        int kv_head = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
        const float* q_head = q + qh * HEAD_DIM;
        float* out_head = attn_out + qh * HEAD_DIM;

        float max_score = -INFINITY;
        float sum_exp = 0.0f;
        float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // Each warp processes a subset of cache positions
        for (int pos = warp_id; pos < cache_len; pos += LDG_NUM_WARPS) {
            const __nv_bfloat16* k_pos = k_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
            const __nv_bfloat16* v_pos = v_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

            // Q @ K with __ldg
            float score = 0.0f;
            for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                score += q_head[d] * __bfloat162float(__ldg(k_pos + d));
            }
            score = ldg_warp_reduce_sum(score) * attn_scale;
            score = __shfl_sync(0xffffffff, score, 0);

            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_diff = expf(old_max - max_score);
            sum_exp = sum_exp * exp_diff + expf(score - max_score);

            float weight = expf(score - max_score);
            #pragma unroll
            for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
                out_acc[j] = out_acc[j] * exp_diff + weight * __bfloat162float(__ldg(v_pos + d));
            }
        }

        // Store each warp's partial results to shared memory
        if (lane_id == 0) {
            s_max_score[warp_id] = max_score;
            s_sum_exp[warp_id] = sum_exp;
        }
        #pragma unroll
        for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
            s_out_acc[warp_id][d] = out_acc[j];
        }
        __syncthreads();

        // Warp 0 combines results from all warps
        if (warp_id == 0) {
            // Find global max across all warps
            float global_max = s_max_score[0];
            for (int w = 1; w < LDG_NUM_WARPS; w++) {
                if (s_max_score[w] > -INFINITY) {  // Only consider warps that processed positions
                    global_max = fmaxf(global_max, s_max_score[w]);
                }
            }

            // Rescale and sum the partial results
            float total_sum_exp = 0.0f;
            float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (int w = 0; w < LDG_NUM_WARPS; w++) {
                if (s_max_score[w] > -INFINITY) {  // Only consider warps that processed positions
                    float scale = expf(s_max_score[w] - global_max);
                    total_sum_exp += s_sum_exp[w] * scale;

                    #pragma unroll
                    for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
                        final_out[j] += s_out_acc[w][d] * scale;
                    }
                }
            }

            // Write final normalized output
            #pragma unroll
            for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
                out_head[d] = final_out[j] / total_sum_exp;
            }
        }
        __syncthreads();
    }

    if (dbg_stage != nullptr) {
        unsigned long long t1 = clock64();
        dbg_stage[DBG_ATTN] = t1 - t0;
    }
    {
        unsigned long long t0s = 0;
        if (dbg_stage != nullptr) t0s = clock64();
        grid.sync();
        if (dbg_stage != nullptr) {
            unsigned long long t1 = clock64();
            dbg_stage[DBG_ATTN_SYNC] = t1 - t0s;
        }
    }
    return;
    }

    // =============================================================================
    // All-block attention: split cache positions across blocks (improves SM utilization on large GPUs).
    //
    // block_id maps to:
    //   qh = block_id % NUM_Q_HEADS
    //   group = block_id / NUM_Q_HEADS  (different groups cover disjoint cache ranges)
    // Each (qh, group) computes a partial online-softmax accumulator over its cache slice,
    // writes it to global scratch, then a single reducer block per head combines groups.
    // =============================================================================

    const int qh = block_id % NUM_Q_HEADS;
    const int group = block_id / NUM_Q_HEADS;
    const int groups = (num_blocks + NUM_Q_HEADS - 1) / NUM_Q_HEADS;
    // For small cache_len, having too many (qh,group) pairs creates lots of empty
    // groups that race to the barrier and spend most of the stage waiting.
    // Clamp the number of active groups so each group has >=1 token when possible.
    const int effective_groups = (cache_len < groups) ? cache_len : groups;
    const int active_blocks = effective_groups * NUM_Q_HEADS;
    if (block_id >= active_blocks) {
        // Repurpose excess blocks to prefetch weights so they don't arrive at the
        // barrier immediately (reduces barrier skew and can warm L2 for later).
        ldg_prefetch_weights_l2(o_weight, Q_SIZE * HIDDEN_SIZE);
        ldg_prefetch_weights_l2(gate_weight, HIDDEN_SIZE * INTERMEDIATE_SIZE);
        ldg_prefetch_weights_l2(up_weight, HIDDEN_SIZE * INTERMEDIATE_SIZE);
        if (dbg_stage != nullptr) {
            unsigned long long t1 = clock64();
            dbg_stage[DBG_ATTN] = t1 - t0;
        }
        unsigned long long t0s = 0;
        if (dbg_stage != nullptr) t0s = clock64();
        grid.sync();
        if (dbg_stage != nullptr) {
            unsigned long long t1 = clock64();
            dbg_stage[DBG_ATTN_SYNC] = t1 - t0s;
        }
        // Wait for reducer blocks too.
        unsigned long long t0s2 = 0;
        if (dbg_stage != nullptr) t0s2 = clock64();
        grid.sync();
        if (dbg_stage != nullptr) {
            unsigned long long t1 = clock64();
            dbg_stage[DBG_ATTN_SYNC] += (t1 - t0s2);
        }
        return;
    }

    const int pos_per_group = (cache_len + groups - 1) / groups;
    const int pos_start = group * pos_per_group;
    const int pos_end = min(pos_start + pos_per_group, cache_len);

    const int kv_head = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
    const float* q_head = q + qh * HEAD_DIM;
    float* out_head = attn_out + qh * HEAD_DIM;

    // Shared memory for cross-warp reduction of online softmax within the block.
    __shared__ float s_max_score[LDG_NUM_WARPS];
    __shared__ float s_sum_exp[LDG_NUM_WARPS];
    __shared__ float s_out_acc[LDG_NUM_WARPS][HEAD_DIM];

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (pos_start < pos_end) {
        for (int pos = pos_start + warp_id; pos < pos_end; pos += LDG_NUM_WARPS) {
            const __nv_bfloat16* k_pos = k_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
            const __nv_bfloat16* v_pos = v_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

            float score = 0.0f;
            for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                score += q_head[d] * __bfloat162float(__ldg(k_pos + d));
            }
            score = ldg_warp_reduce_sum(score) * attn_scale;
            score = __shfl_sync(0xffffffff, score, 0);

            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_diff = expf(old_max - max_score);
            sum_exp = sum_exp * exp_diff + expf(score - max_score);

            float weight = expf(score - max_score);
            #pragma unroll
            for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
                out_acc[j] = out_acc[j] * exp_diff + weight * __bfloat162float(__ldg(v_pos + d));
            }
        }
    }

    if (lane_id == 0) {
        s_max_score[warp_id] = max_score;
        s_sum_exp[warp_id] = sum_exp;
    }
    #pragma unroll
    for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
        s_out_acc[warp_id][d] = out_acc[j];
    }
    __syncthreads();

    if (warp_id == 0) {
        float global_max = s_max_score[0];
        for (int w = 1; w < LDG_NUM_WARPS; w++) {
            if (s_max_score[w] > -INFINITY) {
                global_max = fmaxf(global_max, s_max_score[w]);
            }
        }

        float total_sum_exp = 0.0f;
        float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int w = 0; w < LDG_NUM_WARPS; w++) {
            if (s_max_score[w] > -INFINITY) {
                float scale = expf(s_max_score[w] - global_max);
                total_sum_exp += s_sum_exp[w] * scale;
                #pragma unroll
                for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
                    final_out[j] += s_out_acc[w][d] * scale;
                }
            }
        }

        if (lane_id == 0) {
            attn_partial_max[block_id] = global_max;
            attn_partial_sum[block_id] = total_sum_exp;
        }
        #pragma unroll
        for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
            attn_partial_out[block_id * HEAD_DIM + d] = final_out[j];
        }
    }
    __syncthreads();

    if (dbg_stage != nullptr) {
        unsigned long long t1 = clock64();
        dbg_stage[DBG_ATTN] = t1 - t0;
    }
    unsigned long long t0s = 0;
    if (dbg_stage != nullptr) t0s = clock64();
    grid.sync();
    if (dbg_stage != nullptr) {
        unsigned long long t1 = clock64();
        dbg_stage[DBG_ATTN_SYNC] = t1 - t0s;
    }

    // Reducer: one block per head (the group-0 block for that head).
    if (block_id == qh) {
        __shared__ float s_global_max;
        __shared__ float s_total_sum;

        if (threadIdx.x == 0) {
            float gmax = -INFINITY;
            for (int g = 0; g < effective_groups; g++) {
                int idx = qh + g * NUM_Q_HEADS;
                gmax = fmaxf(gmax, attn_partial_max[idx]);
            }
            s_global_max = gmax;

            float tsum = 0.0f;
            for (int g = 0; g < effective_groups; g++) {
                int idx = qh + g * NUM_Q_HEADS;
                float m = attn_partial_max[idx];
                float s = attn_partial_sum[idx];
                tsum += s * expf(m - gmax);
            }
            s_total_sum = tsum;
        }
        __syncthreads();

        float gmax = s_global_max;
        float tsum = s_total_sum;
        // Avoid NaNs on degenerate slices (shouldn't happen for cache_len>=1).
        float inv_tsum = (tsum > 0.0f) ? (1.0f / tsum) : 0.0f;

        for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
            float acc = 0.0f;
            for (int g = 0; g < effective_groups; g++) {
                int idx = qh + g * NUM_Q_HEADS;
                float scale = expf(attn_partial_max[idx] - gmax);
                acc += attn_partial_out[idx * HEAD_DIM + d] * scale;
            }
            out_head[d] = acc * inv_tsum;
        }
    }

    unsigned long long t0s2 = 0;
    if (dbg_stage != nullptr) t0s2 = clock64();
    grid.sync();
    if (dbg_stage != nullptr) {
        unsigned long long t1 = clock64();
        dbg_stage[DBG_ATTN_SYNC] += (t1 - t0s2);
    }
}
