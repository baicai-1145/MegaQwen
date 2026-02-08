/**
 * Split decode backend (host-orchestrated, non-cooperative) for Qwen3 0.6B dims.
 *
 * Goal (Step 5.1 in docs/ACCEL_GAPS.md):
 * - Remove cooperative grid.sync() stalls by switching decode to a standard
 *   multi-kernel pipeline (cuBLAS GEMM for linears + custom cache-attn).
 * - Make it CUDA-graph-capture friendly by reading `position` on device via
 *   `position_ptr` (no per-step host parameter patching needed).
 *
 * Notes:
 * - We intentionally reuse existing BF16 prefill kernels where possible (DRY).
 * - Attention is decode-only: single query, KV from cache (GQA).
 */

#include "config.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// =============================================================================
// Reuse kernels compiled in fused_prefill.cu (must match signatures)
// =============================================================================
struct PrefillLayerWeights;  // defined in fused_prefill.cu (same extension TU)

__global__ void prefill_silu_mul_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int n
);

__global__ void prefill_lm_head_phase1(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ lm_head_weight,
    float* __restrict__ block_max_vals,
    int* __restrict__ block_max_idxs
);

__global__ void prefill_lm_head_phase2(
    const float* __restrict__ block_max_vals,
    const int* __restrict__ block_max_idxs,
    int* __restrict__ output_token,
    int num_blocks
);

// =============================================================================
// Decode-only kernels: QK norm + RoPE + KV cache, and attention over cache.
// =============================================================================

// Specialized for a single query token (seq_len=1).
// Grid: (max(num_q_heads, num_kv_heads)), Block: 128
__global__ void decode_qk_norm_rope_cache_kernel(
    __nv_bfloat16* __restrict__ q,              // [num_q_heads, head_dim]
    __nv_bfloat16* __restrict__ k,              // [num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ v,        // [num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ q_norm_weight,  // [head_dim]
    const __nv_bfloat16* __restrict__ k_norm_weight,  // [head_dim]
    const __nv_bfloat16* __restrict__ cos_table,      // [max_seq, head_dim]
    const __nv_bfloat16* __restrict__ sin_table,      // [max_seq, head_dim]
    __nv_bfloat16* __restrict__ k_cache,  // [num_kv_heads, max_seq_len, head_dim]
    __nv_bfloat16* __restrict__ v_cache,  // [num_kv_heads, max_seq_len, head_dim]
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    const int* __restrict__ position_ptr  // device scalar
) {
    int head = (int)blockIdx.x;
    int heads = (num_q_heads > num_kv_heads) ? num_q_heads : num_kv_heads;
    if (head >= heads) return;

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // Absolute position for RoPE and cache.
    int position = *position_ptr;
    const __nv_bfloat16* cos_pos = cos_table + position * head_dim;
    const __nv_bfloat16* sin_pos = sin_table + position * head_dim;

    __shared__ float smem_reduce[8];
    __shared__ float smem_normed[HEAD_DIM];

    // Q head: normalize + rope in-place.
    if (head < num_q_heads) {
        __nv_bfloat16* q_head = q + head * head_dim;
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float vv = __bfloat162float(q_head[i]);
            sum_sq += vv * vv;
        }
        sum_sq = warp_reduce_sum(sum_sq);
        if (lane_id == 0 && warp_id < 8) smem_reduce[warp_id] = sum_sq;
        __syncthreads();
        if (warp_id == 0) {
            float sum = (lane_id < (blockDim.x / WARP_SIZE)) ? smem_reduce[lane_id] : 0.0f;
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) smem_reduce[0] = rsqrtf(sum / float(head_dim) + RMS_NORM_EPS);
        }
        __syncthreads();
        float scale = smem_reduce[0];

        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            smem_normed[i] = __bfloat162float(q_head[i]) * scale * __bfloat162float(q_norm_weight[i]);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float cos_v = __bfloat162float(cos_pos[i]);
            float sin_v = __bfloat162float(sin_pos[i]);
            int pair_idx = (i < head_dim / 2) ? (i + head_dim / 2) : (i - head_dim / 2);
            float pair_v = smem_normed[pair_idx];
            float out;
            if (i < head_dim / 2) out = smem_normed[i] * cos_v - pair_v * sin_v;
            else out = pair_v * sin_v + smem_normed[i] * cos_v;
            q_head[i] = __float2bfloat16(out);
        }
    }
    __syncthreads();

    // K head: normalize + rope, write to cache. V: write to cache as-is.
    if (head < num_kv_heads) {
        __nv_bfloat16* k_head = k + head * head_dim;
        const __nv_bfloat16* v_head = v + head * head_dim;

        __nv_bfloat16* k_cache_head = k_cache + head * max_seq_len * head_dim + position * head_dim;
        __nv_bfloat16* v_cache_head = v_cache + head * max_seq_len * head_dim + position * head_dim;

        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float vv = __bfloat162float(k_head[i]);
            sum_sq += vv * vv;
        }
        sum_sq = warp_reduce_sum(sum_sq);
        if (lane_id == 0 && warp_id < 8) smem_reduce[warp_id] = sum_sq;
        __syncthreads();
        if (warp_id == 0) {
            float sum = (lane_id < (blockDim.x / WARP_SIZE)) ? smem_reduce[lane_id] : 0.0f;
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) smem_reduce[0] = rsqrtf(sum / float(head_dim) + RMS_NORM_EPS);
        }
        __syncthreads();
        float scale = smem_reduce[0];

        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            smem_normed[i] = __bfloat162float(k_head[i]) * scale * __bfloat162float(k_norm_weight[i]);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float cos_v = __bfloat162float(cos_pos[i]);
            float sin_v = __bfloat162float(sin_pos[i]);
            int pair_idx = (i < head_dim / 2) ? (i + head_dim / 2) : (i - head_dim / 2);
            float pair_v = smem_normed[pair_idx];
            float out;
            if (i < head_dim / 2) out = smem_normed[i] * cos_v - pair_v * sin_v;
            else out = pair_v * sin_v + smem_normed[i] * cos_v;
            __nv_bfloat16 out_bf16 = __float2bfloat16(out);
            k_head[i] = out_bf16;
            k_cache_head[i] = out_bf16;
            v_cache_head[i] = v_head[i];
        }
    }
}

// Decode attention over KV cache (single query token).
// Legacy kernel: each warp handles one q head; low occupancy for large GPUs.
__global__ void decode_attention_cache_legacy_kernel(
    const __nv_bfloat16* __restrict__ q,             // [num_q_heads, head_dim] (already norm+rope)
    const __nv_bfloat16* __restrict__ k_cache,       // [num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,       // [num_kv_heads, max_seq_len, head_dim]
    __nv_bfloat16* __restrict__ out,                 // [num_q_heads, head_dim]
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    float attn_scale,
    const int* __restrict__ position_ptr             // device scalar
) {
    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_idx >= num_q_heads) return;

    int q_head = warp_idx;
    int kv_head = q_head / (num_q_heads / num_kv_heads);

    int position = *position_ptr;
    int cache_len = position + 1;
    if (cache_len > max_seq_len) cache_len = max_seq_len;

    const __nv_bfloat16* q_vec = q + q_head * head_dim;
    __nv_bfloat16* out_vec = out + q_head * head_dim;

    constexpr int ELEMS_PER_LANE = HEAD_DIM / WARP_SIZE;  // 4 for 128/32
    float q_local[ELEMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; i++) {
        int d = lane_id + i * WARP_SIZE;
        q_local[i] = __bfloat162float(q_vec[d]);
    }

    float m = -INFINITY;
    float s = 0.0f;
    float out_local[ELEMS_PER_LANE] = {0.f, 0.f, 0.f, 0.f};

    const __nv_bfloat16* k_base = k_cache + kv_head * max_seq_len * head_dim;
    const __nv_bfloat16* v_base = v_cache + kv_head * max_seq_len * head_dim;

    for (int t = 0; t < cache_len; t++) {
        const __nv_bfloat16* k_vec = k_base + t * head_dim;
        const __nv_bfloat16* v_vec = v_base + t * head_dim;

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_LANE; i++) {
            int d = lane_id + i * WARP_SIZE;
            dot += q_local[i] * __bfloat162float(k_vec[d]);
        }
        dot = warp_reduce_sum(dot) * attn_scale;
        float score = __shfl_sync(0xffffffff, dot, 0);

        float m_new = fmaxf(m, score);
        float old_scale = expf(m - m_new);
        float new_scale = expf(score - m_new);
        s = s * old_scale + new_scale;

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_LANE; i++) {
            int d = lane_id + i * WARP_SIZE;
            float vv = __bfloat162float(v_vec[d]);
            out_local[i] = out_local[i] * old_scale + vv * new_scale;
        }
        m = m_new;
    }

    float inv_s = 1.0f / s;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; i++) {
        int d = lane_id + i * WARP_SIZE;
        out_vec[d] = __float2bfloat16(out_local[i] * inv_s);
    }
}

// Split-K decode attention over KV cache (single query token).
// One block = one q_head; warps in a block split sequence positions.
// This increases parallelism and improves utilization on wide GPUs (e.g. AD102).
__global__ void decode_attention_cache_splitk_kernel(
    const __nv_bfloat16* __restrict__ q,             // [num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_cache,       // [num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,       // [num_kv_heads, max_seq_len, head_dim]
    __nv_bfloat16* __restrict__ out,                 // [num_q_heads, head_dim]
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    float attn_scale,
    int warps_per_head,
    const int* __restrict__ position_ptr             // device scalar
) {
    constexpr int kMaxWarps = 8;
    constexpr int kElemsPerLane = HEAD_DIM / WARP_SIZE;  // 4 (128 / 32)
    __shared__ float sm_m[kMaxWarps];
    __shared__ float sm_s[kMaxWarps];
    __shared__ float sm_scale[kMaxWarps];
    __shared__ float sm_out[kMaxWarps * HEAD_DIM];
    __shared__ float sm_global_s;
    __shared__ int sm_active_warps;

    int q_head = (int)blockIdx.x;
    if (q_head >= num_q_heads) return;

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (warp_id >= warps_per_head) return;

    int kv_ratio = (num_kv_heads > 0) ? (num_q_heads / num_kv_heads) : 1;
    if (kv_ratio <= 0) kv_ratio = 1;
    int kv_head = q_head / kv_ratio;
    if (kv_head >= num_kv_heads) kv_head = num_kv_heads - 1;

    int position = *position_ptr;
    int cache_len = position + 1;
    if (cache_len > max_seq_len) cache_len = max_seq_len;
    int active_warps = warps_per_head;
    if (active_warps > kMaxWarps) active_warps = kMaxWarps;
    if (active_warps > cache_len && cache_len > 0) active_warps = cache_len;
    if (active_warps <= 0) active_warps = 1;

    if (threadIdx.x == 0) sm_active_warps = active_warps;
    __syncthreads();
    active_warps = sm_active_warps;
    if (warp_id >= active_warps) return;

    const __nv_bfloat16* q_vec = q + q_head * head_dim;
    __nv_bfloat16* out_vec = out + q_head * head_dim;

    float q_local[kElemsPerLane];
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        int d = lane_id + i * WARP_SIZE;
        q_local[i] = __bfloat162float(q_vec[d]);
    }

    float m = -INFINITY;
    float s = 0.0f;
    float out_local[kElemsPerLane] = {0.f, 0.f, 0.f, 0.f};

    const __nv_bfloat16* k_base = k_cache + kv_head * max_seq_len * head_dim;
    const __nv_bfloat16* v_base = v_cache + kv_head * max_seq_len * head_dim;

    for (int t = warp_id; t < cache_len; t += active_warps) {
        const __nv_bfloat16* k_vec = k_base + t * head_dim;
        const __nv_bfloat16* v_vec = v_base + t * head_dim;

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < kElemsPerLane; i++) {
            int d = lane_id + i * WARP_SIZE;
            dot += q_local[i] * __bfloat162float(k_vec[d]);
        }
        dot = warp_reduce_sum(dot) * attn_scale;
        float score = __shfl_sync(0xffffffff, dot, 0);

        float m_new = fmaxf(m, score);
        float old_scale = expf(m - m_new);
        float new_scale = expf(score - m_new);
        s = s * old_scale + new_scale;

        #pragma unroll
        for (int i = 0; i < kElemsPerLane; i++) {
            int d = lane_id + i * WARP_SIZE;
            float vv = __bfloat162float(v_vec[d]);
            out_local[i] = out_local[i] * old_scale + vv * new_scale;
        }
        m = m_new;
    }

    if (lane_id == 0) {
        sm_m[warp_id] = m;
        sm_s[warp_id] = s;
    }
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        int d = lane_id + i * WARP_SIZE;
        sm_out[warp_id * HEAD_DIM + d] = out_local[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float g_m = -INFINITY;
        for (int w = 0; w < active_warps; w++) g_m = fmaxf(g_m, sm_m[w]);

        float g_s = 0.0f;
        for (int w = 0; w < active_warps; w++) {
            float scale = (sm_s[w] > 0.0f) ? expf(sm_m[w] - g_m) : 0.0f;
            sm_scale[w] = scale;
            g_s += sm_s[w] * scale;
        }
        sm_global_s = g_s;
    }
    __syncthreads();

    float inv_s = (sm_global_s > 0.0f) ? (1.0f / sm_global_s) : 0.0f;
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        int d = lane_id + i * WARP_SIZE;
        float acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < kMaxWarps; w++) {
            if (w < active_warps) acc += sm_out[w * HEAD_DIM + d] * sm_scale[w];
        }
        out_vec[d] = __float2bfloat16(acc * inv_s);
    }
}

// Split-K v2 phase-1: each block handles one (q_head, chunk) and produces
// one local softmax summary (m, s, out[head_dim]).
__global__ void decode_attention_cache_splitk_phase1_kernel(
    const __nv_bfloat16* __restrict__ q,             // [num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_cache,       // [num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,       // [num_kv_heads, max_seq_len, head_dim]
    float* __restrict__ partial_m,                   // [max_chunks, num_q_heads]
    float* __restrict__ partial_s,                   // [max_chunks, num_q_heads]
    float* __restrict__ partial_out,                 // [max_chunks, num_q_heads, head_dim]
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    float attn_scale,
    int warps_per_head,
    int chunk_size,
    int max_chunks,
    const int* __restrict__ position_ptr
) {
    constexpr int kMaxWarps = 8;
    constexpr int kElemsPerLane = HEAD_DIM / WARP_SIZE;
    __shared__ float sm_m[kMaxWarps];
    __shared__ float sm_s[kMaxWarps];
    __shared__ float sm_scale[kMaxWarps];
    __shared__ float sm_out[kMaxWarps * HEAD_DIM];
    __shared__ float sm_global_m;
    __shared__ float sm_global_s;
    __shared__ int sm_active_warps;

    if (head_dim != HEAD_DIM) return;
    if (chunk_size <= 0 || max_chunks <= 0) return;

    int q_head = (int)(blockIdx.x % num_q_heads);
    int chunk_idx = (int)(blockIdx.x / num_q_heads);
    if (chunk_idx >= max_chunks) return;

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    int position = *position_ptr;
    int cache_len = position + 1;
    if (cache_len > max_seq_len) cache_len = max_seq_len;
    int t_start = chunk_idx * chunk_size;
    if (t_start >= cache_len) return;
    int t_end = t_start + chunk_size;
    if (t_end > cache_len) t_end = cache_len;

    int active_warps = warps_per_head;
    if (active_warps > kMaxWarps) active_warps = kMaxWarps;
    int chunk_tokens = t_end - t_start;
    if (active_warps > chunk_tokens) active_warps = chunk_tokens;
    if (active_warps <= 0) active_warps = 1;
    if (threadIdx.x == 0) sm_active_warps = active_warps;
    __syncthreads();
    active_warps = sm_active_warps;

    int kv_ratio = (num_kv_heads > 0) ? (num_q_heads / num_kv_heads) : 1;
    if (kv_ratio <= 0) kv_ratio = 1;
    int kv_head = q_head / kv_ratio;
    if (kv_head >= num_kv_heads) kv_head = num_kv_heads - 1;

    const __nv_bfloat16* q_vec = q + q_head * head_dim;
    float q_local[kElemsPerLane];
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        int d = lane_id + i * WARP_SIZE;
        q_local[i] = __bfloat162float(q_vec[d]);
    }

    float m = -INFINITY;
    float s = 0.0f;
    float out_local[kElemsPerLane] = {0.f, 0.f, 0.f, 0.f};

    const __nv_bfloat16* k_base = k_cache + kv_head * max_seq_len * head_dim;
    const __nv_bfloat16* v_base = v_cache + kv_head * max_seq_len * head_dim;

    if (warp_id < active_warps) {
        for (int t = t_start + warp_id; t < t_end; t += active_warps) {
            const __nv_bfloat16* k_vec = k_base + t * head_dim;
            const __nv_bfloat16* v_vec = v_base + t * head_dim;

            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < kElemsPerLane; i++) {
                int d = lane_id + i * WARP_SIZE;
                dot += q_local[i] * __bfloat162float(k_vec[d]);
            }
            dot = warp_reduce_sum(dot) * attn_scale;
            float score = __shfl_sync(0xffffffff, dot, 0);

            float m_new = fmaxf(m, score);
            float old_scale = expf(m - m_new);
            float new_scale = expf(score - m_new);
            s = s * old_scale + new_scale;

            #pragma unroll
            for (int i = 0; i < kElemsPerLane; i++) {
                int d = lane_id + i * WARP_SIZE;
                float vv = __bfloat162float(v_vec[d]);
                out_local[i] = out_local[i] * old_scale + vv * new_scale;
            }
            m = m_new;
        }
    }

    if (lane_id == 0) {
        sm_m[warp_id] = (warp_id < active_warps) ? m : -INFINITY;
        sm_s[warp_id] = (warp_id < active_warps) ? s : 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        int d = lane_id + i * WARP_SIZE;
        sm_out[warp_id * HEAD_DIM + d] = (warp_id < active_warps) ? out_local[i] : 0.0f;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float g_m = -INFINITY;
        for (int w = 0; w < active_warps; w++) g_m = fmaxf(g_m, sm_m[w]);
        float g_s = 0.0f;
        for (int w = 0; w < active_warps; w++) {
            float scale = (sm_s[w] > 0.0f) ? expf(sm_m[w] - g_m) : 0.0f;
            sm_scale[w] = scale;
            g_s += sm_s[w] * scale;
        }
        sm_global_m = g_m;
        sm_global_s = g_s;
    }
    __syncthreads();

    int idx = chunk_idx * num_q_heads + q_head;
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        int d = lane_id + i * WARP_SIZE;
        float acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < kMaxWarps; w++) {
            if (w < active_warps) acc += sm_out[w * HEAD_DIM + d] * sm_scale[w];
        }
        partial_out[idx * HEAD_DIM + d] = acc;
    }
    if (threadIdx.x == 0) {
        partial_m[idx] = sm_global_m;
        partial_s[idx] = sm_global_s;
    }
}

// Split-K v2 phase-2: reduce all chunk summaries of one q_head.
__global__ void decode_attention_cache_splitk_phase2_kernel(
    const float* __restrict__ partial_m,      // [max_chunks, num_q_heads]
    const float* __restrict__ partial_s,      // [max_chunks, num_q_heads]
    const float* __restrict__ partial_out,    // [max_chunks, num_q_heads, head_dim]
    __nv_bfloat16* __restrict__ out,          // [num_q_heads, head_dim]
    int num_q_heads,
    int head_dim,
    int chunk_size,
    int max_chunks,
    const int* __restrict__ position_ptr
) {
    if (head_dim != HEAD_DIM) return;
    int q_head = (int)blockIdx.x;
    if (q_head >= num_q_heads) return;
    if (chunk_size <= 0 || max_chunks <= 0) return;

    __shared__ float sm_m;
    __shared__ float sm_inv_s;
    __shared__ int sm_num_chunks;

    int position = *position_ptr;
    int cache_len = position + 1;
    int num_chunks = (cache_len + chunk_size - 1) / chunk_size;
    if (num_chunks > max_chunks) num_chunks = max_chunks;
    if (num_chunks < 1) num_chunks = 1;

    if (threadIdx.x == 0) {
        sm_num_chunks = num_chunks;
        float g_m = -INFINITY;
        for (int c = 0; c < num_chunks; c++) {
            int idx = c * num_q_heads + q_head;
            g_m = fmaxf(g_m, partial_m[idx]);
        }
        float g_s = 0.0f;
        for (int c = 0; c < num_chunks; c++) {
            int idx = c * num_q_heads + q_head;
            float scale = expf(partial_m[idx] - g_m);
            g_s += partial_s[idx] * scale;
        }
        sm_m = g_m;
        sm_inv_s = (g_s > 0.0f) ? (1.0f / g_s) : 0.0f;
    }
    __syncthreads();

    __nv_bfloat16* out_vec = out + q_head * HEAD_DIM;
    int num_chunks_local = sm_num_chunks;
    float g_m = sm_m;
    float inv_s = sm_inv_s;

    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        float acc = 0.0f;
        for (int c = 0; c < num_chunks_local; c++) {
            int idx = c * num_q_heads + q_head;
            float scale = expf(partial_m[idx] - g_m);
            acc += partial_out[idx * HEAD_DIM + d] * scale;
        }
        out_vec[d] = __float2bfloat16(acc * inv_s);
    }
}

// =============================================================================
// Decode helper kernels (FP32 activation chain for stability)
// =============================================================================

__global__ void split_embed_to_hidden_f32_kernel(
    const int* __restrict__ input_token_id,
    const __nv_bfloat16* __restrict__ embed_weight,
    float* __restrict__ hidden_f32
) {
    int tok = input_token_id[0];
    const __nv_bfloat16* row = embed_weight + tok * HIDDEN_SIZE;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += blockDim.x) {
        hidden_f32[i] = __bfloat162float(row[i]);
    }
}

__global__ void split_rmsnorm_f32_to_bf16_kernel(
    const float* __restrict__ input_f32,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output_bf16,
    float* __restrict__ residual_f32
) {
    constexpr int kBlock = 256;
    constexpr int kWarps = kBlock / WARP_SIZE;
    __shared__ float smem_reduce[kWarps];

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += kBlock) {
        float v = input_f32[i];
        residual_f32[i] = v;
        local_sum_sq += v * v;
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < kWarps) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_NORM_EPS);
        }
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += kBlock) {
        float w = __bfloat162float(weight[i]);
        output_bf16[i] = __float2bfloat16(residual_f32[i] * rstd * w);
    }
}

__global__ void split_residual_add_bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ input_bf16,
    const float* __restrict__ residual_f32,
    float* __restrict__ output_f32,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output_f32[idx] = __bfloat162float(input_bf16[idx]) + residual_f32[idx];
    }
}

__global__ void split_final_norm_f32_to_bf16_kernel(
    const float* __restrict__ hidden_f32,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ final_hidden_bf16
) {
    constexpr int kBlock = 256;
    constexpr int kWarps = kBlock / WARP_SIZE;
    __shared__ float smem_reduce[kWarps];

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += kBlock) {
        float v = hidden_f32[i];
        local_sum_sq += v * v;
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < kWarps) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_NORM_EPS);
        }
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += kBlock) {
        float w = __bfloat162float(weight[i]);
        final_hidden_bf16[i] = __float2bfloat16(hidden_f32[i] * rstd * w);
    }
}

// Experimental FFN fused tail:
// down_proj(matvec) + residual_add in one kernel.
// This avoids writing `down_out_bf16` and launching an extra residual kernel.
__global__ void split_downproj_residual_fused_kernel(
    const __nv_bfloat16* __restrict__ mlp_intermediate_bf16,  // [INTERMEDIATE_SIZE]
    const __nv_bfloat16* __restrict__ down_weight,            // [HIDDEN_SIZE, INTERMEDIATE_SIZE] (row-major)
    const float* __restrict__ residual_f32,                   // [HIDDEN_SIZE]
    float* __restrict__ hidden_f32                            // [HIDDEN_SIZE]
) {
    int out_idx = (int)blockIdx.x;
    if (out_idx >= HIDDEN_SIZE) return;

    constexpr int kBlock = 256;
    constexpr int kWarps = kBlock / WARP_SIZE;
    __shared__ float smem_reduce[kWarps];

    float local_sum = 0.0f;
    const __nv_bfloat16* w_row = down_weight + out_idx * INTERMEDIATE_SIZE;
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += kBlock) {
        float x = __bfloat162float(mlp_intermediate_bf16[i]);
        float w = __bfloat162float(w_row[i]);
        local_sum += x * w;
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < kWarps) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            hidden_f32[out_idx] = sum + residual_f32[out_idx];
        }
    }
}

// Experimental deeper FFN fusion:
// consume packed gateup output directly and fuse:
//   silu(gate) * up -> down_proj(matvec) -> residual_add
// This avoids writing/reading `mlp_intermediate_bf16` and skips a separate silu kernel.
__global__ void split_silu_downproj_residual_fused_kernel(
    const __nv_bfloat16* __restrict__ gateup_out_bf16,  // [2 * INTERMEDIATE_SIZE], [gate | up]
    const __nv_bfloat16* __restrict__ down_weight,      // [HIDDEN_SIZE, INTERMEDIATE_SIZE] (row-major)
    const float* __restrict__ residual_f32,             // [HIDDEN_SIZE]
    float* __restrict__ hidden_f32                      // [HIDDEN_SIZE]
) {
    int out_idx = (int)blockIdx.x;
    if (out_idx >= HIDDEN_SIZE) return;

    constexpr int kBlock = 256;
    constexpr int kWarps = kBlock / WARP_SIZE;
    __shared__ float smem_reduce[kWarps];

    const __nv_bfloat16* gate = gateup_out_bf16;
    const __nv_bfloat16* up = gateup_out_bf16 + INTERMEDIATE_SIZE;
    const __nv_bfloat16* w_row = down_weight + out_idx * INTERMEDIATE_SIZE;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += kBlock) {
        float gate_v = __bfloat162float(gate[i]);
        float up_v = __bfloat162float(up[i]);
        float silu_v = gate_v / (1.0f + expf(-gate_v));
        float w = __bfloat162float(w_row[i]);
        local_sum += (silu_v * up_v) * w;
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < kWarps) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            hidden_f32[out_idx] = sum + residual_f32[out_idx];
        }
    }
}

// Runtime W4A16 matvec (NF4/FP4 compatible codebook+scales):
// - packed_w4: 2 int4 values per byte, row-major over [rows, cols]
// - scales: per-64-element block scale (linear indexing)
// - codebook: 16-entry dequant table
__global__ void split_w4_matvec_bf16_out_kernel(
    const uint8_t* __restrict__ packed_w4,
    const float* __restrict__ scales,
    const float* __restrict__ codebook,
    const __nv_bfloat16* __restrict__ x_bf16,
    __nv_bfloat16* __restrict__ y_bf16,
    int rows,
    int cols
) {
    int row = (int)blockIdx.x;
    if (row >= rows) return;

    constexpr int kBlock = 256;
    constexpr int kWarps = kBlock / WARP_SIZE;
    __shared__ float smem_reduce[kWarps];
    int codebook_base = (rows == (INTERMEDIATE_SIZE + INTERMEDIATE_SIZE) && row >= INTERMEDIATE_SIZE) ? 16 : 0;

    float local_sum = 0.0f;
    int64_t row_base = (int64_t)row * (int64_t)cols;
    for (int col = threadIdx.x; col < cols; col += kBlock) {
        int64_t linear = row_base + (int64_t)col;
        uint8_t pack = packed_w4[linear >> 1];
        int q = (linear & 1) ? int(pack & 0x0F) : int((pack >> 4) & 0x0F);
        float w = codebook[codebook_base + q] * scales[linear >> 6];
        float xv = __bfloat162float(x_bf16[col]);
        local_sum += w * xv;
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < kWarps) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) y_bf16[row] = __float2bfloat16(sum);
    }
}

__global__ void split_w4_qkv_matvec_bf16_out_kernel(
    const uint8_t* __restrict__ q_packed_w4,
    const float* __restrict__ q_scales,
    const float* __restrict__ q_codebook,
    const uint8_t* __restrict__ k_packed_w4,
    const float* __restrict__ k_scales,
    const float* __restrict__ k_codebook,
    const uint8_t* __restrict__ v_packed_w4,
    const float* __restrict__ v_scales,
    const float* __restrict__ v_codebook,
    const __nv_bfloat16* __restrict__ x_bf16,
    __nv_bfloat16* __restrict__ y_bf16,
    int cols
) {
    int out_row = (int)blockIdx.x;
    if (out_row >= (Q_SIZE + KV_SIZE + KV_SIZE)) return;

    const uint8_t* packed_w4 = nullptr;
    const float* scales = nullptr;
    const float* codebook = nullptr;
    int local_row = 0;
    if (out_row < Q_SIZE) {
        packed_w4 = q_packed_w4;
        scales = q_scales;
        codebook = q_codebook;
        local_row = out_row;
    } else if (out_row < (Q_SIZE + KV_SIZE)) {
        packed_w4 = k_packed_w4;
        scales = k_scales;
        codebook = k_codebook;
        local_row = out_row - Q_SIZE;
    } else {
        packed_w4 = v_packed_w4;
        scales = v_scales;
        codebook = v_codebook;
        local_row = out_row - Q_SIZE - KV_SIZE;
    }

    constexpr int kBlock = 256;
    constexpr int kWarps = kBlock / WARP_SIZE;
    __shared__ float smem_reduce[kWarps];

    float local_sum = 0.0f;
    int64_t row_base = (int64_t)local_row * (int64_t)cols;
    for (int col = threadIdx.x; col < cols; col += kBlock) {
        int64_t linear = row_base + (int64_t)col;
        uint8_t pack = packed_w4[linear >> 1];
        int q = (linear & 1) ? int(pack & 0x0F) : int((pack >> 4) & 0x0F);
        float weight = codebook[q] * scales[linear >> 6];
        float x_value = __bfloat162float(x_bf16[col]);
        local_sum += weight * x_value;
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < kWarps) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) y_bf16[out_row] = __float2bfloat16(sum);
    }
}

__global__ void split_w4_downproj_residual_kernel(
    const uint8_t* __restrict__ packed_w4,
    const float* __restrict__ scales,
    const float* __restrict__ codebook,
    const __nv_bfloat16* __restrict__ x_bf16,   // [INTERMEDIATE_SIZE]
    const float* __restrict__ residual_f32,      // [HIDDEN_SIZE]
    float* __restrict__ hidden_f32               // [HIDDEN_SIZE]
) {
    int row = (int)blockIdx.x;
    if (row >= HIDDEN_SIZE) return;

    constexpr int kBlock = 256;
    constexpr int kWarps = kBlock / WARP_SIZE;
    __shared__ float smem_reduce[kWarps];

    float local_sum = 0.0f;
    int64_t row_base = (int64_t)row * (int64_t)INTERMEDIATE_SIZE;
    for (int col = threadIdx.x; col < INTERMEDIATE_SIZE; col += kBlock) {
        int64_t linear = row_base + (int64_t)col;
        uint8_t pack = packed_w4[linear >> 1];
        int q = (linear & 1) ? int(pack & 0x0F) : int((pack >> 4) & 0x0F);
        float w = codebook[q] * scales[linear >> 6];
        float xv = __bfloat162float(x_bf16[col]);
        local_sum += w * xv;
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < kWarps) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) hidden_f32[row] = sum + residual_f32[row];
    }
}

// =============================================================================
// Host entry: one split decode step (single token)
// =============================================================================

#include "split_decode_gemm_host.inl"
