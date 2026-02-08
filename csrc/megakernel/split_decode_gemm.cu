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
#include <cuda_fp8.h>
#include <cuda_fp8.hpp>
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

__device__ __forceinline__ int kv_cache_physical_token(
    int logical_token,
    const int* __restrict__ kv_block_table,
    int kv_block_size
) {
    if (kv_block_table == nullptr || kv_block_size <= 0) return logical_token;
    int logical_block = logical_token / kv_block_size;
    int in_block = logical_token - logical_block * kv_block_size;
    int physical_block = kv_block_table[logical_block];
    return physical_block * kv_block_size + in_block;
}

__device__ __forceinline__ float kv_fp8_max_abs_scale(float amax) {
    constexpr float kFp8E4M3Max = 448.0f;
    if (amax < 1.0e-8f) return 1.0f;
    return amax / kFp8E4M3Max;
}

__device__ __forceinline__ float kv_cache_load_elem(
    const __nv_bfloat16* __restrict__ bf16_vec,
    const __nv_fp8_e4m3* __restrict__ fp8_vec,
    int d,
    float scale,
    int fp8_enabled
) {
    if (fp8_enabled) {
        return static_cast<float>(fp8_vec[d]) * scale;
    }
    return __bfloat162float(bf16_vec[d]);
}

// Quantize existing BF16 KV cache into FP8 cache for [start_pos, end_pos).
// - k/v scale shape: [num_layers, num_kv_heads] (one scale per layer/head)
// - k/v fp8 shape:   [num_layers, num_kv_heads, max_seq_len, head_dim]
__global__ void split_quantize_kv_cache_fp8_kernel(
    const __nv_bfloat16* __restrict__ k_cache_bf16,
    const __nv_bfloat16* __restrict__ v_cache_bf16,
    __nv_fp8_e4m3* __restrict__ k_cache_fp8,
    __nv_fp8_e4m3* __restrict__ v_cache_fp8,
    float* __restrict__ k_scale_cache,
    float* __restrict__ v_scale_cache,
    int num_layers,
    int num_kv_heads,
    int max_seq_len,
    int head_dim,
    int start_pos,
    int end_pos
) {
    if (head_dim != HEAD_DIM) return;
    int span = end_pos - start_pos;
    if (span <= 0) return;
    int idx = (int)blockIdx.x;
    int total = num_layers * num_kv_heads;
    if (idx >= total) return;

    int kv_head = idx % num_kv_heads;
    int layer = idx / num_kv_heads;
    int base = (layer * num_kv_heads + kv_head) * max_seq_len * HEAD_DIM;
    float* k_scale_ptr = k_scale_cache + (layer * num_kv_heads + kv_head);
    float* v_scale_ptr = v_scale_cache + (layer * num_kv_heads + kv_head);

    float local_k_max = 0.0f;
    float local_v_max = 0.0f;
    for (int t = start_pos; t < end_pos; t++) {
        int row = base + t * HEAD_DIM;
        for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
            float kf = fabsf(__bfloat162float(k_cache_bf16[row + d]));
            float vf = fabsf(__bfloat162float(v_cache_bf16[row + d]));
            if (kf > local_k_max) local_k_max = kf;
            if (vf > local_v_max) local_v_max = vf;
        }
    }
    local_k_max = warp_reduce_max(local_k_max);
    local_v_max = warp_reduce_max(local_v_max);

    __shared__ float sm_k_max[8];
    __shared__ float sm_v_max[8];
    __shared__ float sm_k_scale;
    __shared__ float sm_v_scale;
    int lane = threadIdx.x % WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        sm_k_max[warp] = local_k_max;
        sm_v_max[warp] = local_v_max;
    }
    __syncthreads();
    if (warp == 0) {
        float kmax = (lane < (blockDim.x / WARP_SIZE)) ? sm_k_max[lane] : 0.0f;
        float vmax = (lane < (blockDim.x / WARP_SIZE)) ? sm_v_max[lane] : 0.0f;
        kmax = warp_reduce_max(kmax);
        vmax = warp_reduce_max(vmax);
        if (lane == 0) {
            sm_k_scale = kv_fp8_max_abs_scale(kmax);
            sm_v_scale = kv_fp8_max_abs_scale(vmax);
            *k_scale_ptr = sm_k_scale;
            *v_scale_ptr = sm_v_scale;
        }
    }
    __syncthreads();

    float inv_k = 1.0f / sm_k_scale;
    float inv_v = 1.0f / sm_v_scale;
    for (int t = start_pos; t < end_pos; t++) {
        int row = base + t * HEAD_DIM;
        for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
            float kf = __bfloat162float(k_cache_bf16[row + d]) * inv_k;
            float vf = __bfloat162float(v_cache_bf16[row + d]) * inv_v;
            k_cache_fp8[row + d] = __nv_fp8_e4m3(kf);
            v_cache_fp8[row + d] = __nv_fp8_e4m3(vf);
        }
    }
}

extern "C" void launch_split_quantize_kv_cache_fp8(
    const void* k_cache_bf16,
    const void* v_cache_bf16,
    void* k_cache_fp8,
    void* v_cache_fp8,
    void* k_scale_cache,
    void* v_scale_cache,
    int num_layers,
    int num_kv_heads,
    int max_seq_len,
    int head_dim,
    int start_pos,
    int end_pos,
    cudaStream_t stream
) {
    if (k_cache_bf16 == nullptr || v_cache_bf16 == nullptr ||
        k_cache_fp8 == nullptr || v_cache_fp8 == nullptr ||
        k_scale_cache == nullptr || v_scale_cache == nullptr) {
        return;
    }
    int span = end_pos - start_pos;
    if (span <= 0) return;
    int blocks = num_layers * num_kv_heads;
    if (blocks <= 0) return;
    split_quantize_kv_cache_fp8_kernel<<<blocks, 128, 0, stream>>>(
        (const __nv_bfloat16*)k_cache_bf16,
        (const __nv_bfloat16*)v_cache_bf16,
        (__nv_fp8_e4m3*)k_cache_fp8,
        (__nv_fp8_e4m3*)v_cache_fp8,
        (float*)k_scale_cache,
        (float*)v_scale_cache,
        num_layers,
        num_kv_heads,
        max_seq_len,
        head_dim,
        start_pos,
        end_pos
    );
}

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
    __nv_fp8_e4m3* __restrict__ k_cache_fp8,  // optional [num_kv_heads, max_seq_len, head_dim]
    __nv_fp8_e4m3* __restrict__ v_cache_fp8,  // optional [num_kv_heads, max_seq_len, head_dim]
    float* __restrict__ k_scale_cache,        // optional [num_kv_heads]
    float* __restrict__ v_scale_cache,        // optional [num_kv_heads]
    int kv_fp8_enabled,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    const int* __restrict__ kv_block_table,
    int kv_block_size,
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

        int position_physical = kv_cache_physical_token(position, kv_block_table, kv_block_size);
        if (position_physical >= max_seq_len) return;
        __nv_bfloat16* k_cache_head = k_cache + head * max_seq_len * head_dim + position_physical * head_dim;
        __nv_bfloat16* v_cache_head = v_cache + head * max_seq_len * head_dim + position_physical * head_dim;
        __nv_fp8_e4m3* k_cache_head_fp8 = nullptr;
        __nv_fp8_e4m3* v_cache_head_fp8 = nullptr;
        float* k_scale_ptr = nullptr;
        float* v_scale_ptr = nullptr;
        float k_scale = 1.0f;
        float v_scale = 1.0f;
        if (kv_fp8_enabled) {
            k_cache_head_fp8 = k_cache_fp8 + head * max_seq_len * head_dim + position_physical * head_dim;
            v_cache_head_fp8 = v_cache_fp8 + head * max_seq_len * head_dim + position_physical * head_dim;
            k_scale_ptr = k_scale_cache + head;
            v_scale_ptr = v_scale_cache + head;
            k_scale = *k_scale_ptr;
            v_scale = *v_scale_ptr;
            if (k_scale < 1.0e-8f) k_scale = 1.0f;
            if (v_scale < 1.0e-8f) v_scale = 1.0f;
        }

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
            smem_normed[i] = out;
        }
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float out = smem_normed[i];
            __nv_bfloat16 out_bf16 = __float2bfloat16(out);
            k_head[i] = out_bf16;
            k_cache_head[i] = out_bf16;
            __nv_bfloat16 vv_bf16 = v_head[i];
            v_cache_head[i] = vv_bf16;
            if (kv_fp8_enabled) {
                float inv_ks = 1.0f / k_scale;
                float inv_vs = 1.0f / v_scale;
                float vv = __bfloat162float(vv_bf16);
                k_cache_head_fp8[i] = __nv_fp8_e4m3(out * inv_ks);
                v_cache_head_fp8[i] = __nv_fp8_e4m3(vv * inv_vs);
            }
        }
    }
}

// Decode attention over KV cache (single query token).
// Legacy kernel: each warp handles one q head; low occupancy for large GPUs.
__global__ void decode_attention_cache_legacy_kernel(
    const __nv_bfloat16* __restrict__ q,             // [num_q_heads, head_dim] (already norm+rope)
    const __nv_bfloat16* __restrict__ k_cache,       // [num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,       // [num_kv_heads, max_seq_len, head_dim]
    const __nv_fp8_e4m3* __restrict__ k_cache_fp8,   // optional [num_kv_heads, max_seq_len, head_dim]
    const __nv_fp8_e4m3* __restrict__ v_cache_fp8,   // optional [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ k_scale_cache,         // optional [num_kv_heads]
    const float* __restrict__ v_scale_cache,         // optional [num_kv_heads]
    int kv_fp8_enabled,
    __nv_bfloat16* __restrict__ out,                 // [num_q_heads, head_dim]
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    float attn_scale,
    const int* __restrict__ kv_block_table,
    int kv_block_size,
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
    const __nv_fp8_e4m3* k_base_fp8 = kv_fp8_enabled ? (k_cache_fp8 + kv_head * max_seq_len * head_dim) : nullptr;
    const __nv_fp8_e4m3* v_base_fp8 = kv_fp8_enabled ? (v_cache_fp8 + kv_head * max_seq_len * head_dim) : nullptr;
    const float* k_scale_base = kv_fp8_enabled ? (k_scale_cache + kv_head) : nullptr;
    const float* v_scale_base = kv_fp8_enabled ? (v_scale_cache + kv_head) : nullptr;
    float k_scale = kv_fp8_enabled ? *k_scale_base : 1.0f;
    float v_scale = kv_fp8_enabled ? *v_scale_base : 1.0f;
    if (k_scale < 1.0e-8f) k_scale = 1.0f;
    if (v_scale < 1.0e-8f) v_scale = 1.0f;

    for (int t = 0; t < cache_len; t++) {
        int physical_t = kv_cache_physical_token(t, kv_block_table, kv_block_size);
        if (physical_t >= max_seq_len) continue;
        const __nv_bfloat16* k_vec = k_base + physical_t * head_dim;
        const __nv_bfloat16* v_vec = v_base + physical_t * head_dim;
        const __nv_fp8_e4m3* k_vec_fp8 = kv_fp8_enabled ? (k_base_fp8 + physical_t * head_dim) : nullptr;
        const __nv_fp8_e4m3* v_vec_fp8 = kv_fp8_enabled ? (v_base_fp8 + physical_t * head_dim) : nullptr;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_LANE; i++) {
            int d = lane_id + i * WARP_SIZE;
            dot += q_local[i] * kv_cache_load_elem(k_vec, k_vec_fp8, d, k_scale, kv_fp8_enabled);
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
            float vv = kv_cache_load_elem(v_vec, v_vec_fp8, d, v_scale, kv_fp8_enabled);
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
    const __nv_fp8_e4m3* __restrict__ k_cache_fp8,   // optional [num_kv_heads, max_seq_len, head_dim]
    const __nv_fp8_e4m3* __restrict__ v_cache_fp8,   // optional [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ k_scale_cache,         // optional [num_kv_heads]
    const float* __restrict__ v_scale_cache,         // optional [num_kv_heads]
    int kv_fp8_enabled,
    __nv_bfloat16* __restrict__ out,                 // [num_q_heads, head_dim]
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    float attn_scale,
    int warps_per_head,
    const int* __restrict__ kv_block_table,
    int kv_block_size,
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
    if (num_kv_heads <= 0) return;

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
    const __nv_fp8_e4m3* k_base_fp8 = kv_fp8_enabled ? (k_cache_fp8 + kv_head * max_seq_len * head_dim) : nullptr;
    const __nv_fp8_e4m3* v_base_fp8 = kv_fp8_enabled ? (v_cache_fp8 + kv_head * max_seq_len * head_dim) : nullptr;
    const float* k_scale_base = kv_fp8_enabled ? (k_scale_cache + kv_head) : nullptr;
    const float* v_scale_base = kv_fp8_enabled ? (v_scale_cache + kv_head) : nullptr;
    float k_scale = kv_fp8_enabled ? *k_scale_base : 1.0f;
    float v_scale = kv_fp8_enabled ? *v_scale_base : 1.0f;
    if (k_scale < 1.0e-8f) k_scale = 1.0f;
    if (v_scale < 1.0e-8f) v_scale = 1.0f;

    for (int t = warp_id; t < cache_len; t += active_warps) {
        int physical_t = kv_cache_physical_token(t, kv_block_table, kv_block_size);
        if (physical_t >= max_seq_len) continue;
        const __nv_bfloat16* k_vec = k_base + physical_t * head_dim;
        const __nv_bfloat16* v_vec = v_base + physical_t * head_dim;
        const __nv_fp8_e4m3* k_vec_fp8 = kv_fp8_enabled ? (k_base_fp8 + physical_t * head_dim) : nullptr;
        const __nv_fp8_e4m3* v_vec_fp8 = kv_fp8_enabled ? (v_base_fp8 + physical_t * head_dim) : nullptr;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < kElemsPerLane; i++) {
            int d = lane_id + i * WARP_SIZE;
            dot += q_local[i] * kv_cache_load_elem(k_vec, k_vec_fp8, d, k_scale, kv_fp8_enabled);
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
            float vv = kv_cache_load_elem(v_vec, v_vec_fp8, d, v_scale, kv_fp8_enabled);
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
    const __nv_fp8_e4m3* __restrict__ k_cache_fp8,   // optional [num_kv_heads, max_seq_len, head_dim]
    const __nv_fp8_e4m3* __restrict__ v_cache_fp8,   // optional [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ k_scale_cache,         // optional [num_kv_heads]
    const float* __restrict__ v_scale_cache,         // optional [num_kv_heads]
    int kv_fp8_enabled,
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
    const int* __restrict__ kv_block_table,
    int kv_block_size,
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
    const __nv_fp8_e4m3* k_base_fp8 = kv_fp8_enabled ? (k_cache_fp8 + kv_head * max_seq_len * head_dim) : nullptr;
    const __nv_fp8_e4m3* v_base_fp8 = kv_fp8_enabled ? (v_cache_fp8 + kv_head * max_seq_len * head_dim) : nullptr;
    const float* k_scale_base = kv_fp8_enabled ? (k_scale_cache + kv_head) : nullptr;
    const float* v_scale_base = kv_fp8_enabled ? (v_scale_cache + kv_head) : nullptr;
    float k_scale = kv_fp8_enabled ? *k_scale_base : 1.0f;
    float v_scale = kv_fp8_enabled ? *v_scale_base : 1.0f;
    if (k_scale < 1.0e-8f) k_scale = 1.0f;
    if (v_scale < 1.0e-8f) v_scale = 1.0f;

    if (warp_id < active_warps) {
        for (int t = t_start + warp_id; t < t_end; t += active_warps) {
            int physical_t = kv_cache_physical_token(t, kv_block_table, kv_block_size);
            if (physical_t >= max_seq_len) continue;
            const __nv_bfloat16* k_vec = k_base + physical_t * head_dim;
            const __nv_bfloat16* v_vec = v_base + physical_t * head_dim;
            const __nv_fp8_e4m3* k_vec_fp8 = kv_fp8_enabled ? (k_base_fp8 + physical_t * head_dim) : nullptr;
            const __nv_fp8_e4m3* v_vec_fp8 = kv_fp8_enabled ? (v_base_fp8 + physical_t * head_dim) : nullptr;
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < kElemsPerLane; i++) {
                int d = lane_id + i * WARP_SIZE;
                dot += q_local[i] * kv_cache_load_elem(k_vec, k_vec_fp8, d, k_scale, kv_fp8_enabled);
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
                float vv = kv_cache_load_elem(v_vec, v_vec_fp8, d, v_scale, kv_fp8_enabled);
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

struct SplitFlashDecodeDebugRec {
    unsigned long long cycles;
    int cache_len;
    int active_warps;
    int tokens_processed;
    int block_switches;
    int oob_skips;
    int tiles_processed;
};

// Flash-decode v2 phase-1:
// - One block handles one (q_head, chunk).
// - Warps process contiguous token tiles inside the chunk to improve paged-KV locality.
// - Output format is shared with splitk phase-2 reducer.
__global__ void decode_attention_cache_flash_phase1_kernel(
    const __nv_bfloat16* __restrict__ q,             // [num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_cache,       // [num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,       // [num_kv_heads, max_seq_len, head_dim]
    const __nv_fp8_e4m3* __restrict__ k_cache_fp8,   // optional [num_kv_heads, max_seq_len, head_dim]
    const __nv_fp8_e4m3* __restrict__ v_cache_fp8,   // optional [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ k_scale_cache,         // optional [num_kv_heads]
    const float* __restrict__ v_scale_cache,         // optional [num_kv_heads]
    int kv_fp8_enabled,
    float* __restrict__ partial_m,                   // [max_chunks, num_q_heads]
    float* __restrict__ partial_s,                   // [max_chunks, num_q_heads]
    float* __restrict__ partial_out,                 // [max_chunks, num_q_heads, head_dim]
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    float attn_scale,
    int warps_per_head,
    int partitions_per_chunk,
    int chunk_size,
    int max_chunks,
    const int* __restrict__ kv_block_table,
    int kv_block_size,
    const int* __restrict__ position_ptr,
    SplitFlashDecodeDebugRec* __restrict__ debug_out // optional [partitions_per_chunk, max_chunks, num_q_heads]
) {
    constexpr int kMaxWarps = 8;
    constexpr int kElemsPerLane = HEAD_DIM / WARP_SIZE;
    constexpr int kFlashTile = 8;
    __shared__ float sm_m[kMaxWarps];
    __shared__ float sm_s[kMaxWarps];
    __shared__ float sm_scale[kMaxWarps];
    __shared__ float sm_out[kMaxWarps * HEAD_DIM];
    __shared__ float sm_global_m;
    __shared__ float sm_global_s;
    __shared__ int sm_active_warps;
    __shared__ int sm_tokens;
    __shared__ int sm_switches;
    __shared__ int sm_oob;
    __shared__ int sm_tiles;
    __shared__ unsigned long long sm_clock_begin;

    if (head_dim != HEAD_DIM) return;
    if (chunk_size <= 0 || max_chunks <= 0) return;
    if (num_q_heads <= 0 || num_kv_heads <= 0) return;
    if (partitions_per_chunk <= 0) return;

    int block_linear = (int)blockIdx.x;
    int q_head = block_linear % num_q_heads;
    int tmp = block_linear / num_q_heads;
    int chunk_idx = tmp % max_chunks;
    int part_idx = tmp / max_chunks;
    if (chunk_idx >= max_chunks) return;
    if (part_idx >= partitions_per_chunk) return;

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    int position = *position_ptr;
    int cache_len = position + 1;
    if (cache_len > max_seq_len) cache_len = max_seq_len;
    int chunk_start = chunk_idx * chunk_size;
    if (chunk_start >= cache_len) return;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > cache_len) chunk_end = cache_len;
    int chunk_tokens = chunk_end - chunk_start;
    int part_tokens = (chunk_tokens + partitions_per_chunk - 1) / partitions_per_chunk;
    int t_start = chunk_start + part_idx * part_tokens;
    int t_end = t_start + part_tokens;
    if (t_start < chunk_start) t_start = chunk_start;
    if (t_end > chunk_end) t_end = chunk_end;
    if (t_start > chunk_end) t_start = chunk_end;
    if (t_end < t_start) t_end = t_start;

    int active_warps = warps_per_head;
    if (active_warps > kMaxWarps) active_warps = kMaxWarps;
    int local_tokens_span = t_end - t_start;
    if (active_warps > local_tokens_span) active_warps = local_tokens_span;
    if (active_warps <= 0) active_warps = 1;
    if (threadIdx.x == 0) {
        sm_active_warps = active_warps;
        if (debug_out != nullptr) {
            sm_tokens = 0;
            sm_switches = 0;
            sm_oob = 0;
            sm_tiles = 0;
            sm_clock_begin = clock64();
        }
    }
    __syncthreads();
    active_warps = sm_active_warps;

    int kv_ratio = num_q_heads / num_kv_heads;
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
    const __nv_fp8_e4m3* k_base_fp8 = kv_fp8_enabled ? (k_cache_fp8 + kv_head * max_seq_len * head_dim) : nullptr;
    const __nv_fp8_e4m3* v_base_fp8 = kv_fp8_enabled ? (v_cache_fp8 + kv_head * max_seq_len * head_dim) : nullptr;
    const float* k_scale_base = kv_fp8_enabled ? (k_scale_cache + kv_head) : nullptr;
    const float* v_scale_base = kv_fp8_enabled ? (v_scale_cache + kv_head) : nullptr;
    float k_scale = kv_fp8_enabled ? *k_scale_base : 1.0f;
    float v_scale = kv_fp8_enabled ? *v_scale_base : 1.0f;
    if (k_scale < 1.0e-8f) k_scale = 1.0f;
    if (v_scale < 1.0e-8f) v_scale = 1.0f;
    bool paged = (kv_block_table != nullptr && kv_block_size > 0);
    bool use_blocking = (kv_block_size > 0);

    if (warp_id < active_warps) {
        int local_tokens = 0;
        int local_switches = 0;
        int local_oob = 0;
        int local_tiles = 0;
        int block_size = (use_blocking ? kv_block_size : max_seq_len);
        if (block_size <= 0) block_size = max_seq_len;
        if (block_size <= 0) block_size = 1;
        int chunk_block_start = t_start / block_size;
        int chunk_block_end = (t_end + block_size - 1) / block_size;  // exclusive
        int chunk_blocks = chunk_block_end - chunk_block_start;
        int blocks_per_warp = (chunk_blocks + active_warps - 1) / active_warps;
        int warp_block_start = chunk_block_start + warp_id * blocks_per_warp;
        int warp_block_end = warp_block_start + blocks_per_warp;
        if (warp_block_start < chunk_block_start) warp_block_start = chunk_block_start;
        if (warp_block_end > chunk_block_end) warp_block_end = chunk_block_end;

        for (int logical_block = warp_block_start; logical_block < warp_block_end; logical_block++) {
            int block_token_begin = logical_block * block_size;
            int block_token_end = block_token_begin + block_size;
            if (block_token_begin < t_start) block_token_begin = t_start;
            if (block_token_end > t_end) block_token_end = t_end;
            if (block_token_begin >= block_token_end) continue;

            int physical_block = logical_block;
            if (paged) {
                physical_block = kv_block_table[logical_block];
            }
            int physical_block_base = physical_block * block_size;
            local_switches += 1;

            for (int tile_start = block_token_begin; tile_start < block_token_end; tile_start += kFlashTile) {
                local_tiles += 1;
                #pragma unroll
                for (int ti = 0; ti < kFlashTile; ti++) {
                    int t = tile_start + ti;
                    if (t >= block_token_end) break;
                    int in_block = t - logical_block * block_size;
                    int physical_t = physical_block_base + in_block;
                    if (physical_t >= max_seq_len) {
                        local_oob += 1;
                        continue;
                    }
                    local_tokens += 1;

                    const __nv_bfloat16* k_vec = k_base + physical_t * head_dim;
                    const __nv_bfloat16* v_vec = v_base + physical_t * head_dim;
                    const __nv_fp8_e4m3* k_vec_fp8 = kv_fp8_enabled ? (k_base_fp8 + physical_t * head_dim) : nullptr;
                    const __nv_fp8_e4m3* v_vec_fp8 = kv_fp8_enabled ? (v_base_fp8 + physical_t * head_dim) : nullptr;
                    float dot = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < kElemsPerLane; i++) {
                        int d = lane_id + i * WARP_SIZE;
                        dot += q_local[i] * kv_cache_load_elem(k_vec, k_vec_fp8, d, k_scale, kv_fp8_enabled);
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
                        float vv = kv_cache_load_elem(v_vec, v_vec_fp8, d, v_scale, kv_fp8_enabled);
                        out_local[i] = out_local[i] * old_scale + vv * new_scale;
                    }
                    m = m_new;
                }
            }
        }
        if (debug_out != nullptr && lane_id == 0) {
            atomicAdd(&sm_tokens, local_tokens);
            atomicAdd(&sm_switches, local_switches);
            atomicAdd(&sm_oob, local_oob);
            atomicAdd(&sm_tiles, local_tiles);
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

    int idx = (part_idx * max_chunks + chunk_idx) * num_q_heads + q_head;
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
        if (debug_out != nullptr) {
            SplitFlashDecodeDebugRec rec{};
            rec.cycles = clock64() - sm_clock_begin;
            rec.cache_len = cache_len;
            rec.active_warps = active_warps;
            rec.tokens_processed = sm_tokens;
            rec.block_switches = sm_switches;
            rec.oob_skips = sm_oob;
            rec.tiles_processed = sm_tiles;
            debug_out[idx] = rec;
        }
    }
}

__global__ void decode_attention_cache_flash_reduce_parts_kernel(
    const float* __restrict__ part_m,          // [parts, max_chunks, num_q_heads]
    const float* __restrict__ part_s,          // [parts, max_chunks, num_q_heads]
    const float* __restrict__ part_out,        // [parts, max_chunks, num_q_heads, head_dim]
    float* __restrict__ chunk_m,               // [max_chunks, num_q_heads]
    float* __restrict__ chunk_s,               // [max_chunks, num_q_heads]
    float* __restrict__ chunk_out,             // [max_chunks, num_q_heads, head_dim]
    int num_q_heads,
    int head_dim,
    int max_chunks,
    int partitions_per_chunk,
    int chunk_size,
    const int* __restrict__ position_ptr
) {
    if (head_dim != HEAD_DIM) return;
    if (partitions_per_chunk <= 0 || chunk_size <= 0 || max_chunks <= 0) return;
    int q_head = (int)(blockIdx.x % num_q_heads);
    int chunk_idx = (int)(blockIdx.x / num_q_heads);
    if (chunk_idx >= max_chunks) return;

    int position = *position_ptr;
    int cache_len = position + 1;
    int num_chunks = (cache_len + chunk_size - 1) / chunk_size;
    if (num_chunks > max_chunks) num_chunks = max_chunks;
    if (chunk_idx >= num_chunks) return;

    int chunk_base = chunk_idx * num_q_heads + q_head;
    float g_m = -INFINITY;
    for (int p = 0; p < partitions_per_chunk; p++) {
        int idx = (p * max_chunks + chunk_idx) * num_q_heads + q_head;
        g_m = fmaxf(g_m, part_m[idx]);
    }
    float g_s = 0.0f;
    for (int p = 0; p < partitions_per_chunk; p++) {
        int idx = (p * max_chunks + chunk_idx) * num_q_heads + q_head;
        g_s += part_s[idx] * expf(part_m[idx] - g_m);
    }
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        float acc = 0.0f;
        for (int p = 0; p < partitions_per_chunk; p++) {
            int idx = (p * max_chunks + chunk_idx) * num_q_heads + q_head;
            float scale = expf(part_m[idx] - g_m);
            acc += part_out[idx * HEAD_DIM + d] * scale;
        }
        chunk_out[chunk_base * HEAD_DIM + d] = acc;
    }
    if (threadIdx.x == 0) {
        chunk_m[chunk_base] = g_m;
        chunk_s[chunk_base] = g_s;
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

// Flash-decode style kernel (single kernel online softmax):
// - One block per q_head.
// - Warps split cache by contiguous ranges (better locality for paged KV).
// - Per-warp online softmax summary is reduced inside block.
__global__ void decode_attention_cache_flash_decode_kernel(
    const __nv_bfloat16* __restrict__ q,             // [num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_cache,       // [num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,       // [num_kv_heads, max_seq_len, head_dim]
    const __nv_fp8_e4m3* __restrict__ k_cache_fp8,   // optional [num_kv_heads, max_seq_len, head_dim]
    const __nv_fp8_e4m3* __restrict__ v_cache_fp8,   // optional [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ k_scale_cache,         // optional [num_kv_heads]
    const float* __restrict__ v_scale_cache,         // optional [num_kv_heads]
    int kv_fp8_enabled,
    __nv_bfloat16* __restrict__ out,                 // [num_q_heads, head_dim]
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    float attn_scale,
    int warps_per_head,
    const int* __restrict__ kv_block_table,
    int kv_block_size,
    const int* __restrict__ position_ptr,            // device scalar
    SplitFlashDecodeDebugRec* __restrict__ debug_out // optional [num_q_heads]
) {
    if (head_dim != HEAD_DIM) return;
    if (num_q_heads <= 0 || num_kv_heads <= 0) return;

    constexpr int kMaxWarps = 8;
    constexpr int kElemsPerLane = HEAD_DIM / WARP_SIZE;  // 4 (128 / 32)
    constexpr int kFlashTile = 8;  // contiguous token tile per warp
    __shared__ float sm_m[kMaxWarps];
    __shared__ float sm_s[kMaxWarps];
    __shared__ float sm_scale[kMaxWarps];
    __shared__ float sm_out[kMaxWarps * HEAD_DIM];
    __shared__ float sm_global_s;
    __shared__ int sm_active_warps;
    __shared__ int sm_tokens;
    __shared__ int sm_block_switches;
    __shared__ int sm_oob_skips;
    __shared__ int sm_tiles_processed;
    __shared__ unsigned long long sm_clock_begin;

    int q_head = (int)blockIdx.x;
    if (q_head >= num_q_heads) return;

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (warp_id >= warps_per_head) return;

    int kv_ratio = num_q_heads / num_kv_heads;
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
    if (threadIdx.x == 0) {
        sm_active_warps = active_warps;
        if (debug_out != nullptr) {
            sm_tokens = 0;
            sm_block_switches = 0;
            sm_oob_skips = 0;
            sm_tiles_processed = 0;
            sm_clock_begin = clock64();
        }
    }
    __syncthreads();
    active_warps = sm_active_warps;
    if (warp_id >= active_warps) return;

    const __nv_bfloat16* q_vec = q + q_head * HEAD_DIM;
    __nv_bfloat16* out_vec = out + q_head * HEAD_DIM;
    float q_local[kElemsPerLane];
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        int d = lane_id + i * WARP_SIZE;
        q_local[i] = __bfloat162float(q_vec[d]);
    }

    float m = -INFINITY;
    float s = 0.0f;
    float out_local[kElemsPerLane] = {0.f, 0.f, 0.f, 0.f};

    const __nv_bfloat16* k_base = k_cache + kv_head * max_seq_len * HEAD_DIM;
    const __nv_bfloat16* v_base = v_cache + kv_head * max_seq_len * HEAD_DIM;
    const __nv_fp8_e4m3* k_base_fp8 = kv_fp8_enabled ? (k_cache_fp8 + kv_head * max_seq_len * HEAD_DIM) : nullptr;
    const __nv_fp8_e4m3* v_base_fp8 = kv_fp8_enabled ? (v_cache_fp8 + kv_head * max_seq_len * HEAD_DIM) : nullptr;
    const float* k_scale_base = kv_fp8_enabled ? (k_scale_cache + kv_head) : nullptr;
    const float* v_scale_base = kv_fp8_enabled ? (v_scale_cache + kv_head) : nullptr;
    float k_scale = kv_fp8_enabled ? *k_scale_base : 1.0f;
    float v_scale = kv_fp8_enabled ? *v_scale_base : 1.0f;
    if (k_scale < 1.0e-8f) k_scale = 1.0f;
    if (v_scale < 1.0e-8f) v_scale = 1.0f;
    bool paged = (kv_block_table != nullptr && kv_block_size > 0);

    int cached_logical_block = -1;
    int cached_physical_block = 0;
    int local_tokens = 0;
    int local_block_switches = 0;
    int local_oob_skips = 0;
    int local_tiles = 0;
    for (int tile_start = warp_id * kFlashTile;
         tile_start < cache_len;
         tile_start += active_warps * kFlashTile) {
        local_tiles += 1;
        #pragma unroll
        for (int ti = 0; ti < kFlashTile; ti++) {
            int t = tile_start + ti;
            if (t >= cache_len) break;

            int physical_t = t;
            if (paged) {
                int logical_block = t / kv_block_size;
                if (logical_block != cached_logical_block) {
                    cached_logical_block = logical_block;
                    cached_physical_block = kv_block_table[logical_block];
                    local_block_switches += 1;
                }
                int in_block = t - logical_block * kv_block_size;
                physical_t = cached_physical_block * kv_block_size + in_block;
            }
            if (physical_t >= max_seq_len) {
                local_oob_skips += 1;
                continue;
            }
            local_tokens += 1;

            const __nv_bfloat16* k_vec = k_base + physical_t * HEAD_DIM;
            const __nv_bfloat16* v_vec = v_base + physical_t * HEAD_DIM;
            const __nv_fp8_e4m3* k_vec_fp8 = kv_fp8_enabled ? (k_base_fp8 + physical_t * HEAD_DIM) : nullptr;
            const __nv_fp8_e4m3* v_vec_fp8 = kv_fp8_enabled ? (v_base_fp8 + physical_t * HEAD_DIM) : nullptr;
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < kElemsPerLane; i++) {
                int d = lane_id + i * WARP_SIZE;
                dot += q_local[i] * kv_cache_load_elem(k_vec, k_vec_fp8, d, k_scale, kv_fp8_enabled);
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
                float vv = kv_cache_load_elem(v_vec, v_vec_fp8, d, v_scale, kv_fp8_enabled);
                out_local[i] = out_local[i] * old_scale + vv * new_scale;
            }
            m = m_new;
        }
    }

    if (debug_out != nullptr && lane_id == 0) {
        atomicAdd(&sm_tokens, local_tokens);
        atomicAdd(&sm_block_switches, local_block_switches);
        atomicAdd(&sm_oob_skips, local_oob_skips);
        atomicAdd(&sm_tiles_processed, local_tiles);
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
        for (int w = 0; w < active_warps; w++) {
            acc += sm_out[w * HEAD_DIM + d] * sm_scale[w];
        }
        out_vec[d] = __float2bfloat16(acc * inv_s);
    }

    if (debug_out != nullptr && threadIdx.x == 0) {
        SplitFlashDecodeDebugRec rec{};
        rec.cycles = clock64() - sm_clock_begin;
        rec.cache_len = cache_len;
        rec.active_warps = active_warps;
        rec.tokens_processed = sm_tokens;
        rec.block_switches = sm_block_switches;
        rec.oob_skips = sm_oob_skips;
        rec.tiles_processed = sm_tiles_processed;
        debug_out[q_head] = rec;
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

__global__ void split_w4_silu_downproj_residual_kernel(
    const __nv_bfloat16* __restrict__ gateup_out_bf16,  // [2 * INTERMEDIATE_SIZE], [gate | up]
    const uint8_t* __restrict__ down_packed_w4,         // [HIDDEN_SIZE * INTERMEDIATE_SIZE / 2]
    const float* __restrict__ down_scales,              // [HIDDEN_SIZE * INTERMEDIATE_SIZE / 64]
    const float* __restrict__ down_codebook,            // [16]
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

    float local_sum = 0.0f;
    int64_t row_base = (int64_t)out_idx * (int64_t)INTERMEDIATE_SIZE;
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += kBlock) {
        float gate_v = __bfloat162float(gate[i]);
        float up_v = __bfloat162float(up[i]);
        float silu_v = gate_v / (1.0f + expf(-gate_v));

        int64_t linear = row_base + (int64_t)i;
        uint8_t pack = down_packed_w4[linear >> 1];
        int q = (linear & 1) ? int(pack & 0x0F) : int((pack >> 4) & 0x0F);
        float down_w = down_codebook[q] * down_scales[linear >> 6];

        local_sum += (silu_v * up_v) * down_w;
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < kWarps) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) hidden_f32[out_idx] = sum + residual_f32[out_idx];
    }
}

// =============================================================================
// Host entry: one split decode step (single token)
// =============================================================================

#include "split_decode_gemm_host.inl"
