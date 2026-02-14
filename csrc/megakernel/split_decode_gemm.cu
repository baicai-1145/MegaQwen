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

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp8.hpp>
#include <mma.h>
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

struct SplitFloat2 {
    float x;
    float y;
};

__device__ __forceinline__ SplitFloat2 split_warp_reduce_sum2(SplitFloat2 v) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        v.x += __shfl_down_sync(0xffffffff, v.x, offset);
        v.y += __shfl_down_sync(0xffffffff, v.y, offset);
    }
    return v;
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

union SplitFp8x4Pack {
    uint32_t raw;
    __nv_fp8_e4m3 v[4];
};

struct SplitHalf2x2Pack {
    __half2 h01;
    __half2 h23;
};

struct SplitW4x4Pack {
    int q0;
    int q1;
    int q2;
    int q3;
};

struct SplitW4x8Pack {
    int q0;
    int q1;
    int q2;
    int q3;
    int q4;
    int q5;
    int q6;
    int q7;
};

__device__ __forceinline__ SplitFp8x4Pack split_load_fp8x4(
    const __nv_fp8_e4m3* __restrict__ ptr
) {
    SplitFp8x4Pack pack;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    pack.raw = __ldg(reinterpret_cast<const uint32_t*>(ptr));
#else
    pack.raw = *reinterpret_cast<const uint32_t*>(ptr);
#endif
    return pack;
}

__device__ __forceinline__ SplitHalf2x2Pack split_fp8x4_to_half2x2(
    const SplitFp8x4Pack& pack
) {
    SplitHalf2x2Pack out;
    __half2_raw raw01 = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<uint16_t>(pack.raw),
        __NV_E4M3);
    __half2_raw raw23 = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<uint16_t>(pack.raw >> 16U),
        __NV_E4M3);
    out.h01 = __halves2half2(__ushort_as_half(raw01.x), __ushort_as_half(raw01.y));
    out.h23 = __halves2half2(__ushort_as_half(raw23.x), __ushort_as_half(raw23.y));
    return out;
}

__device__ __forceinline__ uint16_t split_load_u16(
    const uint16_t* __restrict__ ptr
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

__device__ __forceinline__ uint32_t split_load_u32(
    const uint32_t* __restrict__ ptr
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

__device__ __forceinline__ SplitW4x4Pack split_decode_w4x4(
    uint16_t packed2
) {
    uint8_t p0 = static_cast<uint8_t>(packed2 & 0x00FFu);
    uint8_t p1 = static_cast<uint8_t>(packed2 >> 8U);
    SplitW4x4Pack out;
    out.q0 = int((p0 >> 4) & 0x0F);
    out.q1 = int(p0 & 0x0F);
    out.q2 = int((p1 >> 4) & 0x0F);
    out.q3 = int(p1 & 0x0F);
    return out;
}

__device__ __forceinline__ SplitW4x8Pack split_decode_w4x8(
    uint32_t packed4
) {
    uint8_t p0 = static_cast<uint8_t>(packed4 & 0x000000FFu);
    uint8_t p1 = static_cast<uint8_t>((packed4 >> 8U) & 0xFFu);
    uint8_t p2 = static_cast<uint8_t>((packed4 >> 16U) & 0xFFu);
    uint8_t p3 = static_cast<uint8_t>((packed4 >> 24U) & 0xFFu);
    SplitW4x8Pack out;
    out.q0 = int((p0 >> 4) & 0x0F);
    out.q1 = int(p0 & 0x0F);
    out.q2 = int((p1 >> 4) & 0x0F);
    out.q3 = int(p1 & 0x0F);
    out.q4 = int((p2 >> 4) & 0x0F);
    out.q5 = int(p2 & 0x0F);
    out.q6 = int((p3 >> 4) & 0x0F);
    out.q7 = int(p3 & 0x0F);
    return out;
}

template <bool FAST_EXP>
__device__ __forceinline__ float split_silu_value(float x) {
    float ex = FAST_EXP ? __expf(-x) : expf(-x);
    return x / (1.0f + ex);
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
    int kv_fp8_only,
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
        bool use_fp8_only = (kv_fp8_enabled && kv_fp8_only);
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
            __nv_bfloat16 vv_bf16 = v_head[i];
            if (!use_fp8_only) {
                k_head[i] = out_bf16;
                k_cache_head[i] = out_bf16;
                v_cache_head[i] = vv_bf16;
            }
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
    unsigned long long map_cycles;
    unsigned long long qk_cycles;
    unsigned long long qk_load_cycles;
    unsigned long long qk_load_mem_cycles;
    unsigned long long qk_load_cvt_cycles;
    unsigned long long qk_load_bf16_cycles;
    unsigned long long qk_fma_cycles;
    unsigned long long qk_fma_bf16_cycles;
    unsigned long long qk_reduce_cycles;
    unsigned long long qk_tc_pack_cycles;
    unsigned long long qk_tc_mma_cycles;
    unsigned long long qk_tc_unpack_cycles;
    unsigned long long softmax_cycles;
    unsigned long long value_cycles;
    unsigned long long value_load_cycles;
    unsigned long long value_fma_cycles;
    unsigned long long value_dequant_cycles;
    unsigned long long value_rescale_cycles;
    unsigned long long value_accum_cycles;
    unsigned long long value_bookkeep_cycles;
    unsigned long long merge_cycles;
    int cache_len;
    int active_warps;
    int tokens_processed;
    int block_switches;
    int oob_skips;
    int tiles_processed;
};

struct SplitFfnW4FineDebugRec {
    unsigned long long gateup_dequant_cycles;
    unsigned long long gateup_fma_cycles;
    unsigned long long gateup_reduce_cycles;
    unsigned long long down_dequant_cycles;
    unsigned long long down_act_cycles;
    unsigned long long down_fma_cycles;
};

// Flash-decode v2 phase-1:
// - One block handles one (q_head, chunk).
// - Warps process contiguous token tiles inside the chunk to improve paged-KV locality.
// - Output format is shared with splitk phase-2 reducer.
template <bool KV_FP8>
__global__ void decode_attention_cache_flash_phase1_kernel_t(
    const __nv_bfloat16* __restrict__ q,             // [num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_cache,       // [num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,       // [num_kv_heads, max_seq_len, head_dim]
    const __nv_fp8_e4m3* __restrict__ k_cache_fp8,   // optional [num_kv_heads, max_seq_len, head_dim]
    const __nv_fp8_e4m3* __restrict__ v_cache_fp8,   // optional [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ k_scale_cache,         // optional [num_kv_heads]
    const float* __restrict__ v_scale_cache,         // optional [num_kv_heads]
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
    __shared__ unsigned long long sm_map_cycles;
    __shared__ unsigned long long sm_qk_cycles;
    __shared__ unsigned long long sm_qk_load_cycles;
    __shared__ unsigned long long sm_qk_load_bf16_cycles;
    __shared__ unsigned long long sm_qk_fma_cycles;
    __shared__ unsigned long long sm_qk_fma_bf16_cycles;
    __shared__ unsigned long long sm_qk_reduce_cycles;
    __shared__ unsigned long long sm_softmax_cycles;
    __shared__ unsigned long long sm_value_cycles;
    __shared__ unsigned long long sm_value_load_cycles;
    __shared__ unsigned long long sm_value_fma_cycles;
    __shared__ unsigned long long sm_merge_cycles;
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
            sm_map_cycles = 0;
            sm_qk_cycles = 0;
            sm_qk_load_cycles = 0;
            sm_qk_load_bf16_cycles = 0;
            sm_qk_fma_cycles = 0;
            sm_qk_fma_bf16_cycles = 0;
            sm_qk_reduce_cycles = 0;
            sm_softmax_cycles = 0;
            sm_value_cycles = 0;
            sm_value_load_cycles = 0;
            sm_value_fma_cycles = 0;
            sm_merge_cycles = 0;
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

    float m = -INFINITY;
    float s = 0.0f;
    float out_local[kElemsPerLane] = {0.f, 0.f, 0.f, 0.f};

    const __nv_bfloat16* k_base = k_cache + kv_head * max_seq_len * head_dim;
    const __nv_bfloat16* v_base = v_cache + kv_head * max_seq_len * head_dim;
    const __nv_fp8_e4m3* k_base_fp8 = nullptr;
    const __nv_fp8_e4m3* v_base_fp8 = nullptr;
    float k_scale = 1.0f;
    float v_scale = 1.0f;
    if constexpr (KV_FP8) {
        k_base_fp8 = k_cache_fp8 + kv_head * max_seq_len * head_dim;
        v_base_fp8 = v_cache_fp8 + kv_head * max_seq_len * head_dim;
        k_scale = k_scale_cache[kv_head];
        v_scale = v_scale_cache[kv_head];
        if (k_scale < 1.0e-8f) k_scale = 1.0f;
        if (v_scale < 1.0e-8f) v_scale = 1.0f;
    }
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        int d = lane_id * kElemsPerLane + i;
        float qv = __bfloat162float(q_vec[d]);
        if constexpr (KV_FP8) qv *= k_scale;
        q_local[i] = qv;
    }
    bool paged = (kv_block_table != nullptr && kv_block_size > 0);
    bool use_blocking = (kv_block_size > 0);

    if (warp_id < active_warps) {
        int local_tokens = 0;
        int local_switches = 0;
        int local_oob = 0;
        int local_tiles = 0;
        unsigned long long local_map_cycles = 0;
        unsigned long long local_qk_cycles = 0;
        unsigned long long local_qk_load_cycles = 0;
        unsigned long long local_qk_load_bf16_cycles = 0;
        unsigned long long local_qk_fma_cycles = 0;
        unsigned long long local_qk_fma_bf16_cycles = 0;
        unsigned long long local_qk_reduce_cycles = 0;
        unsigned long long local_softmax_cycles = 0;
        unsigned long long local_value_cycles = 0;
        unsigned long long local_value_load_cycles = 0;
        unsigned long long local_value_fma_cycles = 0;
        bool debug_lane = (debug_out != nullptr && lane_id == 0);
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
                    unsigned long long t_map_begin = 0;
                    if (debug_lane) t_map_begin = clock64();
                    int t = tile_start + ti;
                    if (t >= block_token_end) break;
                    int in_block = t - logical_block * block_size;
                    int physical_t = physical_block_base + in_block;
                    if (debug_lane) local_map_cycles += (clock64() - t_map_begin);
                    if (physical_t >= max_seq_len) {
                        local_oob += 1;
                        continue;
                    }
                    local_tokens += 1;

                    const __nv_bfloat16* k_vec = k_base + physical_t * head_dim;
                    const __nv_bfloat16* v_vec = v_base + physical_t * head_dim;
                    const __nv_fp8_e4m3* k_vec_fp8 = KV_FP8 ? (k_base_fp8 + physical_t * head_dim) : nullptr;
                    const __nv_fp8_e4m3* v_vec_fp8 = KV_FP8 ? (v_base_fp8 + physical_t * head_dim) : nullptr;
                    unsigned long long t_qk_begin = 0;
                    if (debug_lane) t_qk_begin = clock64();
                    float dot = 0.0f;
                    if constexpr (KV_FP8) {
                        int d0 = lane_id * kElemsPerLane;
                        unsigned long long t_load_begin = 0;
                        if (debug_lane) t_load_begin = clock64();
                        SplitFp8x4Pack kpack = split_load_fp8x4(k_vec_fp8 + d0);
                        if (debug_lane) local_qk_load_cycles += (clock64() - t_load_begin);
                        unsigned long long t_fma_begin = 0;
                        if (debug_lane) t_fma_begin = clock64();
                        dot += q_local[0] * static_cast<float>(kpack.v[0]);
                        dot += q_local[1] * static_cast<float>(kpack.v[1]);
                        dot += q_local[2] * static_cast<float>(kpack.v[2]);
                        dot += q_local[3] * static_cast<float>(kpack.v[3]);
                        if (debug_lane) local_qk_fma_cycles += (clock64() - t_fma_begin);
                    } else {
                        #pragma unroll
                        for (int i = 0; i < kElemsPerLane; i++) {
                            int d = lane_id * kElemsPerLane + i;
                            unsigned long long t_load_begin = 0;
                            if (debug_lane) t_load_begin = clock64();
                            float kval = __bfloat162float(k_vec[d]);
                            if (debug_lane) {
                                unsigned long long bf16_load = (clock64() - t_load_begin);
                                local_qk_load_cycles += bf16_load;
                                local_qk_load_bf16_cycles += bf16_load;
                            }
                            unsigned long long t_fma_begin = 0;
                            if (debug_lane) t_fma_begin = clock64();
                            dot += q_local[i] * kval;
                            if (debug_lane) {
                                unsigned long long bf16_fma = (clock64() - t_fma_begin);
                                local_qk_fma_cycles += bf16_fma;
                                local_qk_fma_bf16_cycles += bf16_fma;
                            }
                        }
                    }
                    unsigned long long t_qk_reduce_begin = 0;
                    if (debug_lane) t_qk_reduce_begin = clock64();
                    dot = warp_reduce_sum(dot) * attn_scale;
                    float score = __shfl_sync(0xffffffff, dot, 0);
                    if (debug_lane) local_qk_reduce_cycles += (clock64() - t_qk_reduce_begin);
                    if (debug_lane) local_qk_cycles += (clock64() - t_qk_begin);

                    unsigned long long t_softmax_begin = 0;
                    if (debug_lane) t_softmax_begin = clock64();
                    float m_new = fmaxf(m, score);
                    float old_scale = expf(m - m_new);
                    float new_scale = expf(score - m_new);
                    s = s * old_scale + new_scale;
                    if (debug_lane) local_softmax_cycles += (clock64() - t_softmax_begin);

                    unsigned long long t_value_begin = 0;
                    if (debug_lane) t_value_begin = clock64();
                    if constexpr (KV_FP8) {
                        int d0 = lane_id * kElemsPerLane;
                        unsigned long long t_vload_begin = 0;
                        if (debug_lane) t_vload_begin = clock64();
                        SplitFp8x4Pack vpack = split_load_fp8x4(v_vec_fp8 + d0);
                        if (debug_lane) local_value_load_cycles += (clock64() - t_vload_begin);
                        unsigned long long t_vfma_begin = 0;
                        if (debug_lane) t_vfma_begin = clock64();
                        out_local[0] = out_local[0] * old_scale + static_cast<float>(vpack.v[0]) * new_scale;
                        out_local[1] = out_local[1] * old_scale + static_cast<float>(vpack.v[1]) * new_scale;
                        out_local[2] = out_local[2] * old_scale + static_cast<float>(vpack.v[2]) * new_scale;
                        out_local[3] = out_local[3] * old_scale + static_cast<float>(vpack.v[3]) * new_scale;
                        if (debug_lane) local_value_fma_cycles += (clock64() - t_vfma_begin);
                    } else {
                        #pragma unroll
                        for (int i = 0; i < kElemsPerLane; i++) {
                            int d = lane_id * kElemsPerLane + i;
                            unsigned long long t_vload_begin = 0;
                            if (debug_lane) t_vload_begin = clock64();
                            float vv = __bfloat162float(v_vec[d]);
                            if (debug_lane) local_value_load_cycles += (clock64() - t_vload_begin);
                            unsigned long long t_vfma_begin = 0;
                            if (debug_lane) t_vfma_begin = clock64();
                            out_local[i] = out_local[i] * old_scale + vv * new_scale;
                            if (debug_lane) local_value_fma_cycles += (clock64() - t_vfma_begin);
                        }
                    }
                    if (debug_lane) local_value_cycles += (clock64() - t_value_begin);
                    m = m_new;
                }
            }
        }
        if (debug_out != nullptr && lane_id == 0) {
            atomicAdd(&sm_tokens, local_tokens);
            atomicAdd(&sm_switches, local_switches);
            atomicAdd(&sm_oob, local_oob);
            atomicAdd(&sm_tiles, local_tiles);
            atomicAdd(&sm_map_cycles, local_map_cycles);
            atomicAdd(&sm_qk_cycles, local_qk_cycles);
            atomicAdd(&sm_qk_load_cycles, local_qk_load_cycles);
            atomicAdd(&sm_qk_load_bf16_cycles, local_qk_load_bf16_cycles);
            atomicAdd(&sm_qk_fma_cycles, local_qk_fma_cycles);
            atomicAdd(&sm_qk_fma_bf16_cycles, local_qk_fma_bf16_cycles);
            atomicAdd(&sm_qk_reduce_cycles, local_qk_reduce_cycles);
            atomicAdd(&sm_softmax_cycles, local_softmax_cycles);
            atomicAdd(&sm_value_cycles, local_value_cycles);
            atomicAdd(&sm_value_load_cycles, local_value_load_cycles);
            atomicAdd(&sm_value_fma_cycles, local_value_fma_cycles);
        }
    }

    if (lane_id == 0) {
        sm_m[warp_id] = (warp_id < active_warps) ? m : -INFINITY;
        sm_s[warp_id] = (warp_id < active_warps) ? s : 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        int d = lane_id * kElemsPerLane + i;
        sm_out[warp_id * HEAD_DIM + d] = (warp_id < active_warps) ? out_local[i] : 0.0f;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned long long merge_begin = 0;
        if (debug_out != nullptr) merge_begin = clock64();
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
        if (debug_out != nullptr) sm_merge_cycles += (clock64() - merge_begin);
    }
    __syncthreads();

    int idx = (part_idx * max_chunks + chunk_idx) * num_q_heads + q_head;
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        int d = lane_id * kElemsPerLane + i;
        float acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < kMaxWarps; w++) {
            if (w < active_warps) acc += sm_out[w * HEAD_DIM + d] * sm_scale[w];
        }
        if constexpr (KV_FP8) acc *= v_scale;
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
            rec.map_cycles = sm_map_cycles;
            rec.qk_cycles = sm_qk_cycles;
            rec.qk_load_cycles = sm_qk_load_cycles;
            rec.qk_load_bf16_cycles = sm_qk_load_bf16_cycles;
            rec.qk_fma_cycles = sm_qk_fma_cycles;
            rec.qk_fma_bf16_cycles = sm_qk_fma_bf16_cycles;
            rec.qk_reduce_cycles = sm_qk_reduce_cycles;
            rec.softmax_cycles = sm_softmax_cycles;
            rec.value_cycles = sm_value_cycles;
            rec.value_load_cycles = sm_value_load_cycles;
            rec.value_fma_cycles = sm_value_fma_cycles;
            rec.merge_cycles = sm_merge_cycles;
            debug_out[idx] = rec;
        }
    }
}

// Flash-decode v2 phase-1 (GQA shared-KV variant):
// - One block handles one (kv_head, chunk), and computes two q_heads sharing the same KV head.
// - This reuses FP8 K/V loads and dequant conversion across grouped q heads (Q/KV ratio = 2).
template <bool K_FP8, bool V_FP8, bool ENABLE_TC_QK, bool ENABLE_DEBUG>
__global__ void decode_attention_cache_flash_phase1_gqa2_kernel_t(
    const __nv_bfloat16* __restrict__ q,             // [num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_cache,       // [num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,       // [num_kv_heads, max_seq_len, head_dim]
    const __nv_fp8_e4m3* __restrict__ k_cache_fp8,   // optional [num_kv_heads, max_seq_len, head_dim]
    const __nv_fp8_e4m3* __restrict__ v_cache_fp8,   // optional [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ k_scale_cache,         // optional [num_kv_heads]
    const float* __restrict__ v_scale_cache,         // optional [num_kv_heads]
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
    int fp8_tc_qk,
    SplitFlashDecodeDebugRec* __restrict__ debug_out // optional [partitions_per_chunk, max_chunks, num_q_heads]
) {
    constexpr int kMaxWarps = 8;
    constexpr int kElemsPerLane = HEAD_DIM / WARP_SIZE;
    constexpr int kFlashTile = 8;
    constexpr int kTcM = 16;
    constexpr int kTcN = 16;
    constexpr int kTcK = 16;
    constexpr size_t kTcWarpElems = size_t(kTcM) * size_t(kTcK);
    constexpr size_t kTcAllWarpElems = size_t(kMaxWarps) * kTcWarpElems;
    __shared__ float sm_m0[kMaxWarps];
    __shared__ float sm_m1[kMaxWarps];
    __shared__ float sm_s0[kMaxWarps];
    __shared__ float sm_s1[kMaxWarps];
    __shared__ float sm_scale0[kMaxWarps];
    __shared__ float sm_scale1[kMaxWarps];
    __shared__ float sm_out0[kMaxWarps * HEAD_DIM];
    __shared__ float sm_out1[kMaxWarps * HEAD_DIM];
    __shared__ float sm_global_m0;
    __shared__ float sm_global_m1;
    __shared__ float sm_global_s0;
    __shared__ float sm_global_s1;
    __shared__ int sm_active_warps;
    __shared__ int sm_tokens;
    __shared__ int sm_switches;
    __shared__ int sm_oob;
    __shared__ int sm_tiles;
    __shared__ unsigned long long sm_map_cycles;
    __shared__ unsigned long long sm_qk_cycles;
    __shared__ unsigned long long sm_qk_load_cycles;
    __shared__ unsigned long long sm_qk_load_mem_cycles;
    __shared__ unsigned long long sm_qk_load_cvt_cycles;
    __shared__ unsigned long long sm_qk_load_bf16_cycles;
    __shared__ unsigned long long sm_qk_fma_cycles;
    __shared__ unsigned long long sm_qk_fma_bf16_cycles;
    __shared__ unsigned long long sm_qk_reduce_cycles;
    __shared__ unsigned long long sm_qk_tc_pack_cycles;
    __shared__ unsigned long long sm_qk_tc_mma_cycles;
    __shared__ unsigned long long sm_qk_tc_unpack_cycles;
    __shared__ unsigned long long sm_softmax_cycles;
    __shared__ unsigned long long sm_value_cycles;
    __shared__ unsigned long long sm_value_load_cycles;
    __shared__ unsigned long long sm_value_fma_cycles;
    __shared__ unsigned long long sm_value_dequant_cycles;
    __shared__ unsigned long long sm_value_rescale_cycles;
    __shared__ unsigned long long sm_value_accum_cycles;
    __shared__ unsigned long long sm_value_bookkeep_cycles;
    __shared__ unsigned long long sm_merge_cycles;
    __shared__ unsigned long long sm_clock_begin;
    __shared__ __half sm_q_tc0[HEAD_DIM];
    __shared__ __half sm_q_tc1[HEAD_DIM];
    extern __shared__ __align__(16) unsigned char sm_tc_storage[];
    __half* sm_tc_a = reinterpret_cast<__half*>(sm_tc_storage);
    __half* sm_tc_b = reinterpret_cast<__half*>(
        sm_tc_storage + sizeof(__half) * kTcAllWarpElems);
    float* sm_tc_c = reinterpret_cast<float*>(
        sm_tc_storage + sizeof(__half) * kTcAllWarpElems * 2);

    if (head_dim != HEAD_DIM) return;
    if (chunk_size <= 0 || max_chunks <= 0) return;
    if (num_q_heads <= 0 || num_kv_heads <= 0) return;
    if (partitions_per_chunk <= 0) return;
    if (num_q_heads != 2 * num_kv_heads) return;
    bool use_fp8_tc_qk = (ENABLE_TC_QK && K_FP8 && fp8_tc_qk != 0);

    int block_linear = (int)blockIdx.x;
    int kv_head = block_linear % num_kv_heads;
    int tmp = block_linear / num_kv_heads;
    int chunk_idx = tmp % max_chunks;
    int part_idx = tmp / max_chunks;
    if (chunk_idx >= max_chunks) return;
    if (part_idx >= partitions_per_chunk) return;

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    int q_head0 = kv_head * 2;
    if (q_head0 >= num_q_heads) return;
    int q_head1 = q_head0 + 1;
    if (q_head1 >= num_q_heads) return;

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
        if constexpr (ENABLE_DEBUG) {
            sm_tokens = 0;
            sm_switches = 0;
            sm_oob = 0;
            sm_tiles = 0;
            sm_map_cycles = 0;
            sm_qk_cycles = 0;
            sm_qk_load_cycles = 0;
            sm_qk_load_mem_cycles = 0;
            sm_qk_load_cvt_cycles = 0;
            sm_qk_load_bf16_cycles = 0;
            sm_qk_fma_cycles = 0;
            sm_qk_fma_bf16_cycles = 0;
            sm_qk_reduce_cycles = 0;
            sm_qk_tc_pack_cycles = 0;
            sm_qk_tc_mma_cycles = 0;
            sm_qk_tc_unpack_cycles = 0;
            sm_softmax_cycles = 0;
            sm_value_cycles = 0;
            sm_value_load_cycles = 0;
            sm_value_fma_cycles = 0;
            sm_value_dequant_cycles = 0;
            sm_value_rescale_cycles = 0;
            sm_value_accum_cycles = 0;
            sm_value_bookkeep_cycles = 0;
            sm_merge_cycles = 0;
            sm_clock_begin = clock64();
        }
    }
    __syncthreads();
    active_warps = sm_active_warps;

    const __nv_bfloat16* q_vec0 = q + q_head0 * head_dim;
    const __nv_bfloat16* q_vec1 = q + q_head1 * head_dim;
    float q_local0[kElemsPerLane];
    float q_local1[kElemsPerLane];
    __half2 q_local0_h2[2];
    __half2 q_local1_h2[2];

    float m0 = -INFINITY;
    float m1 = -INFINITY;
    float s0 = 0.0f;
    float s1 = 0.0f;
    float out_local0[kElemsPerLane] = {0.f, 0.f, 0.f, 0.f};
    float out_local1[kElemsPerLane] = {0.f, 0.f, 0.f, 0.f};

    const __nv_bfloat16* k_base = k_cache + kv_head * max_seq_len * head_dim;
    const __nv_bfloat16* v_base = v_cache + kv_head * max_seq_len * head_dim;
    const __nv_fp8_e4m3* k_base_fp8 = nullptr;
    const __nv_fp8_e4m3* v_base_fp8 = nullptr;
    float k_scale = 1.0f;
    float v_scale = 1.0f;
    if constexpr (K_FP8) {
        k_base_fp8 = k_cache_fp8 + kv_head * max_seq_len * head_dim;
        k_scale = k_scale_cache[kv_head];
        if (k_scale < 1.0e-8f) k_scale = 1.0f;
    }
    if constexpr (V_FP8) {
        v_base_fp8 = v_cache_fp8 + kv_head * max_seq_len * head_dim;
        v_scale = v_scale_cache[kv_head];
        if (v_scale < 1.0e-8f) v_scale = 1.0f;
    }
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        int d = lane_id * kElemsPerLane + i;
        float qv0 = __bfloat162float(q_vec0[d]);
        float qv1 = __bfloat162float(q_vec1[d]);
        if constexpr (K_FP8) {
            qv0 *= k_scale;
            qv1 *= k_scale;
        }
        q_local0[i] = qv0;
        q_local1[i] = qv1;
    }
    q_local0_h2[0] = __floats2half2_rn(q_local0[0], q_local0[1]);
    q_local0_h2[1] = __floats2half2_rn(q_local0[2], q_local0[3]);
    q_local1_h2[0] = __floats2half2_rn(q_local1[0], q_local1[1]);
    q_local1_h2[1] = __floats2half2_rn(q_local1[2], q_local1[3]);
    if (use_fp8_tc_qk) {
        for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
            float qv0 = __bfloat162float(q_vec0[d]) * k_scale;
            sm_q_tc0[d] = __float2half_rn(qv0);
            float qv1 = __bfloat162float(q_vec1[d]) * k_scale;
            sm_q_tc1[d] = __float2half_rn(qv1);
        }
        __syncthreads();
    }
    bool paged = (kv_block_table != nullptr && kv_block_size > 0);
    bool use_blocking = (kv_block_size > 0);

    if (warp_id < active_warps) {
        int warp_a_base = warp_id * kTcM * kTcK;
        int warp_b_base = warp_id * kTcK * kTcN;
        int warp_c_base = warp_id * kTcM * kTcN;
        int local_tokens = 0;
        int local_switches = 0;
        int local_oob = 0;
        int local_tiles = 0;
        unsigned long long local_map_cycles = 0;
        unsigned long long local_qk_cycles = 0;
        unsigned long long local_qk_load_cycles = 0;
        unsigned long long local_qk_load_mem_cycles = 0;
        unsigned long long local_qk_load_cvt_cycles = 0;
        unsigned long long local_qk_load_bf16_cycles = 0;
        unsigned long long local_qk_fma_cycles = 0;
        unsigned long long local_qk_fma_bf16_cycles = 0;
        unsigned long long local_qk_reduce_cycles = 0;
        unsigned long long local_qk_tc_pack_cycles = 0;
        unsigned long long local_qk_tc_mma_cycles = 0;
        unsigned long long local_qk_tc_unpack_cycles = 0;
        unsigned long long local_softmax_cycles = 0;
        unsigned long long local_value_cycles = 0;
        unsigned long long local_value_load_cycles = 0;
        unsigned long long local_value_fma_cycles = 0;
        unsigned long long local_value_dequant_cycles = 0;
        unsigned long long local_value_rescale_cycles = 0;
        unsigned long long local_value_accum_cycles = 0;
        unsigned long long local_value_bookkeep_cycles = 0;
        bool debug_lane = (ENABLE_DEBUG && lane_id == 0);
        int block_size = (use_blocking ? kv_block_size : max_seq_len);
        if (block_size <= 0) block_size = max_seq_len;
        if (block_size <= 0) block_size = 1;
        int chunk_block_start = t_start / block_size;
        int chunk_block_end = (t_end + block_size - 1) / block_size;
        int chunk_blocks = chunk_block_end - chunk_block_start;
        int blocks_per_warp = (chunk_blocks + active_warps - 1) / active_warps;
        int warp_block_start = chunk_block_start + warp_id * blocks_per_warp;
        int warp_block_end = warp_block_start + blocks_per_warp;
        if (warp_block_start < chunk_block_start) warp_block_start = chunk_block_start;
        if (warp_block_end > chunk_block_end) warp_block_end = chunk_block_end;
        if (use_fp8_tc_qk) {
            int a_row_start = 2;
            for (int idx = lane_id; idx < (kTcM - a_row_start) * kTcK; idx += WARP_SIZE) {
                int r = a_row_start + (idx / kTcK);
                int c = idx - (idx / kTcK) * kTcK;
                sm_tc_a[warp_a_base + r * kTcK + c] = __float2half_rn(0.0f);
            }
            for (int idx = lane_id; idx < (kTcN - kFlashTile) * kTcK; idx += WARP_SIZE) {
                int c = kFlashTile + (idx / kTcK);
                int r = idx - (idx / kTcK) * kTcK;
                sm_tc_b[warp_b_base + c * kTcK + r] = __float2half_rn(0.0f);
            }
            __syncwarp();
        }

        for (int logical_block = warp_block_start; logical_block < warp_block_end; logical_block++) {
            int block_token_begin = logical_block * block_size;
            int block_token_end = block_token_begin + block_size;
            if (block_token_begin < t_start) block_token_begin = t_start;
            if (block_token_end > t_end) block_token_end = t_end;
            if (block_token_begin >= block_token_end) continue;

            int physical_block = logical_block;
            if (paged) physical_block = kv_block_table[logical_block];
            int physical_block_base = physical_block * block_size;
            local_switches += 1;

            for (int tile_start = block_token_begin; tile_start < block_token_end; tile_start += kFlashTile) {
                local_tiles += 1;
                bool full_tile = (tile_start + kFlashTile <= block_token_end);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
                if (use_fp8_tc_qk && full_tile) {
                    int physical_tile[kFlashTile];
                    unsigned valid_mask = 0u;
                    #pragma unroll
                    for (int ti = 0; ti < kFlashTile; ti++) {
                        unsigned long long t_map_begin = 0;
                        if (debug_lane) t_map_begin = clock64();
                        int t = tile_start + ti;
                        int in_block = t - logical_block * block_size;
                        int physical_t = physical_block_base + in_block;
                        if (debug_lane) local_map_cycles += (clock64() - t_map_begin);
                        physical_tile[ti] = physical_t;
                        if (physical_t >= max_seq_len) {
                            local_oob += 1;
                            physical_tile[ti] = -1;
                        } else {
                            local_tokens += 1;
                            valid_mask |= (1u << ti);
                        }
                    }

                    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kTcM, kTcN, kTcK, float> c_frag;
                    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
                    constexpr int kTcVec = 4;
                    constexpr int kTcColsPerLane = kTcK / kTcVec;
                    static_assert(kFlashTile * kTcColsPerLane == WARP_SIZE,
                                  "TC pack mapping must fully cover one tile per warp.");
                    constexpr int kTcSteps = HEAD_DIM / kTcK;
                    int lane_pack = lane_id;
                    bool lane_pack_active = (lane_pack < kFlashTile * kTcColsPerLane);
                    int lane_pack_c = 0;
                    int lane_pack_r4 = 0;
                    __half* lane_b_ptr = nullptr;
                    const __nv_fp8_e4m3* lane_k_ptr_base = nullptr;
                    __half2 lane_h01_cache[kTcSteps];
                    __half2 lane_h23_cache[kTcSteps];
                    #pragma unroll
                    for (int kk = 0; kk < kTcSteps; kk++) {
                        lane_h01_cache[kk] = __float2half2_rn(0.0f);
                        lane_h23_cache[kk] = __float2half2_rn(0.0f);
                    }
                    if (lane_pack_active) {
                        lane_pack_c = lane_pack / kTcColsPerLane;   // 0..kFlashTile-1
                        lane_pack_r4 = (lane_pack % kTcColsPerLane) * kTcVec;  // 0/4/8/12
                        lane_b_ptr = sm_tc_b + warp_b_base + lane_pack_c * kTcK + lane_pack_r4;
                        int physical_t = physical_tile[lane_pack_c];
                        if (physical_t >= 0) {
                            lane_k_ptr_base = k_base_fp8 + physical_t * head_dim + lane_pack_r4;
                            #pragma unroll
                            for (int kk = 0; kk < kTcSteps; kk++) {
                                unsigned long long t_kmem_begin = 0;
                                if (debug_lane) t_kmem_begin = clock64();
                                SplitFp8x4Pack kpack = split_load_fp8x4(lane_k_ptr_base + kk * kTcK);
                                if (debug_lane) {
                                    unsigned long long kmem = (clock64() - t_kmem_begin);
                                    local_qk_load_mem_cycles += kmem;
                                    local_qk_load_cycles += kmem;
                                    local_qk_cycles += kmem;
                                }
                                unsigned long long t_kcvt_begin = 0;
                                if (debug_lane) t_kcvt_begin = clock64();
                                SplitHalf2x2Pack kh2 = split_fp8x4_to_half2x2(kpack);
                                lane_h01_cache[kk] = kh2.h01;
                                lane_h23_cache[kk] = kh2.h23;
                                if (debug_lane) {
                                    unsigned long long kcvt = (clock64() - t_kcvt_begin);
                                    local_qk_load_cvt_cycles += kcvt;
                                    local_qk_load_cycles += kcvt;
                                    local_qk_cycles += kcvt;
                                }
                            }
                        }
                    }

                    #pragma unroll
                    for (int kk = 0; kk < kTcSteps; kk++) {
                        unsigned long long t_tc_pack_begin = 0;
                        if (debug_lane) t_tc_pack_begin = clock64();
                        int kk_offset = kk * kTcK;
                        for (int c = lane_id; c < kTcK; c += WARP_SIZE) {
                            sm_tc_a[warp_a_base + c] = sm_q_tc0[kk_offset + c];
                            sm_tc_a[warp_a_base + kTcK + c] = sm_q_tc1[kk_offset + c];
                        }
                        if (lane_pack_active) {
                            *reinterpret_cast<__half2*>(lane_b_ptr) = lane_h01_cache[kk];
                            *reinterpret_cast<__half2*>(lane_b_ptr + 2) = lane_h23_cache[kk];
                        }
                        if (debug_lane) {
                            unsigned long long tc_pack_total = (clock64() - t_tc_pack_begin);
                            local_qk_tc_pack_cycles += tc_pack_total;
                            local_qk_cycles += tc_pack_total;
                        }
                        __syncwarp();
                        unsigned long long t_tc_mma_begin = 0;
                        if (debug_lane) t_tc_mma_begin = clock64();
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kTcM, kTcN, kTcK, __half, nvcuda::wmma::row_major> a_frag;
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kTcM, kTcN, kTcK, __half, nvcuda::wmma::col_major> b_frag;
                        nvcuda::wmma::load_matrix_sync(a_frag, sm_tc_a + warp_a_base, kTcK);
                        nvcuda::wmma::load_matrix_sync(b_frag, sm_tc_b + warp_b_base, kTcK);
                        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                        if (debug_lane) {
                            unsigned long long tc_mma = (clock64() - t_tc_mma_begin);
                            local_qk_tc_mma_cycles += tc_mma;
                            local_qk_cycles += tc_mma;
                        }
                    }
                    unsigned long long t_tc_unpack_begin = 0;
                    if (debug_lane) t_tc_unpack_begin = clock64();
                    nvcuda::wmma::store_matrix_sync(sm_tc_c + warp_c_base, c_frag, kTcN, nvcuda::wmma::mem_row_major);
                    __syncwarp();

                    float score0_lane = -INFINITY;
                    float score1_lane = -INFINITY;
                    if (lane_id < kFlashTile) {
                        if (valid_mask & (1u << lane_id)) {
                            score0_lane = sm_tc_c[warp_c_base + lane_id];
                            score1_lane = sm_tc_c[warp_c_base + kTcN + lane_id];
                        }
                    }
                    if (debug_lane) {
                        unsigned long long tc_unpack = (clock64() - t_tc_unpack_begin);
                        local_qk_tc_unpack_cycles += tc_unpack;
                        local_qk_cycles += tc_unpack;
                    }
                    int d0 = lane_id * kElemsPerLane;
                    SplitFp8x4Pack vpack_prefetch[kFlashTile];
                    unsigned long long t_vprefetch_begin = 0;
                    if (debug_lane) t_vprefetch_begin = clock64();
                    #pragma unroll
                    for (int ti = 0; ti < kFlashTile; ti++) {
                        if ((valid_mask & (1u << ti)) == 0) {
                            vpack_prefetch[ti].raw = 0u;
                            continue;
                        }
                        int physical_t = physical_tile[ti];
                        const __nv_fp8_e4m3* v_vec_fp8 = v_base_fp8 + physical_t * head_dim;
                        vpack_prefetch[ti] = split_load_fp8x4(v_vec_fp8 + d0);
                    }
                    if (debug_lane) local_value_load_cycles += (clock64() - t_vprefetch_begin);
                    #pragma unroll
                    for (int ti = 0; ti < kFlashTile; ti++) {
                        if ((valid_mask & (1u << ti)) == 0) continue;
                        unsigned long long t_qk_reduce_begin = 0;
                        if (debug_lane) t_qk_reduce_begin = clock64();
                        float score0 = __shfl_sync(0xffffffff, score0_lane, ti) * attn_scale;
                        float score1 = __shfl_sync(0xffffffff, score1_lane, ti) * attn_scale;
                        if (debug_lane) {
                            unsigned long long t_qk_reduce = (clock64() - t_qk_reduce_begin);
                            local_qk_reduce_cycles += t_qk_reduce;
                            local_qk_cycles += t_qk_reduce;
                        }

                        unsigned long long t_softmax_begin = 0;
                        if (debug_lane) t_softmax_begin = clock64();
                        float m_new0 = fmaxf(m0, score0);
                        float old_scale0 = expf(m0 - m_new0);
                        float new_scale0 = expf(score0 - m_new0);
                        s0 = s0 * old_scale0 + new_scale0;
                        float m_new1 = m1;
                        float old_scale1 = 1.0f;
                        float new_scale1 = 0.0f;
                        m_new1 = fmaxf(m1, score1);
                        old_scale1 = expf(m1 - m_new1);
                        new_scale1 = expf(score1 - m_new1);
                        s1 = s1 * old_scale1 + new_scale1;
                        if (debug_lane) local_softmax_cycles += (clock64() - t_softmax_begin);

                        unsigned long long t_value_begin = 0;
                        if (debug_lane) t_value_begin = clock64();
                        const SplitFp8x4Pack& vpack = vpack_prefetch[ti];
                        unsigned long long t_vfma_begin = 0;
                        if (debug_lane) t_vfma_begin = clock64();
                        unsigned long long t_vdeq_begin = 0;
                        if (debug_lane) t_vdeq_begin = clock64();
                        float v0 = static_cast<float>(vpack.v[0]);
                        float v1 = static_cast<float>(vpack.v[1]);
                        float v2 = static_cast<float>(vpack.v[2]);
                        float v3 = static_cast<float>(vpack.v[3]);
                        if (debug_lane) local_value_dequant_cycles += (clock64() - t_vdeq_begin);

                        unsigned long long t_vrescale_begin = 0;
                        if (debug_lane) t_vrescale_begin = clock64();
                        float out0_old0 = out_local0[0] * old_scale0;
                        float out0_old1 = out_local0[1] * old_scale0;
                        float out0_old2 = out_local0[2] * old_scale0;
                        float out0_old3 = out_local0[3] * old_scale0;
                        float out1_old0 = out_local1[0] * old_scale1;
                        float out1_old1 = out_local1[1] * old_scale1;
                        float out1_old2 = out_local1[2] * old_scale1;
                        float out1_old3 = out_local1[3] * old_scale1;
                        if (debug_lane) local_value_rescale_cycles += (clock64() - t_vrescale_begin);

                        unsigned long long t_vaccum_begin = 0;
                        if (debug_lane) t_vaccum_begin = clock64();
                        out_local0[0] = fmaf(v0, new_scale0, out0_old0);
                        out_local0[1] = fmaf(v1, new_scale0, out0_old1);
                        out_local0[2] = fmaf(v2, new_scale0, out0_old2);
                        out_local0[3] = fmaf(v3, new_scale0, out0_old3);
                        out_local1[0] = fmaf(v0, new_scale1, out1_old0);
                        out_local1[1] = fmaf(v1, new_scale1, out1_old1);
                        out_local1[2] = fmaf(v2, new_scale1, out1_old2);
                        out_local1[3] = fmaf(v3, new_scale1, out1_old3);
                        if (debug_lane) local_value_accum_cycles += (clock64() - t_vaccum_begin);
                        if (debug_lane) local_value_fma_cycles += (clock64() - t_vfma_begin);
                        if (debug_lane) local_value_cycles += (clock64() - t_value_begin);
                        unsigned long long t_vbook_begin = 0;
                        if (debug_lane) t_vbook_begin = clock64();
                        m0 = m_new0;
                        m1 = m_new1;
                        if (debug_lane) local_value_bookkeep_cycles += (clock64() - t_vbook_begin);
                    }
                } else
#endif
                {
                    #pragma unroll
                    for (int ti = 0; ti < kFlashTile; ti++) {
                        unsigned long long t_map_begin = 0;
                        if (debug_lane) t_map_begin = clock64();
                        int t = tile_start + ti;
                        if (t >= block_token_end) break;
                        int in_block = t - logical_block * block_size;
                        int physical_t = physical_block_base + in_block;
                        if (debug_lane) local_map_cycles += (clock64() - t_map_begin);
                        if (physical_t >= max_seq_len) {
                            local_oob += 1;
                            continue;
                        }
                        local_tokens += 1;

                        const __nv_bfloat16* k_vec = k_base + physical_t * head_dim;
                        const __nv_bfloat16* v_vec = v_base + physical_t * head_dim;
                        const __nv_fp8_e4m3* k_vec_fp8 = K_FP8 ? (k_base_fp8 + physical_t * head_dim) : nullptr;
                        const __nv_fp8_e4m3* v_vec_fp8 = V_FP8 ? (v_base_fp8 + physical_t * head_dim) : nullptr;
                        SplitFp8x4Pack vpack_prefetch{};
                        if constexpr (V_FP8) {
                            int d0 = lane_id * kElemsPerLane;
                            unsigned long long t_vload_begin = 0;
                            if (debug_lane) t_vload_begin = clock64();
                            vpack_prefetch = split_load_fp8x4(v_vec_fp8 + d0);
                            if (debug_lane) local_value_load_cycles += (clock64() - t_vload_begin);
                        }

                        unsigned long long t_qk_begin = 0;
                        if (debug_lane) t_qk_begin = clock64();
                        float dot0 = 0.0f;
                        float dot1 = 0.0f;
                        if constexpr (K_FP8) {
                            int d0 = lane_id * kElemsPerLane;
                            unsigned long long t_load_begin = 0;
                            if (debug_lane) t_load_begin = clock64();
                            SplitFp8x4Pack kpack = split_load_fp8x4(k_vec_fp8 + d0);
                            if (debug_lane) {
                                unsigned long long kmem = (clock64() - t_load_begin);
                                local_qk_load_mem_cycles += kmem;
                                local_qk_load_cycles += kmem;
                            }
                            unsigned long long t_cvt_begin = 0;
                            if (debug_lane) t_cvt_begin = clock64();
                            SplitHalf2x2Pack kh2 = split_fp8x4_to_half2x2(kpack);
                            if (debug_lane) {
                                unsigned long long kcvt = (clock64() - t_cvt_begin);
                                local_qk_load_cvt_cycles += kcvt;
                                local_qk_load_cycles += kcvt;
                            }
                            unsigned long long t_fma_begin = 0;
                            if (debug_lane) t_fma_begin = clock64();
                            __half2 k01 = kh2.h01;
                            __half2 k23 = kh2.h23;
                            __half2 p001 = __hmul2(q_local0_h2[0], k01);
                            __half2 p023 = __hmul2(q_local0_h2[1], k23);
                            dot0 += __half2float(__low2half(p001)) + __half2float(__high2half(p001));
                            dot0 += __half2float(__low2half(p023)) + __half2float(__high2half(p023));
                            __half2 p101 = __hmul2(q_local1_h2[0], k01);
                            __half2 p123 = __hmul2(q_local1_h2[1], k23);
                            dot1 += __half2float(__low2half(p101)) + __half2float(__high2half(p101));
                            dot1 += __half2float(__low2half(p123)) + __half2float(__high2half(p123));
                            if (debug_lane) local_qk_fma_cycles += (clock64() - t_fma_begin);
                        } else {
                            #pragma unroll
                            for (int i = 0; i < kElemsPerLane; i++) {
                                int d = lane_id * kElemsPerLane + i;
                                unsigned long long t_load_begin = 0;
                                if (debug_lane) t_load_begin = clock64();
                                float kval = __bfloat162float(k_vec[d]);
                                if (debug_lane) {
                                    unsigned long long bf16_load = (clock64() - t_load_begin);
                                    local_qk_load_cycles += bf16_load;
                                    local_qk_load_bf16_cycles += bf16_load;
                                }
                                unsigned long long t_fma_begin = 0;
                                if (debug_lane) t_fma_begin = clock64();
                                dot0 += q_local0[i] * kval;
                                dot1 += q_local1[i] * kval;
                                if (debug_lane) {
                                    unsigned long long bf16_fma = (clock64() - t_fma_begin);
                                    local_qk_fma_cycles += bf16_fma;
                                    local_qk_fma_bf16_cycles += bf16_fma;
                                }
                            }
                        }
                        unsigned long long t_qk_reduce_begin = 0;
                        if (debug_lane) t_qk_reduce_begin = clock64();
                        SplitFloat2 dot_pair{dot0, dot1};
                        dot_pair = split_warp_reduce_sum2(dot_pair);
                        dot0 = dot_pair.x * attn_scale;
                        dot1 = dot_pair.y * attn_scale;
                        float score0 = __shfl_sync(0xffffffff, dot0, 0);
                        float score1 = __shfl_sync(0xffffffff, dot1, 0);
                        if (debug_lane) local_qk_reduce_cycles += (clock64() - t_qk_reduce_begin);
                        if (debug_lane) local_qk_cycles += (clock64() - t_qk_begin);

                        unsigned long long t_softmax_begin = 0;
                        if (debug_lane) t_softmax_begin = clock64();
                        float m_new0 = fmaxf(m0, score0);
                        float old_scale0 = expf(m0 - m_new0);
                        float new_scale0 = expf(score0 - m_new0);
                        s0 = s0 * old_scale0 + new_scale0;
                        float m_new1 = m1;
                        float old_scale1 = 1.0f;
                        float new_scale1 = 0.0f;
                        m_new1 = fmaxf(m1, score1);
                        old_scale1 = expf(m1 - m_new1);
                        new_scale1 = expf(score1 - m_new1);
                        s1 = s1 * old_scale1 + new_scale1;
                        if (debug_lane) local_softmax_cycles += (clock64() - t_softmax_begin);

                        unsigned long long t_value_begin = 0;
                        if (debug_lane) t_value_begin = clock64();
                        if constexpr (V_FP8) {
                            const SplitFp8x4Pack& vpack = vpack_prefetch;
                            unsigned long long t_vfma_begin = 0;
                            if (debug_lane) t_vfma_begin = clock64();
                            unsigned long long t_vdeq_begin = 0;
                            if (debug_lane) t_vdeq_begin = clock64();
                            float v0 = static_cast<float>(vpack.v[0]);
                            float v1 = static_cast<float>(vpack.v[1]);
                            float v2 = static_cast<float>(vpack.v[2]);
                            float v3 = static_cast<float>(vpack.v[3]);
                            if (debug_lane) local_value_dequant_cycles += (clock64() - t_vdeq_begin);

                            unsigned long long t_vrescale_begin = 0;
                            if (debug_lane) t_vrescale_begin = clock64();
                            float out0_old0 = out_local0[0] * old_scale0;
                            float out0_old1 = out_local0[1] * old_scale0;
                            float out0_old2 = out_local0[2] * old_scale0;
                            float out0_old3 = out_local0[3] * old_scale0;
                            float out1_old0 = out_local1[0] * old_scale1;
                            float out1_old1 = out_local1[1] * old_scale1;
                            float out1_old2 = out_local1[2] * old_scale1;
                            float out1_old3 = out_local1[3] * old_scale1;
                            if (debug_lane) local_value_rescale_cycles += (clock64() - t_vrescale_begin);

                            unsigned long long t_vaccum_begin = 0;
                            if (debug_lane) t_vaccum_begin = clock64();
                            out_local0[0] = fmaf(v0, new_scale0, out0_old0);
                            out_local0[1] = fmaf(v1, new_scale0, out0_old1);
                            out_local0[2] = fmaf(v2, new_scale0, out0_old2);
                            out_local0[3] = fmaf(v3, new_scale0, out0_old3);
                            out_local1[0] = fmaf(v0, new_scale1, out1_old0);
                            out_local1[1] = fmaf(v1, new_scale1, out1_old1);
                            out_local1[2] = fmaf(v2, new_scale1, out1_old2);
                            out_local1[3] = fmaf(v3, new_scale1, out1_old3);
                            if (debug_lane) local_value_accum_cycles += (clock64() - t_vaccum_begin);
                            if (debug_lane) local_value_fma_cycles += (clock64() - t_vfma_begin);
                        } else {
                            #pragma unroll
                            for (int i = 0; i < kElemsPerLane; i++) {
                                int d = lane_id * kElemsPerLane + i;
                                unsigned long long t_vload_begin = 0;
                                if (debug_lane) t_vload_begin = clock64();
                                float vv = __bfloat162float(v_vec[d]);
                                if (debug_lane) local_value_load_cycles += (clock64() - t_vload_begin);
                                unsigned long long t_vfma_begin = 0;
                                if (debug_lane) t_vfma_begin = clock64();
                                unsigned long long t_vdeq_begin = 0;
                                if (debug_lane) t_vdeq_begin = clock64();
                                if (debug_lane) local_value_dequant_cycles += (clock64() - t_vdeq_begin);
                                unsigned long long t_vrescale_begin = 0;
                                if (debug_lane) t_vrescale_begin = clock64();
                                float out0_old = out_local0[i] * old_scale0;
                                float out1_old = out_local1[i] * old_scale1;
                                if (debug_lane) local_value_rescale_cycles += (clock64() - t_vrescale_begin);
                                unsigned long long t_vaccum_begin = 0;
                                if (debug_lane) t_vaccum_begin = clock64();
                                out_local0[i] = fmaf(vv, new_scale0, out0_old);
                                out_local1[i] = fmaf(vv, new_scale1, out1_old);
                                if (debug_lane) local_value_accum_cycles += (clock64() - t_vaccum_begin);
                                if (debug_lane) local_value_fma_cycles += (clock64() - t_vfma_begin);
                            }
                        }
                        if (debug_lane) local_value_cycles += (clock64() - t_value_begin);
                        unsigned long long t_vbook_begin = 0;
                        if (debug_lane) t_vbook_begin = clock64();
                        m0 = m_new0;
                        m1 = m_new1;
                        if (debug_lane) local_value_bookkeep_cycles += (clock64() - t_vbook_begin);
                    }
                }
            }
        }
        if constexpr (ENABLE_DEBUG) {
            if (lane_id == 0) {
            atomicAdd(&sm_tokens, local_tokens);
            atomicAdd(&sm_switches, local_switches);
            atomicAdd(&sm_oob, local_oob);
            atomicAdd(&sm_tiles, local_tiles);
            atomicAdd(&sm_map_cycles, local_map_cycles);
            atomicAdd(&sm_qk_cycles, local_qk_cycles);
            atomicAdd(&sm_qk_load_cycles, local_qk_load_cycles);
            atomicAdd(&sm_qk_load_mem_cycles, local_qk_load_mem_cycles);
            atomicAdd(&sm_qk_load_cvt_cycles, local_qk_load_cvt_cycles);
            atomicAdd(&sm_qk_load_bf16_cycles, local_qk_load_bf16_cycles);
            atomicAdd(&sm_qk_fma_cycles, local_qk_fma_cycles);
            atomicAdd(&sm_qk_fma_bf16_cycles, local_qk_fma_bf16_cycles);
            atomicAdd(&sm_qk_reduce_cycles, local_qk_reduce_cycles);
            atomicAdd(&sm_qk_tc_pack_cycles, local_qk_tc_pack_cycles);
            atomicAdd(&sm_qk_tc_mma_cycles, local_qk_tc_mma_cycles);
            atomicAdd(&sm_qk_tc_unpack_cycles, local_qk_tc_unpack_cycles);
            atomicAdd(&sm_softmax_cycles, local_softmax_cycles);
            atomicAdd(&sm_value_cycles, local_value_cycles);
            atomicAdd(&sm_value_load_cycles, local_value_load_cycles);
            atomicAdd(&sm_value_fma_cycles, local_value_fma_cycles);
            atomicAdd(&sm_value_dequant_cycles, local_value_dequant_cycles);
            atomicAdd(&sm_value_rescale_cycles, local_value_rescale_cycles);
            atomicAdd(&sm_value_accum_cycles, local_value_accum_cycles);
            atomicAdd(&sm_value_bookkeep_cycles, local_value_bookkeep_cycles);
            }
        }
    }

    if (lane_id == 0) {
        sm_m0[warp_id] = (warp_id < active_warps) ? m0 : -INFINITY;
        sm_s0[warp_id] = (warp_id < active_warps) ? s0 : 0.0f;
        sm_m1[warp_id] = (warp_id < active_warps) ? m1 : -INFINITY;
        sm_s1[warp_id] = (warp_id < active_warps) ? s1 : 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        int d = lane_id * kElemsPerLane + i;
        sm_out0[warp_id * HEAD_DIM + d] = (warp_id < active_warps) ? out_local0[i] : 0.0f;
        sm_out1[warp_id * HEAD_DIM + d] = (warp_id < active_warps) ? out_local1[i] : 0.0f;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned long long merge_begin = 0;
        if constexpr (ENABLE_DEBUG) merge_begin = clock64();
        float g_m0 = -INFINITY;
        for (int w = 0; w < active_warps; w++) g_m0 = fmaxf(g_m0, sm_m0[w]);
        float g_s0 = 0.0f;
        for (int w = 0; w < active_warps; w++) {
            float scale = (sm_s0[w] > 0.0f) ? expf(sm_m0[w] - g_m0) : 0.0f;
            sm_scale0[w] = scale;
            g_s0 += sm_s0[w] * scale;
        }
        sm_global_m0 = g_m0;
        sm_global_s0 = g_s0;

        float g_m1 = -INFINITY;
        float g_s1 = 0.0f;
        for (int w = 0; w < active_warps; w++) g_m1 = fmaxf(g_m1, sm_m1[w]);
        for (int w = 0; w < active_warps; w++) {
            float scale = (sm_s1[w] > 0.0f) ? expf(sm_m1[w] - g_m1) : 0.0f;
            sm_scale1[w] = scale;
            g_s1 += sm_s1[w] * scale;
        }
        sm_global_m1 = g_m1;
        sm_global_s1 = g_s1;
        if constexpr (ENABLE_DEBUG) sm_merge_cycles += (clock64() - merge_begin);
    }
    __syncthreads();

    int idx0 = (part_idx * max_chunks + chunk_idx) * num_q_heads + q_head0;
    int idx1 = idx0 + 1;
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        int d = lane_id * kElemsPerLane + i;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        #pragma unroll
        for (int w = 0; w < kMaxWarps; w++) {
            if (w < active_warps) {
                acc0 += sm_out0[w * HEAD_DIM + d] * sm_scale0[w];
                acc1 += sm_out1[w * HEAD_DIM + d] * sm_scale1[w];
            }
        }
        if constexpr (V_FP8) {
            acc0 *= v_scale;
            acc1 *= v_scale;
        }
        partial_out[idx0 * HEAD_DIM + d] = acc0;
        partial_out[idx1 * HEAD_DIM + d] = acc1;
    }
    if (threadIdx.x == 0) {
        partial_m[idx0] = sm_global_m0;
        partial_s[idx0] = sm_global_s0;
        partial_m[idx1] = sm_global_m1;
        partial_s[idx1] = sm_global_s1;
        if constexpr (ENABLE_DEBUG) {
            SplitFlashDecodeDebugRec rec{};
            rec.cycles = clock64() - sm_clock_begin;
            rec.cache_len = cache_len;
            rec.active_warps = active_warps;
            rec.tokens_processed = sm_tokens;
            rec.block_switches = sm_switches;
            rec.oob_skips = sm_oob;
            rec.tiles_processed = sm_tiles;
            rec.map_cycles = sm_map_cycles;
            rec.qk_cycles = sm_qk_cycles;
            rec.qk_load_cycles = sm_qk_load_cycles;
            rec.qk_load_mem_cycles = sm_qk_load_mem_cycles;
            rec.qk_load_cvt_cycles = sm_qk_load_cvt_cycles;
            rec.qk_load_bf16_cycles = sm_qk_load_bf16_cycles;
            rec.qk_fma_cycles = sm_qk_fma_cycles;
            rec.qk_fma_bf16_cycles = sm_qk_fma_bf16_cycles;
            rec.qk_reduce_cycles = sm_qk_reduce_cycles;
            rec.qk_tc_pack_cycles = sm_qk_tc_pack_cycles;
            rec.qk_tc_mma_cycles = sm_qk_tc_mma_cycles;
            rec.qk_tc_unpack_cycles = sm_qk_tc_unpack_cycles;
            rec.softmax_cycles = sm_softmax_cycles;
            rec.value_cycles = sm_value_cycles;
            rec.value_load_cycles = sm_value_load_cycles;
            rec.value_fma_cycles = sm_value_fma_cycles;
            rec.value_dequant_cycles = sm_value_dequant_cycles;
            rec.value_rescale_cycles = sm_value_rescale_cycles;
            rec.value_accum_cycles = sm_value_accum_cycles;
            rec.value_bookkeep_cycles = sm_value_bookkeep_cycles;
            rec.merge_cycles = sm_merge_cycles;
            debug_out[idx0] = rec;
            debug_out[idx1] = rec;
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
template <bool KV_FP8>
__global__ void decode_attention_cache_flash_decode_kernel_t(
    const __nv_bfloat16* __restrict__ q,             // [num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_cache,       // [num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,       // [num_kv_heads, max_seq_len, head_dim]
    const __nv_fp8_e4m3* __restrict__ k_cache_fp8,   // optional [num_kv_heads, max_seq_len, head_dim]
    const __nv_fp8_e4m3* __restrict__ v_cache_fp8,   // optional [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ k_scale_cache,         // optional [num_kv_heads]
    const float* __restrict__ v_scale_cache,         // optional [num_kv_heads]
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
    __shared__ unsigned long long sm_map_cycles;
    __shared__ unsigned long long sm_qk_cycles;
    __shared__ unsigned long long sm_qk_load_cycles;
    __shared__ unsigned long long sm_qk_load_bf16_cycles;
    __shared__ unsigned long long sm_qk_fma_cycles;
    __shared__ unsigned long long sm_qk_fma_bf16_cycles;
    __shared__ unsigned long long sm_qk_reduce_cycles;
    __shared__ unsigned long long sm_softmax_cycles;
    __shared__ unsigned long long sm_value_cycles;
    __shared__ unsigned long long sm_value_load_cycles;
    __shared__ unsigned long long sm_value_fma_cycles;
    __shared__ unsigned long long sm_merge_cycles;
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
            sm_map_cycles = 0;
            sm_qk_cycles = 0;
            sm_qk_load_cycles = 0;
            sm_qk_load_bf16_cycles = 0;
            sm_qk_fma_cycles = 0;
            sm_qk_fma_bf16_cycles = 0;
            sm_qk_reduce_cycles = 0;
            sm_softmax_cycles = 0;
            sm_value_cycles = 0;
            sm_value_load_cycles = 0;
            sm_value_fma_cycles = 0;
            sm_merge_cycles = 0;
            sm_clock_begin = clock64();
        }
    }
    __syncthreads();
    active_warps = sm_active_warps;
    if (warp_id >= active_warps) return;

    const __nv_bfloat16* q_vec = q + q_head * HEAD_DIM;
    __nv_bfloat16* out_vec = out + q_head * HEAD_DIM;
    float q_local[kElemsPerLane];

    float m = -INFINITY;
    float s = 0.0f;
    float out_local[kElemsPerLane] = {0.f, 0.f, 0.f, 0.f};

    const __nv_bfloat16* k_base = k_cache + kv_head * max_seq_len * HEAD_DIM;
    const __nv_bfloat16* v_base = v_cache + kv_head * max_seq_len * HEAD_DIM;
    const __nv_fp8_e4m3* k_base_fp8 = nullptr;
    const __nv_fp8_e4m3* v_base_fp8 = nullptr;
    float k_scale = 1.0f;
    float v_scale = 1.0f;
    if constexpr (KV_FP8) {
        k_base_fp8 = k_cache_fp8 + kv_head * max_seq_len * HEAD_DIM;
        v_base_fp8 = v_cache_fp8 + kv_head * max_seq_len * HEAD_DIM;
        k_scale = k_scale_cache[kv_head];
        v_scale = v_scale_cache[kv_head];
        if (k_scale < 1.0e-8f) k_scale = 1.0f;
        if (v_scale < 1.0e-8f) v_scale = 1.0f;
    }
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        int d = lane_id * kElemsPerLane + i;
        float qv = __bfloat162float(q_vec[d]);
        if constexpr (KV_FP8) qv *= k_scale;
        q_local[i] = qv;
    }
    bool paged = (kv_block_table != nullptr && kv_block_size > 0);

    int cached_logical_block = -1;
    int cached_physical_block = 0;
    int local_tokens = 0;
    int local_block_switches = 0;
    int local_oob_skips = 0;
    int local_tiles = 0;
    unsigned long long local_map_cycles = 0;
    unsigned long long local_qk_cycles = 0;
    unsigned long long local_qk_load_cycles = 0;
    unsigned long long local_qk_load_bf16_cycles = 0;
    unsigned long long local_qk_fma_cycles = 0;
    unsigned long long local_qk_fma_bf16_cycles = 0;
    unsigned long long local_qk_reduce_cycles = 0;
    unsigned long long local_softmax_cycles = 0;
    unsigned long long local_value_cycles = 0;
    unsigned long long local_value_load_cycles = 0;
    unsigned long long local_value_fma_cycles = 0;
    bool debug_lane = (debug_out != nullptr && lane_id == 0);
    for (int tile_start = warp_id * kFlashTile;
         tile_start < cache_len;
         tile_start += active_warps * kFlashTile) {
        local_tiles += 1;
        #pragma unroll
        for (int ti = 0; ti < kFlashTile; ti++) {
            unsigned long long t_map_begin = 0;
            if (debug_lane) t_map_begin = clock64();
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
            if (debug_lane) local_map_cycles += (clock64() - t_map_begin);
            if (physical_t >= max_seq_len) {
                local_oob_skips += 1;
                continue;
            }
            local_tokens += 1;

            const __nv_bfloat16* k_vec = k_base + physical_t * HEAD_DIM;
            const __nv_bfloat16* v_vec = v_base + physical_t * HEAD_DIM;
            const __nv_fp8_e4m3* k_vec_fp8 = KV_FP8 ? (k_base_fp8 + physical_t * HEAD_DIM) : nullptr;
            const __nv_fp8_e4m3* v_vec_fp8 = KV_FP8 ? (v_base_fp8 + physical_t * HEAD_DIM) : nullptr;
            unsigned long long t_qk_begin = 0;
            if (debug_lane) t_qk_begin = clock64();
            float dot = 0.0f;
            if constexpr (KV_FP8) {
                int d0 = lane_id * kElemsPerLane;
                unsigned long long t_load_begin = 0;
                if (debug_lane) t_load_begin = clock64();
                SplitFp8x4Pack kpack = split_load_fp8x4(k_vec_fp8 + d0);
                if (debug_lane) local_qk_load_cycles += (clock64() - t_load_begin);
                unsigned long long t_fma_begin = 0;
                if (debug_lane) t_fma_begin = clock64();
                dot += q_local[0] * static_cast<float>(kpack.v[0]);
                dot += q_local[1] * static_cast<float>(kpack.v[1]);
                dot += q_local[2] * static_cast<float>(kpack.v[2]);
                dot += q_local[3] * static_cast<float>(kpack.v[3]);
                if (debug_lane) local_qk_fma_cycles += (clock64() - t_fma_begin);
            } else {
                #pragma unroll
                for (int i = 0; i < kElemsPerLane; i++) {
                    int d = lane_id * kElemsPerLane + i;
                    unsigned long long t_load_begin = 0;
                    if (debug_lane) t_load_begin = clock64();
                    float kval = __bfloat162float(k_vec[d]);
                    if (debug_lane) {
                        unsigned long long bf16_load = (clock64() - t_load_begin);
                        local_qk_load_cycles += bf16_load;
                        local_qk_load_bf16_cycles += bf16_load;
                    }
                    unsigned long long t_fma_begin = 0;
                    if (debug_lane) t_fma_begin = clock64();
                    dot += q_local[i] * kval;
                    if (debug_lane) {
                        unsigned long long bf16_fma = (clock64() - t_fma_begin);
                        local_qk_fma_cycles += bf16_fma;
                        local_qk_fma_bf16_cycles += bf16_fma;
                    }
                }
            }
            unsigned long long t_qk_reduce_begin = 0;
            if (debug_lane) t_qk_reduce_begin = clock64();
            dot = warp_reduce_sum(dot) * attn_scale;
            float score = __shfl_sync(0xffffffff, dot, 0);
            if (debug_lane) local_qk_reduce_cycles += (clock64() - t_qk_reduce_begin);
            if (debug_lane) local_qk_cycles += (clock64() - t_qk_begin);

            unsigned long long t_softmax_begin = 0;
            if (debug_lane) t_softmax_begin = clock64();
            float m_new = fmaxf(m, score);
            float old_scale = expf(m - m_new);
            float new_scale = expf(score - m_new);
            s = s * old_scale + new_scale;
            if (debug_lane) local_softmax_cycles += (clock64() - t_softmax_begin);

            unsigned long long t_value_begin = 0;
            if (debug_lane) t_value_begin = clock64();
            if constexpr (KV_FP8) {
                int d0 = lane_id * kElemsPerLane;
                unsigned long long t_vload_begin = 0;
                if (debug_lane) t_vload_begin = clock64();
                SplitFp8x4Pack vpack = split_load_fp8x4(v_vec_fp8 + d0);
                if (debug_lane) local_value_load_cycles += (clock64() - t_vload_begin);
                unsigned long long t_vfma_begin = 0;
                if (debug_lane) t_vfma_begin = clock64();
                out_local[0] = out_local[0] * old_scale + static_cast<float>(vpack.v[0]) * new_scale;
                out_local[1] = out_local[1] * old_scale + static_cast<float>(vpack.v[1]) * new_scale;
                out_local[2] = out_local[2] * old_scale + static_cast<float>(vpack.v[2]) * new_scale;
                out_local[3] = out_local[3] * old_scale + static_cast<float>(vpack.v[3]) * new_scale;
                if (debug_lane) local_value_fma_cycles += (clock64() - t_vfma_begin);
            } else {
                #pragma unroll
                for (int i = 0; i < kElemsPerLane; i++) {
                    int d = lane_id * kElemsPerLane + i;
                    unsigned long long t_vload_begin = 0;
                    if (debug_lane) t_vload_begin = clock64();
                    float vv = __bfloat162float(v_vec[d]);
                    if (debug_lane) local_value_load_cycles += (clock64() - t_vload_begin);
                    unsigned long long t_vfma_begin = 0;
                    if (debug_lane) t_vfma_begin = clock64();
                    out_local[i] = out_local[i] * old_scale + vv * new_scale;
                    if (debug_lane) local_value_fma_cycles += (clock64() - t_vfma_begin);
                }
            }
            if (debug_lane) local_value_cycles += (clock64() - t_value_begin);
            m = m_new;
        }
    }

    if (debug_out != nullptr && lane_id == 0) {
        atomicAdd(&sm_tokens, local_tokens);
        atomicAdd(&sm_block_switches, local_block_switches);
        atomicAdd(&sm_oob_skips, local_oob_skips);
        atomicAdd(&sm_tiles_processed, local_tiles);
        atomicAdd(&sm_map_cycles, local_map_cycles);
        atomicAdd(&sm_qk_cycles, local_qk_cycles);
        atomicAdd(&sm_qk_load_cycles, local_qk_load_cycles);
        atomicAdd(&sm_qk_load_bf16_cycles, local_qk_load_bf16_cycles);
        atomicAdd(&sm_qk_fma_cycles, local_qk_fma_cycles);
        atomicAdd(&sm_qk_fma_bf16_cycles, local_qk_fma_bf16_cycles);
        atomicAdd(&sm_qk_reduce_cycles, local_qk_reduce_cycles);
        atomicAdd(&sm_softmax_cycles, local_softmax_cycles);
        atomicAdd(&sm_value_cycles, local_value_cycles);
        atomicAdd(&sm_value_load_cycles, local_value_load_cycles);
        atomicAdd(&sm_value_fma_cycles, local_value_fma_cycles);
    }

    if (lane_id == 0) {
        sm_m[warp_id] = m;
        sm_s[warp_id] = s;
    }
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        int d = lane_id * kElemsPerLane + i;
        sm_out[warp_id * HEAD_DIM + d] = out_local[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned long long merge_begin = 0;
        if (debug_out != nullptr) merge_begin = clock64();
        float g_m = -INFINITY;
        for (int w = 0; w < active_warps; w++) g_m = fmaxf(g_m, sm_m[w]);
        float g_s = 0.0f;
        for (int w = 0; w < active_warps; w++) {
            float scale = (sm_s[w] > 0.0f) ? expf(sm_m[w] - g_m) : 0.0f;
            sm_scale[w] = scale;
            g_s += sm_s[w] * scale;
        }
        sm_global_s = g_s;
        if (debug_out != nullptr) sm_merge_cycles += (clock64() - merge_begin);
    }
    __syncthreads();

    float inv_s = (sm_global_s > 0.0f) ? (1.0f / sm_global_s) : 0.0f;
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        int d = lane_id * kElemsPerLane + i;
        float acc = 0.0f;
        for (int w = 0; w < active_warps; w++) {
            acc += sm_out[w * HEAD_DIM + d] * sm_scale[w];
        }
        float out_f = acc * inv_s;
        if constexpr (KV_FP8) out_f *= v_scale;
        out_vec[d] = __float2bfloat16(out_f);
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
        rec.map_cycles = sm_map_cycles;
        rec.qk_cycles = sm_qk_cycles;
        rec.qk_load_cycles = sm_qk_load_cycles;
        rec.qk_load_bf16_cycles = sm_qk_load_bf16_cycles;
        rec.qk_fma_cycles = sm_qk_fma_cycles;
        rec.qk_fma_bf16_cycles = sm_qk_fma_bf16_cycles;
        rec.qk_reduce_cycles = sm_qk_reduce_cycles;
        rec.softmax_cycles = sm_softmax_cycles;
        rec.value_cycles = sm_value_cycles;
        rec.value_load_cycles = sm_value_load_cycles;
        rec.value_fma_cycles = sm_value_fma_cycles;
        rec.merge_cycles = sm_merge_cycles;
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
template <int kBlock>
__global__ void split_w4_matvec_bf16_out_kernel_t(
    const uint8_t* __restrict__ packed_w4,
    const float* __restrict__ scales,
    const float* __restrict__ codebook,
    const __nv_bfloat16* __restrict__ x_bf16,
    __nv_bfloat16* __restrict__ y_bf16,
    int rows,
    int cols,
    SplitFfnW4FineDebugRec* __restrict__ fine_debug
) {
    int row = (int)blockIdx.x;
    if (row >= rows) return;

    static_assert(kBlock % WARP_SIZE == 0, "kBlock must be multiple of warp size.");
    constexpr int kWarps = kBlock / WARP_SIZE;
    __shared__ float smem_reduce[kWarps];
    int codebook_base = (rows == (INTERMEDIATE_SIZE + INTERMEDIATE_SIZE) && row >= INTERMEDIATE_SIZE) ? 16 : 0;
    const float* __restrict__ codebook_row = codebook + codebook_base;
    bool debug_fine = (fine_debug != nullptr);
    unsigned long long local_dequant_cycles = 0;
    unsigned long long local_fma_cycles = 0;
    unsigned long long local_reduce_cycles = 0;

    float local_sum = 0.0f;
    int64_t row_base = (int64_t)row * (int64_t)cols;
    int vec_cols = cols & ~7;
    for (int col = threadIdx.x * 8; col < vec_cols; col += kBlock * 8) {
        unsigned long long t_deq = 0;
        if (debug_fine) t_deq = clock64();
        int64_t linear = row_base + (int64_t)col;
        const uint32_t* pack_ptr = reinterpret_cast<const uint32_t*>(packed_w4 + (linear >> 1));
        SplitW4x8Pack qw = split_decode_w4x8(split_load_u32(pack_ptr));
        int scale_idx = int(linear >> 6);
        float scale = scales[scale_idx];
        float w0 = codebook_row[qw.q0] * scale;
        float w1 = codebook_row[qw.q1] * scale;
        float w2 = codebook_row[qw.q2] * scale;
        float w3 = codebook_row[qw.q3] * scale;
        float w4 = codebook_row[qw.q4] * scale;
        float w5 = codebook_row[qw.q5] * scale;
        float w6 = codebook_row[qw.q6] * scale;
        float w7 = codebook_row[qw.q7] * scale;
        if (debug_fine) local_dequant_cycles += (clock64() - t_deq);

        unsigned long long t_fma = 0;
        if (debug_fine) t_fma = clock64();
        float x0 = __bfloat162float(x_bf16[col]);
        float x1 = __bfloat162float(x_bf16[col + 1]);
        float x2 = __bfloat162float(x_bf16[col + 2]);
        float x3 = __bfloat162float(x_bf16[col + 3]);
        float x4 = __bfloat162float(x_bf16[col + 4]);
        float x5 = __bfloat162float(x_bf16[col + 5]);
        float x6 = __bfloat162float(x_bf16[col + 6]);
        float x7 = __bfloat162float(x_bf16[col + 7]);

        local_sum += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3 +
                     w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7;
        if (debug_fine) local_fma_cycles += (clock64() - t_fma);
    }
    for (int col = vec_cols + threadIdx.x; col < cols; col += kBlock) {
        unsigned long long t_deq = 0;
        if (debug_fine) t_deq = clock64();
        int64_t linear = row_base + (int64_t)col;
        uint8_t pack = packed_w4[linear >> 1];
        int q = (linear & 1) ? int(pack & 0x0F) : int((pack >> 4) & 0x0F);
        int scale_idx = int(linear >> 6);
        float w = codebook_row[q] * scales[scale_idx];
        if (debug_fine) local_dequant_cycles += (clock64() - t_deq);
        unsigned long long t_fma = 0;
        if (debug_fine) t_fma = clock64();
        float xv = __bfloat162float(x_bf16[col]);
        local_sum += w * xv;
        if (debug_fine) local_fma_cycles += (clock64() - t_fma);
    }

    unsigned long long t_reduce = 0;
    if (debug_fine) t_reduce = clock64();
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
    if (debug_fine) {
        local_reduce_cycles += (clock64() - t_reduce);
        atomicAdd(&fine_debug->gateup_dequant_cycles, local_dequant_cycles);
        atomicAdd(&fine_debug->gateup_fma_cycles, local_fma_cycles);
        atomicAdd(&fine_debug->gateup_reduce_cycles, local_reduce_cycles);
    }
}

template <int kBlock>
__global__ void split_w4_matvec_warp_per_row_kernel_t(
    const uint8_t* __restrict__ packed_w4,
    const float* __restrict__ scales,
    const float* __restrict__ codebook,
    const __nv_bfloat16* __restrict__ x_bf16,
    __nv_bfloat16* __restrict__ y_bf16,
    int rows,
    int cols,
    SplitFfnW4FineDebugRec* __restrict__ fine_debug
) {
    static_assert(kBlock % WARP_SIZE == 0, "kBlock must be multiple of warp size.");
    constexpr int kWarpsPerBlock = kBlock / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int row = (int)blockIdx.x * kWarpsPerBlock + warp_id;
    if (row >= rows) return;

    int codebook_base = (rows == (INTERMEDIATE_SIZE + INTERMEDIATE_SIZE) && row >= INTERMEDIATE_SIZE) ? 16 : 0;
    const float* __restrict__ codebook_row = codebook + codebook_base;
    bool debug_fine = (fine_debug != nullptr);
    unsigned long long local_dequant_cycles = 0;
    unsigned long long local_fma_cycles = 0;
    unsigned long long local_reduce_cycles = 0;

    float local_sum = 0.0f;
    int64_t row_base = (int64_t)row * (int64_t)cols;
    int vec_cols = cols & ~7;
    for (int col = lane_id * 8; col < vec_cols; col += WARP_SIZE * 8) {
        unsigned long long t_deq = 0;
        if (debug_fine) t_deq = clock64();
        int64_t linear = row_base + (int64_t)col;
        const uint32_t* pack_ptr = reinterpret_cast<const uint32_t*>(packed_w4 + (linear >> 1));
        SplitW4x8Pack qw = split_decode_w4x8(split_load_u32(pack_ptr));
        float scale = scales[linear >> 6];
        float w0 = codebook_row[qw.q0] * scale;
        float w1 = codebook_row[qw.q1] * scale;
        float w2 = codebook_row[qw.q2] * scale;
        float w3 = codebook_row[qw.q3] * scale;
        float w4 = codebook_row[qw.q4] * scale;
        float w5 = codebook_row[qw.q5] * scale;
        float w6 = codebook_row[qw.q6] * scale;
        float w7 = codebook_row[qw.q7] * scale;
        if (debug_fine) local_dequant_cycles += (clock64() - t_deq);

        unsigned long long t_fma = 0;
        if (debug_fine) t_fma = clock64();
        float x0 = __bfloat162float(x_bf16[col]);
        float x1 = __bfloat162float(x_bf16[col + 1]);
        float x2 = __bfloat162float(x_bf16[col + 2]);
        float x3 = __bfloat162float(x_bf16[col + 3]);
        float x4 = __bfloat162float(x_bf16[col + 4]);
        float x5 = __bfloat162float(x_bf16[col + 5]);
        float x6 = __bfloat162float(x_bf16[col + 6]);
        float x7 = __bfloat162float(x_bf16[col + 7]);
        local_sum += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3 +
                     w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7;
        if (debug_fine) local_fma_cycles += (clock64() - t_fma);
    }
    for (int col = vec_cols + lane_id; col < cols; col += WARP_SIZE) {
        unsigned long long t_deq = 0;
        if (debug_fine) t_deq = clock64();
        int64_t linear = row_base + (int64_t)col;
        uint8_t pack = packed_w4[linear >> 1];
        int q = (linear & 1) ? int(pack & 0x0F) : int((pack >> 4) & 0x0F);
        float w = codebook_row[q] * scales[linear >> 6];
        if (debug_fine) local_dequant_cycles += (clock64() - t_deq);

        unsigned long long t_fma = 0;
        if (debug_fine) t_fma = clock64();
        float xv = __bfloat162float(x_bf16[col]);
        local_sum += w * xv;
        if (debug_fine) local_fma_cycles += (clock64() - t_fma);
    }

    unsigned long long t_reduce = 0;
    if (debug_fine) t_reduce = clock64();
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
        y_bf16[row] = __float2bfloat16(local_sum);
    }
    if (debug_fine) {
        local_reduce_cycles += (clock64() - t_reduce);
        atomicAdd(&fine_debug->gateup_dequant_cycles, local_dequant_cycles);
        atomicAdd(&fine_debug->gateup_fma_cycles, local_fma_cycles);
        atomicAdd(&fine_debug->gateup_reduce_cycles, local_reduce_cycles);
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
    int vec_cols = cols & ~3;
    for (int col = threadIdx.x * 4; col < vec_cols; col += kBlock * 4) {
        int64_t linear = row_base + (int64_t)col;
        const uint16_t* pack_ptr = reinterpret_cast<const uint16_t*>(packed_w4 + (linear >> 1));
        SplitW4x4Pack qw = split_decode_w4x4(split_load_u16(pack_ptr));
        float scale = scales[linear >> 6];

        float x0 = __bfloat162float(x_bf16[col]);
        float x1 = __bfloat162float(x_bf16[col + 1]);
        float x2 = __bfloat162float(x_bf16[col + 2]);
        float x3 = __bfloat162float(x_bf16[col + 3]);
        float dot4 = codebook[qw.q0] * x0 +
                     codebook[qw.q1] * x1 +
                     codebook[qw.q2] * x2 +
                     codebook[qw.q3] * x3;
        local_sum += dot4 * scale;
    }
    for (int col = vec_cols + threadIdx.x; col < cols; col += kBlock) {
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
    float* __restrict__ hidden_f32,              // [HIDDEN_SIZE]
    SplitFfnW4FineDebugRec* __restrict__ fine_debug
) {
    int row = (int)blockIdx.x;
    if (row >= HIDDEN_SIZE) return;

    constexpr int kBlock = 256;
    constexpr int kWarps = kBlock / WARP_SIZE;
    __shared__ float smem_reduce[kWarps];
    bool debug_fine = (fine_debug != nullptr);
    unsigned long long local_dequant_cycles = 0;
    unsigned long long local_fma_cycles = 0;

    float local_sum = 0.0f;
    int64_t row_base = (int64_t)row * (int64_t)INTERMEDIATE_SIZE;
    int vec_cols = INTERMEDIATE_SIZE & ~7;
    for (int col = threadIdx.x * 8; col < vec_cols; col += kBlock * 8) {
        unsigned long long t_deq = 0;
        if (debug_fine) t_deq = clock64();
        int64_t linear = row_base + (int64_t)col;
        const uint32_t* pack_ptr = reinterpret_cast<const uint32_t*>(packed_w4 + (linear >> 1));
        SplitW4x8Pack qw = split_decode_w4x8(split_load_u32(pack_ptr));
        float scale = scales[linear >> 6];
        float w0 = codebook[qw.q0] * scale;
        float w1 = codebook[qw.q1] * scale;
        float w2 = codebook[qw.q2] * scale;
        float w3 = codebook[qw.q3] * scale;
        float w4 = codebook[qw.q4] * scale;
        float w5 = codebook[qw.q5] * scale;
        float w6 = codebook[qw.q6] * scale;
        float w7 = codebook[qw.q7] * scale;
        if (debug_fine) local_dequant_cycles += (clock64() - t_deq);

        unsigned long long t_fma = 0;
        if (debug_fine) t_fma = clock64();
        float x0 = __bfloat162float(x_bf16[col]);
        float x1 = __bfloat162float(x_bf16[col + 1]);
        float x2 = __bfloat162float(x_bf16[col + 2]);
        float x3 = __bfloat162float(x_bf16[col + 3]);
        float x4 = __bfloat162float(x_bf16[col + 4]);
        float x5 = __bfloat162float(x_bf16[col + 5]);
        float x6 = __bfloat162float(x_bf16[col + 6]);
        float x7 = __bfloat162float(x_bf16[col + 7]);
        local_sum += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3 +
                     w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7;
        if (debug_fine) local_fma_cycles += (clock64() - t_fma);
    }
    for (int col = vec_cols + threadIdx.x; col < INTERMEDIATE_SIZE; col += kBlock) {
        unsigned long long t_deq = 0;
        if (debug_fine) t_deq = clock64();
        int64_t linear = row_base + (int64_t)col;
        uint8_t pack = packed_w4[linear >> 1];
        int q = (linear & 1) ? int(pack & 0x0F) : int((pack >> 4) & 0x0F);
        float w = codebook[q] * scales[linear >> 6];
        if (debug_fine) local_dequant_cycles += (clock64() - t_deq);
        unsigned long long t_fma = 0;
        if (debug_fine) t_fma = clock64();
        float xv = __bfloat162float(x_bf16[col]);
        local_sum += w * xv;
        if (debug_fine) local_fma_cycles += (clock64() - t_fma);
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
    if (debug_fine) {
        atomicAdd(&fine_debug->down_dequant_cycles, local_dequant_cycles);
        atomicAdd(&fine_debug->down_fma_cycles, local_fma_cycles);
    }
}

template <bool FAST_EXP>
__global__ void split_w4_silu_downproj_residual_kernel_t(
    const __nv_bfloat16* __restrict__ gateup_out_bf16,  // [2 * INTERMEDIATE_SIZE], [gate | up]
    const uint8_t* __restrict__ down_packed_w4,         // [HIDDEN_SIZE * INTERMEDIATE_SIZE / 2]
    const float* __restrict__ down_scales,              // [HIDDEN_SIZE * INTERMEDIATE_SIZE / 64]
    const float* __restrict__ down_codebook,            // [16]
    const float* __restrict__ residual_f32,             // [HIDDEN_SIZE]
    float* __restrict__ hidden_f32,                     // [HIDDEN_SIZE]
    SplitFfnW4FineDebugRec* __restrict__ fine_debug
) {
    int out_idx = (int)blockIdx.x;
    if (out_idx >= HIDDEN_SIZE) return;

    constexpr int kBlock = 256;
    constexpr int kWarps = kBlock / WARP_SIZE;
    __shared__ float smem_reduce[kWarps];

    const __nv_bfloat16* gate = gateup_out_bf16;
    const __nv_bfloat16* up = gateup_out_bf16 + INTERMEDIATE_SIZE;
    bool debug_fine = (fine_debug != nullptr);
    unsigned long long local_dequant_cycles = 0;
    unsigned long long local_act_cycles = 0;
    unsigned long long local_fma_cycles = 0;

    float local_sum = 0.0f;
    int64_t row_base = (int64_t)out_idx * (int64_t)INTERMEDIATE_SIZE;
    int vec_cols = INTERMEDIATE_SIZE & ~7;
    for (int i = threadIdx.x * 8; i < vec_cols; i += kBlock * 8) {
        unsigned long long t_act = 0;
        if (debug_fine) t_act = clock64();
        float gate0 = __bfloat162float(gate[i]);
        float gate1 = __bfloat162float(gate[i + 1]);
        float gate2 = __bfloat162float(gate[i + 2]);
        float gate3 = __bfloat162float(gate[i + 3]);
        float gate4 = __bfloat162float(gate[i + 4]);
        float gate5 = __bfloat162float(gate[i + 5]);
        float gate6 = __bfloat162float(gate[i + 6]);
        float gate7 = __bfloat162float(gate[i + 7]);
        float up0 = __bfloat162float(up[i]);
        float up1 = __bfloat162float(up[i + 1]);
        float up2 = __bfloat162float(up[i + 2]);
        float up3 = __bfloat162float(up[i + 3]);
        float up4 = __bfloat162float(up[i + 4]);
        float up5 = __bfloat162float(up[i + 5]);
        float up6 = __bfloat162float(up[i + 6]);
        float up7 = __bfloat162float(up[i + 7]);
        float silu0 = split_silu_value<FAST_EXP>(gate0);
        float silu1 = split_silu_value<FAST_EXP>(gate1);
        float silu2 = split_silu_value<FAST_EXP>(gate2);
        float silu3 = split_silu_value<FAST_EXP>(gate3);
        float silu4 = split_silu_value<FAST_EXP>(gate4);
        float silu5 = split_silu_value<FAST_EXP>(gate5);
        float silu6 = split_silu_value<FAST_EXP>(gate6);
        float silu7 = split_silu_value<FAST_EXP>(gate7);
        float act0 = silu0 * up0;
        float act1 = silu1 * up1;
        float act2 = silu2 * up2;
        float act3 = silu3 * up3;
        float act4 = silu4 * up4;
        float act5 = silu5 * up5;
        float act6 = silu6 * up6;
        float act7 = silu7 * up7;
        if (debug_fine) local_act_cycles += (clock64() - t_act);

        unsigned long long t_deq = 0;
        if (debug_fine) t_deq = clock64();
        int64_t linear = row_base + (int64_t)i;
        const uint32_t* pack_ptr = reinterpret_cast<const uint32_t*>(down_packed_w4 + (linear >> 1));
        SplitW4x8Pack qw = split_decode_w4x8(split_load_u32(pack_ptr));
        float scale = down_scales[linear >> 6];
        float w0 = down_codebook[qw.q0] * scale;
        float w1 = down_codebook[qw.q1] * scale;
        float w2 = down_codebook[qw.q2] * scale;
        float w3 = down_codebook[qw.q3] * scale;
        float w4 = down_codebook[qw.q4] * scale;
        float w5 = down_codebook[qw.q5] * scale;
        float w6 = down_codebook[qw.q6] * scale;
        float w7 = down_codebook[qw.q7] * scale;
        if (debug_fine) local_dequant_cycles += (clock64() - t_deq);

        unsigned long long t_fma = 0;
        if (debug_fine) t_fma = clock64();
        local_sum += act0 * w0 + act1 * w1 + act2 * w2 + act3 * w3 +
                     act4 * w4 + act5 * w5 + act6 * w6 + act7 * w7;
        if (debug_fine) local_fma_cycles += (clock64() - t_fma);
    }
    for (int i = vec_cols + threadIdx.x; i < INTERMEDIATE_SIZE; i += kBlock) {
        unsigned long long t_act = 0;
        if (debug_fine) t_act = clock64();
        float gate_v = __bfloat162float(gate[i]);
        float up_v = __bfloat162float(up[i]);
        float silu_v = split_silu_value<FAST_EXP>(gate_v);
        float act_v = silu_v * up_v;
        if (debug_fine) local_act_cycles += (clock64() - t_act);

        unsigned long long t_deq = 0;
        if (debug_fine) t_deq = clock64();
        int64_t linear = row_base + (int64_t)i;
        uint8_t pack = down_packed_w4[linear >> 1];
        int q = (linear & 1) ? int(pack & 0x0F) : int((pack >> 4) & 0x0F);
        float down_w = down_codebook[q] * down_scales[linear >> 6];
        if (debug_fine) local_dequant_cycles += (clock64() - t_deq);

        unsigned long long t_fma = 0;
        if (debug_fine) t_fma = clock64();
        local_sum += act_v * down_w;
        if (debug_fine) local_fma_cycles += (clock64() - t_fma);
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
    if (debug_fine) {
        atomicAdd(&fine_debug->down_dequant_cycles, local_dequant_cycles);
        atomicAdd(&fine_debug->down_act_cycles, local_act_cycles);
        atomicAdd(&fine_debug->down_fma_cycles, local_fma_cycles);
    }
}

template <bool FAST_EXP, int ROW_TILE>
__global__ void split_w4_silu_downproj_residual_rowtile_kernel_t(
    const __nv_bfloat16* __restrict__ gateup_out_bf16,
    const uint8_t* __restrict__ down_packed_w4,
    const float* __restrict__ down_scales,
    const float* __restrict__ down_codebook,
    const float* __restrict__ residual_f32,
    float* __restrict__ hidden_f32,
    SplitFfnW4FineDebugRec* __restrict__ fine_debug
) {
    int out_base = (int)blockIdx.x * ROW_TILE;
    if (out_base >= HIDDEN_SIZE) return;

    constexpr int kBlock = 256;
    constexpr int kWarps = kBlock / WARP_SIZE;
    __shared__ float smem_reduce[ROW_TILE][kWarps];

    const __nv_bfloat16* gate = gateup_out_bf16;
    const __nv_bfloat16* up = gateup_out_bf16 + INTERMEDIATE_SIZE;
    bool debug_fine = (fine_debug != nullptr);
    unsigned long long local_dequant_cycles = 0;
    unsigned long long local_act_cycles = 0;
    unsigned long long local_fma_cycles = 0;

    float local_sum[ROW_TILE];
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) local_sum[r] = 0.0f;

    int vec_cols = INTERMEDIATE_SIZE & ~7;
    for (int i = threadIdx.x * 8; i < vec_cols; i += kBlock * 8) {
        unsigned long long t_act = 0;
        if (debug_fine) t_act = clock64();
        float gate0 = __bfloat162float(gate[i]);
        float gate1 = __bfloat162float(gate[i + 1]);
        float gate2 = __bfloat162float(gate[i + 2]);
        float gate3 = __bfloat162float(gate[i + 3]);
        float gate4 = __bfloat162float(gate[i + 4]);
        float gate5 = __bfloat162float(gate[i + 5]);
        float gate6 = __bfloat162float(gate[i + 6]);
        float gate7 = __bfloat162float(gate[i + 7]);
        float up0 = __bfloat162float(up[i]);
        float up1 = __bfloat162float(up[i + 1]);
        float up2 = __bfloat162float(up[i + 2]);
        float up3 = __bfloat162float(up[i + 3]);
        float up4 = __bfloat162float(up[i + 4]);
        float up5 = __bfloat162float(up[i + 5]);
        float up6 = __bfloat162float(up[i + 6]);
        float up7 = __bfloat162float(up[i + 7]);
        float silu0 = split_silu_value<FAST_EXP>(gate0);
        float silu1 = split_silu_value<FAST_EXP>(gate1);
        float silu2 = split_silu_value<FAST_EXP>(gate2);
        float silu3 = split_silu_value<FAST_EXP>(gate3);
        float silu4 = split_silu_value<FAST_EXP>(gate4);
        float silu5 = split_silu_value<FAST_EXP>(gate5);
        float silu6 = split_silu_value<FAST_EXP>(gate6);
        float silu7 = split_silu_value<FAST_EXP>(gate7);
        float act0 = silu0 * up0;
        float act1 = silu1 * up1;
        float act2 = silu2 * up2;
        float act3 = silu3 * up3;
        float act4 = silu4 * up4;
        float act5 = silu5 * up5;
        float act6 = silu6 * up6;
        float act7 = silu7 * up7;
        if (debug_fine) local_act_cycles += (clock64() - t_act);

#pragma unroll
        for (int r = 0; r < ROW_TILE; ++r) {
            int out_idx = out_base + r;
            if (out_idx >= HIDDEN_SIZE) continue;
            unsigned long long t_deq = 0;
            if (debug_fine) t_deq = clock64();
            int64_t linear = (int64_t)out_idx * (int64_t)INTERMEDIATE_SIZE + (int64_t)i;
            const uint32_t* pack_ptr = reinterpret_cast<const uint32_t*>(down_packed_w4 + (linear >> 1));
            SplitW4x8Pack qw = split_decode_w4x8(split_load_u32(pack_ptr));
            float scale = down_scales[linear >> 6];
            float w0 = down_codebook[qw.q0] * scale;
            float w1 = down_codebook[qw.q1] * scale;
            float w2 = down_codebook[qw.q2] * scale;
            float w3 = down_codebook[qw.q3] * scale;
            float w4 = down_codebook[qw.q4] * scale;
            float w5 = down_codebook[qw.q5] * scale;
            float w6 = down_codebook[qw.q6] * scale;
            float w7 = down_codebook[qw.q7] * scale;
            if (debug_fine) local_dequant_cycles += (clock64() - t_deq);

            unsigned long long t_fma = 0;
            if (debug_fine) t_fma = clock64();
            local_sum[r] += act0 * w0 + act1 * w1 + act2 * w2 + act3 * w3 +
                            act4 * w4 + act5 * w5 + act6 * w6 + act7 * w7;
            if (debug_fine) local_fma_cycles += (clock64() - t_fma);
        }
    }
    for (int i = vec_cols + threadIdx.x; i < INTERMEDIATE_SIZE; i += kBlock) {
        unsigned long long t_act = 0;
        if (debug_fine) t_act = clock64();
        float gate_v = __bfloat162float(gate[i]);
        float up_v = __bfloat162float(up[i]);
        float silu_v = split_silu_value<FAST_EXP>(gate_v);
        float act_v = silu_v * up_v;
        if (debug_fine) local_act_cycles += (clock64() - t_act);

#pragma unroll
        for (int r = 0; r < ROW_TILE; ++r) {
            int out_idx = out_base + r;
            if (out_idx >= HIDDEN_SIZE) continue;
            unsigned long long t_deq = 0;
            if (debug_fine) t_deq = clock64();
            int64_t linear = (int64_t)out_idx * (int64_t)INTERMEDIATE_SIZE + (int64_t)i;
            uint8_t pack = down_packed_w4[linear >> 1];
            int q = (linear & 1) ? int(pack & 0x0F) : int((pack >> 4) & 0x0F);
            float down_w = down_codebook[q] * down_scales[linear >> 6];
            if (debug_fine) local_dequant_cycles += (clock64() - t_deq);

            unsigned long long t_fma = 0;
            if (debug_fine) t_fma = clock64();
            local_sum[r] += act_v * down_w;
            if (debug_fine) local_fma_cycles += (clock64() - t_fma);
        }
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
        float sum_r = warp_reduce_sum(local_sum[r]);
        if (lane_id == 0) smem_reduce[r][warp_id] = sum_r;
    }
    __syncthreads();

    if (warp_id == 0) {
#pragma unroll
        for (int r = 0; r < ROW_TILE; ++r) {
            int out_idx = out_base + r;
            if (out_idx >= HIDDEN_SIZE) continue;
            float sum = (lane_id < kWarps) ? smem_reduce[r][lane_id] : 0.0f;
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) hidden_f32[out_idx] = sum + residual_f32[out_idx];
        }
    }
    if (debug_fine) {
        atomicAdd(&fine_debug->down_dequant_cycles, local_dequant_cycles);
        atomicAdd(&fine_debug->down_act_cycles, local_act_cycles);
        atomicAdd(&fine_debug->down_fma_cycles, local_fma_cycles);
    }
}

// =============================================================================
// Host entry: one split decode step (single token)
// =============================================================================

#include "split_decode_gemm_host.inl"
