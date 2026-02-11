#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

#define AUDIO_ATTN_WARP_SIZE 32

__device__ __forceinline__ float audio_warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

__device__ __forceinline__ float audio_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return __shfl_sync(0xffffffff, val, 0);
}

__device__ __forceinline__ float audio_block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % AUDIO_ATTN_WARP_SIZE;
    int wid = threadIdx.x / AUDIO_ATTN_WARP_SIZE;
    int num_warps = (blockDim.x + AUDIO_ATTN_WARP_SIZE - 1) / AUDIO_ATTN_WARP_SIZE;

    val = audio_warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    if (wid == 0) {
        float block_val = (lane < num_warps) ? shared[lane] : -INFINITY;
        block_val = audio_warp_reduce_max(block_val);
        if (lane == 0) shared[0] = block_val;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float audio_block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % AUDIO_ATTN_WARP_SIZE;
    int wid = threadIdx.x / AUDIO_ATTN_WARP_SIZE;
    int num_warps = (blockDim.x + AUDIO_ATTN_WARP_SIZE - 1) / AUDIO_ATTN_WARP_SIZE;

    val = audio_warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    if (wid == 0) {
        float block_val = (lane < num_warps) ? shared[lane] : 0.0f;
        block_val = audio_warp_reduce_sum(block_val);
        if (lane == 0) shared[0] = block_val;
    }
    __syncthreads();
    return shared[0];
}

__global__ void audio_attention_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,   // [B, H, T, D]
    const __nv_bfloat16* __restrict__ k,   // [B, H, T, D]
    const __nv_bfloat16* __restrict__ v,   // [B, H, T, D]
    __nv_bfloat16* __restrict__ out,       // [B, H, T, D]
    const int* __restrict__ cu_seqlens,    // optional [S+1], defines ragged segments in T
    int num_segments,
    int batch,
    int n_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    int h = blockIdx.x;
    int tq = blockIdx.y;
    int b = blockIdx.z;
    if (b >= batch || h >= n_heads || tq >= seq_len) return;

    extern __shared__ float smem[];
    float* scores = smem; // size seq_len
    float* reduce_shared = smem + seq_len; // size num_warps

    int seg_start = 0;
    int seg_end = seq_len;
    if (cu_seqlens != nullptr && num_segments > 0) {
        int lo = 0;
        int hi = num_segments - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            int s = cu_seqlens[mid];
            int e = cu_seqlens[mid + 1];
            if (tq < s) {
                hi = mid - 1;
            } else if (tq >= e) {
                lo = mid + 1;
            } else {
                seg_start = s;
                seg_end = e;
                break;
            }
        }
    }

    int seg_len = seg_end - seg_start;
    if (seg_len <= 0) return;

    const __nv_bfloat16* q_row = q + (((b * n_heads + h) * seq_len + tq) * head_dim);

    float local_max = -INFINITY;
    for (int tk = threadIdx.x; tk < seg_len; tk += blockDim.x) {
        int key_idx = seg_start + tk;
        const __nv_bfloat16* k_row = k + (((b * n_heads + h) * seq_len + key_idx) * head_dim);
        float dot = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < head_dim; d++) {
            dot += __bfloat162float(q_row[d]) * __bfloat162float(k_row[d]);
        }
        dot *= scale;
        scores[tk] = dot;
        local_max = fmaxf(local_max, dot);
    }
    __syncthreads();

    float max_score = audio_block_reduce_max(local_max, reduce_shared);

    float local_sum = 0.0f;
    for (int tk = threadIdx.x; tk < seg_len; tk += blockDim.x) {
        float e = expf(scores[tk] - max_score);
        scores[tk] = e;
        local_sum += e;
    }
    __syncthreads();

    float sum_exp = audio_block_reduce_sum(local_sum, reduce_shared);
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int tk = 0; tk < seg_len; tk++) {
            int key_idx = seg_start + tk;
            const __nv_bfloat16* v_row = v + (((b * n_heads + h) * seq_len + key_idx) * head_dim);
            acc += (scores[tk] * inv_sum) * __bfloat162float(v_row[d]);
        }
        out[(((b * n_heads + h) * seq_len + tq) * head_dim) + d] = __float2bfloat16(acc);
    }
}

extern "C" void launch_audio_attention(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    const int* cu_seqlens,
    int num_segments,
    int batch,
    int n_heads,
    int seq_len,
    int head_dim,
    float scale,
    cudaStream_t stream
) {
    if (batch <= 0 || n_heads <= 0 || seq_len <= 0 || head_dim <= 0) return;
    constexpr int threads = 128;
    dim3 grid(n_heads, seq_len, batch);
    int num_warps = (threads + AUDIO_ATTN_WARP_SIZE - 1) / AUDIO_ATTN_WARP_SIZE;
    size_t shared_bytes = sizeof(float) * size_t(seq_len + num_warps);
    audio_attention_bf16_kernel<<<grid, threads, shared_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<__nv_bfloat16*>(out),
        cu_seqlens,
        num_segments,
        batch,
        n_heads,
        seq_len,
        head_dim,
        scale
    );
}
