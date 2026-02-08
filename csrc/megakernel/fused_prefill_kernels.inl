// =============================================================================

__global__ void prefill_embed_kernel(
    const int* __restrict__ token_ids,            // [seq_len]
    const __nv_bfloat16* __restrict__ embed_table,  // [vocab_size, hidden_size]
    __nv_bfloat16* __restrict__ output,           // [seq_len, hidden_size]
    int seq_len,
    int hidden_size
) {
    int seq_idx = blockIdx.x;
    if (seq_idx >= seq_len) return;

    int token_id = token_ids[seq_idx];
    const __nv_bfloat16* embed_row = embed_table + token_id * hidden_size;
    __nv_bfloat16* out_row = output + seq_idx * hidden_size;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out_row[i] = embed_row[i];
    }
}

// =============================================================================
// Audio embedding patch kernel (BF16 in/out)
// Replaces a contiguous [audio_start_idx, audio_start_idx + audio_len) span in
// the embedding output with externally-computed audio embeddings.
// =============================================================================

__global__ void prefill_patch_audio_kernel(
    __nv_bfloat16* __restrict__ hidden,              // [seq_len, hidden_size]
    const __nv_bfloat16* __restrict__ audio_embeds,  // [audio_len, hidden_size]
    int audio_start_idx,
    int audio_len,
    int hidden_size
) {
    int row = blockIdx.x;  // [0, audio_len)
    if (row >= audio_len) return;

    int seq_row = audio_start_idx + row;
    __nv_bfloat16* dst = hidden + seq_row * hidden_size;
    const __nv_bfloat16* src = audio_embeds + row * hidden_size;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// =============================================================================
// RMSNorm kernel (BF16 input/output)
// Each block handles one token position
// =============================================================================

__global__ void prefill_rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ input,   // [seq_len, hidden_size]
    const __nv_bfloat16* __restrict__ weight,  // [hidden_size]
    __nv_bfloat16* __restrict__ output,        // [seq_len, hidden_size]
    __nv_bfloat16* __restrict__ residual,      // [seq_len, hidden_size] - can be nullptr
    int seq_len,
    int hidden_size
) {
    int seq_idx = blockIdx.x;
    if (seq_idx >= seq_len) return;

    const __nv_bfloat16* in_row = input + seq_idx * hidden_size;
    __nv_bfloat16* out_row = output + seq_idx * hidden_size;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    __shared__ float smem_reduce[PREFILL_NUM_WARPS];

    // Compute sum of squares
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += PREFILL_BLOCK_SIZE) {
        float v = __bfloat162float(in_row[i]);
        // Save residual if requested
        if (residual != nullptr) {
            residual[seq_idx * hidden_size + i] = in_row[i];
        }
        local_sum_sq += v * v;
    }

    local_sum_sq = prefill_warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
        smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < PREFILL_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = prefill_warp_reduce_sum(sum);
        if (lane_id == 0) {
            smem_reduce[0] = rsqrtf(sum / float(hidden_size) + PREFILL_RMS_EPS);
        }
    }
    __syncthreads();

    float rstd = smem_reduce[0];

    // Apply normalization
    for (int i = threadIdx.x; i < hidden_size; i += PREFILL_BLOCK_SIZE) {
        float v = __bfloat162float(in_row[i]);
        float w = __bfloat162float(weight[i]);
        out_row[i] = __float2bfloat16(v * rstd * w);
    }
}

// =============================================================================
// QK Norm + RoPE + KV Cache Kernel
// Operates on BF16 projected Q/K, writes BF16 to cache
// =============================================================================

__global__ void prefill_qk_norm_rope_kernel(
    __nv_bfloat16* __restrict__ q,              // [seq_len, num_q_heads, head_dim]
    __nv_bfloat16* __restrict__ k,              // [seq_len, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ v,        // [seq_len, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ q_norm_weight,  // [head_dim]
    const __nv_bfloat16* __restrict__ k_norm_weight,  // [head_dim]
    const __nv_bfloat16* __restrict__ cos_table,      // [max_seq, head_dim]
    const __nv_bfloat16* __restrict__ sin_table,      // [max_seq, head_dim]
    __nv_bfloat16* __restrict__ k_cache,  // [num_kv_heads, max_seq_len, head_dim]
    __nv_bfloat16* __restrict__ v_cache,  // [num_kv_heads, max_seq_len, head_dim]
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int position_offset  // Starting position in KV cache
) {
    // Each block handles one (position, head) pair
    // Grid: (seq_len, max(num_q_heads, num_kv_heads))
    int pos = blockIdx.x;
    int head = blockIdx.y;

    if (pos >= seq_len) return;

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    const __nv_bfloat16* cos_pos = cos_table + pos * head_dim;
    const __nv_bfloat16* sin_pos = sin_table + pos * head_dim;

    __shared__ float smem_reduce[8];
    __shared__ float smem_normed[HEAD_DIM];

    // Process Q heads
    if (head < num_q_heads) {
        __nv_bfloat16* q_head = q + pos * num_q_heads * head_dim + head * head_dim;

        // RMSNorm for Q head
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float v = __bfloat162float(q_head[i]);
            sum_sq += v * v;
        }

        // Warp reduction
        sum_sq = prefill_warp_reduce_sum(sum_sq);
        if (lane_id == 0 && warp_id < 8) {
            smem_reduce[warp_id] = sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < (blockDim.x / WARP_SIZE)) ? smem_reduce[lane_id] : 0.0f;
            sum = prefill_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(head_dim) + PREFILL_RMS_EPS);
            }
        }
        __syncthreads();

        float scale = smem_reduce[0];

        // Load normalized values to shared memory
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            smem_normed[i] = __bfloat162float(q_head[i]) * scale * __bfloat162float(q_norm_weight[i]);
        }
        __syncthreads();

        // Apply RoPE
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float cos_v = __bfloat162float(cos_pos[i]);
            float sin_v = __bfloat162float(sin_pos[i]);

            int pair_idx = (i < head_dim/2) ? (i + head_dim/2) : (i - head_dim/2);
            float pair_v = smem_normed[pair_idx];

            float result;
            if (i < head_dim/2) {
                result = smem_normed[i] * cos_v - pair_v * sin_v;
            } else {
                result = pair_v * sin_v + smem_normed[i] * cos_v;
            }
            q_head[i] = __float2bfloat16(result);
        }
    }
    __syncthreads();

    // Process K heads (and populate cache)
    if (head < num_kv_heads) {
        __nv_bfloat16* k_head = k + pos * num_kv_heads * head_dim + head * head_dim;
        const __nv_bfloat16* v_head = v + pos * num_kv_heads * head_dim + head * head_dim;

        int cache_pos = position_offset + pos;
        __nv_bfloat16* k_cache_head = k_cache + head * max_seq_len * head_dim + cache_pos * head_dim;
        __nv_bfloat16* v_cache_head = v_cache + head * max_seq_len * head_dim + cache_pos * head_dim;

        // RMSNorm for K head
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float v = __bfloat162float(k_head[i]);
            sum_sq += v * v;
        }

        sum_sq = prefill_warp_reduce_sum(sum_sq);
        if (lane_id == 0 && warp_id < 8) {
            smem_reduce[warp_id] = sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < (blockDim.x / WARP_SIZE)) ? smem_reduce[lane_id] : 0.0f;
            sum = prefill_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(head_dim) + PREFILL_RMS_EPS);
            }
        }
        __syncthreads();

        float scale = smem_reduce[0];

        // Load normalized K to shared memory
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            smem_normed[i] = __bfloat162float(k_head[i]) * scale * __bfloat162float(k_norm_weight[i]);
        }
        __syncthreads();

        // Apply RoPE and write to cache
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float cos_v = __bfloat162float(cos_pos[i]);
            float sin_v = __bfloat162float(sin_pos[i]);

            int pair_idx = (i < head_dim/2) ? (i + head_dim/2) : (i - head_dim/2);
            float pair_v = smem_normed[pair_idx];

            float k_final;
            if (i < head_dim/2) {
                k_final = smem_normed[i] * cos_v - pair_v * sin_v;
            } else {
                k_final = pair_v * sin_v + smem_normed[i] * cos_v;
            }

            k_head[i] = __float2bfloat16(k_final);
            k_cache_head[i] = __float2bfloat16(k_final);
            v_cache_head[i] = v_head[i];  // V doesn't need RoPE
        }
    }
}

// =============================================================================
// Causal Attention Kernel (BF16 input, BF16 output)
// Each warp handles one (query_pos, q_head) pair
// =============================================================================

__global__ void prefill_causal_attention_kernel(
    const __nv_bfloat16* __restrict__ q,       // [seq_len, num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ k,       // [seq_len, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ v,       // [seq_len, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ output,        // [seq_len, num_q_heads, head_dim]
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float attn_scale
) {
    // Each warp handles one (query_pos, q_head) pair
    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int total_work = seq_len * num_q_heads;
    if (warp_idx >= total_work) return;

    int q_pos = warp_idx / num_q_heads;
    int q_head = warp_idx % num_q_heads;
    int kv_head = q_head / (num_q_heads / num_kv_heads);  // GQA mapping

    const __nv_bfloat16* q_vec = q + q_pos * num_q_heads * head_dim + q_head * head_dim;
    __nv_bfloat16* out_vec = output + q_pos * num_q_heads * head_dim + q_head * head_dim;

    // Each lane accumulates for its portion of head_dim
    // head_dim=128, WARP_SIZE=32, so 4 elements per lane
    constexpr int ELEMS_PER_LANE = HEAD_DIM / WARP_SIZE;  // 4

    float out_acc[ELEMS_PER_LANE] = {0.0f};
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // Load Q values
    float q_local[ELEMS_PER_LANE];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_LANE; e++) {
        q_local[e] = __bfloat162float(q_vec[lane_id + e * WARP_SIZE]);
    }

    // Process all KV positions up to and including q_pos (causal)
    for (int kv_pos = 0; kv_pos <= q_pos; kv_pos++) {
        const __nv_bfloat16* k_vec = k + kv_pos * num_kv_heads * head_dim + kv_head * head_dim;
        const __nv_bfloat16* v_vec = v + kv_pos * num_kv_heads * head_dim + kv_head * head_dim;

        // Compute dot product Q @ K
        float score = 0.0f;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
            score += q_local[e] * __bfloat162float(k_vec[lane_id + e * WARP_SIZE]);
        }

        // Warp reduction for dot product
        score = prefill_warp_reduce_sum(score) * attn_scale;
        // Broadcast reduced score to all lanes (only lane 0 has correct value after reduction)
        score = __shfl_sync(0xffffffff, score, 0);

        // Online softmax
        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float exp_diff = expf(old_max - max_score);
        sum_exp = sum_exp * exp_diff + expf(score - max_score);

        // Update output accumulator
        float weight = expf(score - max_score);
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
            out_acc[e] = out_acc[e] * exp_diff + weight * __bfloat162float(v_vec[lane_id + e * WARP_SIZE]);
        }
    }

    // Write output
    float sum_inv = 1.0f / sum_exp;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_LANE; e++) {
        out_vec[lane_id + e * WARP_SIZE] = __float2bfloat16(out_acc[e] * sum_inv);
    }
}

// Flash-style tiled prefill attention.
// One block computes one (q_pos, q_head). K/V are streamed by tiles into shared memory.
// Online softmax is kept in FP32 to preserve quality.
__global__ void prefill_causal_attention_flash_tile_kernel(
    const __nv_bfloat16* __restrict__ q,       // [seq_len, num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ k,       // [seq_len, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ v,       // [seq_len, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ output,        // [seq_len, num_q_heads, head_dim]
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float attn_scale
) {
    constexpr int FLASH_HEAD_DIM = 128;
    constexpr int FLASH_TILE_KV = 16;
    if (head_dim != FLASH_HEAD_DIM) return;

    int query_idx = blockIdx.x;
    int total_queries = seq_len * num_q_heads;
    if (query_idx >= total_queries) return;

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    if (tid >= FLASH_HEAD_DIM) return;

    int q_pos = query_idx / num_q_heads;
    int q_head = query_idx % num_q_heads;
    int kv_head = q_head / (num_q_heads / num_kv_heads);

    const __nv_bfloat16* q_vec = q + q_pos * num_q_heads * head_dim + q_head * head_dim;
    __nv_bfloat16* out_vec = output + q_pos * num_q_heads * head_dim + q_head * head_dim;

    __shared__ float s_q[FLASH_HEAD_DIM];
    __shared__ float s_k[FLASH_TILE_KV * FLASH_HEAD_DIM];
    __shared__ float s_v[FLASH_TILE_KV * FLASH_HEAD_DIM];
    __shared__ float s_warp_sum[PREFILL_NUM_WARPS];
    __shared__ float s_score;
    __shared__ float s_old_scale;
    __shared__ float s_new_scale;
    __shared__ float s_sum_exp;

    s_q[tid] = __bfloat162float(q_vec[tid]);
    __syncthreads();

    float out_acc = 0.0f;
    float running_m = -INFINITY;
    float running_s = 0.0f;

    for (int tile_start = 0; tile_start <= q_pos; tile_start += FLASH_TILE_KV) {
        int tile_len = min(FLASH_TILE_KV, q_pos + 1 - tile_start);
        int tile_elems = tile_len * FLASH_HEAD_DIM;

        for (int idx = tid; idx < tile_elems; idx += FLASH_HEAD_DIM) {
            int r = idx / FLASH_HEAD_DIM;
            int d = idx % FLASH_HEAD_DIM;
            int kv_pos = tile_start + r;
            const __nv_bfloat16* k_vec = k + kv_pos * num_kv_heads * head_dim + kv_head * head_dim;
            const __nv_bfloat16* v_vec = v + kv_pos * num_kv_heads * head_dim + kv_head * head_dim;
            s_k[idx] = __bfloat162float(k_vec[d]);
            s_v[idx] = __bfloat162float(v_vec[d]);
        }
        __syncthreads();

        for (int j = 0; j < tile_len; j++) {
            float dot = s_q[tid] * s_k[j * FLASH_HEAD_DIM + tid];
            dot = prefill_warp_reduce_sum(dot);
            if (lane_id == 0) s_warp_sum[warp_id] = dot;
            __syncthreads();

            if (warp_id == 0) {
                float bsum = (lane_id < num_warps) ? s_warp_sum[lane_id] : 0.0f;
                bsum = prefill_warp_reduce_sum(bsum);
                if (lane_id == 0) {
                    s_score = bsum * attn_scale;
                }
            }
            __syncthreads();

            if (tid == 0) {
                float score = s_score;
                float new_m = fmaxf(running_m, score);
                float old_scale = (running_s > 0.0f) ? expf(running_m - new_m) : 0.0f;
                float new_scale = expf(score - new_m);
                running_s = running_s * old_scale + new_scale;
                running_m = new_m;
                s_old_scale = old_scale;
                s_new_scale = new_scale;
                s_sum_exp = running_s;
            }
            __syncthreads();

            out_acc = out_acc * s_old_scale + s_new_scale * s_v[j * FLASH_HEAD_DIM + tid];
            __syncthreads();
        }
    }

    float denom = (s_sum_exp > 0.0f) ? s_sum_exp : 1e-20f;
    out_vec[tid] = __float2bfloat16(out_acc / denom);
}

__global__ void prefill_attn_compare_stats_kernel(
    const __nv_bfloat16* __restrict__ flash_out,
    const __nv_bfloat16* __restrict__ ref_out,
    int n,
    double* __restrict__ sum_abs_diff,
    double* __restrict__ sum_sq_diff,
    double* __restrict__ sum_abs_ref,
    unsigned int* __restrict__ max_abs_bits,
    int* __restrict__ bad_count,
    int* __restrict__ gt_1e2_count,
    int* __restrict__ gt_5e2_count,
    int* __restrict__ gt_1e1_count
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;

    float fv = __bfloat162float(flash_out[idx]);
    float rv = __bfloat162float(ref_out[idx]);

    bool bad = (!isfinite(fv) || !isfinite(rv));
    if (bad) {
        atomicAdd(bad_count, 1);
        return;
    }

    float ad = fabsf(fv - rv);
    atomicAdd(sum_abs_diff, (double)ad);
    atomicAdd(sum_sq_diff, (double)ad * (double)ad);
    atomicAdd(sum_abs_ref, (double)fabsf(rv));
    atomicMax(max_abs_bits, __float_as_uint(ad));

    if (ad > 1e-2f) atomicAdd(gt_1e2_count, 1);
    if (ad > 5e-2f) atomicAdd(gt_5e2_count, 1);
    if (ad > 1e-1f) atomicAdd(gt_1e1_count, 1);
}

// Split-K prefill attention:
// one block handles one (q_pos, q_head), and multiple warps split kv positions.
// This reduces long-serial kv loops for large seq_len (e.g. ASR prefill).
__global__ void prefill_causal_attention_splitk_kernel(
    const __nv_bfloat16* __restrict__ q,       // [seq_len, num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ k,       // [seq_len, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ v,       // [seq_len, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ output,        // [seq_len, num_q_heads, head_dim]
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float attn_scale,
    int warps_per_block
) {
    int query_idx = blockIdx.x;
    int total_queries = seq_len * num_q_heads;
    if (query_idx >= total_queries) return;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    if (warp_id >= warps_per_block) return;

    int q_pos = query_idx / num_q_heads;
    int q_head = query_idx % num_q_heads;
    int kv_head = q_head / (num_q_heads / num_kv_heads);

    const __nv_bfloat16* q_vec = q + q_pos * num_q_heads * head_dim + q_head * head_dim;
    __nv_bfloat16* out_vec = output + q_pos * num_q_heads * head_dim + q_head * head_dim;

    constexpr int ELEMS_PER_LANE = HEAD_DIM / WARP_SIZE;  // 4 for head_dim=128

    float q_local[ELEMS_PER_LANE];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_LANE; e++) {
        int d = lane_id + e * WARP_SIZE;
        q_local[e] = (d < head_dim) ? __bfloat162float(q_vec[d]) : 0.0f;
    }

    float local_out[ELEMS_PER_LANE] = {0.0f};
    float local_m = -INFINITY;
    float local_s = 0.0f;

    for (int kv_pos = warp_id; kv_pos <= q_pos; kv_pos += warps_per_block) {
        const __nv_bfloat16* k_vec = k + kv_pos * num_kv_heads * head_dim + kv_head * head_dim;
        const __nv_bfloat16* v_vec = v + kv_pos * num_kv_heads * head_dim + kv_head * head_dim;

        float score = 0.0f;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
            int d = lane_id + e * WARP_SIZE;
            if (d < head_dim) {
                score += q_local[e] * __bfloat162float(k_vec[d]);
            }
        }
        score = prefill_warp_reduce_sum(score) * attn_scale;
        score = __shfl_sync(0xffffffff, score, 0);

        float new_m = fmaxf(local_m, score);
        float old_scale = (local_s > 0.0f) ? expf(local_m - new_m) : 0.0f;
        float new_scale = expf(score - new_m);

        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
            int d = lane_id + e * WARP_SIZE;
            float vv = (d < head_dim) ? __bfloat162float(v_vec[d]) : 0.0f;
            local_out[e] = local_out[e] * old_scale + new_scale * vv;
        }
        local_s = local_s * old_scale + new_scale;
        local_m = new_m;
    }

    __shared__ float s_warp_m[PREFILL_NUM_WARPS];
    __shared__ float s_warp_s[PREFILL_NUM_WARPS];
    __shared__ float s_warp_factor[PREFILL_NUM_WARPS];
    __shared__ float s_partial_out[PREFILL_NUM_WARPS * HEAD_DIM];
    __shared__ float s_global_m;
    __shared__ float s_global_s;

    if (lane_id == 0) {
        s_warp_m[warp_id] = local_m;
        s_warp_s[warp_id] = local_s;
    }
    __syncthreads();

    if (warp_id == 0) {
        float m = (lane_id < warps_per_block) ? s_warp_m[lane_id] : -INFINITY;
        m = prefill_warp_reduce_max(m);
        if (lane_id == 0) s_global_m = m;
    }
    __syncthreads();

    float gmax = s_global_m;
    if (lane_id == 0) {
        float f = (local_s > 0.0f) ? expf(local_m - gmax) : 0.0f;
        s_warp_factor[warp_id] = f;
        s_warp_s[warp_id] = local_s * f;
    }
    __syncthreads();

    float w_factor = s_warp_factor[warp_id];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_LANE; e++) {
        local_out[e] *= w_factor;
        int d = lane_id + e * WARP_SIZE;
        if (d < head_dim) {
            s_partial_out[warp_id * HEAD_DIM + d] = local_out[e];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        float s = (lane_id < warps_per_block) ? s_warp_s[lane_id] : 0.0f;
        s = prefill_warp_reduce_sum(s);
        if (lane_id == 0) s_global_s = (s > 0.0f) ? s : 1e-20f;
    }
    __syncthreads();

    float inv_s = 1.0f / s_global_s;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_LANE; e++) {
        int d = lane_id + e * WARP_SIZE;
        if (d >= head_dim) continue;
        float acc = 0.0f;
        for (int w = 0; w < warps_per_block; w++) {
            acc += s_partial_out[w * HEAD_DIM + d];
        }
        out_vec[d] = __float2bfloat16(acc * inv_s);
    }
}

// =============================================================================
// Residual Add Kernel (BF16)
// =============================================================================

__global__ void prefill_residual_add_kernel(
    const __nv_bfloat16* __restrict__ input,     // [seq_len, hidden_size]
    const __nv_bfloat16* __restrict__ residual,  // [seq_len, hidden_size]
    __nv_bfloat16* __restrict__ output,          // [seq_len, hidden_size]
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float a = __bfloat162float(input[idx]);
        float b = __bfloat162float(residual[idx]);
        output[idx] = __float2bfloat16(a + b);
    }
}

// =============================================================================
// SiLU + Element-wise Multiply Kernel (BF16)
// =============================================================================

__global__ void prefill_silu_mul_kernel(
    const __nv_bfloat16* __restrict__ gate,   // [seq_len, intermediate_size]
    const __nv_bfloat16* __restrict__ up,     // [seq_len, intermediate_size]
    __nv_bfloat16* __restrict__ output,       // [seq_len, intermediate_size]
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float g = __bfloat162float(gate[idx]);
        float u = __bfloat162float(up[idx]);
        output[idx] = __float2bfloat16(prefill_silu(g) * u);
    }
}

// =============================================================================
// Final Norm Kernel (only process last token, output BF16)
// =============================================================================

__global__ void prefill_final_norm_kernel(
    const __nv_bfloat16* __restrict__ input,       // [seq_len, hidden_size]
    const __nv_bfloat16* __restrict__ weight,      // [hidden_size]
    __nv_bfloat16* __restrict__ output,            // [hidden_size] - only last token
    int seq_len,
    int hidden_size
) {
    // Only process last token
    const __nv_bfloat16* in_row = input + (seq_len - 1) * hidden_size;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    __shared__ float smem_reduce[PREFILL_NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += PREFILL_BLOCK_SIZE) {
        float v = __bfloat162float(in_row[i]);
        local_sum_sq += v * v;
    }

    local_sum_sq = prefill_warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
        smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < PREFILL_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = prefill_warp_reduce_sum(sum);
        if (lane_id == 0) {
            smem_reduce[0] = rsqrtf(sum / float(hidden_size) + PREFILL_RMS_EPS);
        }
    }
    __syncthreads();

    float rstd = smem_reduce[0];

    for (int i = threadIdx.x; i < hidden_size; i += PREFILL_BLOCK_SIZE) {
        float v = __bfloat162float(in_row[i]);
        float w = __bfloat162float(weight[i]);
        output[i] = __float2bfloat16(v * rstd * w);
    }
}

// =============================================================================
// LM Head with argmax (BF16 input)
// =============================================================================

constexpr int PREFILL_LM_NUM_BLOCKS = 1184;
constexpr int PREFILL_LM_BLOCK_SIZE = 256;
constexpr int PREFILL_VOCAB_SIZE = 151936;

__global__ void prefill_lm_head_phase1(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ block_max_vals,
    int* __restrict__ block_max_idxs
) {
    __shared__ float s_hidden[HIDDEN_SIZE];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += PREFILL_LM_BLOCK_SIZE) {
        s_hidden[i] = __bfloat162float(hidden[i]);
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (PREFILL_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, PREFILL_VOCAB_SIZE);

    float local_max = -INFINITY;
    int local_max_idx = -1;

    for (int m = row_start + warp_id; m < row_end; m += PREFILL_LM_BLOCK_SIZE / WARP_SIZE) {
        const __nv_bfloat16* w_row = weight + m * HIDDEN_SIZE;

        float sum = 0.0f;
        #pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(w_row + k));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

            sum += __bfloat162float(w_ptr[0]) * s_hidden[k] +
                   __bfloat162float(w_ptr[1]) * s_hidden[k+1] +
                   __bfloat162float(w_ptr[2]) * s_hidden[k+2] +
                   __bfloat162float(w_ptr[3]) * s_hidden[k+3];
        }
        sum = prefill_warp_reduce_sum(sum);

        if (lane_id == 0 && sum > local_max) {
            local_max = sum;
            local_max_idx = m;
        }
    }

    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float warp_max[PREFILL_LM_BLOCK_SIZE / WARP_SIZE];
    __shared__ int warp_idx[PREFILL_LM_BLOCK_SIZE / WARP_SIZE];

    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
        warp_idx[warp_id] = local_max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float max_val = (lane_id < PREFILL_LM_BLOCK_SIZE / WARP_SIZE) ? warp_max[lane_id] : -INFINITY;
        int max_idx = (lane_id < PREFILL_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
            if (other_val > max_val) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            block_max_vals[blockIdx.x] = max_val;
            block_max_idxs[blockIdx.x] = max_idx;
        }
    }
}

__global__ void prefill_lm_head_phase2(
    const float* __restrict__ block_max_vals,
    const int* __restrict__ block_max_idxs,
    int* __restrict__ output_token,
    int num_blocks
) {
    __shared__ float s_max_vals[1024];
    __shared__ int s_max_idxs[1024];

    int tid = threadIdx.x;

    float local_max = -INFINITY;
    int local_idx = -1;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float val = block_max_vals[i];
        if (val > local_max) {
            local_max = val;
            local_idx = block_max_idxs[i];
        }
    }

    s_max_vals[tid] = local_max;
    s_max_idxs[tid] = local_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max_vals[tid + s] > s_max_vals[tid]) {
                s_max_vals[tid] = s_max_vals[tid + s];
                s_max_idxs[tid] = s_max_idxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output_token = s_max_idxs[0];
    }
}
