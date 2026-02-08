// =============================================================================
// Main Kernel
// =============================================================================

template <bool kFuseLmHead>
__global__ void __launch_bounds__(LDG_BLOCK_SIZE, MEGAQWEN_LDG_MIN_BLOCKS_PER_SM)
ldg_decode_kernel(
    const int* input_token_id_ptr,
    const __nv_bfloat16* __restrict__ embed_weight,
    const LDGLayerWeights* __restrict__ layer_weights,
    const __nv_bfloat16* __restrict__ final_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ hidden_buffer,
    float* __restrict__ g_activations,
    float* __restrict__ g_residual,
    float* __restrict__ g_q,
    float* __restrict__ g_k,
    float* __restrict__ g_v,
    float* __restrict__ g_attn_out,
    float* __restrict__ g_mlp_intermediate,
    float* __restrict__ g_normalized,
    float* __restrict__ attn_partial_max,
    float* __restrict__ attn_partial_sum,
    float* __restrict__ attn_partial_out,
    int num_layers,
    const int* __restrict__ position_ptr,
    int max_seq_len,
    float attn_scale,
    int attn_all_blocks,
    const __nv_bfloat16* __restrict__ lm_head_weight,
    float* __restrict__ block_max_vals,
    int* __restrict__ block_max_idxs,
    int* __restrict__ output_token_id,
    int fuse_lm_head
) {
    cg::grid_group grid = cg::this_grid();
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;

    const int position = *position_ptr;
    const int cache_len = position + 1;

    int input_token_id = *input_token_id_ptr;
    // Embedding lookup with __ldg
    const __nv_bfloat16* embed_row = embed_weight + input_token_id * HIDDEN_SIZE;
    for (int i = block_id * LDG_BLOCK_SIZE + threadIdx.x; i < HIDDEN_SIZE; i += num_blocks * LDG_BLOCK_SIZE) {
        hidden_buffer[i] = __ldg(embed_row + i);
    }
    grid.sync();

    int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    for (int layer = 0; layer < num_layers; layer++) {
        unsigned long long* dbg_stage = nullptr;
        if (g_ldg_dbg_cycles != nullptr &&
            g_ldg_dbg_nstages == 16 &&
            g_ldg_dbg_num_blocks >= num_blocks &&
            g_ldg_dbg_num_layers == num_layers) {
            const long long idx = ((long long)block_id * (long long)num_layers + (long long)layer) * 16ll;
            dbg_stage = g_ldg_dbg_cycles + idx;
        }

        const LDGLayerWeights& w = layer_weights[layer];
        __nv_bfloat16* layer_k_cache = k_cache + layer * kv_cache_layer_stride;
        __nv_bfloat16* layer_v_cache = v_cache + layer * kv_cache_layer_stride;

        ldg_matvec_qkv(
            grid, hidden_buffer, w.input_layernorm_weight,
            w.q_proj_weight, w.k_proj_weight, w.v_proj_weight,
            g_activations, g_residual, g_q, g_k, g_v,
            dbg_stage
        );

        ldg_qk_norm_rope_cache(
            grid, g_q, g_k, g_v,
            w.q_norm_weight, w.k_norm_weight,
            cos_table, sin_table,
            layer_k_cache, layer_v_cache,
            position, max_seq_len,
            dbg_stage
        );

        ldg_attention(
            grid, g_q, layer_k_cache, layer_v_cache, g_attn_out,
            attn_partial_max, attn_partial_sum, attn_partial_out,
            cache_len, max_seq_len, attn_scale, attn_all_blocks,
            w.o_proj_weight, w.gate_proj_weight, w.up_proj_weight,
            dbg_stage
        );

        ldg_o_proj_postnorm_mlp(
            grid, w.o_proj_weight, w.post_attn_layernorm_weight,
            w.gate_proj_weight, w.up_proj_weight, w.down_proj_weight,
            g_attn_out, g_residual, g_activations, g_mlp_intermediate,
            hidden_buffer,
            dbg_stage
        );
    }

    // Final RMSNorm
    if (block_id == 0) {
        __shared__ float smem_reduce[LDG_NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float v = __bfloat162float(hidden_buffer[i]);
            g_activations[i] = v;
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
            float wt = __bfloat162float(__ldg(final_norm_weight + i));
            g_normalized[i] = g_activations[i] * rstd * wt;
        }
    }

    // NOTE: The fused LM head argmax path uses substantial shared memory.
    // Keep it compiled only into the kFuseLmHead=true instantiation so the
    // default (unfused) decode kernel can reach higher occupancy.
    if constexpr (kFuseLmHead) {
        if (!fuse_lm_head) {
            return;
        }

        // =============================================================================
        // Fused LM head argmax (single cooperative kernel stage).
        // =============================================================================

        grid.sync();  // ensure g_normalized is ready for all blocks

        __shared__ float s_hidden[HIDDEN_SIZE];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_LM_BLOCK_SIZE) {
            s_hidden[i] = g_normalized[i];
        }
        __syncthreads();

        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;

        int rows_per_block = (LDG_VOCAB_SIZE + num_blocks - 1) / num_blocks;
        int row_start = block_id * rows_per_block;
        int row_end = min(row_start + rows_per_block, LDG_VOCAB_SIZE);

        float local_max = -INFINITY;
        int local_max_idx = -1;

        for (int m = row_start + warp_id; m < row_end; m += LDG_LM_BLOCK_SIZE / WARP_SIZE) {
            const __nv_bfloat16* w_row = lm_head_weight + m * HIDDEN_SIZE;

            float sum = 0.0f;
            MEGAQWEN_PRAGMA_UNROLL_N(MEGAQWEN_LDG_UNROLL)
            for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
                uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(w_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

                sum += __bfloat162float(w_ptr[0]) * s_hidden[k] +
                       __bfloat162float(w_ptr[1]) * s_hidden[k + 1] +
                       __bfloat162float(w_ptr[2]) * s_hidden[k + 2] +
                       __bfloat162float(w_ptr[3]) * s_hidden[k + 3];
            }
            sum = ldg_warp_reduce_sum(sum);

            if (lane_id == 0 && sum > local_max) {
                local_max = sum;
                local_max_idx = m;
            }
        }

        local_max = __shfl_sync(0xffffffff, local_max, 0);
        local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

        __shared__ float warp_max[LDG_LM_BLOCK_SIZE / WARP_SIZE];
        __shared__ int warp_idx[LDG_LM_BLOCK_SIZE / WARP_SIZE];

        if (lane_id == 0) {
            warp_max[warp_id] = local_max;
            warp_idx[warp_id] = local_max_idx;
        }
        __syncthreads();

        if (warp_id == 0) {
            float max_val = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? warp_max[lane_id] : -INFINITY;
            int max_idx = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;

            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
                int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
                if (other_val > max_val) {
                    max_val = other_val;
                    max_idx = other_idx;
                }
            }

            if (lane_id == 0) {
                block_max_vals[block_id] = max_val;
                block_max_idxs[block_id] = max_idx;
            }
        }

        grid.sync();

        if (block_id == 0) {
            __shared__ float s_max_vals[1024];
            __shared__ int s_max_idxs[1024];

            int tid = threadIdx.x;
            float bmax = -INFINITY;
            int bidx = -1;

            for (int i = tid; i < num_blocks; i += blockDim.x) {
                float val = block_max_vals[i];
                if (val > bmax) {
                    bmax = val;
                    bidx = block_max_idxs[i];
                }
            }

            s_max_vals[tid] = bmax;
            s_max_idxs[tid] = bidx;
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
                *output_token_id = s_max_idxs[0];
            }
        }
    }
}

