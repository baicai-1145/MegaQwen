// =============================================================================
// Launch function
// =============================================================================

__global__ void ldg_fill_iota_kernel(int* __restrict__ out, int start, int n) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) out[idx] = start + idx;
}

extern "C" void launch_ldg_fill_iota(int* out, int start, int n, cudaStream_t stream) {
    if (n <= 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    ldg_fill_iota_kernel<<<grid, block, 0, stream>>>(out, start, n);
}

// Find the first EOS token index in `tokens[0:n]` against a (small) eos-id list.
// Writes `out_min_idx = n` when no EOS is found.
__global__ void ldg_find_first_eos_kernel(
    const int* __restrict__ tokens,
    int n,
    const int* __restrict__ eos_ids,
    int eos_n,
    int* __restrict__ out_min_idx
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    int t = tokens[i];
    // eos_n is typically tiny (1~3). Keep it simple.
    for (int j = 0; j < eos_n; j++) {
        if (t == eos_ids[j]) {
            atomicMin(out_min_idx, i);
            break;
        }
    }
}

extern "C" void launch_ldg_find_first_eos(
    const int* tokens,
    int n,
    const int* eos_ids,
    int eos_n,
    int* out_min_idx,
    cudaStream_t stream
) {
    if (n <= 0) return;
    int init = n;
    cudaMemcpyAsync(out_min_idx, &init, sizeof(int), cudaMemcpyHostToDevice, stream);
    if (eos_n <= 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    ldg_find_first_eos_kernel<<<grid, block, 0, stream>>>(tokens, n, eos_ids, eos_n, out_min_idx);
}

extern "C" void launch_ldg_decode(
    const int* input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const LDGLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* attn_partial_max,
    void* attn_partial_sum,
    void* attn_partial_out,
    void* block_max_vals,
    void* block_max_idxs,
    int num_layers,
    const int* position_ptr,
    int max_seq_len,
    float attn_scale,
    int decode_num_blocks,
    int lm_num_blocks,
    cudaStream_t stream
) {
    // Choose a safe cooperative grid size: all blocks must be resident concurrently.
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    int fuse_lm_head = 0;
    if (const char* s = std::getenv("MEGAQWEN_FUSE_LM_HEAD")) {
        fuse_lm_head = (std::atoi(s) != 0) ? 1 : 0;
    }
    int attn_all_blocks = 0;
    if (const char* s = std::getenv("MEGAQWEN_ATTN_ALL_BLOCKS")) {
        attn_all_blocks = (std::atoi(s) != 0) ? 1 : 0;
    }

    // IMPORTANT: fused vs unfused decode are different kernel instantiations
    // with different shared/reg footprints; occupancy must be computed on
    // the selected one.
    auto kernel_fn = fuse_lm_head ? ldg_decode_kernel<true> : ldg_decode_kernel<false>;
    int max_blocks_per_sm = 0;
    int max_coop_blocks = 0;
    if (cudaSuccess == cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, kernel_fn, LDG_BLOCK_SIZE, 0) && max_blocks_per_sm > 0) {
        max_coop_blocks = max_blocks_per_sm * prop.multiProcessorCount;
    }

    // Optional one-shot debug print that doesn't require Nsight counters.
    // Useful in containerized environments where ERR_NVGPUCTRPERM blocks ncu/nsys.
    static int s_printed = 0;
    if (!s_printed && std::getenv("MEGAQWEN_DEBUG_LDG_ATTRS")) {
        s_printed = 1;
        cudaFuncAttributes attr;
        if (cudaSuccess == cudaFuncGetAttributes(&attr, kernel_fn)) {
            const int max_threads_sm = prop.maxThreadsPerMultiProcessor;
            const int active_threads_sm = max_blocks_per_sm * LDG_BLOCK_SIZE;
            const float occ = (max_threads_sm > 0) ? (float(active_threads_sm) / float(max_threads_sm)) : 0.0f;
            std::fprintf(
                stderr,
                "[MEGAQWEN_DEBUG] ldg_decode_kernel(%s): cc=%d.%d sm=%d regs=%d smem=%zuB maxBlocks/SM=%d => threads/SM=%d/%d (occ~%.3f)\n",
                fuse_lm_head ? "fused" : "unfused",
                prop.major,
                prop.minor,
                prop.multiProcessorCount,
                attr.numRegs,
                (size_t)attr.sharedSizeBytes,
                max_blocks_per_sm,
                active_threads_sm,
                max_threads_sm,
                occ
            );
        } else {
            std::fprintf(stderr, "[MEGAQWEN_DEBUG] ldg_decode_kernel: cudaFuncGetAttributes failed\n");
        }
    }

    if (decode_num_blocks <= 0) {
        decode_num_blocks = (max_coop_blocks > 0) ? max_coop_blocks : LDG_NUM_BLOCKS_DEFAULT;
    } else if (max_coop_blocks > 0 && decode_num_blocks > max_coop_blocks) {
        // Avoid cudaErrorCooperativeLaunchTooLarge when the user requests too many blocks.
        decode_num_blocks = max_coop_blocks;
    }

    // Keep scratch buffers bounded (see block_max_vals/idxs and attention scratch alloc).
    if (decode_num_blocks > LDG_LM_NUM_BLOCKS_DEFAULT) decode_num_blocks = LDG_LM_NUM_BLOCKS_DEFAULT;
    if (lm_num_blocks <= 0) lm_num_blocks = LDG_LM_NUM_BLOCKS_DEFAULT;
    void* kernel_args[] = {
        (void*)&input_token_id,
        (void*)&embed_weight,
        (void*)&layer_weights,
        (void*)&final_norm_weight,
        (void*)&cos_table,
        (void*)&sin_table,
        (void*)&k_cache,
        (void*)&v_cache,
        (void*)&hidden_buffer,
        (void*)&g_activations,
        (void*)&g_residual,
        (void*)&g_q,
        (void*)&g_k,
        (void*)&g_v,
        (void*)&g_attn_out,
        (void*)&g_mlp_intermediate,
        (void*)&g_normalized,
        (void*)&attn_partial_max,
        (void*)&attn_partial_sum,
        (void*)&attn_partial_out,
        (void*)&num_layers,
        (void*)&position_ptr,
        (void*)&max_seq_len,
        (void*)&attn_scale,
        (void*)&attn_all_blocks,
        (void*)&lm_head_weight,
        (void*)&block_max_vals,
        (void*)&block_max_idxs,
        (void*)&output_token_id,
        (void*)&fuse_lm_head
    };

    cudaLaunchCooperativeKernel(
        (void*)kernel_fn,
        dim3(decode_num_blocks),
        dim3(LDG_BLOCK_SIZE),
        kernel_args,
        0,
        stream
    );

    if (fuse_lm_head) {
        return;
    }

    ldg_lm_head_phase1<<<lm_num_blocks, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );

    ldg_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        lm_num_blocks
    );
}

// Launch function that also outputs full logits (for KL divergence)
extern "C" void launch_ldg_decode_with_logits(
    const int* input_token_id,
    int* output_token_id,
    float* logits_output,
    const void* embed_weight,
    const LDGLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* attn_partial_max,
    void* attn_partial_sum,
    void* attn_partial_out,
    void* block_max_vals,
    void* block_max_idxs,
    int num_layers,
    const int* position_ptr,
    int max_seq_len,
    float attn_scale,
    int decode_num_blocks,
    int lm_num_blocks,
    cudaStream_t stream
) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    // Logits path always uses the unfused decode instantiation.
    auto kernel_fn = ldg_decode_kernel<false>;
    int max_blocks_per_sm = 0;
    int max_coop_blocks = 0;
    if (cudaSuccess == cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, kernel_fn, LDG_BLOCK_SIZE, 0) && max_blocks_per_sm > 0) {
        max_coop_blocks = max_blocks_per_sm * prop.multiProcessorCount;
    }

    static int s_printed = 0;
    if (!s_printed && std::getenv("MEGAQWEN_DEBUG_LDG_ATTRS")) {
        s_printed = 1;
        cudaFuncAttributes attr;
        if (cudaSuccess == cudaFuncGetAttributes(&attr, kernel_fn)) {
            const int max_threads_sm = prop.maxThreadsPerMultiProcessor;
            const int active_threads_sm = max_blocks_per_sm * LDG_BLOCK_SIZE;
            const float occ = (max_threads_sm > 0) ? (float(active_threads_sm) / float(max_threads_sm)) : 0.0f;
            std::fprintf(
                stderr,
                "[MEGAQWEN_DEBUG] ldg_decode_kernel(unfused,logits): cc=%d.%d sm=%d regs=%d smem=%zuB maxBlocks/SM=%d => threads/SM=%d/%d (occ~%.3f)\n",
                prop.major,
                prop.minor,
                prop.multiProcessorCount,
                attr.numRegs,
                (size_t)attr.sharedSizeBytes,
                max_blocks_per_sm,
                active_threads_sm,
                max_threads_sm,
                occ
            );
        } else {
            std::fprintf(stderr, "[MEGAQWEN_DEBUG] ldg_decode_kernel(logits): cudaFuncGetAttributes failed\n");
        }
    }

    if (decode_num_blocks <= 0) {
        decode_num_blocks = (max_coop_blocks > 0) ? max_coop_blocks : LDG_NUM_BLOCKS_DEFAULT;
    } else if (max_coop_blocks > 0 && decode_num_blocks > max_coop_blocks) {
        decode_num_blocks = max_coop_blocks;
    }
    if (decode_num_blocks > LDG_LM_NUM_BLOCKS_DEFAULT) decode_num_blocks = LDG_LM_NUM_BLOCKS_DEFAULT;
    if (lm_num_blocks <= 0) lm_num_blocks = LDG_LM_NUM_BLOCKS_DEFAULT;
    int fuse_lm_head = 0;  // keep unfused for logits path
    int attn_all_blocks = 0;
    if (const char* s = std::getenv("MEGAQWEN_ATTN_ALL_BLOCKS")) {
        attn_all_blocks = (std::atoi(s) != 0) ? 1 : 0;
    }
    void* kernel_args[] = {
        (void*)&input_token_id,
        (void*)&embed_weight,
        (void*)&layer_weights,
        (void*)&final_norm_weight,
        (void*)&cos_table,
        (void*)&sin_table,
        (void*)&k_cache,
        (void*)&v_cache,
        (void*)&hidden_buffer,
        (void*)&g_activations,
        (void*)&g_residual,
        (void*)&g_q,
        (void*)&g_k,
        (void*)&g_v,
        (void*)&g_attn_out,
        (void*)&g_mlp_intermediate,
        (void*)&g_normalized,
        (void*)&attn_partial_max,
        (void*)&attn_partial_sum,
        (void*)&attn_partial_out,
        (void*)&num_layers,
        (void*)&position_ptr,
        (void*)&max_seq_len,
        (void*)&attn_scale,
        (void*)&attn_all_blocks,
        (void*)&lm_head_weight,
        (void*)&block_max_vals,
        (void*)&block_max_idxs,
        (void*)&output_token_id,
        (void*)&fuse_lm_head
    };

    cudaLaunchCooperativeKernel(
        (void*)kernel_fn,
        dim3(decode_num_blocks),
        dim3(LDG_BLOCK_SIZE),
        kernel_args,
        0,
        stream
    );

    // Compute full logits
    ldg_lm_head_logits<<<lm_num_blocks, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        logits_output
    );

    // Also compute argmax for the token output
    ldg_lm_head_phase1<<<lm_num_blocks, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );

    ldg_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        lm_num_blocks
    );
}
