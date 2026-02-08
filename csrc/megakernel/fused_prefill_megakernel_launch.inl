// =============================================================================
// Launch Function
// =============================================================================

extern "C" void launch_prefill_megakernel(
    const int* token_ids,
    int* output_token_id,
    int seq_len,
    const void* embed_weight,
    const void* audio_embeds,        // [audio_len, HIDDEN_SIZE] bf16 on device (optional)
    int audio_start_idx,
    int audio_len,
    const PrefillMKLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    // Intermediate buffers (float32)
    void* hidden,
    void* residual,
    void* normalized,
    void* q_proj,
    void* k_proj,
    void* v_proj,
    void* attn_out,
    void* mlp_intermediate,
    void* final_hidden,
    void* hidden_bf16_out,
    void* block_max_vals,
    void* block_max_idxs,
    int num_layers,
    int max_seq_len,
    float attn_scale,
    cudaStream_t stream
) {
    bool debug_stage = _prefill_mk_debug_take_ticket();
    cudaEvent_t evt_start[PREFILL_MK_STAGE_COUNT] = {nullptr, nullptr, nullptr};
    cudaEvent_t evt_stop[PREFILL_MK_STAGE_COUNT] = {nullptr, nullptr, nullptr};
    auto dbg_create = [&](int st) {
        if (!debug_stage) return;
        if (cudaEventCreateWithFlags(&evt_start[st], cudaEventDefault) != cudaSuccess) {
            evt_start[st] = nullptr;
            evt_stop[st] = nullptr;
            return;
        }
        if (cudaEventCreateWithFlags(&evt_stop[st], cudaEventDefault) != cudaSuccess) {
            cudaEventDestroy(evt_start[st]);
            evt_start[st] = nullptr;
            evt_stop[st] = nullptr;
        }
    };
    auto dbg_begin = [&](int st) {
        if (!debug_stage || evt_start[st] == nullptr) return;
        cudaEventRecord(evt_start[st], stream);
    };
    auto dbg_end = [&](int st) {
        if (!debug_stage || evt_stop[st] == nullptr) return;
        cudaEventRecord(evt_stop[st], stream);
    };
    if (debug_stage) {
        for (int s = 0; s < PREFILL_MK_STAGE_COUNT; s++) dbg_create(s);
    }

    // Basic validation to avoid OOB writes in the patch phase.
    if (audio_embeds != nullptr && audio_len > 0) {
        if (audio_start_idx < 0 || audio_start_idx + audio_len > seq_len) {
            // In this repo we prefer failing fast over silent corruption.
            // (This function is called from a C++ extension; errors will surface
            // as an invalid output token / CUDA error downstream.)
            return;
        }
    }

    void* kernel_args[] = {
        (void*)&token_ids,
        (void*)&embed_weight,
        (void*)&audio_embeds,
        (void*)&audio_start_idx,
        (void*)&audio_len,
        (void*)&layer_weights,
        (void*)&final_norm_weight,
        (void*)&cos_table,
        (void*)&sin_table,
        (void*)&k_cache,
        (void*)&v_cache,
        (void*)&hidden,
        (void*)&residual,
        (void*)&normalized,
        (void*)&q_proj,
        (void*)&k_proj,
        (void*)&v_proj,
        (void*)&attn_out,
        (void*)&mlp_intermediate,
        (void*)&final_hidden,
        (void*)&hidden_bf16_out,
        (void*)&seq_len,
        (void*)&num_layers,
        (void*)&max_seq_len,
        (void*)&attn_scale
    };

    dbg_begin(PREFILL_MK_STAGE_CORE);
    cudaLaunchCooperativeKernel(
        (void*)prefill_megakernel,
        dim3(PREFILL_MK_NUM_BLOCKS),
        dim3(PREFILL_MK_BLOCK_SIZE),
        kernel_args,
        0,
        stream
    );
    dbg_end(PREFILL_MK_STAGE_CORE);

    // LM Head (2-phase argmax)
    dbg_begin(PREFILL_MK_STAGE_LM1);
    prefill_mk_lm_head_phase1<<<PREFILL_MK_LM_NUM_BLOCKS, PREFILL_MK_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)final_hidden,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );
    dbg_end(PREFILL_MK_STAGE_LM1);

    dbg_begin(PREFILL_MK_STAGE_LM2);
    prefill_mk_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        PREFILL_MK_LM_NUM_BLOCKS
    );
    dbg_end(PREFILL_MK_STAGE_LM2);

    if (debug_stage) {
        cudaStreamSynchronize(stream);
        float stage_ms[PREFILL_MK_STAGE_COUNT] = {0.0f, 0.0f, 0.0f};
        float total_ms = 0.0f;
        for (int s = 0; s < PREFILL_MK_STAGE_COUNT; s++) {
            if (evt_start[s] != nullptr && evt_stop[s] != nullptr) {
                cudaEventElapsedTime(&stage_ms[s], evt_start[s], evt_stop[s]);
                total_ms += stage_ms[s];
            }
        }
        printf("[MEGAQWEN_DEBUG] PREFILL(megakernel) stage timing seq_len=%d layers=%d\n", seq_len, num_layers);
        for (int s = 0; s < PREFILL_MK_STAGE_COUNT; s++) {
            float pct = (total_ms > 0.0f) ? (stage_ms[s] * 100.0f / total_ms) : 0.0f;
            printf("  %-16s total=%8.3f ms  (%5.1f%%)\n", kPrefillMKStageNames[s], stage_ms[s], pct);
        }
        printf("  %-16s total=%8.3f ms\n", "prefill_sum", total_ms);
        for (int s = 0; s < PREFILL_MK_STAGE_COUNT; s++) {
            if (evt_start[s] != nullptr) cudaEventDestroy(evt_start[s]);
            if (evt_stop[s] != nullptr) cudaEventDestroy(evt_stop[s]);
        }
    }
}
