#include "split_decode_gemm_host_utils.inl"
extern "C" void launch_split_decode_gemm(
    const int* input_token_id,    // device [1]
    int* output_token_id,         // device [1]
    const void* embed_weight,     // bf16
    const PrefillLayerWeights* layer_weights,  // host array (device pointers)
    const void* const* qkv_weight_packed_ptrs,     // host [num_layers], each [Q+KV+KV, H]
    const void* const* gateup_weight_packed_ptrs,  // host [num_layers], each [2*I, H]
    const void* const* down_weight_ptrs,           // host [num_layers], each [H, I]
    const void* const* gateup_w4_packed_ptrs,      // host [num_layers], each uint8 packed [2*I*H/2]
    const void* const* gateup_w4_scales_ptrs,      // host [num_layers], float [2*I*H/64]
    const void* const* gateup_w4_codebook_ptrs,    // host [num_layers], float [16]
    const void* const* down_w4_packed_ptrs,        // host [num_layers], each uint8 packed [H*I/2]
    const void* const* down_w4_scales_ptrs,        // host [num_layers], float [H*I/64]
    const void* const* down_w4_codebook_ptrs,      // host [num_layers], float [16]
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,        // bf16 [max_seq, head_dim]
    const void* sin_table,        // bf16 [max_seq, head_dim]
    void* k_cache,                // bf16
    void* v_cache,                // bf16
    void* split_attn_partial_m,   // [split_max_chunks, num_q_heads] float
    void* split_attn_partial_s,   // [split_max_chunks, num_q_heads] float
    void* split_attn_partial_out, // [split_max_chunks, num_q_heads, head_dim] float
    int split_attn_chunk_size,
    int split_attn_max_chunks,
    // Scratch: FP32 activation chain + BF16 GEMM I/O buffers
    void* hidden_f32,             // [hidden] float
    void* residual_f32,           // [hidden] float
    void* normalized_bf16,        // [hidden]
    void* qkv_proj_bf16,          // [q_size + kv_size + kv_size]
    void* attn_out_bf16,          // [q_size]
    void* o_proj_out_bf16,        // [hidden]
    void* mlp_norm_bf16,          // [hidden]
    void* gateup_out_bf16,        // [2*intermediate]
    void* mlp_intermediate_bf16,  // [intermediate]
    void* down_out_bf16,          // [hidden]
    void* final_hidden_bf16,      // [hidden]
    void* block_max_vals,         // [PREFILL_LM_NUM_BLOCKS] float
    void* block_max_idxs,         // [PREFILL_LM_NUM_BLOCKS] int
    int num_layers,
    const int* position_ptr,      // device scalar (absolute position)
    int max_seq_len,
    float attn_scale,
    cublasHandle_t cublas_handle,
    cublasLtHandle_t cublaslt_handle,
    void* cublaslt_workspace,
    size_t cublaslt_workspace_bytes,
    cudaStream_t stream
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    constexpr int QKV_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
    constexpr int GATEUP_ROWS = INTERMEDIATE_SIZE + INTERMEDIATE_SIZE;
    static SplitLtMatmulPlan qkv_lt_plan;
    static SplitLtMatmulPlan o_lt_plan;
    static SplitLtMatmulPlan gateup_lt_plan;
    static SplitLtMatmulPlan down_lt_plan;
    constexpr int kMaxDebugEvents = 2048;
    SplitDebugEventRec debug_events[kMaxDebugEvents];
    int debug_event_count = 0;
    bool debug_stage_detail = _split_debug_take_ticket();
    bool debug_stage_avg = _split_debug_avg_enabled();
    bool debug_stage_avg_exact = _split_debug_avg_exact();
    int debug_stage_avg_stride = _split_debug_avg_stride();
    if (debug_stage_avg_stride < 1) debug_stage_avg_stride = 1;
    bool debug_stage_avg_sample = false;
    if (debug_stage_avg) {
        auto& agg = _split_debug_agg();
        long long tok_idx = agg.tokens;
        agg.tokens += 1;
        debug_stage_avg_sample =
            debug_stage_avg_exact ||
            debug_stage_avg_stride <= 1 ||
            (tok_idx % static_cast<long long>(debug_stage_avg_stride) == 0);
    }
    bool debug_stage = (debug_stage_detail || debug_stage_avg_sample);
    auto dbg_begin = [&](int stage, int layer_id = -1) -> int {
        if (!debug_stage) return -1;
        if (debug_event_count >= kMaxDebugEvents) return -1;
        int idx = debug_event_count++;
        debug_events[idx].stage = stage;
        debug_events[idx].layer = layer_id;
        if (cudaEventCreateWithFlags(&debug_events[idx].start, cudaEventDefault) != cudaSuccess) {
            debug_events[idx].start = nullptr;
            debug_events[idx].stop = nullptr;
            return -1;
        }
        if (cudaEventCreateWithFlags(&debug_events[idx].stop, cudaEventDefault) != cudaSuccess) {
            cudaEventDestroy(debug_events[idx].start);
            debug_events[idx].start = nullptr;
            debug_events[idx].stop = nullptr;
            return -1;
        }
        cudaEventRecord(debug_events[idx].start, stream);
        return idx;
    };
    auto dbg_end = [&](int idx) {
        if (idx < 0 || idx >= debug_event_count) return;
        if (debug_events[idx].stop == nullptr) return;
        cudaEventRecord(debug_events[idx].stop, stream);
    };

    if (!_cublas_check(cublasSetStream(cublas_handle, stream), "cublasSetStream")) return;
    if (qkv_weight_packed_ptrs == nullptr || gateup_weight_packed_ptrs == nullptr || down_weight_ptrs == nullptr) {
        printf("[MEGAQWEN_SPLIT] packed weight ptrs are null\n");
        return;
    }

    // 0) Embed lookup -> hidden_f32
    int dbg_embed = dbg_begin(SPLIT_STAGE_EMBED);
    split_embed_to_hidden_f32_kernel<<<1, 256, 0, stream>>>(
        input_token_id,
        (const __nv_bfloat16*)embed_weight,
        (float*)hidden_f32
    );
    dbg_end(dbg_embed);

    int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    for (int layer = 0; layer < num_layers; layer++) {
        const PrefillLayerWeights& w = layer_weights[layer];
        const __nv_bfloat16* qkv_weight = (const __nv_bfloat16*)qkv_weight_packed_ptrs[layer];
        const __nv_bfloat16* gateup_weight = (const __nv_bfloat16*)gateup_weight_packed_ptrs[layer];
        const __nv_bfloat16* down_weight = (const __nv_bfloat16*)down_weight_ptrs[layer];
        const uint8_t* gateup_w4_packed = gateup_w4_packed_ptrs ? (const uint8_t*)gateup_w4_packed_ptrs[layer] : nullptr;
        const float* gateup_w4_scales = gateup_w4_scales_ptrs ? (const float*)gateup_w4_scales_ptrs[layer] : nullptr;
        const float* gateup_w4_codebook = gateup_w4_codebook_ptrs ? (const float*)gateup_w4_codebook_ptrs[layer] : nullptr;
        const uint8_t* down_w4_packed = down_w4_packed_ptrs ? (const uint8_t*)down_w4_packed_ptrs[layer] : nullptr;
        const float* down_w4_scales = down_w4_scales_ptrs ? (const float*)down_w4_scales_ptrs[layer] : nullptr;
        const float* down_w4_codebook = down_w4_codebook_ptrs ? (const float*)down_w4_codebook_ptrs[layer] : nullptr;
        bool ffn_w4 = _split_ffn_w4_enabled() &&
                      gateup_w4_packed != nullptr && gateup_w4_scales != nullptr && gateup_w4_codebook != nullptr &&
                      down_w4_packed != nullptr && down_w4_scales != nullptr && down_w4_codebook != nullptr;

        __nv_bfloat16* layer_k_cache = (__nv_bfloat16*)k_cache + layer * kv_cache_layer_stride;
        __nv_bfloat16* layer_v_cache = (__nv_bfloat16*)v_cache + layer * kv_cache_layer_stride;

        // 1) RMSNorm (FP32 chain, BF16 output for GEMM)
        int dbg_rms1 = dbg_begin(SPLIT_STAGE_RMS1, layer);
        split_rmsnorm_f32_to_bf16_kernel<<<1, 256, 0, stream>>>(
            (const float*)hidden_f32,
            (const __nv_bfloat16*)w.input_layernorm_weight,
            (__nv_bfloat16*)normalized_bf16,
            (float*)residual_f32
        );
        dbg_end(dbg_rms1);

        // 2) QKV projection (merged): [Q+KV+KV, H] x [H, 1] -> [Q+KV+KV, 1]
        int dbg_qkv = dbg_begin(SPLIT_STAGE_QKV_GEMM, layer);
        bool qkv_done = false;
        bool qkv_lt_done = false;
        bool qkv_gemmex_done = false;
        bool qkv_gemv_done = false;
        bool qkv_use_gemv = _split_qkv_use_gemv();
        bool qkv_try_lt = _split_qkv_try_cublaslt();
        if (!qkv_use_gemv && qkv_try_lt) {
            qkv_done = _split_lt_matmul(
                cublaslt_handle,
                qkv_lt_plan,
                QKV_ROWS,
                HIDDEN_SIZE,
                qkv_weight,
                (const __nv_bfloat16*)normalized_bf16,
                (__nv_bfloat16*)qkv_proj_bf16,
                cublaslt_workspace,
                cublaslt_workspace_bytes,
                stream
            );
            qkv_lt_done = qkv_done;
        }
        if (!qkv_done) {
            if (qkv_use_gemv) {
                qkv_gemv_done = _split_try_gemv_matvec(
                    cublas_handle,
                    qkv_weight,
                    QKV_ROWS,
                    HIDDEN_SIZE,
                    (const __nv_bfloat16*)normalized_bf16,
                    (__nv_bfloat16*)qkv_proj_bf16,
                    "qkv_gemv"
                );
                qkv_done = qkv_gemv_done;
            }
            if (!qkv_done) {
                qkv_gemmex_done = true;
                if (!_cublas_check(cublasGemmEx(
                    cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    QKV_ROWS, 1, HIDDEN_SIZE,
                    &alpha,
                    qkv_weight, CUDA_R_16BF, HIDDEN_SIZE,
                    normalized_bf16, CUDA_R_16BF, HIDDEN_SIZE,
                    &beta,
                    qkv_proj_bf16, CUDA_R_16BF, QKV_ROWS,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ), "qkv_gemm")) return;
            }
        }
        if (debug_stage_avg) {
            auto& agg = _split_debug_agg();
            if (!qkv_use_gemv && qkv_try_lt) {
                if (qkv_lt_done) agg.qkv_lt_ok += 1;
                else agg.qkv_lt_fb += 1;
            }
            if (qkv_gemv_done) agg.qkv_gemv += 1;
            else if (qkv_gemmex_done) agg.qkv_gemmex += 1;
        }
        dbg_end(dbg_qkv);

        __nv_bfloat16* q_proj_bf16 = (__nv_bfloat16*)qkv_proj_bf16;
        __nv_bfloat16* k_proj_bf16 = q_proj_bf16 + Q_SIZE;
        __nv_bfloat16* v_proj_bf16 = k_proj_bf16 + KV_SIZE;

        // 3) QK norm + RoPE + KV cache (position dynamic via position_ptr)
        int dbg_qk = dbg_begin(SPLIT_STAGE_QK_ROPE_CACHE, layer);
        int heads = (NUM_Q_HEADS > NUM_KV_HEADS) ? NUM_Q_HEADS : NUM_KV_HEADS;
        decode_qk_norm_rope_cache_kernel<<<heads, 128, 0, stream>>>(
            (__nv_bfloat16*)q_proj_bf16,
            (__nv_bfloat16*)k_proj_bf16,
            (const __nv_bfloat16*)v_proj_bf16,
            (const __nv_bfloat16*)w.q_norm_weight,
            (const __nv_bfloat16*)w.k_norm_weight,
            (const __nv_bfloat16*)cos_table,
            (const __nv_bfloat16*)sin_table,
            layer_k_cache,
            layer_v_cache,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            max_seq_len,
            position_ptr
        );
        dbg_end(dbg_qk);

        // 4) Decode attention over cache (BF16 in/out)
        int dbg_attn = dbg_begin(SPLIT_STAGE_ATTN, layer);
        int attn_warps = _split_attn_warps_per_head();
        int attn_impl = _split_attn_impl();
        if (attn_impl == 2 && split_attn_chunk_size > 0 && split_attn_max_chunks > 0 &&
            split_attn_partial_m != nullptr && split_attn_partial_s != nullptr && split_attn_partial_out != nullptr) {
            int blocks = NUM_Q_HEADS * split_attn_max_chunks;
            int threads = attn_warps * WARP_SIZE;
            decode_attention_cache_splitk_phase1_kernel<<<blocks, threads, 0, stream>>>(
                (const __nv_bfloat16*)q_proj_bf16,
                (const __nv_bfloat16*)layer_k_cache,
                (const __nv_bfloat16*)layer_v_cache,
                (float*)split_attn_partial_m,
                (float*)split_attn_partial_s,
                (float*)split_attn_partial_out,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                max_seq_len,
                attn_scale,
                attn_warps,
                split_attn_chunk_size,
                split_attn_max_chunks,
                position_ptr
            );
            decode_attention_cache_splitk_phase2_kernel<<<NUM_Q_HEADS, 128, 0, stream>>>(
                (const float*)split_attn_partial_m,
                (const float*)split_attn_partial_s,
                (const float*)split_attn_partial_out,
                (__nv_bfloat16*)attn_out_bf16,
                NUM_Q_HEADS,
                HEAD_DIM,
                split_attn_chunk_size,
                split_attn_max_chunks,
                position_ptr
            );
        } else if (attn_impl == 1) {
            int threads = attn_warps * WARP_SIZE;
            decode_attention_cache_splitk_kernel<<<NUM_Q_HEADS, threads, 0, stream>>>(
                (const __nv_bfloat16*)q_proj_bf16,
                (const __nv_bfloat16*)layer_k_cache,
                (const __nv_bfloat16*)layer_v_cache,
                (__nv_bfloat16*)attn_out_bf16,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                max_seq_len,
                attn_scale,
                attn_warps,
                position_ptr
            );
        } else {
            int warps_per_block = 8;
            int threads_per_block = warps_per_block * WARP_SIZE;  // 256
            int num_warps = NUM_Q_HEADS;
            int num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;
            decode_attention_cache_legacy_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
                (const __nv_bfloat16*)q_proj_bf16,
                (const __nv_bfloat16*)layer_k_cache,
                (const __nv_bfloat16*)layer_v_cache,
                (__nv_bfloat16*)attn_out_bf16,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                max_seq_len,
                attn_scale,
                position_ptr
            );
        }
        dbg_end(dbg_attn);

        // 5) O projection: [1,q_size] @ [hidden,q_size]^T -> [1,hidden]
        int dbg_o = dbg_begin(SPLIT_STAGE_O_GEMM, layer);
        bool o_done = false;
        bool o_lt_done = false;
        bool o_gemmex_done = false;
        bool o_gemv_done = false;
        bool o_use_gemv = _split_o_use_gemv();
        bool o_try_lt = _split_o_try_cublaslt();
        if (!o_use_gemv && o_try_lt) {
            o_done = _split_lt_matmul(
                cublaslt_handle,
                o_lt_plan,
                HIDDEN_SIZE,
                Q_SIZE,
                (const __nv_bfloat16*)w.o_proj_weight,
                (const __nv_bfloat16*)attn_out_bf16,
                (__nv_bfloat16*)o_proj_out_bf16,
                cublaslt_workspace,
                cublaslt_workspace_bytes,
                stream
            );
            o_lt_done = o_done;
        }
        if (!o_done) {
            if (o_use_gemv) {
                o_gemv_done = _split_try_gemv_matvec(
                    cublas_handle,
                    (const __nv_bfloat16*)w.o_proj_weight,
                    HIDDEN_SIZE,
                    Q_SIZE,
                    (const __nv_bfloat16*)attn_out_bf16,
                    (__nv_bfloat16*)o_proj_out_bf16,
                    "o_gemv"
                );
                o_done = o_gemv_done;
            }
            if (!o_done) {
                o_gemmex_done = true;
                if (!_cublas_check(cublasGemmEx(
                    cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    HIDDEN_SIZE, 1, Q_SIZE,
                    &alpha,
                    w.o_proj_weight, CUDA_R_16BF, Q_SIZE,
                    attn_out_bf16, CUDA_R_16BF, Q_SIZE,
                    &beta,
                    o_proj_out_bf16, CUDA_R_16BF, HIDDEN_SIZE,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ), "o_gemm")) return;
            }
        }
        if (debug_stage_avg) {
            auto& agg = _split_debug_agg();
            if (!o_use_gemv && o_try_lt) {
                if (o_lt_done) agg.o_lt_ok += 1;
                else agg.o_lt_fb += 1;
            }
            if (o_gemv_done) agg.o_gemv += 1;
            else if (o_gemmex_done) agg.o_gemmex += 1;
        }
        dbg_end(dbg_o);

        // 6) Residual add -> hidden
        int dbg_resid1 = dbg_begin(SPLIT_STAGE_RESID1, layer);
        split_residual_add_bf16_to_f32_kernel<<<(HIDDEN_SIZE + 255) / 256, 256, 0, stream>>>(
            (const __nv_bfloat16*)o_proj_out_bf16,
            (const float*)residual_f32,
            (float*)hidden_f32,
            HIDDEN_SIZE
        );
        dbg_end(dbg_resid1);

        // 7) Post-attn RMSNorm (FP32 chain, BF16 output for GEMM)
        int dbg_rms2 = dbg_begin(SPLIT_STAGE_RMS2, layer);
        split_rmsnorm_f32_to_bf16_kernel<<<1, 256, 0, stream>>>(
            (const float*)hidden_f32,
            (const __nv_bfloat16*)w.post_attn_layernorm_weight,
            (__nv_bfloat16*)mlp_norm_bf16,
            (float*)residual_f32
        );
        dbg_end(dbg_rms2);

        // 8) GateUp projection (merged): [2*I, H] x [H, 1] -> [2*I, 1]
        int dbg_gateup = dbg_begin(SPLIT_STAGE_GATEUP_GEMM, layer);
        bool gateup_done = false;
        bool gateup_lt_done = false;
        bool gateup_gemmex_done = false;
        bool gateup_gemv_done = false;
        bool gateup_use_gemv = _split_gateup_use_gemv();
        bool gateup_try_lt = _split_gateup_try_cublaslt();
        if (ffn_w4) {
            split_w4_matvec_bf16_out_kernel<<<GATEUP_ROWS, 256, 0, stream>>>(
                gateup_w4_packed,
                gateup_w4_scales,
                gateup_w4_codebook,
                (const __nv_bfloat16*)mlp_norm_bf16,
                (__nv_bfloat16*)gateup_out_bf16,
                GATEUP_ROWS,
                HIDDEN_SIZE
            );
            gateup_done = true;
            if (debug_stage_avg) {
                auto& agg = _split_debug_agg();
                agg.gateup_w4 += 1;
            }
        } else if (!gateup_use_gemv && gateup_try_lt) {
            gateup_done = _split_lt_matmul(
                cublaslt_handle,
                gateup_lt_plan,
                GATEUP_ROWS,
                HIDDEN_SIZE,
                gateup_weight,
                (const __nv_bfloat16*)mlp_norm_bf16,
                (__nv_bfloat16*)gateup_out_bf16,
                cublaslt_workspace,
                cublaslt_workspace_bytes,
                stream
            );
            gateup_lt_done = gateup_done;
        }
        if (!gateup_done) {
            if (gateup_use_gemv) {
                gateup_gemv_done = _split_try_gemv_matvec(
                    cublas_handle,
                    gateup_weight,
                    GATEUP_ROWS,
                    HIDDEN_SIZE,
                    (const __nv_bfloat16*)mlp_norm_bf16,
                    (__nv_bfloat16*)gateup_out_bf16,
                    "gateup_gemv"
                );
                gateup_done = gateup_gemv_done;
            }
            if (!gateup_done) {
                gateup_gemmex_done = true;
                if (!_cublas_check(cublasGemmEx(
                    cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    GATEUP_ROWS, 1, HIDDEN_SIZE,
                    &alpha,
                    gateup_weight, CUDA_R_16BF, HIDDEN_SIZE,
                    mlp_norm_bf16, CUDA_R_16BF, HIDDEN_SIZE,
                    &beta,
                    gateup_out_bf16, CUDA_R_16BF, GATEUP_ROWS,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ), "gateup_gemm")) return;
            }
        }
        if (debug_stage_avg) {
            auto& agg = _split_debug_agg();
            if (!gateup_use_gemv && gateup_try_lt) {
                if (gateup_lt_done) agg.gateup_lt_ok += 1;
                else agg.gateup_lt_fb += 1;
            }
            if (gateup_gemv_done) agg.gateup_gemv += 1;
            else if (gateup_gemmex_done) agg.gateup_gemmex += 1;
        }
        dbg_end(dbg_gateup);

        __nv_bfloat16* gate_out_bf16 = (__nv_bfloat16*)gateup_out_bf16;
        __nv_bfloat16* up_out_bf16 = gate_out_bf16 + INTERMEDIATE_SIZE;

        int ffn_impl = _split_ffn_impl();
        bool ffn_fused_tail = (ffn_impl != SPLIT_FFN_CUBLAS);

        // 9) SiLU(gate) * up
        int dbg_silu = dbg_begin(SPLIT_STAGE_SILU_MUL, layer);
        if (ffn_impl != SPLIT_FFN_FUSED_SILU_DOWN) {
            prefill_silu_mul_kernel<<<(INTERMEDIATE_SIZE + 255) / 256, 256, 0, stream>>>(
                (const __nv_bfloat16*)gate_out_bf16,
                (const __nv_bfloat16*)up_out_bf16,
                (__nv_bfloat16*)mlp_intermediate_bf16,
                INTERMEDIATE_SIZE
            );
        }
        dbg_end(dbg_silu);

        // 10) Down projection (or fused down+residual tail)
        int dbg_down = dbg_begin(SPLIT_STAGE_DOWN_GEMM, layer);
        bool down_done = false;
        bool down_lt_done = false;
        bool down_gemmex_done = false;
        bool down_gemv_done = false;
        bool down_try_lt = false;
        bool down_use_gemv = false;
        bool down_w4_done = false;
        if (ffn_impl == SPLIT_FFN_FUSED_SILU_DOWN) {
            split_silu_downproj_residual_fused_kernel<<<HIDDEN_SIZE, 256, 0, stream>>>(
                (const __nv_bfloat16*)gateup_out_bf16,
                down_weight,
                (const float*)residual_f32,
                (float*)hidden_f32
            );
            down_done = true;
            if (debug_stage_avg) {
                auto& agg = _split_debug_agg();
                agg.down_fused_silu += 1;
            }
        } else if (ffn_fused_tail) {
            split_downproj_residual_fused_kernel<<<HIDDEN_SIZE, 256, 0, stream>>>(
                (const __nv_bfloat16*)mlp_intermediate_bf16,
                down_weight,
                (const float*)residual_f32,
                (float*)hidden_f32
            );
            down_done = true;
            if (debug_stage_avg) {
                auto& agg = _split_debug_agg();
                agg.down_fused_tail += 1;
            }
        } else {
            if (ffn_w4) {
                split_w4_downproj_residual_kernel<<<HIDDEN_SIZE, 256, 0, stream>>>(
                    down_w4_packed,
                    down_w4_scales,
                    down_w4_codebook,
                    (const __nv_bfloat16*)mlp_intermediate_bf16,
                    (const float*)residual_f32,
                    (float*)hidden_f32
                );
                down_done = true;
                down_w4_done = true;
                if (debug_stage_avg) {
                    auto& agg = _split_debug_agg();
                    agg.down_w4 += 1;
                }
            } else {
                down_try_lt = _split_down_try_cublaslt();
                down_use_gemv = _split_down_use_gemv();
            }
            if (!down_done && !down_use_gemv && down_try_lt) {
                down_done = _split_lt_matmul(
                    cublaslt_handle,
                    down_lt_plan,
                    HIDDEN_SIZE,
                    INTERMEDIATE_SIZE,
                    down_weight,
                    (const __nv_bfloat16*)mlp_intermediate_bf16,
                    (__nv_bfloat16*)down_out_bf16,
                    cublaslt_workspace,
                    cublaslt_workspace_bytes,
                    stream
                );
                down_lt_done = down_done;
            }
            if (!down_done) {
                if (down_use_gemv) {
                    down_gemv_done = _split_try_gemv_matvec(
                        cublas_handle,
                        down_weight,
                        HIDDEN_SIZE,
                        INTERMEDIATE_SIZE,
                        (const __nv_bfloat16*)mlp_intermediate_bf16,
                        (__nv_bfloat16*)down_out_bf16,
                        "down_gemv"
                    );
                    down_done = down_gemv_done;
                }
                if (!down_done) {
                    down_gemmex_done = true;
                    if (!_cublas_check(cublasGemmEx(
                        cublas_handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        HIDDEN_SIZE, 1, INTERMEDIATE_SIZE,
                        &alpha,
                        down_weight, CUDA_R_16BF, INTERMEDIATE_SIZE,
                        mlp_intermediate_bf16, CUDA_R_16BF, INTERMEDIATE_SIZE,
                        &beta,
                        down_out_bf16, CUDA_R_16BF, HIDDEN_SIZE,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                    ), "down_gemm")) return;
                }
            }
            if (debug_stage_avg && !down_w4_done) {
                auto& agg = _split_debug_agg();
                if (!down_use_gemv && down_try_lt) {
                    if (down_lt_done) agg.down_lt_ok += 1;
                    else agg.down_lt_fb += 1;
                }
                if (down_gemv_done) agg.down_gemv += 1;
                else if (down_gemmex_done) agg.down_gemmex += 1;
            }
        }
        dbg_end(dbg_down);

        // 11) Residual add -> hidden
        int dbg_resid2 = dbg_begin(SPLIT_STAGE_RESID2, layer);
        if (!ffn_fused_tail && !down_w4_done) {
            split_residual_add_bf16_to_f32_kernel<<<(HIDDEN_SIZE + 255) / 256, 256, 0, stream>>>(
                (const __nv_bfloat16*)down_out_bf16,
                (const float*)residual_f32,
                (float*)hidden_f32,
                HIDDEN_SIZE
            );
        }
        dbg_end(dbg_resid2);
    }

    // Final RMSNorm
    int dbg_final_norm = dbg_begin(SPLIT_STAGE_FINAL_NORM);
    split_final_norm_f32_to_bf16_kernel<<<1, 256, 0, stream>>>(
        (const float*)hidden_f32,
        (const __nv_bfloat16*)final_norm_weight,
        (__nv_bfloat16*)final_hidden_bf16
    );
    dbg_end(dbg_final_norm);

    // LM head argmax (reuse prefill kernels)
    int dbg_lm1 = dbg_begin(SPLIT_STAGE_LM_PHASE1);
    prefill_lm_head_phase1<<<PREFILL_LM_NUM_BLOCKS, PREFILL_LM_BLOCK_SIZE, 0, stream>>>(
        (const __nv_bfloat16*)final_hidden_bf16,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );
    dbg_end(dbg_lm1);

    int dbg_lm2 = dbg_begin(SPLIT_STAGE_LM_PHASE2);
    prefill_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        PREFILL_LM_NUM_BLOCKS
    );
    dbg_end(dbg_lm2);

    if (debug_stage) {
        cudaStreamSynchronize(stream);
        float stage_ms[SPLIT_STAGE_COUNT] = {0.0f};
        int stage_count[SPLIT_STAGE_COUNT] = {0};
        for (int i = 0; i < debug_event_count; i++) {
            if (debug_events[i].start == nullptr || debug_events[i].stop == nullptr) continue;
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, debug_events[i].start, debug_events[i].stop);
            int st = debug_events[i].stage;
            if (st >= 0 && st < SPLIT_STAGE_COUNT) {
                stage_ms[st] += ms;
                stage_count[st] += 1;
            }
            int layer_id = debug_events[i].layer;
            if (st >= 0 && st < SPLIT_STAGE_COUNT && layer_id >= 0 && layer_id < SPLIT_DEBUG_MAX_LAYERS) {
                auto& agg = _split_debug_agg();
                agg.stage_layer_ms[st][layer_id] += double(ms);
                agg.stage_layer_calls[st][layer_id] += 1;
                long long seen = (long long)layer_id + 1;
                if (seen > agg.max_layer_seen) agg.max_layer_seen = seen;
            }
            cudaEventDestroy(debug_events[i].start);
            cudaEventDestroy(debug_events[i].stop);
        }

        float total_ms = 0.0f;
        for (int s = 0; s < SPLIT_STAGE_COUNT; s++) total_ms += stage_ms[s];

        if (debug_stage_avg && (debug_stage_avg_sample || debug_stage_detail)) {
            auto& agg = _split_debug_agg();
            agg.sampled_tokens += 1;
            agg.total_ms += double(total_ms);
            for (int s = 0; s < SPLIT_STAGE_COUNT; s++) {
                agg.stage_ms[s] += double(stage_ms[s]);
                agg.stage_calls[s] += static_cast<long long>(stage_count[s]);
            }
        }

        if (debug_stage_detail) {
            const char* impl = _split_use_cublaslt() ? "cublasLt(+fallback)" : "cublasGemmEx";
            const char* qkv_impl = _split_qkv_impl_name();
            const char* attn_impl = _split_attn_impl_name();
            float wk_mib = float(cublaslt_workspace_bytes) / (1024.0f * 1024.0f);
            printf("[MEGAQWEN_DEBUG] SPLIT stage timing (first decode token) impl=%s qkv=%s attn=%s chunk=%d chunks=%d layers=%d lt_workspace=%.1f MiB\n",
                   impl, qkv_impl, attn_impl, split_attn_chunk_size, split_attn_max_chunks, num_layers, wk_mib);
            for (int s = 0; s < SPLIT_STAGE_COUNT; s++) {
                if (stage_count[s] == 0) continue;
                float pct = (total_ms > 0.0f) ? (stage_ms[s] * 100.0f / total_ms) : 0.0f;
                float avg_ms = stage_ms[s] / float(stage_count[s]);
                printf("  %-16s total=%8.3f ms  avg=%7.4f ms  calls=%3d  (%5.1f%%)\n",
                       kSplitStageNames[s], stage_ms[s], avg_ms, stage_count[s], pct);
            }
            printf("  %-16s total=%8.3f ms\n", "decode_step_sum", total_ms);
        }
    }
}
