#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <tuple>
#include <cstdlib>
#include <cstring>
#include <cstdio>

// Must match the struct in fused_prefill.cu
struct PrefillLayerWeights {
    const void* input_layernorm_weight;
    const void* q_proj_weight;
    const void* k_proj_weight;
    const void* v_proj_weight;
    const void* q_norm_weight;
    const void* k_norm_weight;
    const void* o_proj_weight;
    const void* post_attn_layernorm_weight;
    const void* gate_proj_weight;
    const void* up_proj_weight;
    const void* down_proj_weight;
};

// Must match the struct in fused_prefill_megakernel.cu
struct PrefillMKLayerWeights {
    const void* input_layernorm_weight;
    const void* q_proj_weight;
    const void* k_proj_weight;
    const void* v_proj_weight;
    const void* q_norm_weight;
    const void* k_norm_weight;
    const void* o_proj_weight;
    const void* post_attn_layernorm_weight;
    const void* gate_proj_weight;
    const void* up_proj_weight;
    const void* down_proj_weight;
};

extern "C" void launch_prefill_float(
    const int* input_token_ids,
    int* output_token_id,
    int seq_len,
    const void* embed_weight,
    const void* audio_embeds,
    int audio_start_idx,
    int audio_len,
    const PrefillLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_float,
    void* residual,
    void* normalized,
    void* q_proj,
    void* k_proj,
    void* v_proj,
    void* attn_out,
    void* o_proj_out,
    void* mlp_norm,
    void* gate_out,
    void* up_out,
    void* mlp_intermediate,
    void* down_out,
    void* final_hidden,
    void* block_max_vals,
    void* block_max_idxs,
    void* hidden_bf16_out,
    int num_layers,
    int max_seq_len,
    float attn_scale,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
);

extern "C" void launch_prefill_megakernel(
    const int* token_ids,
    int* output_token_id,
    int seq_len,
    const void* embed_weight,
    const void* audio_embeds,
    int audio_start_idx,
    int audio_len,
    const PrefillMKLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
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
);

// For decode continuation after prefill
struct LDGLayerWeights {
    const void* input_layernorm_weight;
    const void* q_proj_weight;
    const void* k_proj_weight;
    const void* v_proj_weight;
    const void* q_norm_weight;
    const void* k_norm_weight;
    const void* o_proj_weight;
    const void* post_attn_layernorm_weight;
    const void* gate_proj_weight;
    const void* up_proj_weight;
    const void* down_proj_weight;
};

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
		);

extern "C" void launch_split_decode_gemm(
    const int* input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const PrefillLayerWeights* layer_weights_host,
    const void* const* qkv_weight_packed_ptrs,
    const void* const* gateup_weight_packed_ptrs,
    const void* const* down_weight_ptrs,
    const void* const* q_w4_packed_ptrs,
    const void* const* q_w4_scales_ptrs,
    const void* const* q_w4_codebook_ptrs,
    const void* const* k_w4_packed_ptrs,
    const void* const* k_w4_scales_ptrs,
    const void* const* k_w4_codebook_ptrs,
    const void* const* v_w4_packed_ptrs,
    const void* const* v_w4_scales_ptrs,
    const void* const* v_w4_codebook_ptrs,
    const void* const* o_w4_packed_ptrs,
    const void* const* o_w4_scales_ptrs,
    const void* const* o_w4_codebook_ptrs,
    const void* const* gateup_w4_packed_ptrs,
    const void* const* gateup_w4_scales_ptrs,
    const void* const* gateup_w4_codebook_ptrs,
    const void* const* down_w4_packed_ptrs,
    const void* const* down_w4_scales_ptrs,
    const void* const* down_w4_codebook_ptrs,
    const int* kv_block_table,
    int kv_block_size,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* split_attn_partial_m,
    void* split_attn_partial_s,
    void* split_attn_partial_out,
    int split_attn_chunk_size,
    int split_attn_max_chunks,
    void* hidden_f32,
    void* residual_f32,
    void* normalized_bf16,
    void* qkv_proj_bf16,
    void* attn_out_bf16,
    void* o_proj_out_bf16,
    void* mlp_norm_bf16,
    void* gateup_out_bf16,
    void* mlp_intermediate_bf16,
    void* down_out_bf16,
    void* final_hidden_bf16,
    void* block_max_vals,
    void* block_max_idxs,
    int num_layers,
    const int* position_ptr,
    int max_seq_len,
    float attn_scale,
    cublasHandle_t cublas_handle,
    cublasLtHandle_t cublaslt_handle,
    void* cublaslt_workspace,
    size_t cublaslt_workspace_bytes,
    cudaStream_t stream
);
extern "C" void split_debug_stage_reset();
extern "C" void split_debug_stage_print_summary();

extern "C" void launch_ldg_fill_iota(int* out, int start, int n, cudaStream_t stream);
extern "C" void launch_ldg_find_first_eos(
    const int* tokens,
    int n,
    const int* eos_ids,
    int eos_n,
    int* out_min_idx,
    cudaStream_t stream
);
extern "C" void launch_ldg_set_stage_debug(
    unsigned long long* buf,
    int num_blocks,
    int num_layers,
    int nstages,
    cudaStream_t stream
);
extern "C" void launch_ldg_disable_stage_debug(cudaStream_t stream);

class MegakernelPrefillDecoder {
public:
    MegakernelPrefillDecoder(
        torch::Tensor embed_weight,
        std::vector<torch::Tensor> layer_weights_flat,
        torch::Tensor final_norm_weight,
        torch::Tensor lm_head_weight,
        torch::Tensor cos_table,
        torch::Tensor sin_table,
        std::vector<torch::Tensor> split_q_w4_packed,
        std::vector<torch::Tensor> split_q_w4_scales,
        std::vector<torch::Tensor> split_q_w4_codebook,
        std::vector<torch::Tensor> split_k_w4_packed,
        std::vector<torch::Tensor> split_k_w4_scales,
        std::vector<torch::Tensor> split_k_w4_codebook,
        std::vector<torch::Tensor> split_v_w4_packed,
        std::vector<torch::Tensor> split_v_w4_scales,
        std::vector<torch::Tensor> split_v_w4_codebook,
        std::vector<torch::Tensor> split_o_w4_packed,
        std::vector<torch::Tensor> split_o_w4_scales,
        std::vector<torch::Tensor> split_o_w4_codebook,
        std::vector<torch::Tensor> split_gateup_w4_packed,
        std::vector<torch::Tensor> split_gateup_w4_scales,
        std::vector<torch::Tensor> split_gateup_w4_codebook,
        std::vector<torch::Tensor> split_down_w4_packed,
        std::vector<torch::Tensor> split_down_w4_scales,
        std::vector<torch::Tensor> split_down_w4_codebook,
        int num_layers,
        int max_seq_len,
        int max_prefill_len = 512
    ) : num_layers_(num_layers), max_seq_len_(max_seq_len), max_prefill_len_(max_prefill_len) {

        embed_weight_ = embed_weight;
        final_norm_weight_ = final_norm_weight;
        lm_head_weight_ = lm_head_weight;
        cos_table_ = cos_table;
        sin_table_ = sin_table;

        layer_weights_tensors_ = layer_weights_flat;

        // Build layer weights structs (same layout for both prefill and decode)
        prefill_layer_weights_.resize(num_layers);
        decode_layer_weights_.resize(num_layers);

        for (int i = 0; i < num_layers; i++) {
            prefill_layer_weights_[i].input_layernorm_weight = layer_weights_flat[i * 11 + 0].data_ptr();
            prefill_layer_weights_[i].q_proj_weight = layer_weights_flat[i * 11 + 1].data_ptr();
            prefill_layer_weights_[i].k_proj_weight = layer_weights_flat[i * 11 + 2].data_ptr();
            prefill_layer_weights_[i].v_proj_weight = layer_weights_flat[i * 11 + 3].data_ptr();
            prefill_layer_weights_[i].q_norm_weight = layer_weights_flat[i * 11 + 4].data_ptr();
            prefill_layer_weights_[i].k_norm_weight = layer_weights_flat[i * 11 + 5].data_ptr();
            prefill_layer_weights_[i].o_proj_weight = layer_weights_flat[i * 11 + 6].data_ptr();
            prefill_layer_weights_[i].post_attn_layernorm_weight = layer_weights_flat[i * 11 + 7].data_ptr();
            prefill_layer_weights_[i].gate_proj_weight = layer_weights_flat[i * 11 + 8].data_ptr();
            prefill_layer_weights_[i].up_proj_weight = layer_weights_flat[i * 11 + 9].data_ptr();
            prefill_layer_weights_[i].down_proj_weight = layer_weights_flat[i * 11 + 10].data_ptr();

            // Decode uses same layout
            decode_layer_weights_[i].input_layernorm_weight = layer_weights_flat[i * 11 + 0].data_ptr();
            decode_layer_weights_[i].q_proj_weight = layer_weights_flat[i * 11 + 1].data_ptr();
            decode_layer_weights_[i].k_proj_weight = layer_weights_flat[i * 11 + 2].data_ptr();
            decode_layer_weights_[i].v_proj_weight = layer_weights_flat[i * 11 + 3].data_ptr();
            decode_layer_weights_[i].q_norm_weight = layer_weights_flat[i * 11 + 4].data_ptr();
            decode_layer_weights_[i].k_norm_weight = layer_weights_flat[i * 11 + 5].data_ptr();
            decode_layer_weights_[i].o_proj_weight = layer_weights_flat[i * 11 + 6].data_ptr();
            decode_layer_weights_[i].post_attn_layernorm_weight = layer_weights_flat[i * 11 + 7].data_ptr();
            decode_layer_weights_[i].gate_proj_weight = layer_weights_flat[i * 11 + 8].data_ptr();
            decode_layer_weights_[i].up_proj_weight = layer_weights_flat[i * 11 + 9].data_ptr();
            decode_layer_weights_[i].down_proj_weight = layer_weights_flat[i * 11 + 10].data_ptr();
        }

        // Copy prefill layer weights to device
        d_prefill_layer_weights_ = torch::empty({num_layers * (int)sizeof(PrefillLayerWeights)},
                                                 torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_prefill_layer_weights_.data_ptr(), prefill_layer_weights_.data(),
                   num_layers * sizeof(PrefillLayerWeights), cudaMemcpyHostToDevice);

        // Copy decode layer weights to device
        d_decode_layer_weights_ = torch::empty({num_layers * (int)sizeof(LDGLayerWeights)},
                                                torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_decode_layer_weights_.data_ptr(), decode_layer_weights_.data(),
                   num_layers * sizeof(LDGLayerWeights), cudaMemcpyHostToDevice);

        // Create cuBLAS handle
        cublasCreate(&cublas_handle_);

        // Allocate KV cache
        int kv_heads = 8;
        int head_dim = 128;
        k_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        v_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));

        // Allocate prefill buffers (sized for max_prefill_len) - all BF16 for cuBLAS
        int hidden_size = 1024;
        int q_size = 16 * 128;
        int kv_size = 8 * 128;
        int intermediate_size = 3072;

        hidden_float_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        residual_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        normalized_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        q_proj_ = torch::empty({max_prefill_len, q_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        k_proj_ = torch::empty({max_prefill_len, kv_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        v_proj_ = torch::empty({max_prefill_len, kv_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        attn_out_ = torch::empty({max_prefill_len, q_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        o_proj_out_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        mlp_norm_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        gate_out_ = torch::empty({max_prefill_len, intermediate_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        up_out_ = torch::empty({max_prefill_len, intermediate_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        mlp_intermediate_ = torch::empty({max_prefill_len, intermediate_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        down_out_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        final_hidden_ = torch::empty({hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        split_qkv_out_ = torch::empty({q_size + kv_size + kv_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        split_gateup_out_ = torch::empty({intermediate_size + intermediate_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));

        // Decode buffers (single token)
        hidden_buffer_ = torch::empty({hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        g_activations_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_residual_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_q_ = torch::empty({q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_k_ = torch::empty({kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_v_ = torch::empty({kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_attn_out_ = torch::empty({q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_mlp_intermediate_ = torch::empty({intermediate_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_normalized_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        attn_partial_max_ = torch::empty({1184}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        attn_partial_sum_ = torch::empty({1184}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        attn_partial_out_ = torch::empty({1184 * 128}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

        block_max_vals_ = torch::empty({1184}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        block_max_idxs_ = torch::empty({1184}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        output_token_ = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        input_token_ = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        position_token_ = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        input_tokens_ = torch::empty({max_prefill_len}, torch::dtype(torch::kInt32).device(torch::kCUDA));

        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        decode_num_blocks_ = 0;
        lm_num_blocks_ = (int)block_max_vals_.numel();
        if (const char* s = std::getenv("MEGAQWEN_LDG_NUM_BLOCKS")) {
            int v = std::atoi(s);
            if (v > 0) decode_num_blocks_ = v;
        }
	        if (const char* s = std::getenv("MEGAQWEN_LDG_LM_NUM_BLOCKS")) {
	            int v = std::atoi(s);
	            if (v > 0 && v <= lm_num_blocks_) lm_num_blocks_ = v;
	        }
	        int max_decode_blocks = (int)block_max_vals_.numel();
	        if (decode_num_blocks_ > max_decode_blocks) decode_num_blocks_ = max_decode_blocks;

            // Prefill backend:
            // - "cublas" (default): fused_prefill.cu + cuBLAS GEMM path
            // - "megakernel": fused_prefill_megakernel.cu path (no cuBLAS)
            prefill_backend_ = 0;
            if (const char* s = std::getenv("MEGAQWEN_PREFILL_BACKEND")) {
                if (std::strcmp(s, "megakernel") == 0 || std::strcmp(s, "fused") == 0 || std::strcmp(s, "mk") == 0) {
                    prefill_backend_ = 1;
                }
            }

            if (prefill_backend_ == 1) {
                mk_hidden_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
                mk_residual_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
                mk_normalized_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
                mk_q_proj_ = torch::empty({max_prefill_len, q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
                mk_k_proj_ = torch::empty({max_prefill_len, kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
                mk_v_proj_ = torch::empty({max_prefill_len, kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
                mk_attn_out_ = torch::empty({max_prefill_len, q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
                mk_mlp_intermediate_ = torch::empty({max_prefill_len, intermediate_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
                mk_final_hidden_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
            }

            // Decode backend:
            // - "coop" (default): cooperative-group LDG megakernel
            // - "split_gemm": non-cooperative split pipeline (chunked greedy eos-check)
            // - "split_gemm_v2": non-cooperative split pipeline (chunked greedy + sparse EOS checks)
            decode_backend_ = 0;
            if (const char* s = std::getenv("MEGAQWEN_DECODE_BACKEND")) {
                if (std::strcmp(s, "split_gemm") == 0 || std::strcmp(s, "split") == 0) {
                    decode_backend_ = 1;
                } else if (std::strcmp(s, "split_gemm_v2") == 0 ||
                           std::strcmp(s, "split_v2") == 0 ||
                           std::strcmp(s, "v2") == 0) {
                    decode_backend_ = 2;
                }
            }

            split_kv_paged_ = false;
            if (const char* s = std::getenv("MEGAQWEN_SPLIT_KV_LAYOUT")) {
                if (std::strcmp(s, "paged") == 0 || std::strcmp(s, "page") == 0) {
                    split_kv_paged_ = true;
                }
            }
            if (const char* s = std::getenv("MEGAQWEN_SPLIT_KV_PAGED")) {
                if (std::strcmp(s, "") != 0 &&
                    std::strcmp(s, "0") != 0 &&
                    std::strcmp(s, "false") != 0 &&
                    std::strcmp(s, "False") != 0) {
                    split_kv_paged_ = true;
                }
            }
            split_kv_block_size_ = 16;
            if (const char* s = std::getenv("MEGAQWEN_SPLIT_KV_BLOCK_SIZE")) {
                int v = std::atoi(s);
                if (v > 0) split_kv_block_size_ = v;
            }

            // Build packed weights for split backend:
            // QKV:   [4096, 1024] = cat([Q, K, V], dim=0)
            // GateUp:[6144, 1024] = cat([gate, up], dim=0)
            if (decode_backend_ == 1 || decode_backend_ == 2) {
                cublasLtCreate(&cublaslt_handle_);
                size_t lt_workspace_bytes = 8ull * 1024ull * 1024ull;  // default 8 MiB
                if (const char* s = std::getenv("MEGAQWEN_SPLIT_LTWK_MB")) {
                    int mb = std::atoi(s);
                    if (mb > 0) lt_workspace_bytes = (size_t)mb * 1024ull * 1024ull;
                }
                split_cublaslt_workspace_bytes_ = lt_workspace_bytes;
                split_cublaslt_workspace_ = torch::empty(
                    {(long long)lt_workspace_bytes},
                    torch::dtype(torch::kUInt8).device(torch::kCUDA)
                );

                split_attn_chunk_size_ = 64;
                if (const char* s = std::getenv("MEGAQWEN_SPLIT_ATTN_CHUNK")) {
                    int v = std::atoi(s);
                    if (v > 0) split_attn_chunk_size_ = v;
                }
                if (split_attn_chunk_size_ < 16) split_attn_chunk_size_ = 16;
                if (split_attn_chunk_size_ > max_seq_len_) split_attn_chunk_size_ = max_seq_len_;
                split_attn_max_chunks_ = (max_seq_len_ + split_attn_chunk_size_ - 1) / split_attn_chunk_size_;
                if (split_attn_max_chunks_ < 1) split_attn_max_chunks_ = 1;
                split_attn_partial_m_ = torch::empty(
                    {split_attn_max_chunks_, 16},
                    torch::dtype(torch::kFloat32).device(torch::kCUDA)
                );
                split_attn_partial_s_ = torch::empty(
                    {split_attn_max_chunks_, 16},
                    torch::dtype(torch::kFloat32).device(torch::kCUDA)
                );
                split_attn_partial_out_ = torch::empty(
                    {split_attn_max_chunks_, 16, 128},
                    torch::dtype(torch::kFloat32).device(torch::kCUDA)
                );
                if (split_kv_paged_) {
                    split_kv_num_blocks_ = (max_seq_len_ + split_kv_block_size_ - 1) / split_kv_block_size_;
                    if (split_kv_num_blocks_ < 1) split_kv_num_blocks_ = 1;
                    split_kv_block_table_ = torch::empty(
                        {split_kv_num_blocks_},
                        torch::dtype(torch::kInt32).device(torch::kCUDA)
                    );
                    launch_ldg_fill_iota(
                        (int*)split_kv_block_table_.data_ptr<int>(),
                        0,
                        split_kv_num_blocks_,
                        c10::cuda::getCurrentCUDAStream().stream()
                    );
                    std::printf("[MEGAQWEN_SPLIT] kv layout=paged block=%d blocks=%d\n",
                                split_kv_block_size_, split_kv_num_blocks_);
                } else {
                    split_kv_block_table_ = torch::Tensor();
                    split_kv_num_blocks_ = 0;
                }

                split_qkv_weight_tensors_.reserve(num_layers_);
                split_gateup_weight_tensors_.reserve(num_layers_);
                split_down_weight_tensors_.reserve(num_layers_);
                split_qkv_weight_ptrs_.reserve(num_layers_);
                split_gateup_weight_ptrs_.reserve(num_layers_);
                split_down_weight_ptrs_.reserve(num_layers_);
                for (int i = 0; i < num_layers_; i++) {
                    auto wq = layer_weights_flat[i * 11 + 1];
                    auto wk = layer_weights_flat[i * 11 + 2];
                    auto wv = layer_weights_flat[i * 11 + 3];
                    auto wgate = layer_weights_flat[i * 11 + 8];
                    auto wup = layer_weights_flat[i * 11 + 9];
                    auto wdown = layer_weights_flat[i * 11 + 10];

                    auto qkv_packed = torch::cat({wq, wk, wv}, 0).contiguous();
                    auto gateup_packed = torch::cat({wgate, wup}, 0).contiguous();
                    auto down_packed = wdown.contiguous();

                    split_qkv_weight_tensors_.push_back(qkv_packed);
                    split_gateup_weight_tensors_.push_back(gateup_packed);
                    split_down_weight_tensors_.push_back(down_packed);
                    split_qkv_weight_ptrs_.push_back(split_qkv_weight_tensors_.back().data_ptr());
                    split_gateup_weight_ptrs_.push_back(split_gateup_weight_tensors_.back().data_ptr());
                    split_down_weight_ptrs_.push_back(split_down_weight_tensors_.back().data_ptr());
                }

                if ((int)split_q_w4_packed.size() == num_layers_ &&
                    (int)split_q_w4_scales.size() == num_layers_ &&
                    (int)split_q_w4_codebook.size() == num_layers_ &&
                    (int)split_k_w4_packed.size() == num_layers_ &&
                    (int)split_k_w4_scales.size() == num_layers_ &&
                    (int)split_k_w4_codebook.size() == num_layers_ &&
                    (int)split_v_w4_packed.size() == num_layers_ &&
                    (int)split_v_w4_scales.size() == num_layers_ &&
                    (int)split_v_w4_codebook.size() == num_layers_ &&
                    (int)split_o_w4_packed.size() == num_layers_ &&
                    (int)split_o_w4_scales.size() == num_layers_ &&
                    (int)split_o_w4_codebook.size() == num_layers_) {
                    split_q_w4_packed_tensors_ = std::move(split_q_w4_packed);
                    split_q_w4_scales_tensors_ = std::move(split_q_w4_scales);
                    split_q_w4_codebook_tensors_ = std::move(split_q_w4_codebook);
                    split_k_w4_packed_tensors_ = std::move(split_k_w4_packed);
                    split_k_w4_scales_tensors_ = std::move(split_k_w4_scales);
                    split_k_w4_codebook_tensors_ = std::move(split_k_w4_codebook);
                    split_v_w4_packed_tensors_ = std::move(split_v_w4_packed);
                    split_v_w4_scales_tensors_ = std::move(split_v_w4_scales);
                    split_v_w4_codebook_tensors_ = std::move(split_v_w4_codebook);
                    split_o_w4_packed_tensors_ = std::move(split_o_w4_packed);
                    split_o_w4_scales_tensors_ = std::move(split_o_w4_scales);
                    split_o_w4_codebook_tensors_ = std::move(split_o_w4_codebook);
                    split_q_w4_packed_ptrs_.reserve(num_layers_);
                    split_q_w4_scales_ptrs_.reserve(num_layers_);
                    split_q_w4_codebook_ptrs_.reserve(num_layers_);
                    split_k_w4_packed_ptrs_.reserve(num_layers_);
                    split_k_w4_scales_ptrs_.reserve(num_layers_);
                    split_k_w4_codebook_ptrs_.reserve(num_layers_);
                    split_v_w4_packed_ptrs_.reserve(num_layers_);
                    split_v_w4_scales_ptrs_.reserve(num_layers_);
                    split_v_w4_codebook_ptrs_.reserve(num_layers_);
                    split_o_w4_packed_ptrs_.reserve(num_layers_);
                    split_o_w4_scales_ptrs_.reserve(num_layers_);
                    split_o_w4_codebook_ptrs_.reserve(num_layers_);
                    for (int i = 0; i < num_layers_; i++) {
                        split_q_w4_packed_ptrs_.push_back(split_q_w4_packed_tensors_[i].data_ptr());
                        split_q_w4_scales_ptrs_.push_back(split_q_w4_scales_tensors_[i].data_ptr());
                        split_q_w4_codebook_ptrs_.push_back(split_q_w4_codebook_tensors_[i].data_ptr());
                        split_k_w4_packed_ptrs_.push_back(split_k_w4_packed_tensors_[i].data_ptr());
                        split_k_w4_scales_ptrs_.push_back(split_k_w4_scales_tensors_[i].data_ptr());
                        split_k_w4_codebook_ptrs_.push_back(split_k_w4_codebook_tensors_[i].data_ptr());
                        split_v_w4_packed_ptrs_.push_back(split_v_w4_packed_tensors_[i].data_ptr());
                        split_v_w4_scales_ptrs_.push_back(split_v_w4_scales_tensors_[i].data_ptr());
                        split_v_w4_codebook_ptrs_.push_back(split_v_w4_codebook_tensors_[i].data_ptr());
                        split_o_w4_packed_ptrs_.push_back(split_o_w4_packed_tensors_[i].data_ptr());
                        split_o_w4_scales_ptrs_.push_back(split_o_w4_scales_tensors_[i].data_ptr());
                        split_o_w4_codebook_ptrs_.push_back(split_o_w4_codebook_tensors_[i].data_ptr());
                    }
                    std::printf("[MEGAQWEN_W4] split decode runtime QKV/O W4 enabled for %d layers\n", num_layers_);
                }

                if ((int)split_gateup_w4_packed.size() == num_layers_ &&
                    (int)split_gateup_w4_scales.size() == num_layers_ &&
                    (int)split_gateup_w4_codebook.size() == num_layers_ &&
                    (int)split_down_w4_packed.size() == num_layers_ &&
                    (int)split_down_w4_scales.size() == num_layers_ &&
                    (int)split_down_w4_codebook.size() == num_layers_) {
                    split_gateup_w4_packed_tensors_ = std::move(split_gateup_w4_packed);
                    split_gateup_w4_scales_tensors_ = std::move(split_gateup_w4_scales);
                    split_gateup_w4_codebook_tensors_ = std::move(split_gateup_w4_codebook);
                    split_down_w4_packed_tensors_ = std::move(split_down_w4_packed);
                    split_down_w4_scales_tensors_ = std::move(split_down_w4_scales);
                    split_down_w4_codebook_tensors_ = std::move(split_down_w4_codebook);
                    split_gateup_w4_packed_ptrs_.reserve(num_layers_);
                    split_gateup_w4_scales_ptrs_.reserve(num_layers_);
                    split_gateup_w4_codebook_ptrs_.reserve(num_layers_);
                    split_down_w4_packed_ptrs_.reserve(num_layers_);
                    split_down_w4_scales_ptrs_.reserve(num_layers_);
                    split_down_w4_codebook_ptrs_.reserve(num_layers_);
                    for (int i = 0; i < num_layers_; i++) {
                        split_gateup_w4_packed_ptrs_.push_back(split_gateup_w4_packed_tensors_[i].data_ptr());
                        split_gateup_w4_scales_ptrs_.push_back(split_gateup_w4_scales_tensors_[i].data_ptr());
                        split_gateup_w4_codebook_ptrs_.push_back(split_gateup_w4_codebook_tensors_[i].data_ptr());
                        split_down_w4_packed_ptrs_.push_back(split_down_w4_packed_tensors_[i].data_ptr());
                        split_down_w4_scales_ptrs_.push_back(split_down_w4_scales_tensors_[i].data_ptr());
                        split_down_w4_codebook_ptrs_.push_back(split_down_w4_codebook_tensors_[i].data_ptr());
                    }
                    std::printf("[MEGAQWEN_W4] split decode runtime FFN W4 enabled for %d layers\n", num_layers_);
                }
            }

	        position_ = 0;
	        attn_scale_ = 1.0f / sqrtf(128.0f);
	    }

    ~MegakernelPrefillDecoder() {
        // Best-effort cleanup; errors are ignored in destructor.
        if (decode_graph_exec_ != nullptr) {
            cudaGraphExecDestroy(decode_graph_exec_);
            decode_graph_exec_ = nullptr;
        }
        if (decode_graph_ != nullptr) {
            cudaGraphDestroy(decode_graph_);
            decode_graph_ = nullptr;
        }
        if (cublas_handle_ != nullptr) {
            cublasDestroy(cublas_handle_);
            cublas_handle_ = nullptr;
        }
        if (cublaslt_handle_ != nullptr) {
            cublasLtDestroy(cublaslt_handle_);
            cublaslt_handle_ = nullptr;
        }
    }

    int prefill_step(torch::Tensor input_token_ids) {
        // input_token_ids: 1D tensor of token IDs [seq_len]
        int seq_len = input_token_ids.size(0);
        if (seq_len > max_prefill_len_) {
            throw std::runtime_error("Prefill sequence length exceeds maximum");
        }
        if (seq_len == 0) {
            throw std::runtime_error("Empty input sequence");
        }

        // Copy input tokens to device
        input_tokens_.narrow(0, 0, seq_len).copy_(input_token_ids.to(torch::kCUDA));

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        constexpr int kPrefillMKMaxSeqLen = 64;
        bool use_mk_prefill = (prefill_backend_ == 1 && seq_len <= kPrefillMKMaxSeqLen);
        if (prefill_backend_ == 1 && !use_mk_prefill) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::printf("[MEGAQWEN_PREFILL] fallback to cublas prefill (seq_len=%d > %d); megakernel prefill is short-seq only\\n",
                            seq_len, kPrefillMKMaxSeqLen);
            }
        }

        if (use_mk_prefill) {
            launch_prefill_megakernel(
                (const int*)input_tokens_.data_ptr(),
                (int*)output_token_.data_ptr(),
                seq_len,
                embed_weight_.data_ptr(),
                nullptr,
                0,
                0,
                (const PrefillMKLayerWeights*)d_prefill_layer_weights_.data_ptr(),
                final_norm_weight_.data_ptr(),
                lm_head_weight_.data_ptr(),
                cos_table_.data_ptr(),
                sin_table_.data_ptr(),
                k_cache_.data_ptr(),
                v_cache_.data_ptr(),
                mk_hidden_.data_ptr(),
                mk_residual_.data_ptr(),
                mk_normalized_.data_ptr(),
                mk_q_proj_.data_ptr(),
                mk_k_proj_.data_ptr(),
                mk_v_proj_.data_ptr(),
                mk_attn_out_.data_ptr(),
                mk_mlp_intermediate_.data_ptr(),
                mk_final_hidden_.data_ptr(),
                hidden_buffer_.data_ptr(),
                block_max_vals_.data_ptr(),
                block_max_idxs_.data_ptr(),
                num_layers_,
                max_seq_len_,
                attn_scale_,
                stream
            );
        } else {
            launch_prefill_float(
                (const int*)input_tokens_.data_ptr(),
                (int*)output_token_.data_ptr(),
                seq_len,
                embed_weight_.data_ptr(),
                nullptr,
                0,
                0,
                prefill_layer_weights_.data(),  // Use HOST pointer, not device
                final_norm_weight_.data_ptr(),
                lm_head_weight_.data_ptr(),
                cos_table_.data_ptr(),
                sin_table_.data_ptr(),
                k_cache_.data_ptr(),
                v_cache_.data_ptr(),
                hidden_float_.data_ptr(),
                residual_.data_ptr(),
                normalized_.data_ptr(),
                q_proj_.data_ptr(),
                k_proj_.data_ptr(),
                v_proj_.data_ptr(),
                attn_out_.data_ptr(),
                o_proj_out_.data_ptr(),
                mlp_norm_.data_ptr(),
                gate_out_.data_ptr(),
                up_out_.data_ptr(),
                mlp_intermediate_.data_ptr(),
                down_out_.data_ptr(),
                final_hidden_.data_ptr(),
                block_max_vals_.data_ptr(),
                block_max_idxs_.data_ptr(),
                hidden_buffer_.data_ptr(),  // bf16 output for decode continuation
                num_layers_,
                max_seq_len_,
                attn_scale_,
                cublas_handle_,
                stream
            );
        }

        position_ = seq_len;
        return output_token_.item<int>();
    }

    int prefill_asr(torch::Tensor input_token_ids, torch::Tensor audio_embeds, int audio_start_idx) {
        // input_token_ids: 1D tensor of token IDs [seq_len]
        // audio_embeds: [audio_len, hidden_size] BF16 on CUDA
        int seq_len = input_token_ids.size(0);
        if (seq_len > max_prefill_len_) {
            throw std::runtime_error("Prefill sequence length exceeds maximum");
        }
        if (seq_len == 0) {
            throw std::runtime_error("Empty input sequence");
        }
        if (!audio_embeds.is_cuda() || audio_embeds.dtype() != torch::kBFloat16 || audio_embeds.dim() != 2) {
            throw std::runtime_error("audio_embeds must be a CUDA bfloat16 2D tensor [audio_len, hidden_size]");
        }
        int audio_len = audio_embeds.size(0);
        int hidden_size = audio_embeds.size(1);
        if (hidden_size != 1024) {
            throw std::runtime_error("audio_embeds hidden_size must be 1024 for Qwen3-0.6B");
        }
        if (audio_len <= 0) {
            throw std::runtime_error("audio_embeds must have audio_len > 0");
        }
        if (audio_start_idx < 0 || audio_start_idx + audio_len > seq_len) {
            throw std::runtime_error("Invalid audio_start_idx/audio_len for the given seq_len");
        }

        // Copy input tokens to device
        input_tokens_.narrow(0, 0, seq_len).copy_(input_token_ids.to(torch::kCUDA));

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        constexpr int kPrefillMKMaxSeqLen = 64;
        bool use_mk_prefill = (prefill_backend_ == 1 && seq_len <= kPrefillMKMaxSeqLen);
        if (prefill_backend_ == 1 && !use_mk_prefill) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::printf("[MEGAQWEN_PREFILL] fallback to cublas prefill (seq_len=%d > %d); megakernel prefill is short-seq only\\n",
                            seq_len, kPrefillMKMaxSeqLen);
            }
        }

        if (use_mk_prefill) {
            launch_prefill_megakernel(
                (const int*)input_tokens_.data_ptr(),
                (int*)output_token_.data_ptr(),
                seq_len,
                embed_weight_.data_ptr(),
                audio_embeds.data_ptr(),
                audio_start_idx,
                audio_len,
                (const PrefillMKLayerWeights*)d_prefill_layer_weights_.data_ptr(),
                final_norm_weight_.data_ptr(),
                lm_head_weight_.data_ptr(),
                cos_table_.data_ptr(),
                sin_table_.data_ptr(),
                k_cache_.data_ptr(),
                v_cache_.data_ptr(),
                mk_hidden_.data_ptr(),
                mk_residual_.data_ptr(),
                mk_normalized_.data_ptr(),
                mk_q_proj_.data_ptr(),
                mk_k_proj_.data_ptr(),
                mk_v_proj_.data_ptr(),
                mk_attn_out_.data_ptr(),
                mk_mlp_intermediate_.data_ptr(),
                mk_final_hidden_.data_ptr(),
                hidden_buffer_.data_ptr(),
                block_max_vals_.data_ptr(),
                block_max_idxs_.data_ptr(),
                num_layers_,
                max_seq_len_,
                attn_scale_,
                stream
            );
        } else {
            launch_prefill_float(
                (const int*)input_tokens_.data_ptr(),
                (int*)output_token_.data_ptr(),
                seq_len,
                embed_weight_.data_ptr(),
                audio_embeds.data_ptr(),
                audio_start_idx,
                audio_len,
                prefill_layer_weights_.data(),  // Use HOST pointer, not device
                final_norm_weight_.data_ptr(),
                lm_head_weight_.data_ptr(),
                cos_table_.data_ptr(),
                sin_table_.data_ptr(),
                k_cache_.data_ptr(),
                v_cache_.data_ptr(),
                hidden_float_.data_ptr(),
                residual_.data_ptr(),
                normalized_.data_ptr(),
                q_proj_.data_ptr(),
                k_proj_.data_ptr(),
                v_proj_.data_ptr(),
                attn_out_.data_ptr(),
                o_proj_out_.data_ptr(),
                mlp_norm_.data_ptr(),
                gate_out_.data_ptr(),
                up_out_.data_ptr(),
                mlp_intermediate_.data_ptr(),
                down_out_.data_ptr(),
                final_hidden_.data_ptr(),
                block_max_vals_.data_ptr(),
                block_max_idxs_.data_ptr(),
                hidden_buffer_.data_ptr(),  // bf16 output for decode continuation
                num_layers_,
                max_seq_len_,
                attn_scale_,
                cublas_handle_,
                stream
            );
        }

        position_ = seq_len;
        return output_token_.item<int>();
    }

    int decode_step(int input_token_id) {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        cudaMemcpyAsync(input_token_.data_ptr(), &input_token_id, sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(position_token_.data_ptr(), &position_, sizeof(int), cudaMemcpyHostToDevice, stream);

        launch_decode_step_backend(
            (const int*)input_token_.data_ptr(),
            (int*)output_token_.data_ptr(),
            (const int*)position_token_.data_ptr(),
            stream
        );

        position_++;
        return output_token_.item<int>();
    }

    // Debug helper: run one decode step with stage timing enabled.
    // Returns a CUDA int64 tensor of shape [decode_num_blocks, num_layers, 16]
    // with per-block cycle counts for each stage.
    torch::Tensor debug_ldg_decode_cycles(int input_token_id) {
        constexpr int kStages = 16;
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        if (decode_backend_ != 0) {
            throw std::runtime_error("debug_ldg_decode_cycles is only supported on MEGAQWEN_DECODE_BACKEND=coop");
        }

        // We don't know the final cooperative grid size if decode_num_blocks_==0 (auto).
        // Allocate at the maximum scratch capacity and allow the kernel to write into
        // the prefix [0:gridDim.x). Unused tail blocks remain zero.
        int cap_blocks = lm_num_blocks_;
        if (cap_blocks <= 0) cap_blocks = 1184;

        auto cycles = torch::zeros({cap_blocks, num_layers_, kStages}, torch::dtype(torch::kInt64).device(torch::kCUDA));
        launch_ldg_set_stage_debug(
            (unsigned long long*)cycles.data_ptr<int64_t>(),
            cap_blocks,
            num_layers_,
            kStages,
            stream
        );

        cudaMemcpyAsync(input_token_.data_ptr(), &input_token_id, sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(position_token_.data_ptr(), &position_, sizeof(int), cudaMemcpyHostToDevice, stream);

        launch_ldg_decode(
            (const int*)input_token_.data_ptr(),
            (int*)output_token_.data_ptr(),
            embed_weight_.data_ptr(),
            (const LDGLayerWeights*)d_decode_layer_weights_.data_ptr(),
            final_norm_weight_.data_ptr(),
            lm_head_weight_.data_ptr(),
            cos_table_.data_ptr(),
            sin_table_.data_ptr(),
            k_cache_.data_ptr(),
            v_cache_.data_ptr(),
            hidden_buffer_.data_ptr(),
            g_activations_.data_ptr(),
            g_residual_.data_ptr(),
            g_q_.data_ptr(),
            g_k_.data_ptr(),
            g_v_.data_ptr(),
            g_attn_out_.data_ptr(),
            g_mlp_intermediate_.data_ptr(),
            g_normalized_.data_ptr(),
            attn_partial_max_.data_ptr(),
            attn_partial_sum_.data_ptr(),
            attn_partial_out_.data_ptr(),
            block_max_vals_.data_ptr(),
            block_max_idxs_.data_ptr(),
            num_layers_,
            (const int*)position_token_.data_ptr(),
            max_seq_len_,
            attn_scale_,
            decode_num_blocks_,
            lm_num_blocks_,
            stream
        );

        launch_ldg_disable_stage_debug(stream);

        position_++;
        return cycles;
    }

    torch::Tensor decode_steps_cuda(int first_token_id, int num_steps) {
        if (num_steps <= 0) {
            return torch::empty({0}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        }
        if (position_ + num_steps >= max_seq_len_) {
            throw std::runtime_error("decode_steps would exceed max_seq_len");
        }

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        auto tokens = torch::empty({num_steps + 1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        auto positions = torch::empty({num_steps}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        int* tokens_ptr = (int*)tokens.data_ptr();
        int* pos_ptr = (int*)positions.data_ptr();
        cudaMemcpyAsync(tokens_ptr, &first_token_id, sizeof(int), cudaMemcpyHostToDevice, stream);
        launch_ldg_fill_iota(pos_ptr, position_, num_steps, stream);

        for (int i = 0; i < num_steps; i++) {
            const int* in_ptr = (const int*)(tokens_ptr + i);
            int* out_ptr = (int*)(tokens_ptr + i + 1);
            const int* p_ptr = (const int*)(pos_ptr + i);
            launch_decode_step_backend(in_ptr, out_ptr, p_ptr, stream);
        }

        position_ += num_steps;
        return tokens.slice(0, 1, num_steps + 1);
    }

    // CUDA Graph version of decode_steps_cuda to reduce per-token launch overhead.
    // Captures a fixed-length decode chain: tokens[i] -> tokens[i+1], positions[i] -> one step.
    // Notes:
    // - Graph is rebuilt if num_steps changes.
    // - Input token id and positions are filled *outside* the captured graph.
    torch::Tensor decode_steps_cuda_graph(int first_token_id, int num_steps) {
        if (num_steps <= 0) {
            return torch::empty({0}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        }
        // split_gemm backend currently does not use CUDA Graph capture.
        // cuBLAS host-scalar GEMM in graph mode is not stable yet.
        if (decode_backend_ != 0) {
            return decode_steps_cuda(first_token_id, num_steps);
        }
        if (position_ + num_steps >= max_seq_len_) {
            throw std::runtime_error("decode_steps_cuda_graph would exceed max_seq_len");
        }

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        // (Re)build graph if needed.
        if (decode_graph_exec_ == nullptr || decode_graph_steps_ != num_steps) {
            if (decode_graph_exec_ != nullptr) {
                cudaGraphExecDestroy(decode_graph_exec_);
                decode_graph_exec_ = nullptr;
            }
            if (decode_graph_ != nullptr) {
                cudaGraphDestroy(decode_graph_);
                decode_graph_ = nullptr;
            }

            decode_graph_steps_ = num_steps;
            graph_tokens_ = torch::empty({num_steps + 1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
            graph_positions_ = torch::empty({num_steps}, torch::dtype(torch::kInt32).device(torch::kCUDA));

            int* tokens_ptr = (int*)graph_tokens_.data_ptr();
            int* pos_ptr = (int*)graph_positions_.data_ptr();

            cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
            for (int i = 0; i < num_steps; i++) {
                const int* in_ptr = (const int*)(tokens_ptr + i);
                int* out_ptr = (int*)(tokens_ptr + i + 1);
                const int* p_ptr = (const int*)(pos_ptr + i);
                launch_decode_step_backend(in_ptr, out_ptr, p_ptr, stream);
            }
            cudaError_t end_err = cudaStreamEndCapture(stream, &decode_graph_);
            if (end_err != cudaSuccess || decode_graph_ == nullptr) {
                decode_graph_steps_ = 0;
                decode_graph_ = nullptr;
                decode_graph_exec_ = nullptr;
                // Fall back to the non-graph path.
                return decode_steps_cuda(first_token_id, num_steps);
            }

            cudaError_t inst_err = cudaGraphInstantiate(&decode_graph_exec_, decode_graph_, nullptr, nullptr, 0);
            if (inst_err != cudaSuccess || decode_graph_exec_ == nullptr) {
                cudaGraphDestroy(decode_graph_);
                decode_graph_ = nullptr;
                decode_graph_steps_ = 0;
                decode_graph_exec_ = nullptr;
                return decode_steps_cuda(first_token_id, num_steps);
            }
        }

        int* tokens_ptr = (int*)graph_tokens_.data_ptr();
        int* pos_ptr = (int*)graph_positions_.data_ptr();
        cudaMemcpyAsync(tokens_ptr, &first_token_id, sizeof(int), cudaMemcpyHostToDevice, stream);
        launch_ldg_fill_iota(pos_ptr, position_, num_steps, stream);

        cudaGraphLaunch(decode_graph_exec_, stream);

        position_ += num_steps;
        return graph_tokens_.slice(0, 1, num_steps + 1);
    }

    // Greedy decode loop fully inside the extension:
    // - keeps "next token" on GPU (no `.item()` per chunk)
    // - EOS scan is GPU-side; per chunk only copies 1 int back to host
    // - uses chunked decode (wastes <chunk tokens after EOS, same as the Python chunk loop)
    //
    // Returns: (generated_tokens_cuda, generated_len, stop_reason)
    //   stop_reason: 0=max_new_tokens, 1=eos
    std::tuple<torch::Tensor, int, int> decode_greedy_cuda(
        int first_token_id,
        int max_new_tokens,
        torch::Tensor eos_ids,
        int chunk
    ) {
        int max_new = max_new_tokens;
        if (max_new <= 0) {
            auto empty = torch::empty({0}, torch::dtype(torch::kInt32).device(torch::kCUDA));
            return std::make_tuple(empty, 0, 0);
        }
        if (chunk <= 0) chunk = 32;

        if (!eos_ids.defined() || eos_ids.numel() <= 0) {
            throw std::runtime_error("eos_ids must be a non-empty int32 tensor");
        }
        if (!eos_ids.is_cuda()) {
            eos_ids = eos_ids.to(torch::kCUDA);
        }
        eos_ids = eos_ids.to(torch::kInt32).contiguous();

        int eos_n = (int)eos_ids.numel();
        const int* eos_ptr = (const int*)eos_ids.data_ptr();

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        // Workspace (reused across calls).
        if (!greedy_tokens_.defined() || greedy_tokens_.numel() < (chunk + 1)) {
            greedy_tokens_ = torch::empty({chunk + 1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        }
        if (!greedy_positions_.defined() || greedy_positions_.numel() < chunk) {
            greedy_positions_ = torch::empty({chunk}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        }
        if (!greedy_eos_idx_.defined() || greedy_eos_idx_.numel() != 1) {
            greedy_eos_idx_ = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        }
        if (!greedy_next_token_.defined() || greedy_next_token_.numel() != 1) {
            greedy_next_token_ = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        }

        // Initialize next token (H2D once).
        cudaMemcpyAsync(greedy_next_token_.data_ptr(), &first_token_id, sizeof(int), cudaMemcpyHostToDevice, stream);

        auto out = torch::empty({max_new}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        int* out_ptr = (int*)out.data_ptr();

        int gen_len = 0;
        int stop_reason = 0;

        int* tokens_ptr = (int*)greedy_tokens_.data_ptr();
        int* pos_ptr = (int*)greedy_positions_.data_ptr();
        int* eos_idx_ptr = (int*)greedy_eos_idx_.data_ptr();
        int* next_ptr = (int*)greedy_next_token_.data_ptr();
        if (decode_backend_ != 0) {
            split_debug_stage_reset();
        }

        if (decode_backend_ == 2) {
            // Decode v2: chunked decode + sparse EOS checking.
            // Compared with backend=1:
            // - fewer host sync points (check EOS every N chunks)
            // - bounded wasted compute after EOS (< N * chunk)
            int check_stride = 4;  // default: check every 4 chunks
            if (const char* s = std::getenv("MEGAQWEN_DECODE_V2_CHECK_STRIDE")) {
                int v = std::atoi(s);
                if (v > 0) check_stride = v;
            }
            if (check_stride < 1) check_stride = 1;
            if (check_stride > 64) check_stride = 64;

            int chunks_since_check = 0;
            int checked_len = 1;  // out[0] is the initial token, never EOS for stopping.
            int wasted_tokens = 0;

            while (gen_len < max_new) {
                int steps = std::min(chunk, max_new - gen_len);

                // positions: [position_, position_+1, ...]
                launch_ldg_fill_iota(pos_ptr, position_, steps, stream);

                // tokens[0] = next_token (D2D)
                cudaMemcpyAsync(tokens_ptr, next_ptr, sizeof(int), cudaMemcpyDeviceToDevice, stream);

                // Decode chain (writes tokens[1..steps]).
                for (int i = 0; i < steps; i++) {
                    const int* in_ptr = (const int*)(tokens_ptr + i);
                    int* out_tok_ptr = (int*)(tokens_ptr + i + 1);
                    const int* p_ptr = (const int*)(pos_ptr + i);
                    launch_decode_step_backend(in_ptr, out_tok_ptr, p_ptr, stream);
                }

                // Append outputs for this chunk: tokens[0..steps-1].
                cudaMemcpyAsync(out_ptr + gen_len, tokens_ptr, sizeof(int) * steps, cudaMemcpyDeviceToDevice, stream);

                position_ += steps;
                gen_len += steps;
                chunks_since_check += 1;

                bool do_check = (chunks_since_check >= check_stride) || (gen_len >= max_new);
                if (do_check && gen_len > checked_len) {
                    int span = gen_len - checked_len;
                    launch_ldg_find_first_eos(out_ptr + checked_len, span, eos_ptr, eos_n, eos_idx_ptr, stream);
                    int eos_local_host = span;
                    cudaMemcpyAsync(&eos_local_host, eos_idx_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    if (eos_local_host < span) {
                        int eos_abs = checked_len + eos_local_host;
                        int kept = eos_abs;  // exclude EOS itself
                        wasted_tokens = gen_len - kept;
                        gen_len = kept;
                        stop_reason = 1;
                        break;
                    }
                    checked_len = gen_len;
                    chunks_since_check = 0;
                }

                // next_token = tokens[steps] (D2D)
                cudaMemcpyAsync(next_ptr, tokens_ptr + steps, sizeof(int), cudaMemcpyDeviceToDevice, stream);
            }

            const char* dbg = std::getenv("MEGAQWEN_DEBUG_SPLIT_STAGE_AVG");
            if (stop_reason == 1 &&
                dbg != nullptr && std::strcmp(dbg, "") != 0 &&
                std::strcmp(dbg, "0") != 0 &&
                std::strcmp(dbg, "false") != 0 &&
                std::strcmp(dbg, "False") != 0 &&
                wasted_tokens > 0) {
                std::printf("[MEGAQWEN_DEBUG] split_gemm_v2 wasted decode tokens after EOS: %d\n", wasted_tokens);
            }
        } else {
            while (gen_len < max_new) {
                int steps = std::min(chunk, max_new - gen_len);

                // positions: [position_, position_+1, ...]
                launch_ldg_fill_iota(pos_ptr, position_, steps, stream);

                // tokens[0] = next_token (D2D)
                cudaMemcpyAsync(tokens_ptr, next_ptr, sizeof(int), cudaMemcpyDeviceToDevice, stream);

                // Decode chain (writes tokens[1..steps]).
                for (int i = 0; i < steps; i++) {
                    const int* in_ptr = (const int*)(tokens_ptr + i);
                    int* out_tok_ptr = (int*)(tokens_ptr + i + 1);
                    const int* p_ptr = (const int*)(pos_ptr + i);
                    launch_decode_step_backend(in_ptr, out_tok_ptr, p_ptr, stream);
                }

                // Append outputs for this chunk: tokens[0..steps-1] (excludes tokens[steps], which becomes next token).
                cudaMemcpyAsync(out_ptr + gen_len, tokens_ptr, sizeof(int) * steps, cudaMemcpyDeviceToDevice, stream);

                // EOS scan in tokens[1..steps-1] (exclude current token and exclude the chunk tail token).
                int stop_pos = -1;
                if (steps > 1) {
                    launch_ldg_find_first_eos(tokens_ptr + 1, steps - 1, eos_ptr, eos_n, eos_idx_ptr, stream);
                    int eos_idx_host = steps - 1;
                    cudaMemcpyAsync(&eos_idx_host, eos_idx_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    if (eos_idx_host < (steps - 1)) {
                        stop_pos = eos_idx_host + 1;  // number of tokens to keep from this chunk (excludes EOS itself)
                    }
                }

                // KV cache already advanced by `steps` due to launched kernels; keep position_ consistent.
                position_ += steps;

                if (stop_pos >= 0) {
                    gen_len += stop_pos;
                    stop_reason = 1;
                    break;
                }

                gen_len += steps;
                // next_token = tokens[steps] (D2D)
                cudaMemcpyAsync(next_ptr, tokens_ptr + steps, sizeof(int), cudaMemcpyDeviceToDevice, stream);
            }
        }

        if (decode_backend_ != 0) {
            split_debug_stage_print_summary();
        }

        return std::make_tuple(out.narrow(0, 0, gen_len), gen_len, stop_reason);
    }

    torch::Tensor decode_steps(int first_token_id, int num_steps) {
        return decode_steps_cuda(first_token_id, num_steps).cpu();
    }

    void reset() {
        position_ = 0;
        k_cache_.zero_();
        v_cache_.zero_();
    }

    int position() const { return position_; }

    torch::Tensor get_k_cache() const { return k_cache_; }
    torch::Tensor get_v_cache() const { return v_cache_; }

    void launch_decode_step_backend(const int* in_ptr, int* out_ptr, const int* p_ptr, cudaStream_t stream) {
        if (decode_backend_ != 0) {
            // Split GEMM backend family (non-cooperative; cuBLAS + cache-attn).
            launch_split_decode_gemm(
                in_ptr,
                out_ptr,
                embed_weight_.data_ptr(),
                (const PrefillLayerWeights*)prefill_layer_weights_.data(),  // host pointer
                split_qkv_weight_ptrs_.empty() ? nullptr : split_qkv_weight_ptrs_.data(),
                split_gateup_weight_ptrs_.empty() ? nullptr : split_gateup_weight_ptrs_.data(),
                split_down_weight_ptrs_.empty() ? nullptr : split_down_weight_ptrs_.data(),
                split_q_w4_packed_ptrs_.empty() ? nullptr : split_q_w4_packed_ptrs_.data(),
                split_q_w4_scales_ptrs_.empty() ? nullptr : split_q_w4_scales_ptrs_.data(),
                split_q_w4_codebook_ptrs_.empty() ? nullptr : split_q_w4_codebook_ptrs_.data(),
                split_k_w4_packed_ptrs_.empty() ? nullptr : split_k_w4_packed_ptrs_.data(),
                split_k_w4_scales_ptrs_.empty() ? nullptr : split_k_w4_scales_ptrs_.data(),
                split_k_w4_codebook_ptrs_.empty() ? nullptr : split_k_w4_codebook_ptrs_.data(),
                split_v_w4_packed_ptrs_.empty() ? nullptr : split_v_w4_packed_ptrs_.data(),
                split_v_w4_scales_ptrs_.empty() ? nullptr : split_v_w4_scales_ptrs_.data(),
                split_v_w4_codebook_ptrs_.empty() ? nullptr : split_v_w4_codebook_ptrs_.data(),
                split_o_w4_packed_ptrs_.empty() ? nullptr : split_o_w4_packed_ptrs_.data(),
                split_o_w4_scales_ptrs_.empty() ? nullptr : split_o_w4_scales_ptrs_.data(),
                split_o_w4_codebook_ptrs_.empty() ? nullptr : split_o_w4_codebook_ptrs_.data(),
                split_gateup_w4_packed_ptrs_.empty() ? nullptr : split_gateup_w4_packed_ptrs_.data(),
                split_gateup_w4_scales_ptrs_.empty() ? nullptr : split_gateup_w4_scales_ptrs_.data(),
                split_gateup_w4_codebook_ptrs_.empty() ? nullptr : split_gateup_w4_codebook_ptrs_.data(),
                split_down_w4_packed_ptrs_.empty() ? nullptr : split_down_w4_packed_ptrs_.data(),
                split_down_w4_scales_ptrs_.empty() ? nullptr : split_down_w4_scales_ptrs_.data(),
                split_down_w4_codebook_ptrs_.empty() ? nullptr : split_down_w4_codebook_ptrs_.data(),
                split_kv_block_table_.defined() ? (const int*)split_kv_block_table_.data_ptr<int>() : nullptr,
                split_kv_block_size_,
                final_norm_weight_.data_ptr(),
                lm_head_weight_.data_ptr(),
                cos_table_.data_ptr(),
                sin_table_.data_ptr(),
                k_cache_.data_ptr(),
                v_cache_.data_ptr(),
                split_attn_partial_m_.defined() ? split_attn_partial_m_.data_ptr() : nullptr,
                split_attn_partial_s_.defined() ? split_attn_partial_s_.data_ptr() : nullptr,
                split_attn_partial_out_.defined() ? split_attn_partial_out_.data_ptr() : nullptr,
                split_attn_chunk_size_,
                split_attn_max_chunks_,
                g_activations_.data_ptr(),
                g_residual_.data_ptr(),
                normalized_.data_ptr(),
                split_qkv_out_.data_ptr(),
                attn_out_.data_ptr(),
                o_proj_out_.data_ptr(),
                mlp_norm_.data_ptr(),
                split_gateup_out_.data_ptr(),
                mlp_intermediate_.data_ptr(),
                down_out_.data_ptr(),
                final_hidden_.data_ptr(),
                block_max_vals_.data_ptr(),
                block_max_idxs_.data_ptr(),
                num_layers_,
                p_ptr,
                max_seq_len_,
                attn_scale_,
                cublas_handle_,
                cublaslt_handle_,
                split_cublaslt_workspace_.defined() ? split_cublaslt_workspace_.data_ptr() : nullptr,
                split_cublaslt_workspace_bytes_,
                stream
            );
            return;
        }

        // Cooperative-group LDG megakernel backend (default).
        launch_ldg_decode(
            in_ptr,
            out_ptr,
            embed_weight_.data_ptr(),
            (const LDGLayerWeights*)d_decode_layer_weights_.data_ptr(),
            final_norm_weight_.data_ptr(),
            lm_head_weight_.data_ptr(),
            cos_table_.data_ptr(),
            sin_table_.data_ptr(),
            k_cache_.data_ptr(),
            v_cache_.data_ptr(),
            hidden_buffer_.data_ptr(),
            g_activations_.data_ptr(),
            g_residual_.data_ptr(),
            g_q_.data_ptr(),
            g_k_.data_ptr(),
            g_v_.data_ptr(),
            g_attn_out_.data_ptr(),
            g_mlp_intermediate_.data_ptr(),
            g_normalized_.data_ptr(),
            attn_partial_max_.data_ptr(),
            attn_partial_sum_.data_ptr(),
            attn_partial_out_.data_ptr(),
            block_max_vals_.data_ptr(),
            block_max_idxs_.data_ptr(),
            num_layers_,
            p_ptr,
            max_seq_len_,
            attn_scale_,
            decode_num_blocks_,
            lm_num_blocks_,
            stream
        );
    }

private:
    int num_layers_;
    int max_seq_len_;
    int max_prefill_len_;
    int position_;
    int prefill_backend_;  // 0=cublas, 1=megakernel
    int decode_num_blocks_;
    int lm_num_blocks_;
    int decode_backend_;  // 0=coop, 1=split_gemm(chunked), 2=split_gemm_v2(chunked+sparse EOS check)
    float attn_scale_;

    cublasHandle_t cublas_handle_{nullptr};
    cublasLtHandle_t cublaslt_handle_{nullptr};

    // Decode CUDA graph (optional).
    cudaGraph_t decode_graph_{nullptr};
    cudaGraphExec_t decode_graph_exec_{nullptr};
    int decode_graph_steps_{0};
    torch::Tensor graph_tokens_;
    torch::Tensor graph_positions_;

    torch::Tensor embed_weight_;
    torch::Tensor final_norm_weight_;
    torch::Tensor lm_head_weight_;
    torch::Tensor cos_table_;
    torch::Tensor sin_table_;
    torch::Tensor d_prefill_layer_weights_;
    torch::Tensor d_decode_layer_weights_;

    std::vector<torch::Tensor> layer_weights_tensors_;
    std::vector<PrefillLayerWeights> prefill_layer_weights_;
    std::vector<LDGLayerWeights> decode_layer_weights_;

    torch::Tensor k_cache_, v_cache_;

    // Prefill buffers
    torch::Tensor hidden_float_, residual_, normalized_;
    torch::Tensor q_proj_, k_proj_, v_proj_;
    torch::Tensor attn_out_, o_proj_out_, mlp_norm_;
    torch::Tensor gate_out_, up_out_, mlp_intermediate_, down_out_;
    torch::Tensor final_hidden_;
    torch::Tensor mk_hidden_, mk_residual_, mk_normalized_;
    torch::Tensor mk_q_proj_, mk_k_proj_, mk_v_proj_;
    torch::Tensor mk_attn_out_, mk_mlp_intermediate_, mk_final_hidden_;
    torch::Tensor split_qkv_out_, split_gateup_out_;
    torch::Tensor input_tokens_;

    // Decode buffers
    torch::Tensor hidden_buffer_, g_activations_, g_residual_;
    torch::Tensor g_q_, g_k_, g_v_, g_attn_out_;
    torch::Tensor g_mlp_intermediate_, g_normalized_;
    torch::Tensor attn_partial_max_, attn_partial_sum_, attn_partial_out_;

    // Shared
    torch::Tensor block_max_vals_, block_max_idxs_, output_token_;
    torch::Tensor input_token_;
    torch::Tensor position_token_;

    // Greedy decode workspace (GPU-side loop to reduce host synchronizations).
    torch::Tensor greedy_tokens_;
    torch::Tensor greedy_positions_;
    torch::Tensor greedy_eos_idx_;
    torch::Tensor greedy_next_token_;

    // Split backend packed weights (host-side pointer arrays; tensors keep storage alive).
    std::vector<torch::Tensor> split_qkv_weight_tensors_;
    std::vector<torch::Tensor> split_gateup_weight_tensors_;
    std::vector<torch::Tensor> split_down_weight_tensors_;
    std::vector<torch::Tensor> split_q_w4_packed_tensors_;
    std::vector<torch::Tensor> split_q_w4_scales_tensors_;
    std::vector<torch::Tensor> split_q_w4_codebook_tensors_;
    std::vector<torch::Tensor> split_k_w4_packed_tensors_;
    std::vector<torch::Tensor> split_k_w4_scales_tensors_;
    std::vector<torch::Tensor> split_k_w4_codebook_tensors_;
    std::vector<torch::Tensor> split_v_w4_packed_tensors_;
    std::vector<torch::Tensor> split_v_w4_scales_tensors_;
    std::vector<torch::Tensor> split_v_w4_codebook_tensors_;
    std::vector<torch::Tensor> split_o_w4_packed_tensors_;
    std::vector<torch::Tensor> split_o_w4_scales_tensors_;
    std::vector<torch::Tensor> split_o_w4_codebook_tensors_;
    std::vector<torch::Tensor> split_gateup_w4_packed_tensors_;
    std::vector<torch::Tensor> split_gateup_w4_scales_tensors_;
    std::vector<torch::Tensor> split_gateup_w4_codebook_tensors_;
    std::vector<torch::Tensor> split_down_w4_packed_tensors_;
    std::vector<torch::Tensor> split_down_w4_scales_tensors_;
    std::vector<torch::Tensor> split_down_w4_codebook_tensors_;
    std::vector<const void*> split_qkv_weight_ptrs_;
    std::vector<const void*> split_gateup_weight_ptrs_;
    std::vector<const void*> split_down_weight_ptrs_;
    std::vector<const void*> split_q_w4_packed_ptrs_;
    std::vector<const void*> split_q_w4_scales_ptrs_;
    std::vector<const void*> split_q_w4_codebook_ptrs_;
    std::vector<const void*> split_k_w4_packed_ptrs_;
    std::vector<const void*> split_k_w4_scales_ptrs_;
    std::vector<const void*> split_k_w4_codebook_ptrs_;
    std::vector<const void*> split_v_w4_packed_ptrs_;
    std::vector<const void*> split_v_w4_scales_ptrs_;
    std::vector<const void*> split_v_w4_codebook_ptrs_;
    std::vector<const void*> split_o_w4_packed_ptrs_;
    std::vector<const void*> split_o_w4_scales_ptrs_;
    std::vector<const void*> split_o_w4_codebook_ptrs_;
    std::vector<const void*> split_gateup_w4_packed_ptrs_;
    std::vector<const void*> split_gateup_w4_scales_ptrs_;
    std::vector<const void*> split_gateup_w4_codebook_ptrs_;
    std::vector<const void*> split_down_w4_packed_ptrs_;
    std::vector<const void*> split_down_w4_scales_ptrs_;
    std::vector<const void*> split_down_w4_codebook_ptrs_;
    torch::Tensor split_cublaslt_workspace_;
    size_t split_cublaslt_workspace_bytes_{0};
    torch::Tensor split_attn_partial_m_;
    torch::Tensor split_attn_partial_s_;
    torch::Tensor split_attn_partial_out_;
    int split_attn_chunk_size_{0};
    int split_attn_max_chunks_{0};
    bool split_kv_paged_{false};
    int split_kv_block_size_{16};
    int split_kv_num_blocks_{0};
    torch::Tensor split_kv_block_table_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MegakernelPrefillDecoder>(m, "MegakernelPrefillDecoder")
        .def(py::init<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor,
                      std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
                      std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
                      std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
                      std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
                      std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
                      std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
                      int, int, int>(),
             py::arg("embed_weight"),
             py::arg("layer_weights_flat"),
             py::arg("final_norm_weight"),
             py::arg("lm_head_weight"),
             py::arg("cos_table"),
             py::arg("sin_table"),
             py::arg("split_q_w4_packed") = std::vector<torch::Tensor>{},
             py::arg("split_q_w4_scales") = std::vector<torch::Tensor>{},
             py::arg("split_q_w4_codebook") = std::vector<torch::Tensor>{},
             py::arg("split_k_w4_packed") = std::vector<torch::Tensor>{},
             py::arg("split_k_w4_scales") = std::vector<torch::Tensor>{},
             py::arg("split_k_w4_codebook") = std::vector<torch::Tensor>{},
             py::arg("split_v_w4_packed") = std::vector<torch::Tensor>{},
             py::arg("split_v_w4_scales") = std::vector<torch::Tensor>{},
             py::arg("split_v_w4_codebook") = std::vector<torch::Tensor>{},
             py::arg("split_o_w4_packed") = std::vector<torch::Tensor>{},
             py::arg("split_o_w4_scales") = std::vector<torch::Tensor>{},
             py::arg("split_o_w4_codebook") = std::vector<torch::Tensor>{},
             py::arg("split_gateup_w4_packed") = std::vector<torch::Tensor>{},
             py::arg("split_gateup_w4_scales") = std::vector<torch::Tensor>{},
             py::arg("split_gateup_w4_codebook") = std::vector<torch::Tensor>{},
             py::arg("split_down_w4_packed") = std::vector<torch::Tensor>{},
             py::arg("split_down_w4_scales") = std::vector<torch::Tensor>{},
             py::arg("split_down_w4_codebook") = std::vector<torch::Tensor>{},
             py::arg("num_layers"),
             py::arg("max_seq_len"),
             py::arg("max_prefill_len") = 512)
        .def("prefill_step", &MegakernelPrefillDecoder::prefill_step)
        .def("prefill_asr", &MegakernelPrefillDecoder::prefill_asr,
             py::arg("input_token_ids"),
             py::arg("audio_embeds"),
             py::arg("audio_start_idx"))
        .def("decode_step", &MegakernelPrefillDecoder::decode_step)
        .def("decode_steps", &MegakernelPrefillDecoder::decode_steps,
             py::arg("first_token_id"),
             py::arg("num_steps"))
        .def("decode_steps_cuda", &MegakernelPrefillDecoder::decode_steps_cuda,
             py::arg("first_token_id"),
             py::arg("num_steps"))
        .def("decode_steps_cuda_graph", &MegakernelPrefillDecoder::decode_steps_cuda_graph,
             py::arg("first_token_id"),
             py::arg("num_steps"))
        .def("decode_greedy_cuda", &MegakernelPrefillDecoder::decode_greedy_cuda,
             py::arg("first_token_id"),
             py::arg("max_new_tokens"),
             py::arg("eos_ids"),
             py::arg("chunk") = 32)
        .def("debug_ldg_decode_cycles", &MegakernelPrefillDecoder::debug_ldg_decode_cycles,
             py::arg("input_token_id"))
        .def("reset", &MegakernelPrefillDecoder::reset)
        .def("position", &MegakernelPrefillDecoder::position)
        .def("get_k_cache", &MegakernelPrefillDecoder::get_k_cache)
        .def("get_v_cache", &MegakernelPrefillDecoder::get_v_cache);
}
