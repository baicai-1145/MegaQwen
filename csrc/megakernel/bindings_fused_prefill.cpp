#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

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

extern "C" void launch_ldg_fill_iota(int* out, int start, int n, cudaStream_t stream);

class MegakernelFusedPrefillDecoder {
public:
    MegakernelFusedPrefillDecoder(
        torch::Tensor embed_weight,
        std::vector<torch::Tensor> layer_weights_flat,
        torch::Tensor final_norm_weight,
        torch::Tensor lm_head_weight,
        torch::Tensor cos_table,
        torch::Tensor sin_table,
        int num_layers,
        int max_seq_len,
        int max_prefill_len = 64
    ) : num_layers_(num_layers), max_seq_len_(max_seq_len), max_prefill_len_(max_prefill_len) {

        embed_weight_ = embed_weight;
        final_norm_weight_ = final_norm_weight;
        lm_head_weight_ = lm_head_weight;
        cos_table_ = cos_table;
        sin_table_ = sin_table;

        layer_weights_tensors_ = layer_weights_flat;

        // Build layer weights structs
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
        d_prefill_layer_weights_ = torch::empty({num_layers * (int)sizeof(PrefillMKLayerWeights)},
                                                 torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_prefill_layer_weights_.data_ptr(), prefill_layer_weights_.data(),
                   num_layers * sizeof(PrefillMKLayerWeights), cudaMemcpyHostToDevice);

        // Copy decode layer weights to device
        d_decode_layer_weights_ = torch::empty({num_layers * (int)sizeof(LDGLayerWeights)},
                                                torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_decode_layer_weights_.data_ptr(), decode_layer_weights_.data(),
                   num_layers * sizeof(LDGLayerWeights), cudaMemcpyHostToDevice);

        // Allocate KV cache
        int kv_heads = 8;
        int head_dim = 128;
        k_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        v_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));

        // Allocate prefill buffers (all float32 for megakernel)
        int hidden_size = 1024;
        int q_size = 16 * 128;
        int kv_size = 8 * 128;
        int intermediate_size = 3072;

        hidden_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        residual_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        normalized_ = torch::empty({max_prefill_len, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        q_proj_ = torch::empty({max_prefill_len, q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        k_proj_ = torch::empty({max_prefill_len, kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        v_proj_ = torch::empty({max_prefill_len, kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        attn_out_ = torch::empty({max_prefill_len, q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        mlp_intermediate_ = torch::empty({max_prefill_len, intermediate_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        final_hidden_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

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

        block_max_vals_ = torch::empty({1184}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        block_max_idxs_ = torch::empty({1184}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        attn_partial_max_ = torch::empty({1184}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        attn_partial_sum_ = torch::empty({1184}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        attn_partial_out_ = torch::empty({1184 * 128}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
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

	        position_ = 0;
	        attn_scale_ = 1.0f / sqrtf(128.0f);
	    }

    int prefill_step(torch::Tensor input_token_ids) {
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
            hidden_.data_ptr(),
            residual_.data_ptr(),
            normalized_.data_ptr(),
            q_proj_.data_ptr(),
            k_proj_.data_ptr(),
            v_proj_.data_ptr(),
            attn_out_.data_ptr(),
            mlp_intermediate_.data_ptr(),
            final_hidden_.data_ptr(),
            hidden_buffer_.data_ptr(),  // bf16 output for decode continuation
            block_max_vals_.data_ptr(),
            block_max_idxs_.data_ptr(),
            num_layers_,
            max_seq_len_,
            attn_scale_,
            stream
        );

        position_ = seq_len;
        return output_token_.item<int>();
    }

    int prefill_asr(torch::Tensor input_token_ids, torch::Tensor audio_embeds, int audio_start_idx) {
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

        input_tokens_.narrow(0, 0, seq_len).copy_(input_token_ids.to(torch::kCUDA));

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
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
            hidden_.data_ptr(),
            residual_.data_ptr(),
            normalized_.data_ptr(),
            q_proj_.data_ptr(),
            k_proj_.data_ptr(),
            v_proj_.data_ptr(),
            attn_out_.data_ptr(),
            mlp_intermediate_.data_ptr(),
            final_hidden_.data_ptr(),
            hidden_buffer_.data_ptr(),  // bf16 output for decode continuation
            block_max_vals_.data_ptr(),
            block_max_idxs_.data_ptr(),
            num_layers_,
            max_seq_len_,
            attn_scale_,
            stream
        );
        position_ = seq_len;
        return output_token_.item<int>();
    }

    int decode_step(int input_token_id) {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
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

        position_++;
        return output_token_.item<int>();
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

        position_ += num_steps;
        return tokens.slice(0, 1, num_steps + 1);
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
    int max_prefill_len() const { return max_prefill_len_; }

    torch::Tensor get_k_cache() const { return k_cache_; }
    torch::Tensor get_v_cache() const { return v_cache_; }

private:
    int num_layers_;
    int max_seq_len_;
    int max_prefill_len_;
    int position_;
    int decode_num_blocks_;
    int lm_num_blocks_;
    float attn_scale_;

    torch::Tensor embed_weight_;
    torch::Tensor final_norm_weight_;
    torch::Tensor lm_head_weight_;
    torch::Tensor cos_table_;
    torch::Tensor sin_table_;
    torch::Tensor d_prefill_layer_weights_;
    torch::Tensor d_decode_layer_weights_;

    std::vector<torch::Tensor> layer_weights_tensors_;
    std::vector<PrefillMKLayerWeights> prefill_layer_weights_;
    std::vector<LDGLayerWeights> decode_layer_weights_;

    torch::Tensor k_cache_, v_cache_;

    // Prefill buffers (float32)
    torch::Tensor hidden_, residual_, normalized_;
    torch::Tensor q_proj_, k_proj_, v_proj_;
    torch::Tensor attn_out_, mlp_intermediate_;
    torch::Tensor final_hidden_;
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
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MegakernelFusedPrefillDecoder>(m, "MegakernelFusedPrefillDecoder")
        .def(py::init<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor, int, int, int>(),
             py::arg("embed_weight"),
             py::arg("layer_weights_flat"),
             py::arg("final_norm_weight"),
             py::arg("lm_head_weight"),
             py::arg("cos_table"),
             py::arg("sin_table"),
             py::arg("num_layers"),
             py::arg("max_seq_len"),
             py::arg("max_prefill_len") = 64)
        .def("prefill_step", &MegakernelFusedPrefillDecoder::prefill_step)
        .def("prefill_asr", &MegakernelFusedPrefillDecoder::prefill_asr,
             py::arg("input_token_ids"),
             py::arg("audio_embeds"),
             py::arg("audio_start_idx"))
        .def("decode_step", &MegakernelFusedPrefillDecoder::decode_step)
        .def("decode_steps", &MegakernelFusedPrefillDecoder::decode_steps,
             py::arg("first_token_id"),
             py::arg("num_steps"))
        .def("decode_steps_cuda", &MegakernelFusedPrefillDecoder::decode_steps_cuda,
             py::arg("first_token_id"),
             py::arg("num_steps"))
        .def("reset", &MegakernelFusedPrefillDecoder::reset)
        .def("position", &MegakernelFusedPrefillDecoder::position)
        .def("max_prefill_len", &MegakernelFusedPrefillDecoder::max_prefill_len)
        .def("get_k_cache", &MegakernelFusedPrefillDecoder::get_k_cache)
        .def("get_v_cache", &MegakernelFusedPrefillDecoder::get_v_cache);
}
