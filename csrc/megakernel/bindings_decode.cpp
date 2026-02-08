#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <cstdlib>

// Must match the struct in fused_decode_ldg.cu
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

class MegakernelDecoder {
public:
    MegakernelDecoder(
        torch::Tensor embed_weight,
        std::vector<torch::Tensor> layer_weights_flat,
        torch::Tensor final_norm_weight,
        torch::Tensor lm_head_weight,
        torch::Tensor cos_table,
        torch::Tensor sin_table,
        int num_layers,
        int max_seq_len
    ) : num_layers_(num_layers), max_seq_len_(max_seq_len) {

        embed_weight_ = embed_weight;
        final_norm_weight_ = final_norm_weight;
        lm_head_weight_ = lm_head_weight;
        cos_table_ = cos_table;
        sin_table_ = sin_table;

        // Store layer weights
        layer_weights_tensors_ = layer_weights_flat;

        // Build layer weights structs
        layer_weights_.resize(num_layers);
        for (int i = 0; i < num_layers; i++) {
            layer_weights_[i].input_layernorm_weight = layer_weights_flat[i * 11 + 0].data_ptr();
            layer_weights_[i].q_proj_weight = layer_weights_flat[i * 11 + 1].data_ptr();
            layer_weights_[i].k_proj_weight = layer_weights_flat[i * 11 + 2].data_ptr();
            layer_weights_[i].v_proj_weight = layer_weights_flat[i * 11 + 3].data_ptr();
            layer_weights_[i].q_norm_weight = layer_weights_flat[i * 11 + 4].data_ptr();
            layer_weights_[i].k_norm_weight = layer_weights_flat[i * 11 + 5].data_ptr();
            layer_weights_[i].o_proj_weight = layer_weights_flat[i * 11 + 6].data_ptr();
            layer_weights_[i].post_attn_layernorm_weight = layer_weights_flat[i * 11 + 7].data_ptr();
            layer_weights_[i].gate_proj_weight = layer_weights_flat[i * 11 + 8].data_ptr();
            layer_weights_[i].up_proj_weight = layer_weights_flat[i * 11 + 9].data_ptr();
            layer_weights_[i].down_proj_weight = layer_weights_flat[i * 11 + 10].data_ptr();
        }

        // Copy layer weights to device
        d_layer_weights_ = torch::empty({num_layers * (int)sizeof(LDGLayerWeights)},
                                         torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_layer_weights_.data_ptr(), layer_weights_.data(),
                   num_layers * sizeof(LDGLayerWeights), cudaMemcpyHostToDevice);

        // Allocate KV cache
        int kv_heads = 8;
        int head_dim = 128;
        k_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        v_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));

        // Allocate intermediate buffers
        int hidden_size = 1024;
        int q_size = 16 * 128;
        int kv_size = 8 * 128;
        int intermediate_size = 3072;

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
        // Attention scratch (for optional all-block attention path)
        attn_partial_max_ = torch::empty({1184}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        attn_partial_sum_ = torch::empty({1184}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        attn_partial_out_ = torch::empty({1184 * 128}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        block_max_vals_ = torch::empty({1184}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
	        block_max_idxs_ = torch::empty({1184}, torch::dtype(torch::kInt32).device(torch::kCUDA));
	        output_token_ = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
	        input_token_ = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
	        position_token_ = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));

        // Default to one cooperative block per SM (matches the original 3090 tuning).
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        // Let the CUDA side pick an occupancy-safe cooperative grid size by default.
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

            // Decode backend:
            // - "coop" (default): cooperative-group LDG megakernel (launch_ldg_decode)
            // - "split_gemm": non-cooperative split pipeline (cuBLAS GEMM + cache-attn)
            decode_backend_ = 0;
            if (const char* s = std::getenv("MEGAQWEN_DECODE_BACKEND")) {
                if (std::strcmp(s, "split_gemm") == 0 || std::strcmp(s, "split") == 0) {
                    decode_backend_ = 1;
                }
            }
	    }

	    int decode_step(int input_token_id) {
	        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
	        cudaMemcpyAsync(input_token_.data_ptr(), &input_token_id, sizeof(int), cudaMemcpyHostToDevice, stream);
	        cudaMemcpyAsync(position_token_.data_ptr(), &position_, sizeof(int), cudaMemcpyHostToDevice, stream);

	        launch_ldg_decode(
	            (const int*)input_token_.data_ptr(),
	            (int*)output_token_.data_ptr(),
	            embed_weight_.data_ptr(),
	            (const LDGLayerWeights*)d_layer_weights_.data_ptr(),
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

    void reset() {
        position_ = 0;
        k_cache_.zero_();
        v_cache_.zero_();
    }

    int position() const { return position_; }

private:
    int num_layers_;
    int max_seq_len_;
    int position_;
    int decode_num_blocks_;
    int lm_num_blocks_;
    int decode_backend_;  // 0=coop, 1=split_gemm
    float attn_scale_;

    torch::Tensor embed_weight_;
    torch::Tensor final_norm_weight_;
    torch::Tensor lm_head_weight_;
    torch::Tensor cos_table_;
    torch::Tensor sin_table_;
    torch::Tensor d_layer_weights_;

    std::vector<torch::Tensor> layer_weights_tensors_;
    std::vector<LDGLayerWeights> layer_weights_;

    torch::Tensor k_cache_, v_cache_;
    torch::Tensor hidden_buffer_, g_activations_, g_residual_;
    torch::Tensor g_q_, g_k_, g_v_, g_attn_out_;
    torch::Tensor g_mlp_intermediate_, g_normalized_;
    torch::Tensor attn_partial_max_, attn_partial_sum_, attn_partial_out_;
    torch::Tensor block_max_vals_, block_max_idxs_, output_token_;
    torch::Tensor input_token_;
    torch::Tensor position_token_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MegakernelDecoder>(m, "MegakernelDecoder")
        .def(py::init<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor, int, int>())
        .def("decode_step", &MegakernelDecoder::decode_step)
        .def("reset", &MegakernelDecoder::reset)
        .def("position", &MegakernelDecoder::position);
}
