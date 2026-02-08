/**
 * Fused Prefill Kernel for Qwen3-0.6B
 *
 * Processes multiple tokens in parallel using batched matrix multiplications.
 * Uses cuBLAS for compute-bound GEMM operations (BF16 x BF16 with FP32 accumulation)
 * and custom kernels for:
 * - RMSNorm (batched)
 * - RoPE (all positions)
 * - Causal attention with online softmax
 * - SiLU activation
 * - KV cache population
 */

#include "config.cuh"
#include <ATen/ops/scaled_dot_product_attention.h>
#include <torch/extension.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cg = cooperative_groups;

// =============================================================================
// Configuration
// =============================================================================

constexpr int PREFILL_BLOCK_SIZE = 256;
constexpr int PREFILL_NUM_WARPS = PREFILL_BLOCK_SIZE / WARP_SIZE;
constexpr float PREFILL_RMS_EPS = 1e-6f;

struct PrefillLayerWeights {
    const __nv_bfloat16* input_layernorm_weight;
    const __nv_bfloat16* q_proj_weight;
    const __nv_bfloat16* k_proj_weight;
    const __nv_bfloat16* v_proj_weight;
    const __nv_bfloat16* q_norm_weight;
    const __nv_bfloat16* k_norm_weight;
    const __nv_bfloat16* o_proj_weight;
    const __nv_bfloat16* post_attn_layernorm_weight;
    const __nv_bfloat16* gate_proj_weight;
    const __nv_bfloat16* up_proj_weight;
    const __nv_bfloat16* down_proj_weight;
};

enum PrefillStageId : int {
    PREFILL_STAGE_EMBED_PATCH = 0,
    PREFILL_STAGE_RMS1 = 1,
    PREFILL_STAGE_QKV_GEMM = 2,
    PREFILL_STAGE_QK_ROPE_CACHE = 3,
    PREFILL_STAGE_ATTN = 4,
    PREFILL_STAGE_O_GEMM = 5,
    PREFILL_STAGE_RESID1 = 6,
    PREFILL_STAGE_RMS2 = 7,
    PREFILL_STAGE_GATEUP_GEMM = 8,
    PREFILL_STAGE_SILU = 9,
    PREFILL_STAGE_DOWN_GEMM = 10,
    PREFILL_STAGE_RESID2 = 11,
    PREFILL_STAGE_FINAL_NORM = 12,
    PREFILL_STAGE_HIDDEN_COPY = 13,
    PREFILL_STAGE_LM1 = 14,
    PREFILL_STAGE_LM2 = 15,
    PREFILL_STAGE_COUNT = 16
};

static const char* kPrefillStageNames[PREFILL_STAGE_COUNT] = {
    "embed_patch",
    "rmsnorm_1",
    "qkv_gemm",
    "qk_rope_cache",
    "attn",
    "o_gemm",
    "residual_1",
    "rmsnorm_2",
    "gateup_gemm",
    "silu_mul",
    "down_gemm",
    "residual_2",
    "final_norm",
    "hidden_copy",
    "lm_head_phase1",
    "lm_head_phase2"
};

static inline int _prefill_debug_budget() {
    static int budget = -999;
    if (budget != -999) return budget;
    const char* s = std::getenv("MEGAQWEN_DEBUG_PREFILL_STAGE");
    if (s == nullptr || std::strcmp(s, "") == 0 || std::strcmp(s, "0") == 0 ||
        std::strcmp(s, "false") == 0 || std::strcmp(s, "False") == 0) {
        budget = 0;
    } else if (std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0 || std::strcmp(s, "True") == 0) {
        budget = 1;
    } else if (std::strcmp(s, "all") == 0 || std::strcmp(s, "inf") == 0) {
        budget = -1;
    } else {
        int v = std::atoi(s);
        budget = (v > 0) ? v : 0;
    }
    return budget;
}

static inline int _prefill_debug_skip() {
    static int skip = -999;
    if (skip != -999) return skip;
    const char* s = std::getenv("MEGAQWEN_DEBUG_PREFILL_STAGE_SKIP");
    if (s == nullptr || std::strcmp(s, "") == 0) {
        skip = 0;
    } else {
        int v = std::atoi(s);
        skip = (v > 0) ? v : 0;
    }
    return skip;
}

static inline bool _prefill_debug_take_ticket() {
    static int budget = _prefill_debug_budget();
    static int skip = _prefill_debug_skip();
    if (skip > 0) {
        skip--;
        return false;
    }
    if (budget == 0) return false;
    if (budget > 0) budget--;
    return true;
}

// Attention implementation:
// - legacy: one warp handles one (q_pos, q_head), serial over kv_pos (stable default).
// - splitk: one block handles one (q_pos, q_head), multiple warps split kv_pos (experimental).
// - auto: use splitk for long sequence (also experimental; requires explicit enable).
// - flash: tiled shared-memory flash-style kernel (experimental).
// - flash_ext: route to PyTorch SDPA/FlashAttention dispatch (experimental).
static inline int _prefill_attn_impl_mode() {
    static int mode = -1;  // -1 unknown, 0 legacy, 1 splitk, 2 auto, 3 flash, 4 flash_ext
    if (mode >= 0) return mode;
    const char* s = std::getenv("MEGAQWEN_PREFILL_ATTN_IMPL");
    if (s == nullptr || std::strcmp(s, "") == 0 || std::strcmp(s, "legacy") == 0 ||
        std::strcmp(s, "0") == 0 || std::strcmp(s, "false") == 0 || std::strcmp(s, "False") == 0) {
        mode = 0;
    } else if (std::strcmp(s, "flash_ext") == 0 || std::strcmp(s, "sdpa") == 0 ||
               std::strcmp(s, "flash_sdpa") == 0 || std::strcmp(s, "4") == 0) {
        mode = 4;
    } else if (std::strcmp(s, "flash") == 0 || std::strcmp(s, "flash2") == 0 ||
               std::strcmp(s, "fa") == 0 || std::strcmp(s, "3") == 0) {
        mode = 3;
    } else if (std::strcmp(s, "auto") == 0) {
        mode = 2;
    } else if (std::strcmp(s, "splitk") == 0 || std::strcmp(s, "v2") == 0 ||
               std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0 || std::strcmp(s, "True") == 0) {
        mode = 1;
    } else {
        mode = 0;
    }
    return mode;
}

static inline int _prefill_attn_auto_min_seq() {
    static int min_seq = -1;
    if (min_seq >= 0) return min_seq;
    min_seq = 192;
    if (const char* s = std::getenv("MEGAQWEN_PREFILL_ATTN_AUTO_MIN_SEQ")) {
        int v = std::atoi(s);
        if (v > 0) min_seq = v;
    }
    return min_seq;
}

static inline int _prefill_attn_warps() {
    static int warps = -1;
    if (warps > 0) return warps;
    warps = 8;
    if (const char* s = std::getenv("MEGAQWEN_PREFILL_ATTN_WARPS")) {
        int v = std::atoi(s);
        if (v > 0) warps = v;
    }
    if (warps < 1) warps = 1;
    if (warps > PREFILL_NUM_WARPS) warps = PREFILL_NUM_WARPS;
    return warps;
}

static inline int _prefill_legacy_attn_warps() {
    static int warps = -1;
    if (warps > 0) return warps;
    warps = 4;
    if (const char* s = std::getenv("MEGAQWEN_PREFILL_LEGACY_ATTN_WARPS")) {
        int v = std::atoi(s);
        if (v > 0) warps = v;
    }
    if (warps < 1) warps = 1;
    if (warps > PREFILL_NUM_WARPS) warps = PREFILL_NUM_WARPS;
    return warps;
}

static inline bool _prefill_attn_experimental_enabled() {
    const char* s = std::getenv("MEGAQWEN_PREFILL_ATTN_EXPERIMENTAL");
    if (s == nullptr || std::strcmp(s, "") == 0 || std::strcmp(s, "0") == 0 ||
        std::strcmp(s, "false") == 0 || std::strcmp(s, "False") == 0) {
        return false;
    }
    return true;
}

static inline const char* _prefill_attn_impl_name(bool use_splitk, bool use_flash, bool use_flash_ext) {
    if (use_flash_ext) return "flash_ext";
    if (use_flash) return "flash";
    if (use_splitk) return "splitk";
    return "legacy";
}

// Flash debug budget:
// - 0/false/empty: disabled
// - 1/true: print once
// - N: print first N times
// - all/inf: always
static inline int _prefill_flash_debug_budget() {
    static int budget = -999;
    if (budget != -999) return budget;
    const char* s = std::getenv("MEGAQWEN_DEBUG_PREFILL_FLASH");
    if (s == nullptr || std::strcmp(s, "") == 0 || std::strcmp(s, "0") == 0 ||
        std::strcmp(s, "false") == 0 || std::strcmp(s, "False") == 0) {
        budget = 0;
    } else if (std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0 || std::strcmp(s, "True") == 0) {
        budget = 1;
    } else if (std::strcmp(s, "all") == 0 || std::strcmp(s, "inf") == 0) {
        budget = -1;
    } else {
        int v = std::atoi(s);
        budget = (v > 0) ? v : 0;
    }
    return budget;
}

static inline int _prefill_flash_debug_skip() {
    static int skip = -999;
    if (skip != -999) return skip;
    const char* s = std::getenv("MEGAQWEN_DEBUG_PREFILL_FLASH_SKIP");
    if (s == nullptr || std::strcmp(s, "") == 0) {
        skip = 0;
    } else {
        int v = std::atoi(s);
        skip = (v > 0) ? v : 0;
    }
    return skip;
}

static inline bool _prefill_flash_debug_take_ticket() {
    static int budget = _prefill_flash_debug_budget();
    static int skip = _prefill_flash_debug_skip();
    if (skip > 0) {
        skip--;
        return false;
    }
    if (budget == 0) return false;
    if (budget > 0) budget--;
    return true;
}

// flash_ext quality knob:
// - 0/false/empty: keep BF16 Q/K/V for max speed
// - 1/true: cast Q/K/V to FP32 before SDPA for better numerical stability
static inline bool _prefill_flash_ext_force_fp32() {
    const char* s = std::getenv("MEGAQWEN_PREFILL_FLASH_EXT_FORCE_FP32");
    if (s == nullptr || std::strcmp(s, "") == 0 || std::strcmp(s, "0") == 0 ||
        std::strcmp(s, "false") == 0 || std::strcmp(s, "False") == 0) {
        return false;
    }
    return true;
}

// flash_ext hybrid knob:
// keep most layers on flash_ext, fallback tail layers to legacy attention.
// 0 means all layers use flash_ext.
static inline int _prefill_flash_ext_tail_legacy_layers() {
    static int tail = -1;
    if (tail >= 0) return tail;
    tail = 0;
    if (const char* s = std::getenv("MEGAQWEN_PREFILL_FLASH_EXT_TAIL_LEGACY_LAYERS")) {
        int v = std::atoi(s);
        if (v > 0) tail = v;
    }
    return tail;
}

struct PrefillDebugEventRec {
    int stage{-1};
    cudaEvent_t start{nullptr};
    cudaEvent_t stop{nullptr};
};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float prefill_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float prefill_warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float prefill_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =============================================================================
// Embedding lookup kernel (outputs BF16)
