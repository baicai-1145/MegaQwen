/**
 * Fused Prefill Megakernel for Qwen3-0.6B
 *
 * Strategy: Replace cuBLAS-based prefill with a fully fused megakernel that
 * processes multiple tokens in parallel using cooperative groups.
 *
 * For seq_len 4-64 tokens, treat multi-token projection as batched GEMV:
 * - Each output element is an independent dot product
 * - Parallelize across both tokens AND output rows
 * - Reuses proven decode kernel patterns (vec4 loads, warp reduction)
 *
 * Configuration:
 * - PREFILL_MK_NUM_BLOCKS = 82 (one per SM)
 * - PREFILL_MK_BLOCK_SIZE = 256 (8 warps)
 * - MAX_PREFILL_SEQ_LEN = 64
 */

#include "config.cuh"
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cg = cooperative_groups;

// =============================================================================
// Configuration
// =============================================================================

constexpr int PREFILL_MK_NUM_BLOCKS = 82;
constexpr int PREFILL_MK_BLOCK_SIZE = 256;
constexpr int PREFILL_MK_NUM_WARPS = PREFILL_MK_BLOCK_SIZE / WARP_SIZE;
constexpr int MAX_PREFILL_SEQ_LEN = 64;
constexpr float PREFILL_MK_RMS_EPS = 1e-6f;

// LM head configuration
constexpr int PREFILL_MK_LM_NUM_BLOCKS = 1184;
constexpr int PREFILL_MK_LM_BLOCK_SIZE = 256;
constexpr int PREFILL_MK_VOCAB_SIZE = 151936;

// Shared memory layout for caching normalized inputs
// Cache up to 8 tokens at a time for batched operations
constexpr int PREFILL_MK_TOKEN_CACHE = 8;

struct PrefillMKLayerWeights {
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

enum PrefillMKStageId : int {
    PREFILL_MK_STAGE_CORE = 0,
    PREFILL_MK_STAGE_LM1 = 1,
    PREFILL_MK_STAGE_LM2 = 2,
    PREFILL_MK_STAGE_COUNT = 3
};

static const char* kPrefillMKStageNames[PREFILL_MK_STAGE_COUNT] = {
    "prefill_megakernel",
    "lm_head_phase1",
    "lm_head_phase2"
};

static inline int _prefill_mk_debug_budget() {
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

static inline int _prefill_mk_debug_skip() {
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

static inline bool _prefill_mk_debug_take_ticket() {
    static int budget = _prefill_mk_debug_budget();
    static int skip = _prefill_mk_debug_skip();
    if (skip > 0) {
        skip--;
        return false;
    }
    if (budget == 0) return false;
    if (budget > 0) budget--;
    return true;
}

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float prefill_mk_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float prefill_mk_warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float prefill_mk_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =============================================================================
// Phase 1: Embedding Lookup (distributed across grid)
