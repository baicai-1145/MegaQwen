static inline bool _cublas_check(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        // Keep it simple: surface as CUDA runtime error; caller will see failure.
        printf("[MEGAQWEN_SPLIT] cuBLAS error (%s): %d\n", what, (int)st);
        return false;
    }
    return true;
}

static inline bool _split_use_cublaslt() {
    static int mode = -1;  // -1 unknown, 0 off, 1 on
    if (mode >= 0) return mode == 1;
    const char* s = std::getenv("MEGAQWEN_SPLIT_GEMM_IMPL");
    if (s == nullptr || std::strcmp(s, "") == 0) {
        mode = 1;
    } else if (std::strcmp(s, "0") == 0 ||
               std::strcmp(s, "false") == 0 ||
               std::strcmp(s, "False") == 0 ||
               std::strcmp(s, "gemmex") == 0 ||
               std::strcmp(s, "cublas") == 0) {
        mode = 0;
    } else {
        mode = 1;
    }
    return mode == 1;
}

static inline bool _split_is_sm89_like() {
    static int cached = -1;  // -1 unknown, 0 false, 1 true
    if (cached >= 0) return cached == 1;
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        cached = 0;
        return false;
    }
    cached = (prop.major == 8 && prop.minor == 9) ? 1 : 0;
    return cached == 1;
}

enum SplitLinearImpl : int {
    SPLIT_LINEAR_GEMMEX = 0,
    SPLIT_LINEAR_LT = 1,
    SPLIT_LINEAR_GEMV = 2
};

static inline int _split_parse_linear_impl_env(
    const char* env_name,
    int auto_mode
) {
    const char* s = std::getenv(env_name);
    if (s == nullptr || std::strcmp(s, "") == 0 || std::strcmp(s, "auto") == 0) {
        return auto_mode;
    }
    if (std::strcmp(s, "lt") == 0 || std::strcmp(s, "cublaslt") == 0 ||
        std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0 || std::strcmp(s, "True") == 0) {
        return SPLIT_LINEAR_LT;
    }
    if (std::strcmp(s, "gemv") == 0 || std::strcmp(s, "matvec") == 0) {
        return SPLIT_LINEAR_GEMV;
    }
    return SPLIT_LINEAR_GEMMEX;
}

static inline const char* _split_linear_impl_name(int mode) {
    if (mode == SPLIT_LINEAR_LT) return "lt";
    if (mode == SPLIT_LINEAR_GEMV) return "gemv";
    return "gemmex";
}

static inline int _split_qkv_impl() {
    // qkv_gemm is [4096 x 1024] x [1024 x 1]. Default keeps GemmEx as baseline.
    static int mode = -1;  // -1 unknown
    if (mode >= 0) return mode;
    mode = _split_parse_linear_impl_env(
        "MEGAQWEN_SPLIT_QKV_GEMM_IMPL",
        SPLIT_LINEAR_GEMMEX
    );
    return mode;
}

static inline bool _split_qkv_try_cublaslt() {
    return _split_qkv_impl() == SPLIT_LINEAR_LT;
}

static inline bool _split_qkv_use_gemv() {
    return _split_qkv_impl() == SPLIT_LINEAR_GEMV;
}

static inline const char* _split_qkv_impl_name() {
    return _split_linear_impl_name(_split_qkv_impl());
}

static inline int _split_gateup_impl() {
    // gateup_gemm can be tuned independently from qkv.
    // auto => follow global split gemm impl.
    static int mode = -1;  // -1 unknown
    if (mode >= 0) return mode;
    mode = _split_parse_linear_impl_env(
        "MEGAQWEN_SPLIT_GATEUP_GEMM_IMPL",
        _split_use_cublaslt() ? SPLIT_LINEAR_LT : SPLIT_LINEAR_GEMMEX
    );
    return mode;
}

static inline bool _split_gateup_try_cublaslt() {
    return _split_gateup_impl() == SPLIT_LINEAR_LT;
}

static inline bool _split_gateup_use_gemv() {
    return _split_gateup_impl() == SPLIT_LINEAR_GEMV;
}

static inline const char* _split_gateup_impl_name() {
    return _split_linear_impl_name(_split_gateup_impl());
}

static inline int _split_down_impl() {
    // down_gemm can be tuned independently from qkv/gateup.
    // auto => follow global split gemm impl.
    static int mode = -1;  // -1 unknown
    if (mode >= 0) return mode;
    mode = _split_parse_linear_impl_env(
        "MEGAQWEN_SPLIT_DOWN_GEMM_IMPL",
        _split_use_cublaslt() ? SPLIT_LINEAR_LT : SPLIT_LINEAR_GEMMEX
    );
    return mode;
}

static inline bool _split_down_try_cublaslt() {
    return _split_down_impl() == SPLIT_LINEAR_LT;
}

static inline bool _split_down_use_gemv() {
    return _split_down_impl() == SPLIT_LINEAR_GEMV;
}

static inline const char* _split_down_impl_name() {
    return _split_linear_impl_name(_split_down_impl());
}

static inline int _split_o_impl() {
    // o_gemm defaults to the same global rule, but still can be overridden.
    static int mode = -1;  // -1 unknown
    if (mode >= 0) return mode;
    mode = _split_parse_linear_impl_env(
        "MEGAQWEN_SPLIT_O_GEMM_IMPL",
        _split_use_cublaslt() ? SPLIT_LINEAR_LT : SPLIT_LINEAR_GEMMEX
    );
    return mode;
}

static inline bool _split_o_try_cublaslt() {
    return _split_o_impl() == SPLIT_LINEAR_LT;
}

static inline bool _split_o_use_gemv() {
    return _split_o_impl() == SPLIT_LINEAR_GEMV;
}

static inline const char* _split_o_impl_name() {
    return _split_linear_impl_name(_split_o_impl());
}

enum SplitFfnImpl : int {
    SPLIT_FFN_CUBLAS = 0,
    SPLIT_FFN_FUSED_TAIL = 1,
    SPLIT_FFN_FUSED_SILU_DOWN = 2
};

static inline int _split_ffn_impl() {
    // 0/default: original path (silu kernel + down_gemm + residual kernel)
    // 1/fused/fused_tail: fused down_proj+residual
    // 2/fused_full/fused_silu_down: fused (silu*up) + down_proj + residual
    static int mode = -1;  // -1 unknown
    if (mode >= 0) return mode;
    const char* s = std::getenv("MEGAQWEN_SPLIT_FFN_IMPL");
    if (s == nullptr || std::strcmp(s, "") == 0 ||
        std::strcmp(s, "0") == 0 ||
        std::strcmp(s, "false") == 0 ||
        std::strcmp(s, "False") == 0 ||
        std::strcmp(s, "cublas") == 0) {
        mode = SPLIT_FFN_CUBLAS;
    } else if (std::strcmp(s, "2") == 0 ||
               std::strcmp(s, "fused_full") == 0 ||
               std::strcmp(s, "fused_silu_down") == 0) {
        mode = SPLIT_FFN_FUSED_SILU_DOWN;
    } else {
        mode = SPLIT_FFN_FUSED_TAIL;
    }
    return mode;
}

static inline bool _split_ffn_fused_tail_enabled() {
    return _split_ffn_impl() != SPLIT_FFN_CUBLAS;
}

static inline const char* _split_ffn_impl_name() {
    int mode = _split_ffn_impl();
    if (mode == SPLIT_FFN_FUSED_SILU_DOWN) return "fused_silu_down";
    if (mode == SPLIT_FFN_FUSED_TAIL) return "fused_tail";
    return "cublas";
}

static inline bool _split_ffn_w4_enabled() {
    static int mode = -1;
    if (mode >= 0) return mode == 1;
    const char* s = std::getenv("MEGAQWEN_SPLIT_FFN_W4");
    if (s == nullptr || std::strcmp(s, "") == 0 ||
        std::strcmp(s, "0") == 0 ||
        std::strcmp(s, "false") == 0 ||
        std::strcmp(s, "False") == 0) {
        mode = 0;
    } else {
        mode = 1;
    }
    return mode == 1;
}

static inline bool _split_ffn_w4_fused_silu_down_enabled() {
    // default on: for W4 FFN, fuse silu+down+residual to reduce launch and DRAM traffic.
    static int mode = -1;
    if (mode >= 0) return mode == 1;
    const char* s = std::getenv("MEGAQWEN_SPLIT_FFN_W4_FUSED");
    if (s == nullptr || std::strcmp(s, "") == 0 ||
        std::strcmp(s, "1") == 0 ||
        std::strcmp(s, "true") == 0 ||
        std::strcmp(s, "True") == 0) {
        mode = 1;
    } else {
        mode = 0;
    }
    return mode == 1;
}

static inline bool _split_qkv_w4_enabled() {
    static int mode = -1;
    if (mode >= 0) return mode == 1;
    const char* s = std::getenv("MEGAQWEN_SPLIT_QKV_W4");
    if (s == nullptr || std::strcmp(s, "") == 0 ||
        std::strcmp(s, "0") == 0 ||
        std::strcmp(s, "false") == 0 ||
        std::strcmp(s, "False") == 0) {
        mode = 0;
    } else {
        mode = 1;
    }
    return mode == 1;
}

static inline bool _split_o_w4_enabled() {
    static int mode = -1;
    if (mode >= 0) return mode == 1;
    const char* s = std::getenv("MEGAQWEN_SPLIT_O_W4");
    if (s == nullptr || std::strcmp(s, "") == 0 ||
        std::strcmp(s, "0") == 0 ||
        std::strcmp(s, "false") == 0 ||
        std::strcmp(s, "False") == 0) {
        mode = 0;
    } else {
        mode = 1;
    }
    return mode == 1;
}

static inline bool _split_kv_layout_paged_enabled() {
    static int mode = -1;
    if (mode >= 0) return mode == 1;
    const char* layout = std::getenv("MEGAQWEN_SPLIT_KV_LAYOUT");
    if (layout != nullptr &&
        (std::strcmp(layout, "paged") == 0 ||
         std::strcmp(layout, "page") == 0)) {
        mode = 1;
        return true;
    }
    const char* s = std::getenv("MEGAQWEN_SPLIT_KV_PAGED");
    if (s == nullptr || std::strcmp(s, "") == 0 ||
        std::strcmp(s, "0") == 0 ||
        std::strcmp(s, "false") == 0 ||
        std::strcmp(s, "False") == 0) {
        mode = 0;
    } else {
        mode = 1;
    }
    return mode == 1;
}

static inline int _split_kv_block_size() {
    static int v = -1;
    if (v > 0) return v;
    v = 16;
    if (const char* s = std::getenv("MEGAQWEN_SPLIT_KV_BLOCK_SIZE")) {
        int p = std::atoi(s);
        if (p > 0) v = p;
    }
    return v;
}

static inline int _split_attn_impl() {
    // 0=legacy, 1=splitk(v1, one block per head), 2=splitk2(seq-split two-phase), 3=flash_decode(tile-online)
    static int mode = -1;
    if (mode >= 0) return mode;
    const char* s = std::getenv("MEGAQWEN_SPLIT_ATTN_IMPL");
    if (s == nullptr || std::strcmp(s, "") == 0) {
        mode = 2;
    } else if (std::strcmp(s, "flash_decode") == 0 || std::strcmp(s, "flashdecode") == 0 ||
               std::strcmp(s, "flash") == 0 || std::strcmp(s, "3") == 0) {
        mode = 3;
    } else if (std::strcmp(s, "splitk2") == 0 || std::strcmp(s, "v2") == 0 ||
               std::strcmp(s, "2") == 0) {
        mode = 2;
    } else if (std::strcmp(s, "splitk") == 0 || std::strcmp(s, "1") == 0 ||
               std::strcmp(s, "true") == 0 || std::strcmp(s, "True") == 0) {
        mode = 1;
    } else if (std::strcmp(s, "legacy") == 0 ||
               std::strcmp(s, "0") == 0 ||
               std::strcmp(s, "false") == 0 ||
               std::strcmp(s, "False") == 0) {
        mode = 0;
    } else {
        mode = 2;
    }
    return mode;
}

static inline const char* _split_attn_impl_name() {
    int impl = _split_attn_impl();
    if (impl == 0) return "legacy";
    if (impl == 1) return "splitk";
    if (impl == 3) return "flash_decode";
    return "splitk2";
}

static inline int _split_attn_warps_per_head() {
    static int warps = -1;
    if (warps > 0) return warps;
    const char* s = std::getenv("MEGAQWEN_SPLIT_ATTN_WARPS");
    int v = 8;
    if (s != nullptr && std::strcmp(s, "") != 0) {
        int parsed = std::atoi(s);
        if (parsed > 0) v = parsed;
    }
    if (v < 1) v = 1;
    if (v > 8) v = 8;
    warps = v;
    return warps;
}

static inline int _split_flash_attn_warps_per_head() {
    static int warps = -1;
    if (warps > 0) return warps;
    int v = 4;  // flash-decode on AD102 usually prefers fewer warps than splitk2.
    const char* s = std::getenv("MEGAQWEN_SPLIT_FLASH_WARPS");
    if (s != nullptr && std::strcmp(s, "") != 0) {
        int parsed = std::atoi(s);
        if (parsed > 0) v = parsed;
    } else {
        const char* s_attn = std::getenv("MEGAQWEN_SPLIT_ATTN_WARPS");
        if (s_attn != nullptr && std::strcmp(s_attn, "") != 0) {
            int parsed = std::atoi(s_attn);
            if (parsed > 0) v = parsed;
        }
    }
    if (v < 1) v = 1;
    if (v > 8) v = 8;
    warps = v;
    return warps;
}

static inline int _split_debug_budget() {
    static int budget = -999;  // uninitialized
    if (budget != -999) return budget;
    const char* s = std::getenv("MEGAQWEN_DEBUG_SPLIT_STAGE");
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

static inline int _split_debug_skip_budget() {
    static int skip = -999;
    if (skip != -999) return skip;
    const char* s = std::getenv("MEGAQWEN_DEBUG_SPLIT_STAGE_SKIP");
    if (s == nullptr || std::strcmp(s, "") == 0) {
        skip = 0;
    } else {
        int v = std::atoi(s);
        skip = (v > 0) ? v : 0;
    }
    return skip;
}

static inline bool _split_debug_take_ticket() {
    static int budget = _split_debug_budget();
    static int skip = _split_debug_skip_budget();
    if (skip > 0) {
        skip--;
        return false;
    }
    if (budget == 0) return false;
    if (budget > 0) budget--;
    return true;
}

static inline bool _split_debug_avg_enabled() {
    const char* s = std::getenv("MEGAQWEN_DEBUG_SPLIT_STAGE_AVG");
    if (s == nullptr || std::strcmp(s, "") == 0 ||
        std::strcmp(s, "0") == 0 ||
        std::strcmp(s, "false") == 0 ||
        std::strcmp(s, "False") == 0) {
        return false;
    }
    return true;
}

static inline bool _split_debug_avg_exact() {
    const char* s = std::getenv("MEGAQWEN_DEBUG_SPLIT_STAGE_AVG_EXACT");
    if (s == nullptr || std::strcmp(s, "") == 0 ||
        std::strcmp(s, "0") == 0 ||
        std::strcmp(s, "false") == 0 ||
        std::strcmp(s, "False") == 0) {
        return false;
    }
    return true;
}

static inline int _split_debug_avg_stride() {
    static int stride = -1;
    if (stride > 0) return stride;
    const char* s = std::getenv("MEGAQWEN_DEBUG_SPLIT_STAGE_AVG_STRIDE");
    int v = 16;
    if (s != nullptr && std::strcmp(s, "") != 0) {
        int parsed = std::atoi(s);
        if (parsed > 0) v = parsed;
    }
    if (v < 1) v = 1;
    if (v > 256) v = 256;
    stride = v;
    return stride;
}

static inline int _split_debug_layer_topk() {
    static int topk = -1;
    if (topk > 0) return topk;
    const char* s = std::getenv("MEGAQWEN_DEBUG_SPLIT_LAYER_TOPK");
    int v = 5;
    if (s != nullptr && std::strcmp(s, "") != 0) {
        int parsed = std::atoi(s);
        if (parsed > 0) v = parsed;
    }
    if (v < 1) v = 1;
    if (v > 16) v = 16;
    topk = v;
    return topk;
}

static inline bool _split_debug_paged_kv_enabled() {
    static int mode = -1;
    if (mode >= 0) return mode == 1;
    const char* s = std::getenv("MEGAQWEN_DEBUG_PAGED_KV");
    if (s == nullptr || std::strcmp(s, "") == 0 ||
        std::strcmp(s, "0") == 0 ||
        std::strcmp(s, "false") == 0 ||
        std::strcmp(s, "False") == 0) {
        mode = 0;
    } else {
        mode = 1;
    }
    return mode == 1;
}

static inline int _split_debug_paged_kv_tokens() {
    static int v = -1;
    if (v >= 0) return v;
    v = 8;
    const char* s = std::getenv("MEGAQWEN_DEBUG_PAGED_KV_TOKENS");
    if (s == nullptr || std::strcmp(s, "") == 0) return v;
    if (std::strcmp(s, "all") == 0) {
        v = 1000000000;
        return v;
    }
    int p = std::atoi(s);
    if (p >= 0) v = p;
    return v;
}

static inline bool _split_debug_paged_kv_take_ticket() {
    if (!_split_debug_paged_kv_enabled()) return false;
    static long long counter = 0;
    long long idx = counter++;
    return idx < static_cast<long long>(_split_debug_paged_kv_tokens());
}

static inline bool _split_debug_flash_decode_enabled() {
    static int mode = -1;
    if (mode >= 0) return mode == 1;
    const char* s = std::getenv("MEGAQWEN_DEBUG_FLASH_DECODE");
    if (s == nullptr || std::strcmp(s, "") == 0 ||
        std::strcmp(s, "0") == 0 ||
        std::strcmp(s, "false") == 0 ||
        std::strcmp(s, "False") == 0) {
        mode = 0;
    } else {
        mode = 1;
    }
    return mode == 1;
}

static inline int _split_debug_flash_decode_tokens() {
    static int v = -1;
    if (v >= 0) return v;
    v = 1;
    const char* s = std::getenv("MEGAQWEN_DEBUG_FLASH_TOKENS");
    if (s == nullptr || std::strcmp(s, "") == 0) return v;
    if (std::strcmp(s, "all") == 0) {
        v = 1000000000;
        return v;
    }
    int p = std::atoi(s);
    if (p >= 0) v = p;
    return v;
}

static inline bool _split_debug_flash_decode_take_ticket() {
    if (!_split_debug_flash_decode_enabled()) return false;
    static long long counter = 0;
    long long idx = counter++;
    return idx < static_cast<long long>(_split_debug_flash_decode_tokens());
}

static inline int _split_debug_flash_head_topk() {
    static int v = -1;
    if (v > 0) return v;
    v = 8;
    const char* s = std::getenv("MEGAQWEN_DEBUG_FLASH_HEAD_TOPK");
    if (s != nullptr && std::strcmp(s, "") != 0) {
        int p = std::atoi(s);
        if (p > 0) v = p;
    }
    if (v < 1) v = 1;
    if (v > 16) v = 16;
    return v;
}

enum SplitStageId : int {
    SPLIT_STAGE_EMBED = 0,
    SPLIT_STAGE_RMS1 = 1,
    SPLIT_STAGE_QKV_GEMM = 2,
    SPLIT_STAGE_QK_ROPE_CACHE = 3,
    SPLIT_STAGE_ATTN = 4,
    SPLIT_STAGE_O_GEMM = 5,
    SPLIT_STAGE_RESID1 = 6,
    SPLIT_STAGE_RMS2 = 7,
    SPLIT_STAGE_GATEUP_GEMM = 8,
    SPLIT_STAGE_SILU_MUL = 9,
    SPLIT_STAGE_DOWN_GEMM = 10,
    SPLIT_STAGE_RESID2 = 11,
    SPLIT_STAGE_FINAL_NORM = 12,
    SPLIT_STAGE_LM_PHASE1 = 13,
    SPLIT_STAGE_LM_PHASE2 = 14,
    SPLIT_STAGE_COUNT = 15
};

static const char* kSplitStageNames[SPLIT_STAGE_COUNT] = {
    "embed_lookup",
    "rmsnorm_1",
    "qkv_gemm",
    "qk_rope_cache",
    "attn_cache",
    "o_gemm",
    "residual_1",
    "rmsnorm_2",
    "gateup_gemm",
    "silu_mul",
    "down_gemm",
    "residual_2",
    "final_norm",
    "lm_head_phase1",
    "lm_head_phase2"
};

constexpr int SPLIT_DEBUG_MAX_LAYERS = 96;

struct SplitDebugAgg {
    double stage_ms[SPLIT_STAGE_COUNT];
    long long stage_calls[SPLIT_STAGE_COUNT];
    double stage_layer_ms[SPLIT_STAGE_COUNT][SPLIT_DEBUG_MAX_LAYERS];
    long long stage_layer_calls[SPLIT_STAGE_COUNT][SPLIT_DEBUG_MAX_LAYERS];
    double total_ms;
    long long tokens;
    long long sampled_tokens;
    long long sample_stride;
    long long max_layer_seen;
    long long qkv_lt_ok;
    long long qkv_lt_fb;
    long long qkv_gemmex;
    long long qkv_gemv;
    long long o_lt_ok;
    long long o_lt_fb;
    long long o_gemmex;
    long long o_gemv;
    long long gateup_lt_ok;
    long long gateup_lt_fb;
    long long gateup_gemmex;
    long long gateup_gemv;
    long long down_lt_ok;
    long long down_lt_fb;
    long long down_gemmex;
    long long down_gemv;
    long long down_fused_tail;
    long long down_fused_silu;
    long long down_fused_silu_w4;
    long long qkv_w4;
    long long o_w4;
    long long gateup_w4;
    long long down_w4;
};

static inline SplitDebugAgg& _split_debug_agg() {
    static SplitDebugAgg agg{};
    return agg;
}

extern "C" void split_debug_stage_reset() {
    auto& agg = _split_debug_agg();
    std::memset(&agg, 0, sizeof(agg));
    agg.sample_stride = static_cast<long long>(_split_debug_avg_stride());
}

extern "C" void split_debug_stage_print_summary() {
    if (!_split_debug_avg_enabled()) return;
    auto& agg = _split_debug_agg();
    if (agg.tokens <= 0 || agg.total_ms <= 0.0) return;

    bool sampled = (agg.sampled_tokens > 0 && agg.sampled_tokens < agg.tokens);
    if (sampled) {
        printf("[MEGAQWEN_DEBUG] SPLIT stage timing (sampled avg over decode tokens) tokens=%lld sampled=%lld stride=%lld\n",
               agg.tokens, agg.sampled_tokens, (agg.sample_stride > 0 ? agg.sample_stride : 1));
    } else {
        printf("[MEGAQWEN_DEBUG] SPLIT stage timing (avg over all decode tokens) tokens=%lld\n", agg.tokens);
    }
    double denom_tokens = (agg.sampled_tokens > 0) ? double(agg.sampled_tokens) : double(agg.tokens);
    bool paged_focus = _split_debug_paged_kv_enabled();
    for (int s = 0; s < SPLIT_STAGE_COUNT; s++) {
        if (paged_focus && (s == SPLIT_STAGE_GATEUP_GEMM || s == SPLIT_STAGE_DOWN_GEMM)) continue;
        if (agg.stage_calls[s] <= 0) continue;
        double total = agg.stage_ms[s];
        double avg_token = total / denom_tokens;
        double avg_call = total / double(agg.stage_calls[s]);
        double pct = (agg.total_ms > 0.0) ? (total * 100.0 / agg.total_ms) : 0.0;
        printf("  %-16s total=%8.3f ms  avg/token=%7.4f ms  avg/call=%7.4f ms  calls=%5lld  (%5.1f%%)\n",
               kSplitStageNames[s], total, avg_token, avg_call, agg.stage_calls[s], pct);
    }
    printf("  %-16s total=%8.3f ms  avg/token=%7.4f ms\n",
           "decode_step_sum", agg.total_ms, agg.total_ms / denom_tokens);
    if (sampled) {
        printf("[MEGAQWEN_DEBUG] SPLIT note: avg/token values above are sampled estimates.\n");
    }
    printf("[MEGAQWEN_DEBUG] SPLIT gemm route summary\n");
    printf("  qkv    impl=%-6s lt_ok=%5lld lt_fb=%5lld gemmex=%5lld\n",
           _split_qkv_impl_name(), agg.qkv_lt_ok, agg.qkv_lt_fb, agg.qkv_gemmex);
    printf("         gemv=%5lld\n", agg.qkv_gemv);
    printf("  o      impl=%-6s lt_ok=%5lld lt_fb=%5lld gemmex=%5lld\n",
           _split_o_impl_name(), agg.o_lt_ok, agg.o_lt_fb, agg.o_gemmex);
    printf("         gemv=%5lld\n", agg.o_gemv);
    printf("  gateup impl=%-6s lt_ok=%5lld lt_fb=%5lld gemmex=%5lld\n",
           _split_gateup_impl_name(), agg.gateup_lt_ok, agg.gateup_lt_fb, agg.gateup_gemmex);
    printf("         gemv=%5lld\n", agg.gateup_gemv);
    printf("  down   impl=%-6s lt_ok=%5lld lt_fb=%5lld gemmex=%5lld\n",
           _split_down_impl_name(), agg.down_lt_ok, agg.down_lt_fb, agg.down_gemmex);
    printf("         gemv=%5lld\n", agg.down_gemv);
    printf("  ffn    impl=%-16s down_fused_tail=%5lld down_fused_silu=%5lld down_fused_silu_w4=%5lld\n",
           _split_ffn_impl_name(), agg.down_fused_tail, agg.down_fused_silu, agg.down_fused_silu_w4);
    printf("         kv_layout=%s kv_block=%d qkv_w4_enabled=%d qkv_w4=%5lld o_w4_enabled=%d o_w4=%5lld ffn_w4_enabled=%d ffn_w4_fused=%d gateup_w4=%5lld down_w4=%5lld\n",
           _split_kv_layout_paged_enabled() ? "paged" : "contiguous",
           _split_kv_block_size(),
           _split_qkv_w4_enabled() ? 1 : 0,
           agg.qkv_w4,
           _split_o_w4_enabled() ? 1 : 0,
           agg.o_w4,
           _split_ffn_w4_enabled() ? 1 : 0,
           _split_ffn_w4_fused_silu_down_enabled() ? 1 : 0,
           agg.gateup_w4,
           agg.down_w4);

    int layer_n = (int)agg.max_layer_seen;
    if (layer_n > SPLIT_DEBUG_MAX_LAYERS) layer_n = SPLIT_DEBUG_MAX_LAYERS;
    if (layer_n > 0) {
        auto print_stage_topk = [&](int stage, const char* name) {
            struct Pair { double v; int i; };
            Pair buf[SPLIT_DEBUG_MAX_LAYERS];
            int n = 0;
            for (int l = 0; l < layer_n; l++) {
                long long calls = agg.stage_layer_calls[stage][l];
                if (calls <= 0) continue;
                buf[n++] = Pair{agg.stage_layer_ms[stage][l] / double(calls), l};
            }
            if (n <= 0) return;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    if (buf[j].v > buf[i].v) {
                        Pair t = buf[i];
                        buf[i] = buf[j];
                        buf[j] = t;
                    }
                }
            }
            int topk = _split_debug_layer_topk();
            if (topk > n) topk = n;
            printf("[MEGAQWEN_DEBUG] SPLIT layer hotspot (%s, avg/call ms):", name);
            for (int i = 0; i < topk; i++) {
                printf(" L%02d=%.4f", buf[i].i, buf[i].v);
            }
            printf("\n");
        };

        print_stage_topk(SPLIT_STAGE_QKV_GEMM, "qkv_gemm");
        print_stage_topk(SPLIT_STAGE_ATTN, "attn_cache");
        if (!_split_debug_paged_kv_enabled()) {
            print_stage_topk(SPLIT_STAGE_GATEUP_GEMM, "gateup_gemm");
            print_stage_topk(SPLIT_STAGE_DOWN_GEMM, "down_gemm");
        }
    }
}

struct SplitDebugEventRec {
    int stage{-1};
    int layer{-1};
    cudaEvent_t start{nullptr};
    cudaEvent_t stop{nullptr};
};

struct SplitLtMatmulPlan {
    bool initialized{false};
    bool has_algo{false};
    int m{0};
    int k{0};
    cublasLtMatmulDesc_t op_desc{nullptr};
    cublasLtMatrixLayout_t a_desc{nullptr};
    cublasLtMatrixLayout_t b_desc{nullptr};
    cublasLtMatrixLayout_t c_desc{nullptr};
    cublasLtMatrixLayout_t d_desc{nullptr};
    cublasLtMatmulAlgo_t algo{};
};

static inline bool _split_lt_init_plan(
    cublasLtHandle_t lt_handle,
    SplitLtMatmulPlan& plan,
    int m,
    int k,
    size_t workspace_bytes
) {
    if (plan.initialized) return plan.has_algo;

    plan.m = m;
    plan.k = k;
    plan.initialized = true;
    plan.has_algo = false;

    if (lt_handle == nullptr) return false;

    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;

    if (cublasLtMatmulDescCreate(&plan.op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) goto fail;
    if (cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)) != CUBLAS_STATUS_SUCCESS) goto fail;
    if (cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)) != CUBLAS_STATUS_SUCCESS) goto fail;

    if (cublasLtMatrixLayoutCreate(&plan.a_desc, CUDA_R_16BF, m, k, k) != CUBLAS_STATUS_SUCCESS) goto fail;
    if (cublasLtMatrixLayoutCreate(&plan.b_desc, CUDA_R_16BF, k, 1, 1) != CUBLAS_STATUS_SUCCESS) goto fail;
    if (cublasLtMatrixLayoutCreate(&plan.c_desc, CUDA_R_16BF, m, 1, 1) != CUBLAS_STATUS_SUCCESS) goto fail;
    if (cublasLtMatrixLayoutCreate(&plan.d_desc, CUDA_R_16BF, m, 1, 1) != CUBLAS_STATUS_SUCCESS) goto fail;

    if (cublasLtMatrixLayoutSetAttribute(plan.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)) != CUBLAS_STATUS_SUCCESS) goto fail;
    if (cublasLtMatrixLayoutSetAttribute(plan.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)) != CUBLAS_STATUS_SUCCESS) goto fail;
    if (cublasLtMatrixLayoutSetAttribute(plan.c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)) != CUBLAS_STATUS_SUCCESS) goto fail;
    if (cublasLtMatrixLayoutSetAttribute(plan.d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)) != CUBLAS_STATUS_SUCCESS) goto fail;

    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) goto fail;
    if (cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)) != CUBLAS_STATUS_SUCCESS) goto fail;

    if (cublasLtMatmulAlgoGetHeuristic(
            lt_handle,
            plan.op_desc,
            plan.a_desc,
            plan.b_desc,
            plan.c_desc,
            plan.d_desc,
            pref,
            1,
            &heuristic,
            &returned
        ) != CUBLAS_STATUS_SUCCESS) {
        returned = 0;
    }

    if (pref != nullptr) {
        cublasLtMatmulPreferenceDestroy(pref);
        pref = nullptr;
    }

    if (returned <= 0) goto fail;
    plan.algo = heuristic.algo;
    plan.has_algo = true;
    return true;

fail:
    if (pref != nullptr) cublasLtMatmulPreferenceDestroy(pref);
    if (plan.a_desc != nullptr) { cublasLtMatrixLayoutDestroy(plan.a_desc); plan.a_desc = nullptr; }
    if (plan.b_desc != nullptr) { cublasLtMatrixLayoutDestroy(plan.b_desc); plan.b_desc = nullptr; }
    if (plan.c_desc != nullptr) { cublasLtMatrixLayoutDestroy(plan.c_desc); plan.c_desc = nullptr; }
    if (plan.d_desc != nullptr) { cublasLtMatrixLayoutDestroy(plan.d_desc); plan.d_desc = nullptr; }
    if (plan.op_desc != nullptr) { cublasLtMatmulDescDestroy(plan.op_desc); plan.op_desc = nullptr; }
    plan.has_algo = false;
    return false;
}

static inline bool _split_lt_matmul(
    cublasLtHandle_t lt_handle,
    SplitLtMatmulPlan& plan,
    int m,
    int k,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* c,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream
) {
    if (!_split_use_cublaslt()) return false;
    if (!_split_lt_init_plan(lt_handle, plan, m, k, workspace_bytes)) return false;
    if (!plan.has_algo) return false;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasLtMatmul(
        lt_handle,
        plan.op_desc,
        &alpha,
        a, plan.a_desc,
        b, plan.b_desc,
        &beta,
        c, plan.c_desc,
        c, plan.d_desc,
        &plan.algo,
        workspace,
        workspace_bytes,
        stream
    );
    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("[MEGAQWEN_SPLIT] cuBLASLt matmul failed (m=%d,k=%d): %d, fallback to cublasGemmEx\n", m, k, (int)st);
        return false;
    }
    return true;
}

static inline bool _split_try_gemv_matvec(
    cublasHandle_t cublas_handle,
    const __nv_bfloat16* weight_row_major,
    int out_rows,
    int in_cols,
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    const char* tag
) {
    // GEMV-compat route:
    // some CUDA toolchains in this repo environment do not expose cublasGemvEx
    // for BF16, so we keep a matvec-specialized GemmEx variant as fallback.
    // It differs from the default path by using CUBLAS_GEMM_DEFAULT (no tensor-op hint).
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out_rows,
        1,
        in_cols,
        &alpha,
        weight_row_major,
        CUDA_R_16BF,
        in_cols,
        x,
        CUDA_R_16BF,
        1,
        &beta,
        y,
        CUDA_R_16BF,
        out_rows,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    if (st != CUBLAS_STATUS_SUCCESS) {
        static int warned = 0;
        if (warned < 4) {
            warned += 1;
            printf("[MEGAQWEN_SPLIT] cuBLAS GEMV failed (%s): %d, fallback to GemmEx\n", tag, (int)st);
        }
        return false;
    }
    return true;
}
