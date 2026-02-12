import os
import json
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

_decode_kernel_cache: dict[tuple[int, ...], object] = {}
_prefill_kernel_cache: dict[tuple[int, ...], object] = {}
_fused_prefill_kernel_cache: dict[tuple[int, ...], object] = {}
_KERNEL_ABI_VERSION = 2

_DEFAULT_MODEL_SPEC = {
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_q_heads": 16,
    "num_kv_heads": 8,
    "head_dim": 128,
    "num_layers": 28,
    "vocab_size": 151936,
}
_ACTIVE_MODEL_SPEC = dict(_DEFAULT_MODEL_SPEC)


def _spec_key(spec: dict[str, int]) -> tuple[int, ...]:
    return (
        int(spec["hidden_size"]),
        int(spec["intermediate_size"]),
        int(spec["num_q_heads"]),
        int(spec["num_kv_heads"]),
        int(spec["head_dim"]),
        int(spec["num_layers"]),
        int(spec["vocab_size"]),
    )


def _sync_compat_constants() -> None:
    global HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, Q_SIZE, KV_SIZE, NUM_LAYERS, VOCAB_SIZE
    HIDDEN_SIZE = int(_ACTIVE_MODEL_SPEC["hidden_size"])
    INTERMEDIATE_SIZE = int(_ACTIVE_MODEL_SPEC["intermediate_size"])
    NUM_Q_HEADS = int(_ACTIVE_MODEL_SPEC["num_q_heads"])
    NUM_KV_HEADS = int(_ACTIVE_MODEL_SPEC["num_kv_heads"])
    HEAD_DIM = int(_ACTIVE_MODEL_SPEC["head_dim"])
    Q_SIZE = NUM_Q_HEADS * HEAD_DIM
    KV_SIZE = NUM_KV_HEADS * HEAD_DIM
    NUM_LAYERS = int(_ACTIVE_MODEL_SPEC["num_layers"])
    VOCAB_SIZE = int(_ACTIVE_MODEL_SPEC["vocab_size"])


def _normalize_model_spec(spec: dict[str, int]) -> dict[str, int]:
    out = {
        "hidden_size": int(spec["hidden_size"]),
        "intermediate_size": int(spec["intermediate_size"]),
        "num_q_heads": int(spec["num_q_heads"]),
        "num_kv_heads": int(spec["num_kv_heads"]),
        "head_dim": int(spec["head_dim"]),
        "num_layers": int(spec["num_layers"]),
        "vocab_size": int(spec["vocab_size"]),
    }
    if out["hidden_size"] <= 0 or out["intermediate_size"] <= 0:
        raise ValueError(f"Invalid model spec: {out}")
    if out["num_q_heads"] <= 0 or out["num_kv_heads"] <= 0 or out["head_dim"] <= 0:
        raise ValueError(f"Invalid attention spec: {out}")
    if out["num_q_heads"] % out["num_kv_heads"] != 0:
        raise ValueError(f"num_q_heads must be divisible by num_kv_heads: {out}")
    # NOTE:
    # For Qwen3-ASR-0.6B, hidden_size=1024 while num_q_heads*head_dim=2048.
    # This is valid because q_proj maps [hidden_size] -> [num_q_heads*head_dim].
    # So we intentionally do NOT require hidden_size == num_q_heads * head_dim.
    if out["num_layers"] <= 0 or out["vocab_size"] <= 0:
        raise ValueError(f"Invalid model spec: {out}")
    return out


def get_active_model_spec() -> dict[str, int]:
    return dict(_ACTIVE_MODEL_SPEC)


def configure_model_spec(*, hidden_size: int, intermediate_size: int, num_q_heads: int, num_kv_heads: int, head_dim: int, num_layers: int, vocab_size: int) -> dict[str, int]:
    global _ACTIVE_MODEL_SPEC
    _ACTIVE_MODEL_SPEC = _normalize_model_spec(
        {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "num_layers": num_layers,
            "vocab_size": vocab_size,
        }
    )
    _sync_compat_constants()
    return get_active_model_spec()


def configure_model_spec_from_model_dir(model_dir: str | os.PathLike[str]) -> dict[str, int]:
    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    thinker = cfg.get("thinker_config", {})
    text_cfg = thinker.get("text_config", {})
    return configure_model_spec(
        hidden_size=int(text_cfg["hidden_size"]),
        intermediate_size=int(text_cfg["intermediate_size"]),
        num_q_heads=int(text_cfg["num_attention_heads"]),
        num_kv_heads=int(text_cfg["num_key_value_heads"]),
        head_dim=int(text_cfg["head_dim"]),
        num_layers=int(text_cfg["num_hidden_layers"]),
        vocab_size=int(text_cfg.get("vocab_size", cfg.get("vocab_size", _DEFAULT_MODEL_SPEC["vocab_size"]))),
    )


def _cuda_arch_flag(default: str = "-arch=sm_86") -> str:
    arch_env = os.environ.get("MEGAQWEN_CUDA_ARCH", "").strip()
    if arch_env:
        if arch_env.startswith("-arch="):
            return arch_env
        if arch_env.startswith("sm_"):
            return f"-arch={arch_env}"
        digits = arch_env.replace(".", "")
        if digits.isdigit():
            return f"-arch=sm_{digits}"
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"-arch=sm_{major}{minor}"
    return default


def _kernel_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _get_cuda_source(filename: str) -> str:
    with open(os.path.join(_kernel_dir(), filename), encoding="utf-8") as f:
        return f.read()


def _get_cpp_source(filename: str) -> str:
    with open(os.path.join(_kernel_dir(), filename), encoding="utf-8") as f:
        return f.read()


def _spec_define_flags(spec: dict[str, int]) -> list[str]:
    return [
        f"-DMEGAQWEN_HIDDEN_SIZE={int(spec['hidden_size'])}",
        f"-DMEGAQWEN_INTERMEDIATE_SIZE={int(spec['intermediate_size'])}",
        f"-DMEGAQWEN_NUM_Q_HEADS={int(spec['num_q_heads'])}",
        f"-DMEGAQWEN_NUM_KV_HEADS={int(spec['num_kv_heads'])}",
        f"-DMEGAQWEN_HEAD_DIM={int(spec['head_dim'])}",
        f"-DMEGAQWEN_NUM_LAYERS={int(spec['num_layers'])}",
        f"-DMEGAQWEN_VOCAB_SIZE={int(spec['vocab_size'])}",
    ]


def _extra_cflags(spec: dict[str, int]) -> list[str]:
    flags: list[str] = [
        "-O3",
        "-std=c++17",
    ]
    flags.extend(_spec_define_flags(spec))
    return flags


def _extra_cuda_cflags(kernel_dir: str, spec: dict[str, int]) -> tuple[list[str], bool]:
    flags: list[str] = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        _cuda_arch_flag(),
        "--expt-relaxed-constexpr",
        "-I" + kernel_dir,
    ]
    flags.extend(_spec_define_flags(spec))

    auto_ldg_unroll: int | None = None
    auto_ldg_min_blocks: int | None = None
    auto_maxrregcount: int | None = None
    if torch.cuda.is_available():
        try:
            prop = torch.cuda.get_device_properties(0)
            if int(prop.major) == 8 and int(prop.minor) == 9 and int(getattr(prop, "multi_processor_count", 0)) >= 96:
                auto_ldg_unroll = 4
                auto_ldg_min_blocks = 3
                auto_maxrregcount = 80
        except Exception:
            pass

    if (s := os.environ.get("MEGAQWEN_MAXRREGCOUNT")):
        flags.append(f"--maxrregcount={int(s)}")
    elif auto_maxrregcount is not None:
        flags.append(f"--maxrregcount={auto_maxrregcount}")

    if (s := os.environ.get("MEGAQWEN_LDG_UNROLL")):
        flags.append(f"-DMEGAQWEN_LDG_UNROLL={int(s)}")
    elif auto_ldg_unroll is not None:
        flags.append(f"-DMEGAQWEN_LDG_UNROLL={auto_ldg_unroll}")

    if (s := os.environ.get("MEGAQWEN_LDG_MIN_BLOCKS_PER_SM")):
        flags.append(f"-DMEGAQWEN_LDG_MIN_BLOCKS_PER_SM={int(s)}")
    elif auto_ldg_min_blocks is not None:
        flags.append(f"-DMEGAQWEN_LDG_MIN_BLOCKS_PER_SM={auto_ldg_min_blocks}")

    verbose = False
    if os.environ.get("MEGAQWEN_PTXAS_VERBOSE"):
        flags.append("-Xptxas=-v")
        verbose = True
    return flags, verbose


def _kernel_module_name(prefix: str, spec: dict[str, int]) -> str:
    return (
        f"{prefix}_abi{_KERNEL_ABI_VERSION}_h{int(spec['hidden_size'])}"
        f"_i{int(spec['intermediate_size'])}"
        f"_q{int(spec['num_q_heads'])}"
        f"_kv{int(spec['num_kv_heads'])}"
        f"_d{int(spec['head_dim'])}"
        f"_l{int(spec['num_layers'])}"
    )


def _compile_decode_kernel():
    spec = get_active_model_spec()
    key = _spec_key(spec)
    if key in _decode_kernel_cache:
        return _decode_kernel_cache[key]
    kernel_dir = _kernel_dir()
    extra_cuda_cflags, verbose = _extra_cuda_cflags(kernel_dir, spec)
    extra_cflags = _extra_cflags(spec)
    kernel = load_inline(
        name=_kernel_module_name("megakernel_decode", spec),
        cpp_sources=[_get_cpp_source("bindings_decode.cpp")],
        cuda_sources=[_get_cuda_source("fused_decode_ldg.cu")],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )
    _decode_kernel_cache[key] = kernel
    return kernel


def _compile_prefill_kernel():
    spec = get_active_model_spec()
    key = _spec_key(spec)
    if key in _prefill_kernel_cache:
        return _prefill_kernel_cache[key]
    kernel_dir = _kernel_dir()
    extra_cuda_cflags, verbose = _extra_cuda_cflags(kernel_dir, spec)
    extra_cflags = _extra_cflags(spec)
    kernel = load_inline(
        name=_kernel_module_name("megakernel_prefill", spec),
        cpp_sources=[_get_cpp_source("bindings_prefill.cpp")],
        cuda_sources=[
            _get_cuda_source("fused_prefill.cu"),
            _get_cuda_source("fused_prefill_megakernel.cu"),
            _get_cuda_source("fused_decode_ldg.cu"),
            _get_cuda_source("split_decode_gemm.cu"),
        ],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=["-lcublas", "-lcublasLt"],
        verbose=verbose,
    )
    _prefill_kernel_cache[key] = kernel
    return kernel


def _compile_fused_prefill_kernel():
    spec = get_active_model_spec()
    key = _spec_key(spec)
    if key in _fused_prefill_kernel_cache:
        return _fused_prefill_kernel_cache[key]
    kernel_dir = _kernel_dir()
    extra_cuda_cflags, verbose = _extra_cuda_cflags(kernel_dir, spec)
    extra_cflags = _extra_cflags(spec)
    kernel = load_inline(
        name=_kernel_module_name("megakernel_fused_prefill", spec),
        cpp_sources=[_get_cpp_source("bindings_fused_prefill.cpp")],
        cuda_sources=[_get_cuda_source("fused_prefill_megakernel.cu"), _get_cuda_source("fused_decode_ldg.cu")],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )
    _fused_prefill_kernel_cache[key] = kernel
    return kernel


_sync_compat_constants()
