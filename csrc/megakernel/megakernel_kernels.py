import os

import torch
from torch.utils.cpp_extension import load_inline

_decode_kernel = None
_prefill_kernel = None
_fused_prefill_kernel = None

HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
Q_SIZE = NUM_Q_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
NUM_LAYERS = 28
VOCAB_SIZE = 151936


def _cuda_arch_flag(default: str = "-arch=sm_86") -> str:
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


def _extra_cuda_cflags(kernel_dir: str) -> tuple[list[str], bool]:
    flags: list[str] = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        _cuda_arch_flag(),
        "--expt-relaxed-constexpr",
        "-I" + kernel_dir,
    ]

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


def _compile_decode_kernel():
    global _decode_kernel
    if _decode_kernel is not None:
        return _decode_kernel

    kernel_dir = _kernel_dir()
    extra_cuda_cflags, verbose = _extra_cuda_cflags(kernel_dir)
    _decode_kernel = load_inline(
        name="megakernel_decode",
        cpp_sources=[_get_cpp_source("bindings_decode.cpp")],
        cuda_sources=[_get_cuda_source("fused_decode_ldg.cu")],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )
    return _decode_kernel


def _compile_prefill_kernel():
    global _prefill_kernel
    if _prefill_kernel is not None:
        return _prefill_kernel

    kernel_dir = _kernel_dir()
    extra_cuda_cflags, verbose = _extra_cuda_cflags(kernel_dir)
    _prefill_kernel = load_inline(
        name="megakernel_prefill",
        cpp_sources=[_get_cpp_source("bindings_prefill.cpp")],
        cuda_sources=[
            _get_cuda_source("fused_prefill.cu"),
            _get_cuda_source("fused_prefill_megakernel.cu"),
            _get_cuda_source("fused_decode_ldg.cu"),
            _get_cuda_source("split_decode_gemm.cu"),
        ],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=["-lcublas", "-lcublasLt"],
        verbose=verbose,
    )
    return _prefill_kernel


def _compile_fused_prefill_kernel():
    global _fused_prefill_kernel
    if _fused_prefill_kernel is not None:
        return _fused_prefill_kernel

    kernel_dir = _kernel_dir()
    extra_cuda_cflags, verbose = _extra_cuda_cflags(kernel_dir)
    _fused_prefill_kernel = load_inline(
        name="megakernel_fused_prefill",
        cpp_sources=[_get_cpp_source("bindings_fused_prefill.cpp")],
        cuda_sources=[_get_cuda_source("fused_prefill_megakernel.cu"), _get_cuda_source("fused_decode_ldg.cu")],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )
    return _fused_prefill_kernel

