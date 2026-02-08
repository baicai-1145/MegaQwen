"""
Megakernel python entrypoint.

This module now only composes smaller units:
- `megakernel_kernels.py`: CUDA extension compilation + kernel constants
- `megakernel_weights.py`: model/safetensors loading + RoPE tables
- `megakernel_generators.py`: high-level generator wrappers
"""

from megakernel_generators import (
    MegakernelFusedPrefillGenerator,
    MegakernelGenerator,
    MegakernelPrefillGenerator,
    main as _generator_main,
)
from megakernel_kernels import (
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    KV_SIZE,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_Q_HEADS,
    Q_SIZE,
    VOCAB_SIZE,
    _compile_decode_kernel,
    _compile_fused_prefill_kernel,
    _compile_prefill_kernel,
)
from megakernel_weights import load_qwen3_weights

__all__ = [
    "HIDDEN_SIZE",
    "INTERMEDIATE_SIZE",
    "NUM_Q_HEADS",
    "NUM_KV_HEADS",
    "HEAD_DIM",
    "Q_SIZE",
    "KV_SIZE",
    "NUM_LAYERS",
    "VOCAB_SIZE",
    "_compile_decode_kernel",
    "_compile_prefill_kernel",
    "_compile_fused_prefill_kernel",
    "load_qwen3_weights",
    "MegakernelGenerator",
    "MegakernelPrefillGenerator",
    "MegakernelFusedPrefillGenerator",
]


if __name__ == "__main__":
    _generator_main()

