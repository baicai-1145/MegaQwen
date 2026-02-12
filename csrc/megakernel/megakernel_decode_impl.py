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
    _compile_decode_kernel,
    _compile_fused_prefill_kernel,
    _compile_prefill_kernel,
    configure_model_spec as _configure_model_spec,
    configure_model_spec_from_model_dir as _configure_model_spec_from_model_dir,
    get_active_model_spec as _get_active_model_spec,
)
from megakernel_weights import load_qwen3_weights


def _refresh_spec_aliases() -> None:
    spec = _get_active_model_spec()
    globals()["HIDDEN_SIZE"] = int(spec["hidden_size"])
    globals()["INTERMEDIATE_SIZE"] = int(spec["intermediate_size"])
    globals()["NUM_Q_HEADS"] = int(spec["num_q_heads"])
    globals()["NUM_KV_HEADS"] = int(spec["num_kv_heads"])
    globals()["HEAD_DIM"] = int(spec["head_dim"])
    globals()["Q_SIZE"] = int(spec["num_q_heads"]) * int(spec["head_dim"])
    globals()["KV_SIZE"] = int(spec["num_kv_heads"]) * int(spec["head_dim"])
    globals()["NUM_LAYERS"] = int(spec["num_layers"])
    globals()["VOCAB_SIZE"] = int(spec["vocab_size"])


_refresh_spec_aliases()


def get_active_model_spec() -> dict[str, int]:
    return _get_active_model_spec()


def configure_model_spec(**kwargs) -> dict[str, int]:
    spec = _configure_model_spec(**kwargs)
    _refresh_spec_aliases()
    return spec


def configure_model_spec_from_model_dir(model_dir: str) -> dict[str, int]:
    spec = _configure_model_spec_from_model_dir(model_dir)
    _refresh_spec_aliases()
    return spec

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
    "configure_model_spec",
    "configure_model_spec_from_model_dir",
    "get_active_model_spec",
    "_refresh_spec_aliases",
    "load_qwen3_weights",
    "MegakernelGenerator",
    "MegakernelPrefillGenerator",
    "MegakernelFusedPrefillGenerator",
]


if __name__ == "__main__":
    _generator_main()
