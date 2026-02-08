import json
from pathlib import Path

import torch

from megakernel_kernels import HEAD_DIM, HIDDEN_SIZE, NUM_LAYERS


def _collect_layer_keys(text_prefix: str, lm_head_key: str) -> list[str]:
    keys = [text_prefix + "embed_tokens.weight", text_prefix + "norm.weight", lm_head_key]
    for i in range(NUM_LAYERS):
        p = f"{text_prefix}layers.{i}."
        keys.extend(
            [
                p + "input_layernorm.weight",
                p + "self_attn.q_proj.weight",
                p + "self_attn.k_proj.weight",
                p + "self_attn.v_proj.weight",
                p + "self_attn.q_norm.weight",
                p + "self_attn.k_norm.weight",
                p + "self_attn.o_proj.weight",
                p + "post_attention_layernorm.weight",
                p + "mlp.gate_proj.weight",
                p + "mlp.up_proj.weight",
                p + "mlp.down_proj.weight",
            ]
        )
    return keys


def _collect_layer_weights(tensors: dict[str, torch.Tensor], text_prefix: str) -> list[torch.Tensor]:
    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"{text_prefix}layers.{i}."
        layer_weights.extend(
            [
                tensors[p + "input_layernorm.weight"].contiguous(),
                tensors[p + "self_attn.q_proj.weight"].contiguous(),
                tensors[p + "self_attn.k_proj.weight"].contiguous(),
                tensors[p + "self_attn.v_proj.weight"].contiguous(),
                tensors[p + "self_attn.q_norm.weight"].contiguous(),
                tensors[p + "self_attn.k_norm.weight"].contiguous(),
                tensors[p + "self_attn.o_proj.weight"].contiguous(),
                tensors[p + "post_attention_layernorm.weight"].contiguous(),
                tensors[p + "mlp.gate_proj.weight"].contiguous(),
                tensors[p + "mlp.up_proj.weight"].contiguous(),
                tensors[p + "mlp.down_proj.weight"].contiguous(),
            ]
        )
    return layer_weights


def _build_rope_tables(max_seq_len: int, rope_theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, HEAD_DIM, 2, device="cuda", dtype=torch.float32) / HEAD_DIM))
    positions = torch.arange(max_seq_len, device="cuda", dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).contiguous()
    return cos_table, sin_table


def _resolve_local_model_dir(model_name: str) -> Path | None:
    model_dir = Path(model_name)
    if (not model_dir.exists()) and ("/" in str(model_name)):
        fallback = Path(str(model_name).split("/")[-1])
        if fallback.exists() and fallback.is_dir():
            model_dir = fallback
    if model_dir.exists() and model_dir.is_dir():
        return model_dir
    return None


def load_qwen3_weights(model_name: str = "Qwen/Qwen3-0.6B", max_seq_len: int = 2048):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = _resolve_local_model_dir(model_name)
    if model_dir is not None:
        print(f"Loading local weights: {model_dir} ...")
        cfg_path = model_dir / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        model_type = cfg.get("model_type")

        if model_type == "qwen3_asr":
            text_prefix = "thinker.model."
            lm_head_key = "thinker.lm_head.weight"
            hidden = cfg["thinker_config"]["text_config"].get("hidden_size")
            if int(hidden) != HIDDEN_SIZE:
                raise ValueError(
                    f"Unsupported for megakernel (expected hidden_size={HIDDEN_SIZE}): {model_dir} hidden_size={hidden}"
                )
            rope_theta = float(cfg["thinker_config"]["text_config"].get("rope_theta", 1000000.0))
        else:
            text_prefix = "model."
            lm_head_key = "lm_head.weight"
            rope_theta = float(cfg.get("rope_theta", 1000000.0))

        try:
            from megaqwen.safetensors_loader import load_tensors
        except Exception as e:
            raise RuntimeError("Local safetensors loader import failed; ensure repo root is on PYTHONPATH") from e

        tensors = load_tensors(model_dir, _collect_layer_keys(text_prefix, lm_head_key), device="cuda")
        sample_key = text_prefix + "layers.0.self_attn.q_proj.weight"
        sample_dtype = tensors[sample_key].dtype
        if sample_dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise RuntimeError(
                f"MegaK decode kernels currently expect floating-point linear weights, got {sample_dtype} for {sample_key}. "
                f"Quantized checkpoints like bitsandbytes 4bit/8bit are not directly supported yet. "
                f"Please use original BF16 checkpoint for MegaK, or run official transformers backend for quantized inference."
            )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        cos_table, sin_table = _build_rope_tables(max_seq_len=max_seq_len, rope_theta=rope_theta)
        return {
            "embed_weight": tensors[text_prefix + "embed_tokens.weight"].contiguous(),
            "layer_weights": _collect_layer_weights(tensors, text_prefix=text_prefix),
            "final_norm_weight": tensors[text_prefix + "norm.weight"].contiguous(),
            "lm_head_weight": tensors[lm_head_key].contiguous(),
            "cos_table": cos_table,
            "sin_table": sin_table,
            "tokenizer": tokenizer,
        }

    print(f"Loading {model_name} (HuggingFace) ...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    rope_theta = float(getattr(getattr(model, "config", None), "rope_theta", 1000000.0))
    cos_table, sin_table = _build_rope_tables(max_seq_len=max_seq_len, rope_theta=rope_theta)
    layer_weights = _collect_layer_weights(state, text_prefix="model.")

    del model
    torch.cuda.empty_cache()

    return {
        "embed_weight": state["model.embed_tokens.weight"].contiguous(),
        "layer_weights": layer_weights,
        "final_norm_weight": state["model.norm.weight"].contiguous(),
        "lm_head_weight": state["lm_head.weight"].contiguous(),
        "cos_table": cos_table,
        "sin_table": sin_table,
        "tokenizer": tokenizer,
    }
