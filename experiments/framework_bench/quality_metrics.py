"""Quality metrics for Qwen3-0.6B across frameworks."""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_NAME = "Qwen/Qwen3-0.6B"
TEST_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In machine learning, gradient descent is",
    "The capital of France is",
]


@dataclasses.dataclass
class QualityResult:
    name: str
    supported: bool
    kl_div: Optional[float]
    perplexity: Optional[float]
    error: Optional[str] = None


def _format_float(value: Optional[float], precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _format_markdown_table(results: List[QualityResult]) -> str:
    lines = [
        "| Framework | KL Divergence | Perplexity | Status |",
        "|---|---:|---:|---|",
    ]
    for r in results:
        status = "ok" if r.supported else (r.error or "skipped")
        lines.append(
            f"| {r.name} | {_format_float(r.kl_div)} | {_format_float(r.perplexity)} | {status} |"
        )
    return "\n".join(lines)


def _hf_load():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def _compute_logits_hf(model, tokenizer, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    return out.logits, input_ids


def _kl_divergence(ref_logits: torch.Tensor, tgt_logits: torch.Tensor) -> float:
    """Compute KL divergence between reference and target logits."""
    ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
    tgt_log_probs = F.log_softmax(tgt_logits.float(), dim=-1)
    ref_probs = ref_log_probs.exp()

    kl = (ref_probs * (ref_log_probs - tgt_log_probs)).sum(dim=-1).mean()
    return float(kl.item())


def _perplexity_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """Compute perplexity from logits and input_ids."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return float(torch.exp(nll.mean()).item())


def _hf_quality(prompts: List[str]) -> Tuple[QualityResult, List[torch.Tensor], List[torch.Tensor]]:
    """Get HuggingFace reference logits and perplexity."""
    if not torch.cuda.is_available():
        return QualityResult("HuggingFace", False, None, None, "cuda not available"), [], []

    try:
        model, tokenizer = _hf_load()

        all_logits = []
        all_input_ids = []
        total_ppl = 0.0

        for prompt in prompts:
            logits, input_ids = _compute_logits_hf(model, tokenizer, prompt)
            all_logits.append(logits)
            all_input_ids.append(input_ids)
            total_ppl += _perplexity_from_logits(logits, input_ids)

        avg_ppl = total_ppl / len(prompts)

        del model
        torch.cuda.empty_cache()

        return QualityResult("HuggingFace", True, 0.0, avg_ppl), all_logits, all_input_ids
    except Exception as exc:
        return QualityResult("HuggingFace", False, None, None, f"error: {exc}"), [], []


def _megakernel_quality(prompts: List[str], ref_logits: List[torch.Tensor]) -> QualityResult:
    """
    Compare megakernel outputs against HuggingFace reference.

    Note: The megakernel doesn't expose full logits, only argmax tokens.
    We can only verify that the top-1 prediction matches.
    """
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, project_root)
        from chat import MegakernelChat, NUM_LAYERS, MAX_SEQ_LEN
    except Exception as exc:
        return QualityResult("Megakernel", False, None, None, f"import error: {exc}")

    try:
        chat = MegakernelChat()
    except Exception as exc:
        return QualityResult("Megakernel", False, None, None, f"init error: {exc}")

    # Compare argmax predictions
    matches = 0
    total = 0

    for i, prompt in enumerate(prompts):
        chat.k_cache.zero_()
        chat.v_cache.zero_()
        input_ids = chat.tokenizer.encode(prompt)

        # Run through megakernel
        for pos, token_id in enumerate(input_ids[:-1]):
            chat.kernel.decode_ldg(
                token_id, chat.final_norm_weight, chat.lm_head_weight,
                chat.cos_table, chat.sin_table,
                chat.k_cache, chat.v_cache,
                chat.hidden_buffer, chat.g_activations, chat.g_residual,
                chat.g_q, chat.g_k, chat.g_v, chat.g_attn_out,
                chat.g_mlp_intermediate, chat.g_normalized,
                chat.block_max_vals, chat.block_max_idxs,
                NUM_LAYERS, pos, pos + 1, MAX_SEQ_LEN,
            )

        # Get final prediction
        mega_pred = chat.kernel.decode_ldg(
            input_ids[-1], chat.final_norm_weight, chat.lm_head_weight,
            chat.cos_table, chat.sin_table,
            chat.k_cache, chat.v_cache,
            chat.hidden_buffer, chat.g_activations, chat.g_residual,
            chat.g_q, chat.g_k, chat.g_v, chat.g_attn_out,
            chat.g_mlp_intermediate, chat.g_normalized,
            chat.block_max_vals, chat.block_max_idxs,
            NUM_LAYERS, len(input_ids) - 1, len(input_ids), MAX_SEQ_LEN,
        )

        # Compare with HF argmax
        hf_pred = ref_logits[i][:, -1, :].argmax(dim=-1).item()

        if mega_pred == hf_pred:
            matches += 1
        total += 1

    del chat
    torch.cuda.empty_cache()

    match_rate = matches / total if total > 0 else 0.0

    # KL divergence can't be computed without full logits
    # Report match rate instead
    return QualityResult(
        "Megakernel",
        True,
        None,  # Can't compute KL without logits
        None,  # Can't compute perplexity without logits
        f"argmax match: {match_rate:.1%}"
    )


def main() -> int:
    prompts = TEST_PROMPTS

    print("=" * 60)
    print("QUALITY METRICS - Qwen3-0.6B")
    print("=" * 60)
    print(f"Test prompts: {len(prompts)}")
    print()

    # Get HuggingFace reference
    print("Computing HuggingFace reference...")
    hf_result, ref_logits, ref_input_ids = _hf_quality(prompts)

    results = [hf_result]

    if hf_result.supported:
        print("Comparing Megakernel...")
        results.append(_megakernel_quality(prompts, ref_logits))

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(_format_markdown_table(results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
