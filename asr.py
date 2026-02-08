"""Minimal offline transcription demo for Qwen3-ASR models.

Examples:
  python asr.py --model Qwen3-ASR-0.6B --audio sample.wav
  python asr.py --model Qwen3-ASR-1.7B --audio sample.wav --device cuda --dtype bf16
  python asr.py --model Qwen3-ASR-0.6B --audio sample.wav --backend megakernel --device cuda --dtype bf16
"""

import argparse

from megaqwen.qwen3_asr import Qwen3ASRModel
from megaqwen.qwen3_asr_megakernel import Qwen3ASRMegakernelModel, Qwen3ASRMegakernelOptions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Local model dir, e.g. Qwen3-ASR-0.6B or Qwen3-ASR-1.7B")
    ap.add_argument("--audio", required=True, help="Path to a mono/stereo WAV file (PCM16 or float32).")
    ap.add_argument("--backend", default="torch", choices=["torch", "megakernel"])
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--dtype", default=None, help="bf16|fp16|fp32 (default: auto)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--language", default=None, help='Force language, e.g. "Chinese" or "English" (default: auto)')
    ap.add_argument("--show-language", action="store_true", help="Print detected/forced language header if available")
    ap.add_argument("--profile", action="store_true", help="Print per-stage timing breakdown")
    ap.add_argument("--max-seq-len", type=int, default=4096, help="(megakernel) RoPE table length")
    ap.add_argument("--max-prefill-len", type=int, default=4096, help="(megakernel) buffer capacity for prefill")
    args = ap.parse_args()

    timer = None
    if args.profile:
        from megaqwen.profiler import StageTimer

        timer = StageTimer(enabled=True)

    if args.backend == "megakernel":
        opts = Qwen3ASRMegakernelOptions(
            max_seq_len=args.max_seq_len,
            max_prefill_len=args.max_prefill_len,
            max_new_tokens=args.max_new_tokens,
        )
        model = Qwen3ASRMegakernelModel.from_pretrained(args.model, device=args.device, dtype=args.dtype, opts=opts)
        out = model.transcribe(
            args.audio,
            max_new_tokens=args.max_new_tokens,
            language=args.language,
            return_language=args.show_language,
            timer=timer,
        )
    else:
        model = Qwen3ASRModel.from_pretrained(args.model, device=args.device, dtype=args.dtype)
        out = model.transcribe(
            args.audio,
            max_new_tokens=args.max_new_tokens,
            language=args.language,
            return_language=args.show_language,
            timer=timer,
        )

    if isinstance(out, tuple):
        lang, text = out
        print("language:", lang)
        print(text)
    else:
        print(out)

    if timer is not None:
        print(timer.format())


if __name__ == "__main__":
    main()
