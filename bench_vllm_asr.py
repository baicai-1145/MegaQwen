"""Benchmark vLLM OpenAI-compatible server on a local WAV file (ASR).

This script sends `input_audio` (base64) via `/v1/chat/completions` and measures
end-to-end request latency. It avoids extra deps (uses stdlib only).

Prereqs:
  - Start vLLM server in another shell:
      vllm serve Qwen3-ASR-0.6B --dtype bfloat16 --max-model-len 4096

Usage:
  python bench_vllm_asr.py --audio test1_0_30.wav --model Qwen3-ASR-0.6B
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
import urllib.error
import wave


def _wav_duration_s(path: str) -> float | None:
    try:
        with wave.open(path, "rb") as wf:
            n = wf.getnframes()
            sr = wf.getframerate()
        return float(n) / float(sr) if sr > 0 else None
    except Exception:
        return None


def _post_json(url: str, payload: dict, timeout_s: float = 600.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read()
        # Print the server's error body; it's usually the fastest way to fix payload/schema issues.
        raise RuntimeError(
            f"HTTP {e.code} {e.reason} from {url}\n"
            f"Response body:\n{body.decode('utf-8', errors='replace')}\n"
            f"Request payload keys: {sorted(payload.keys())}"
        ) from None


def _get_json(url: str, timeout_s: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000", help="vLLM OpenAI server base URL")
    ap.add_argument("--model", default="Qwen3-ASR-0.6B", help="Model name/path as known to the server")
    ap.add_argument("--audio", required=True, help="Local WAV path (recommend: 16kHz mono PCM16)")
    ap.add_argument("--prompt", default="Transcribe the audio clip into text.", help="Text instruction")
    ap.add_argument("--language", default=None, help='Force language via system prompt, e.g. "Chinese" or "English"')
    ap.add_argument(
        "--count-tokens",
        action="store_true",
        help="Count output tokens locally with Transformers tokenizer (best-effort, for fair decode comparisons).",
    )
    ap.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Local tokenizer dir for --count-tokens (default: use --model if it's a local directory).",
    )
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--audio-mode",
        default="audio_url",
        choices=["audio_url", "input_audio"],
        help="OpenAI content schema; some vLLM builds accept only audio_url.",
    )
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    audio_dur = _wav_duration_s(args.audio)
    with open(args.audio, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    # Quick connectivity check.
    _get_json(args.base_url.rstrip("/") + "/v1/models")

    if args.audio_mode == "audio_url":
        audio_part = {
            "type": "audio_url",
            "audio_url": {"url": "data:audio/wav;base64," + audio_b64},
        }
    else:
        audio_part = {
            "type": "input_audio",
            "input_audio": {"data": audio_b64, "format": "wav"},
        }

    payload = {
        "model": args.model,
        "messages": [],
        "temperature": 0,
        # vLLM usually supports `max_tokens` (OpenAI chat completions).
        "max_tokens": args.max_tokens,
    }
    if args.language is not None:
        # The model's shipped chat_template.json uses the system prompt to carry auxiliary info.
        payload["messages"].append({"role": "system", "content": f"language: {args.language}\n"})
    payload["messages"].append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                audio_part,
            ],
        }
    )

    # Warmup + timed runs.
    times = []
    out_text = None
    usage = None
    finish_reason = None
    for i in range(args.runs):
        t0 = time.perf_counter()
        resp = _post_json(url, payload)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        out_text = resp["choices"][0]["message"]["content"]
        finish_reason = resp["choices"][0].get("finish_reason")
        usage = resp.get("usage")
        if audio_dur is not None:
            print(f"run{i}: {(t1 - t0) * 1000:.1f} ms (RTF={(t1 - t0) / audio_dur:.6f})")
        else:
            print(f"run{i}: {(t1 - t0) * 1000:.1f} ms")

    print("output:", out_text)
    if finish_reason is not None:
        print("finish_reason:", finish_reason)
    if usage is not None:
        print("usage:", usage)
    print(f"avg: {sum(times)/len(times)*1000:.1f} ms over {len(times)} runs")
    if audio_dur is not None:
        print(f"audio: {audio_dur:.3f} s")
        print(f"avg RTF: {(sum(times)/len(times)) / audio_dur:.6f}")

    if args.count_tokens and out_text is not None:
        tok_dir = args.tokenizer_dir
        if tok_dir is None:
            # If --model looks like a local dir, reuse it for tokenizer.
            import os

            tok_dir = args.model if os.path.isdir(args.model) else None
        if tok_dir is None:
            raise SystemExit("--count-tokens needs --tokenizer-dir (or pass a local directory in --model)")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tok_dir, fix_mistral_regex=True)
        out_ids = tokenizer.encode(out_text, add_special_tokens=False)
        print("output_tokens:", len(out_ids))


if __name__ == "__main__":
    main()
