from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ASRResult:
    text: str
    language: str | None
    generated_tokens: int
    stop_reason: str  # "eos" | "max_new_tokens"

