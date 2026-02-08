from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from .safetensors_loader import load_tensors
from .profiler import StageTimer
from .whisper_fbank_torch import WhisperFbankConfig, WhisperFbankTorch
from .asr_result import ASRResult


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_device_dtype(device: str | None, dtype: str | None) -> tuple[torch.device, torch.dtype]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    if dtype is None:
        dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    else:
        dt = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }[dtype.lower()]
        if dev.type == "cpu" and dt in (torch.float16,):
            # CPU fp16 is painful / often unsupported in many ops.
            dt = torch.float32
    return dev, dt


def _linear_resample(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return audio
    if audio.size == 0:
        return audio.astype(np.float32)
    ratio = tgt_sr / src_sr
    n = int(round(audio.shape[0] * ratio))
    x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False, dtype=np.float32)
    return np.interp(x_new, x_old, audio.astype(np.float32)).astype(np.float32)


def load_wav_mono(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load PCM WAV via stdlib `wave` (no external deps).

    Limitations:
    - WAV only (no mp3/flac)
    - Supports 16-bit PCM and 32-bit float PCM
    """
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.float32).astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes (path={path})")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if sr != target_sr:
        audio = _linear_resample(audio, sr, target_sr)
        sr = target_sr

    return audio, sr


class Qwen3ASRAudioTower(nn.Module):
    def __init__(self, *, audio_config: dict, out_hidden: int):
        super().__init__()
        try:
            from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
            from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder
        except Exception as e:
            raise ImportError(
                "Official qwen_asr package is required for accurate audio tower. "
                "Please ensure local Qwen3-ASR package is installed in this venv."
            ) from e

        cfg_dict = dict(audio_config)
        cfg_dict["output_dim"] = int(out_hidden)
        cfg = Qwen3ASRAudioEncoderConfig(**cfg_dict)
        self.inner = Qwen3ASRAudioEncoder(cfg)

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if input_features.dim() != 3:
            raise ValueError(f"input_features must be [B, 128, T], got shape={tuple(input_features.shape)}")
        bsz = int(input_features.shape[0])
        if bsz <= 0:
            raise ValueError("Empty batch for audio tower")

        if attention_mask is not None:
            feature_lens = attention_mask.to(torch.long).sum(dim=-1)
        else:
            feature_lens = torch.full(
                (bsz,),
                int(input_features.shape[-1]),
                dtype=torch.long,
                device=input_features.device,
            )

        out_list: list[torch.Tensor] = []
        for idx in range(bsz):
            fl = int(feature_lens[idx].item())
            one_feat = input_features[idx, :, :fl]
            out = self.inner(one_feat, feature_lens=feature_lens[idx : idx + 1]).last_hidden_state
            out_list.append(out)
        return nn.utils.rnn.pad_sequence(out_list, batch_first=True)

    def prepare_for_inference(self) -> None:
        pass

    def state_dict(self, *args, **kwargs):
        return self.inner.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.inner.load_state_dict(state_dict, strict=strict)


@dataclass(frozen=True)
class Qwen3ASRSpecialIds:
    im_start: int
    im_end: int
    audio_start: int
    audio_pad: int
    audio_end: int
    asr_text: int | None


class Qwen3ASRModel:
    """Offline Qwen3-ASR inference (minimal, single-utterance, greedy decode).

    Notes:
    - This is NOT the official `qwen-asr` toolkit. It is a lightweight port
      intended for this repo's offline use and experimentation.
    - Focuses on correctness and simplicity. Performance work can be layered
      on later (e.g. flash-attn, megakernel decode for the text backbone).
    """

    def __init__(self, model_dir: str | Path, *, device: torch.device, dtype: torch.dtype):
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype

        cfg = _read_json(self.model_dir / "config.json")
        thinker = cfg["thinker_config"]
        self._audio_cfg = thinker["audio_config"]
        self._text_cfg = thinker["text_config"]

        # Some recent tokenizers ship with a known-bad regex; `fix_mistral_regex`
        # (when available) patches it to avoid incorrect tokenization.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), fix_mistral_regex=True)
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(str(self.model_dir))
        self._fbank_torch: WhisperFbankTorch | None = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            # Keep feature extraction fully on GPU to cut CPU cost and avoid D2H/H2D ping-pong.
            self._fbank_torch = WhisperFbankTorch(
                cfg=WhisperFbankConfig(
                    n_fft=int(self.feature_extractor.n_fft),
                    hop_length=int(self.feature_extractor.hop_length),
                    n_samples=int(self.feature_extractor.n_samples),
                    nb_max_frames=int(self.feature_extractor.nb_max_frames),
                    dither=float(getattr(self.feature_extractor, "dither", 0.0) or 0.0),
                ),
                mel_filters=self.feature_extractor.mel_filters,
                device=self.device,
            )

        self.special = Qwen3ASRSpecialIds(
            im_start=self.tokenizer.convert_tokens_to_ids("<|im_start|>"),
            im_end=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            audio_start=self.tokenizer.convert_tokens_to_ids("<|audio_start|>"),
            audio_pad=self.tokenizer.convert_tokens_to_ids("<|audio_pad|>"),
            audio_end=self.tokenizer.convert_tokens_to_ids("<|audio_end|>"),
            asr_text=self._maybe_token_id("<asr_text>"),
        )

        # Build text backbone from nested config.
        text_cfg = {k: v for k, v in self._text_cfg.items() if v is not None}
        # Qwen3-ASR checkpoints may include extra mRoPE bookkeeping keys that
        # current Transformers' Qwen3Config doesn't validate for rope_type=default.
        # They are not needed for this minimal offline inference.
        if isinstance(text_cfg.get("rope_scaling"), dict):
            text_cfg.pop("rope_scaling", None)
        self.text_config = Qwen3Config(**text_cfg)
        self.text_model = Qwen3ForCausalLM(self.text_config)

        # Build audio tower.
        out_hidden = int(self._text_cfg["hidden_size"])
        self.audio_tower = Qwen3ASRAudioTower(audio_config=self._audio_cfg, out_hidden=out_hidden)

        # Load weights from local safetensors.
        self._load_weights()

        self.text_model.eval()
        self.audio_tower.eval()

        self.text_model.to(device=self.device, dtype=self.dtype)
        self.audio_tower.to(device=self.device, dtype=self.dtype)
        self.audio_tower.prepare_for_inference()

    def _maybe_token_id(self, token: str) -> int | None:
        tok_id = int(self.tokenizer.convert_tokens_to_ids(token))
        unk = getattr(self.tokenizer, "unk_token_id", None)
        if unk is not None and tok_id == int(unk):
            return None
        return tok_id

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, *, device: str | None = None, dtype: str | None = None) -> "Qwen3ASRModel":
        dev, dt = _pick_device_dtype(device, dtype)
        return cls(model_dir, device=dev, dtype=dt)

    def _load_weights(self) -> None:
        # Audio tower: we keep key names aligned with our module's state_dict.
        audio_prefix = "thinker.audio_tower."
        text_prefix = "thinker.model."

        # Collect keys for audio tower.
        audio_keys = []
        for k in self.audio_tower.state_dict().keys():
            audio_keys.append(audio_prefix + k)

        # Build the full key list by mapping from our model state_dict.
        mapped_text_keys = []
        # Only request *parameters* from checkpoint; buffers (e.g. rotary inv_freq)
        # are not stored in safetensors.
        for k, _ in self.text_model.named_parameters():
            if k.startswith("model."):
                mapped_text_keys.append(text_prefix + k[len("model."):])
            elif k == "lm_head.weight":
                mapped_text_keys.append("thinker.lm_head.weight")
            else:
                # Buffers (rotary) etc. We ignore.
                pass
        # When tie_word_embeddings=True, `lm_head.weight` is not a Parameter, but
        # it still appears in `state_dict()` and is expected by `load_state_dict`.
        if "lm_head.weight" in self.text_model.state_dict():
            mapped_text_keys.append("thinker.lm_head.weight")

        # Load tensors.
        tensors = load_tensors(self.model_dir, audio_keys + mapped_text_keys, device="cpu")

        # Audio load
        audio_sd = {k[len(audio_prefix):]: v for k, v in tensors.items() if k.startswith(audio_prefix)}
        missing, unexpected = self.audio_tower.load_state_dict(audio_sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Audio tower state mismatch: missing={missing}, unexpected={unexpected}")

        # Text load (rename keys into HF Qwen3 layout)
        text_sd: dict[str, torch.Tensor] = {}
        for k, v in tensors.items():
            if k.startswith(text_prefix):
                text_sd["model." + k[len(text_prefix):]] = v
            elif k == "thinker.lm_head.weight":
                text_sd["lm_head.weight"] = v

        info = self.text_model.load_state_dict(text_sd, strict=False)
        # Allow some buffers / tied-weight differences, but fail if we miss actual parameters.
        missing_params = [k for k in info.missing_keys if ".rotary_emb." not in k]
        if self.text_config.tie_word_embeddings:
            missing_params = [k for k in missing_params if k != "lm_head.weight"]
        if missing_params:
            raise RuntimeError(f"Text model missing keys (unexpected): {missing_params[:10]}")

    def _build_prompt_ids(
        self,
        audio_token_count: int,
        *,
        system_text: str = "",
        force_language: str | None = None,
    ) -> list[int]:
        # Match official template:
        # <|im_start|>system\n{context}<|im_end|>\n
        # <|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n
        # <|im_start|>assistant\n
        # If forcing language, append: language X<asr_text>
        prefix = "<|im_start|>system\n" + (system_text or "") + "<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
        ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        ids.append(self.special.audio_start)
        ids.extend([self.special.audio_pad] * audio_token_count)
        ids.append(self.special.audio_end)
        ids.extend(self.tokenizer.encode(suffix, add_special_tokens=False))
        if force_language:
            ids.extend(self.tokenizer.encode(f"language {force_language}", add_special_tokens=False))
            if self.special.asr_text is not None:
                ids.append(self.special.asr_text)
            else:
                ids.extend(self.tokenizer.encode("<asr_text>", add_special_tokens=False))
        return ids

    @torch.no_grad()
    def transcribe(
        self,
        wav_path: str | Path,
        *,
        max_new_tokens: int = 256,
        language: str | None = None,
        return_language: bool = False,
        timer: StageTimer | None = None,
    ) -> str | tuple[str | None, str]:
        timer = timer if timer is not None else StageTimer(enabled=False)

        with timer.cpu("wav_load"):
            audio, sr = load_wav_mono(wav_path, target_sr=16000)

        if self._fbank_torch is not None:
            # Compute log-mel on GPU (WhisperFeatureExtractor-compatible).
            with timer.cuda("feature_extract"):
                input_features, attention_mask = self._fbank_torch.extract(audio, sampling_rate=sr)
            # Cast for the audio tower.
            with timer.cuda("h2d_features"):
                input_features = input_features.to(dtype=self.dtype)
        else:
            with timer.cpu("feature_extract"):
                feats = self.feature_extractor(
                    audio,
                    sampling_rate=sr,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

            # H2D copy is async; use CUDA events for more realistic timing.
            with timer.cuda("h2d_features"):
                input_features = feats["input_features"].to(self.device, dtype=self.dtype)  # [1, 128, frames]
                attention_mask = feats.get("attention_mask")
                if attention_mask is not None:
                    # If the mask is all-valid, drop it to unlock faster SDPA kernels.
                    if int(attention_mask.sum().item()) == int(attention_mask.numel()):
                        attention_mask = None
                    else:
                        attention_mask = attention_mask.to(self.device)

        with timer.cuda("audio_tower"):
            audio_embeds = self.audio_tower(input_features, attention_mask)  # [1, T', hidden]
        b, t_audio, hidden = audio_embeds.shape

        with timer.cpu("prompt_build"):
            prompt_ids = self._build_prompt_ids(audio_token_count=t_audio, system_text="", force_language=language)

        with timer.cuda("h2d_input_ids"):
            input_ids = torch.tensor(prompt_ids, device=self.device, dtype=torch.long).unsqueeze(0)  # [1, L]

        with timer.cuda("inputs_embed_patch"):
            token_embeds = self.text_model.get_input_embeddings()(input_ids)  # [1, L, hidden]
            audio_pos = (input_ids == self.special.audio_pad).nonzero(as_tuple=False)  # [t_audio, 2]
            if audio_pos.shape[0] != t_audio:
                raise RuntimeError(f"Audio placeholder count mismatch: ids has {audio_pos.shape[0]} pads, audio_embeds has {t_audio}")
            token_embeds[0, audio_pos[:, 1]] = audio_embeds[0]

        # Prefill
        seq_len = token_embeds.shape[1]
        cache_position = torch.arange(seq_len, device=self.device, dtype=torch.long)
        with timer.cuda("text_prefill"):
            out = self.text_model(
                inputs_embeds=token_embeds,
                use_cache=True,
                cache_position=cache_position,
            )
            past = out.past_key_values

        eos_ids = set()
        gen_cfg_path = self.model_dir / "generation_config.json"
        if gen_cfg_path.exists():
            gen_cfg = _read_json(gen_cfg_path)
            eos = gen_cfg.get("eos_token_id")
            if isinstance(eos, list):
                eos_ids.update(int(x) for x in eos)
            elif eos is not None:
                eos_ids.add(int(eos))
        if not eos_ids:
            eos_ids.add(int(self.tokenizer.eos_token_id))

        next_token = torch.argmax(out.logits[:, -1, :], dim=-1)  # [1]
        generated: list[int] = []

        with timer.cuda("text_decode"):
            for step in range(max_new_tokens):
                tok = int(next_token.item())
                if tok in eos_ids:
                    break
                generated.append(tok)

                cache_position = torch.tensor([seq_len + step], device=self.device, dtype=torch.long)
                out = self.text_model(
                    input_ids=next_token.view(1, 1),
                    past_key_values=past,
                    use_cache=True,
                    cache_position=cache_position,
                )
                past = out.past_key_values
                next_token = torch.argmax(out.logits[:, -1, :], dim=-1)

        with timer.cpu("token_decode"):
            raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        lang, text = _parse_asr_output(raw, user_language=language)
        return (lang, text) if return_language else text

    @torch.no_grad()
    def transcribe_result(
        self,
        wav_path: str | Path,
        *,
        max_new_tokens: int = 256,
        language: str | None = None,
        timer: StageTimer | None = None,
    ) -> ASRResult:
        timer = timer if timer is not None else StageTimer(enabled=False)

        with timer.cpu("wav_load"):
            audio, sr = load_wav_mono(wav_path, target_sr=16000)

        if self._fbank_torch is not None:
            with timer.cuda("feature_extract"):
                input_features, attention_mask = self._fbank_torch.extract(audio, sampling_rate=sr)
            with timer.cuda("h2d_features"):
                input_features = input_features.to(dtype=self.dtype)
        else:
            with timer.cpu("feature_extract"):
                feats = self.feature_extractor(
                    audio,
                    sampling_rate=sr,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
            with timer.cuda("h2d_features"):
                input_features = feats["input_features"].to(self.device, dtype=self.dtype)
                attention_mask = feats.get("attention_mask")
                if attention_mask is not None:
                    if int(attention_mask.sum().item()) == int(attention_mask.numel()):
                        attention_mask = None
                    else:
                        attention_mask = attention_mask.to(self.device)

        with timer.cuda("audio_tower"):
            audio_embeds = self.audio_tower(input_features, attention_mask)  # [1, T', hidden]
        _, t_audio, _ = audio_embeds.shape

        with timer.cpu("prompt_build"):
            prompt_ids = self._build_prompt_ids(audio_token_count=t_audio, system_text="", force_language=language)
        with timer.cuda("h2d_input_ids"):
            input_ids = torch.tensor(prompt_ids, device=self.device, dtype=torch.long).unsqueeze(0)

        with timer.cuda("inputs_embed_patch"):
            token_embeds = self.text_model.get_input_embeddings()(input_ids)
            audio_pos = (input_ids == self.special.audio_pad).nonzero(as_tuple=False)
            if audio_pos.shape[0] != t_audio:
                raise RuntimeError(f"Audio placeholder count mismatch: ids has {audio_pos.shape[0]} pads, audio_embeds has {t_audio}")
            token_embeds[0, audio_pos[:, 1]] = audio_embeds[0]

        seq_len = token_embeds.shape[1]
        cache_position = torch.arange(seq_len, device=self.device, dtype=torch.long)
        with timer.cuda("text_prefill"):
            out = self.text_model(inputs_embeds=token_embeds, use_cache=True, cache_position=cache_position)
            past = out.past_key_values

        eos_ids = set()
        gen_cfg_path = self.model_dir / "generation_config.json"
        if gen_cfg_path.exists():
            gen_cfg = _read_json(gen_cfg_path)
            eos = gen_cfg.get("eos_token_id")
            if isinstance(eos, list):
                eos_ids.update(int(x) for x in eos)
            elif eos is not None:
                eos_ids.add(int(eos))
        if not eos_ids:
            eos_ids.add(int(self.tokenizer.eos_token_id))

        next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
        generated: list[int] = []
        stop_reason = "max_new_tokens"

        with timer.cuda("text_decode"):
            for step in range(max_new_tokens):
                tok = int(next_token.item())
                if tok in eos_ids:
                    stop_reason = "eos"
                    break
                generated.append(tok)
                cache_position = torch.tensor([seq_len + step], device=self.device, dtype=torch.long)
                out = self.text_model(input_ids=next_token.view(1, 1), past_key_values=past, use_cache=True, cache_position=cache_position)
                past = out.past_key_values
                next_token = torch.argmax(out.logits[:, -1, :], dim=-1)

        with timer.cpu("token_decode"):
            raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        lang, text = _parse_asr_output(raw, user_language=language)
        return ASRResult(text=text, language=lang, generated_tokens=len(generated), stop_reason=stop_reason)


def _parse_asr_output(s: str, user_language: str | None = None) -> tuple[str | None, str]:
    """Parse common Qwen3-ASR output format: 'language {LANG}<asr_text>{TEXT}'.

    Some backends return the language header as plain text tokens. For most users,
    returning only the transcription is more convenient.
    """
    s = (s or "").strip()
    if not s:
        return (user_language or None), ""

    marker = "<asr_text>"
    if user_language:
        # Official forced-language mode expects text-only output; if model still emits
        # a language header, strip it for robustness.
        if marker in s:
            _, tail = s.split(marker, 1)
            return user_language, tail.strip()
        lower = s.lower()
        if lower.startswith("language "):
            newline = s.find("\n")
            if newline >= 0:
                return user_language, s[newline + 1 :].strip()
        return user_language, s

    lower = s.lower()
    if lower.startswith("language ") and marker in s:
        head, tail = s.split(marker, 1)
        lang = head.split(" ", 1)[1].strip() if " " in head else head[len("language ") :].strip()
        return (lang or None), tail.strip()

    # Fallback: strip marker if it appears without the "language " prefix.
    if marker in s:
        _, tail = s.split(marker, 1)
        return None, tail.strip()

    return None, s
