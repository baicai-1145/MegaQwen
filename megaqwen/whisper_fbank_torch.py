from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class WhisperFbankConfig:
    n_fft: int
    hop_length: int
    n_samples: int
    nb_max_frames: int
    dither: float


class WhisperFbankTorch:
    """GPU-friendly Whisper-style log-mel extractor matching Transformers' implementation.

    This mirrors `transformers.models.whisper.feature_extraction_whisper.WhisperFeatureExtractor._torch_extract_fbank_features`,
    but keeps outputs on the target device (no GPU->CPU roundtrip).
    """

    def __init__(self, *, cfg: WhisperFbankConfig, mel_filters: np.ndarray, device: torch.device):
        self.cfg = cfg
        self.device = device
        # Keep these in fp32 for numerical stability (matches upstream).
        self.window = torch.hann_window(cfg.n_fft, device=device, dtype=torch.float32)
        self.mel_filters = torch.from_numpy(mel_filters).to(device=device, dtype=torch.float32)  # [n_mels, n_freqs]

    def extract(self, audio: np.ndarray, *, sampling_rate: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (input_features, attention_mask) on self.device.

        - input_features: [1, n_mels, nb_max_frames] float32
        - attention_mask: [1, nb_max_frames] int64 (1=valid, 0=pad)
        """
        if sampling_rate != 16000:
            raise ValueError(f"Expected sampling_rate=16000, got {sampling_rate}")

        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio 1D array, got shape={audio.shape}")

        # WhisperFeatureExtractor pads/truncates waveform to `n_samples` and returns a frame-level mask.
        n = int(audio.shape[0])
        n_valid_samples = min(n, self.cfg.n_samples)
        valid_frames = int((n_valid_samples + self.cfg.hop_length - 1) // self.cfg.hop_length)
        valid_frames = min(valid_frames, self.cfg.nb_max_frames)

        # Pad / truncate to fixed window.
        if n >= self.cfg.n_samples:
            audio_fixed = audio[: self.cfg.n_samples]
        else:
            audio_fixed = np.pad(audio, (0, self.cfg.n_samples - n), mode="constant", constant_values=0.0)

        waveform = torch.from_numpy(audio_fixed).to(device=self.device, dtype=torch.float32)

        if self.cfg.dither != 0.0:
            waveform = waveform + float(self.cfg.dither) * torch.randn_like(waveform)

        stft = torch.stft(
            waveform,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            window=self.window,
            return_complex=True,
        )
        # Match upstream: drop the last frame to get exactly nb_max_frames for 30s inputs.
        magnitudes = stft[..., :-1].abs().pow(2.0)  # [n_freqs, n_frames]

        mel_spec = self.mel_filters.T @ magnitudes  # [n_mels, n_frames]
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0  # [n_mels, n_frames]

        # Ensure fixed frame length (right pad if needed, though for fixed n_samples it should match).
        n_frames = int(log_spec.shape[-1])
        if n_frames < self.cfg.nb_max_frames:
            log_spec = torch.nn.functional.pad(log_spec, (0, self.cfg.nb_max_frames - n_frames))
        elif n_frames > self.cfg.nb_max_frames:
            log_spec = log_spec[..., : self.cfg.nb_max_frames]

        # For full windows (e.g. 30s chunk), avoid returning an attention mask.
        # Passing an all-valid mask into SDPA can prevent fast attention kernels.
        if valid_frames >= self.cfg.nb_max_frames:
            return log_spec.unsqueeze(0), None

        attn = torch.zeros((self.cfg.nb_max_frames,), device=self.device, dtype=torch.long)
        if valid_frames > 0:
            attn[:valid_frames] = 1

        return log_spec.unsqueeze(0), attn.unsqueeze(0)
