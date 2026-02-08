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

        - input_features: [1, n_mels, n_frames] float32
        - attention_mask: None (single-sample path, no frame padding)
        """
        if sampling_rate != 16000:
            raise ValueError(f"Expected sampling_rate=16000, got {sampling_rate}")

        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio 1D array, got shape={audio.shape}")

        waveform = torch.from_numpy(audio).to(device=self.device, dtype=torch.float32)

        if self.cfg.dither != 0.0:
            waveform = waveform + float(self.cfg.dither) * torch.randn_like(waveform)

        stft = torch.stft(
            waveform,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            window=self.window,
            return_complex=True,
        )
        magnitudes = stft.abs().pow(2.0)  # [n_freqs, n_frames]
        if magnitudes.shape[-1] > 1:
            # Keep behavior consistent with WhisperFeatureExtractor for normal-length inputs.
            magnitudes = magnitudes[..., :-1]

        mel_spec = self.mel_filters.T @ magnitudes  # [n_mels, n_frames]
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0  # [n_mels, n_frames]

        return log_spec.unsqueeze(0), None
