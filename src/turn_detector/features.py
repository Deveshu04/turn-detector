"""Whisper-compatible log-mel frontend, windowed to the LAST 8 seconds.

Reimplements transformers' WhisperFeatureExtractor math in torch (batchable,
GPU-capable) instead of numpy: STFT(n_fft=400, hop=160, hann, center/reflect),
drop last frame, slaney mel (80 bins, 0-8kHz), log10 -> clamp to max-8 ->
(x+4)/4. Parity with the HF extractor is asserted in tests/test_features.py.

Turn detection cares about how speech ENDS, so windows are right-aligned:
the final audio sample always lands at the last mel frame; short clips are
zero-padded on the LEFT.
"""

import numpy as np
import torch
import torch.nn as nn

SAMPLE_RATE = 16000
WINDOW_SECONDS = 8.0
N_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)   # 128000
N_FFT = 400
HOP = 160
N_MELS = 80
N_FRAMES = N_SAMPLES // HOP                      # 800 mel frames
N_ENCODER_POSITIONS = N_FRAMES // 2              # 400 after Whisper's stride-2 conv


def right_align(wav: np.ndarray, n_samples: int = N_SAMPLES) -> np.ndarray:
    """Keep the last n_samples; left-pad with zeros if shorter."""
    wav = wav[-n_samples:]
    if len(wav) < n_samples:
        wav = np.concatenate([np.zeros(n_samples - len(wav), dtype=wav.dtype), wav])
    return wav.astype(np.float32)


class LogMel(nn.Module):
    """waveform (B, 128000) float32 -> log-mel (B, 80, 800)."""

    def __init__(self):
        super().__init__()
        from transformers.audio_utils import mel_filter_bank
        filters = mel_filter_bank(
            num_frequency_bins=1 + N_FFT // 2,
            num_mel_filters=N_MELS,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=SAMPLE_RATE,
            norm="slaney",
            mel_scale="slaney",
        )  # (201, 80)
        self.register_buffer("mel_filters", torch.from_numpy(filters).float())
        self.register_buffer("window", torch.hann_window(N_FFT, periodic=True))

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        stft = torch.stft(
            wav, N_FFT, HOP, window=self.window,
            center=True, pad_mode="reflect", return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2                  # (B, 201, 800)
        mel = self.mel_filters.T @ magnitudes                   # (B, 80, 800)
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(
            log_spec, log_spec.amax(dim=(1, 2), keepdim=True) - 8.0
        )
        return (log_spec + 4.0) / 4.0
