"""Raw-waveform front end for the browser demo: log-mel as exportable ops.

features.LogMel calls torch.stft, which does not lower to a portable ONNX
graph. MelDFT recomputes the identical spectrogram from explicit reflect
padding, a pair of fixed Conv1d DFT kernels and a matmul against the same
slaney mel filters, so the browser can hand the model raw 16 kHz samples and
run zero DSP of its own. Parity with LogMel is asserted in
tests/test_melgraph.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from turn_detector.common import HOP, N_FFT, N_FRAMES

LN10 = math.log(10.0)
N_FREQS = 1 + N_FFT // 2


class MelDFT(nn.Module):
    """waveform (B, 128000) float32 -> log-mel (B, 80, 800), ONNX-friendly."""

    def __init__(self):
        super().__init__()
        from turn_detector.infer import NumpyLogMel
        filters = torch.from_numpy(NumpyLogMel().filters).float()   # (201, 80)
        self.register_buffer("mel_filters", filters.T.contiguous())

        # DFT as cross-correlation: conv1d is not conjugated, so the cos bank
        # gives Re(X) and the negated sin bank gives Im(X). Built in float64
        # to keep the kernels as close to torch.stft's twiddles as float32 allows.
        window = torch.hann_window(N_FFT, periodic=True, dtype=torch.float64)
        n = torch.arange(N_FFT, dtype=torch.float64)
        k = torch.arange(N_FREQS, dtype=torch.float64).unsqueeze(1)
        angle = 2.0 * math.pi * k * n / N_FFT                       # (201, 400)
        self.register_buffer(
            "cos_kernel", (window * torch.cos(angle)).float().unsqueeze(1))
        self.register_buffer(
            "sin_kernel", (-window * torch.sin(angle)).float().unsqueeze(1))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        x = F.pad(wav.unsqueeze(1), (N_FFT // 2, N_FFT // 2), mode="reflect")
        real = F.conv1d(x, self.cos_kernel, stride=HOP)             # (B, 201, 801)
        imag = F.conv1d(x, self.sin_kernel, stride=HOP)
        power = (real * real + imag * imag)[..., :N_FRAMES]         # drop last frame
        mel = self.mel_filters @ power                              # (B, 80, 800)
        log_spec = torch.log(torch.clamp(mel, min=1e-10)) / LN10
        log_spec = torch.maximum(
            log_spec, log_spec.amax(dim=(1, 2), keepdim=True) - 8.0
        )
        return (log_spec + 4.0) / 4.0


class FullTurnModel(nn.Module):
    """waveform (1, 128000) float32 -> turn-completion logit (1,)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.mel = MelDFT()
        self.model = model

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.model(self.mel(wav))
