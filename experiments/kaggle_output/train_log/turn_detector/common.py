"""Torch-free audio constants + windowing shared by training and inference."""

import numpy as np

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
