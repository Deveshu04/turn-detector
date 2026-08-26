"""CPU inference wrapper around the exported ONNX model.

Torch-free: mel extraction reproduces Whisper's feature math in numpy against a
bundled mel filterbank (mel_filters.npz), so the Gradio Space needs only numpy +
onnxruntime — no torch, no transformers. transformers.audio_utils is used solely
as a fallback to build the filters when the .npz is absent.
"""

import time
from pathlib import Path

import numpy as np

from turn_detector.common import (
    HOP, N_FFT, N_MELS, N_SAMPLES, SAMPLE_RATE, right_align,
)


class NumpyLogMel:
    """Numpy twin of features.LogMel (same math, no torch).

    Mel filters come from a bundled .npz when available (keeps the demo Space
    free of transformers), else transformers.audio_utils computes them.
    """

    def __init__(self, filters_npz: str | Path | None = None):
        import os
        candidates = [
            filters_npz, os.environ.get("MEL_FILTERS_NPZ"),
            Path(__file__).parent / "mel_filters.npz",
        ]
        path = next((p for p in candidates if p and Path(p).exists()), None)
        if path is not None:
            self.filters = np.load(path)["filters"]
        else:
            from transformers.audio_utils import mel_filter_bank
            self.filters = mel_filter_bank(
                num_frequency_bins=1 + N_FFT // 2, num_mel_filters=N_MELS,
                min_frequency=0.0, max_frequency=8000.0,
                sampling_rate=SAMPLE_RATE, norm="slaney", mel_scale="slaney",
            )
        self.window = np.hanning(N_FFT + 1)[:-1]  # periodic hann

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        # vectorized STFT (~10x faster than transformers.audio_utils.spectrogram)
        pad = N_FFT // 2
        x = np.pad(wav.astype(np.float64), pad, mode="reflect")
        n_frames = 1 + (len(x) - N_FFT) // HOP
        frames = np.lib.stride_tricks.as_strided(
            x, shape=(n_frames, N_FFT),
            strides=(x.strides[0] * HOP, x.strides[0]),
        )
        fft = np.fft.rfft(frames * self.window, axis=1)
        power = np.abs(fft[:-1]) ** 2                       # drop last frame
        mel = power @ self.filters                          # (frames, 80)
        log_spec = np.log10(np.clip(mel, 1e-10, None)).T    # (80, frames)
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        return ((log_spec + 4.0) / 4.0).astype(np.float32)


class TurnDetector:
    """detector = TurnDetector("model_int8.onnx"); detector.predict(wav16k)"""

    def __init__(self, onnx_path: str | Path, threshold: float = 0.5,
                 num_threads: int = 1):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.mel = NumpyLogMel()
        self.threshold = threshold

    def predict(self, wav: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
        """wav: mono float array at 16kHz (resample before calling if not)."""
        t0 = time.perf_counter()
        wav = right_align(np.asarray(wav, dtype=np.float32), N_SAMPLES)
        mel = self.mel(wav)[None]                       # (1, 80, 800)
        t1 = time.perf_counter()
        logit = self.session.run(None, {"mel": mel})[0].item()
        t2 = time.perf_counter()
        prob = 1 / (1 + np.exp(-logit))
        return {
            "prob_complete": float(prob),
            "is_complete": bool(prob >= self.threshold),
            "mel_ms": (t1 - t0) * 1000,
            "model_ms": (t2 - t1) * 1000,
            "total_ms": (t2 - t0) * 1000,
        }

    def sliding_probs(self, wav: np.ndarray, step_s: float = 0.24) -> list[dict]:
        """P(complete) as the clip 'plays' — for the demo's streaming view."""
        results = []
        step = int(step_s * SAMPLE_RATE)
        for end in range(step, len(wav) + step, step):
            r = self.predict(wav[: min(end, len(wav))])
            results.append({"t": min(end, len(wav)) / SAMPLE_RATE,
                            "prob": r["prob_complete"]})
        return results


def benchmark(onnx_path: str | Path, n_iter: int = 50,
              num_threads: int = 1) -> dict:
    det = TurnDetector(onnx_path, num_threads=num_threads)
    rng = np.random.default_rng(0)
    wav = rng.normal(0, 0.1, N_SAMPLES).astype(np.float32)
    det.predict(wav)  # warmup
    totals, models = [], []
    for _ in range(n_iter):
        r = det.predict(wav)
        totals.append(r["total_ms"])
        models.append(r["model_ms"])
    return {
        "threads": num_threads,
        "total_ms_p50": round(float(np.percentile(totals, 50)), 2),
        "total_ms_p95": round(float(np.percentile(totals, 95)), 2),
        "model_ms_p50": round(float(np.percentile(models, 50)), 2),
        "size_mb": round(Path(onnx_path).stat().st_size / 1e6, 2),
    }
