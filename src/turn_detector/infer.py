"""CPU inference wrapper around the exported ONNX model.

Torch-free: mel extraction uses transformers' numpy WhisperFeatureExtractor
math via audio_utils, so the Gradio Space only needs numpy + onnxruntime +
transformers (no torch install).
"""

import time
from pathlib import Path

import numpy as np

from turn_detector.features import (
    HOP, N_FFT, N_MELS, N_SAMPLES, SAMPLE_RATE, right_align,
)


class NumpyLogMel:
    """Numpy twin of features.LogMel (same math, no torch)."""

    def __init__(self):
        from transformers.audio_utils import mel_filter_bank, window_function
        self.filters = mel_filter_bank(
            num_frequency_bins=1 + N_FFT // 2, num_mel_filters=N_MELS,
            min_frequency=0.0, max_frequency=8000.0,
            sampling_rate=SAMPLE_RATE, norm="slaney", mel_scale="slaney",
        )
        self.window = window_function(N_FFT, "hann")

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        from transformers.audio_utils import spectrogram
        log_spec = spectrogram(
            wav, self.window, frame_length=N_FFT, hop_length=HOP,
            power=2.0, mel_filters=self.filters, log_mel="log10",
            mel_floor=1e-10,
        )[:, :-1]
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
