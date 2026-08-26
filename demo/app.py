"""Gradio demo: is the speaker done, or just pausing?

Record or upload speech (Hinglish welcome!), get P(turn complete), latency,
and a "streaming view" showing how the decision evolves as the clip plays —
watch the probability dip on fillers (matlab, umm...) and mid-thought pauses.

Torch-free: onnxruntime + numpy only. MODEL_PATH env overrides the default.
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
for candidate in (ROOT / "turn_detector", ROOT.parent / "src" / "turn_detector"):
    if candidate.exists():
        sys.path.insert(0, str(candidate.parent))
        break

import gradio as gr  # noqa: E402

from turn_detector.infer import TurnDetector  # noqa: E402

SR = 16000
MODEL_PATH = os.environ.get("MODEL_PATH") or next(
    (str(p) for p in (ROOT / "model_int8.onnx",
                      ROOT.parent / "models" / "model_int8.onnx") if p.exists()),
    None,
)
if MODEL_PATH is None:
    raise FileNotFoundError(
        "No ONNX model found. Set MODEL_PATH or place model_int8.onnx in demo/ "
        "or models/."
    )
detector = TurnDetector(MODEL_PATH, num_threads=2)

# default decision threshold: tuned on the validation split during training
DEFAULT_THRESHOLD = 0.5
for metrics_candidate in (Path(MODEL_PATH).parent / "metrics.json",):
    if metrics_candidate.exists():
        import json
        try:
            DEFAULT_THRESHOLD = round(
                float(json.loads(metrics_candidate.read_text())["threshold"]), 2)
        except (KeyError, ValueError, json.JSONDecodeError):
            pass


def to_16k_mono(sr: int, wav: np.ndarray) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if wav.dtype.kind in "iu":
        wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
    wav = wav.astype(np.float32)
    if sr != SR:
        try:
            import soxr
            wav = soxr.resample(wav, sr, SR)
        except ImportError:
            x_new = np.linspace(0, len(wav) - 1, int(len(wav) * SR / sr))
            wav = np.interp(x_new, np.arange(len(wav)), wav).astype(np.float32)
    return wav


def analyze(audio: tuple[int, np.ndarray] | None, threshold: float):
    if audio is None:
        return "—", None, "Record or upload a clip first.", None
    sr, wav = audio
    wav = to_16k_mono(sr, wav)
    if len(wav) < SR // 4:
        return "—", None, "Clip too short (need ≥0.25 s).", None

    detector.threshold = threshold
    result = detector.predict(wav)
    verdict = "✅ Done speaking" if result["is_complete"] else "⏳ Still speaking…"
    confidences = {
        "complete (done speaking)": result["prob_complete"],
        "incomplete (just pausing)": 1 - result["prob_complete"],
    }
    latency = (
        f"**Latency** (CPU): mel {result['mel_ms']:.1f} ms + "
        f"model {result['model_ms']:.1f} ms = **{result['total_ms']:.1f} ms** · "
        f"model file {Path(MODEL_PATH).stat().st_size / 1e6:.1f} MB"
    )

    # streaming simulation
    points = detector.sliding_probs(wav, step_s=0.24)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 2.6), dpi=110)
    ts = [p["t"] for p in points]
    ps = [p["prob"] for p in points]
    ax.plot(ts, ps, marker="o", markersize=3)
    ax.axhline(threshold, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("time heard so far (s)")
    ax.set_ylabel("P(complete)")
    ax.set_title("Decision as the clip streams in")
    fig.tight_layout()
    return verdict, confidences, latency, fig


examples_dir = ROOT / "examples"
example_files = sorted(str(p) for p in examples_dir.glob("*.flac")) if examples_dir.exists() else []

with gr.Blocks(title="Hinglish Turn Detection") as app:
    gr.Markdown(
        "# 🎙️ Tiny Hinglish Turn Detector\n"
        "Speak (Hinglish, Hindi, or English) and the model decides: **done "
        "talking, or just pausing?** Try trailing off with *\"matlab…\"*, "
        "*\"aur…\"* or a mid-thought pause — a good turn detector should wait."
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy",
                                label="Your speech")
            threshold = gr.Slider(0.05, 0.95, value=DEFAULT_THRESHOLD, step=0.01,
                                  label="Decision threshold (tuned on validation)")
            btn = gr.Button("Detect turn", variant="primary")
        with gr.Column(scale=1):
            verdict = gr.Textbox(label="Verdict", interactive=False)
            probs = gr.Label(label="Probabilities", num_top_classes=2)
            latency = gr.Markdown()
    stream_plot = gr.Plot(label="Streaming simulation")
    if example_files:
        gr.Examples(examples=example_files, inputs=audio_in)

    btn.click(analyze, [audio_in, threshold],
              [verdict, probs, latency, stream_plot])
    audio_in.stop_recording(analyze, [audio_in, threshold],
                            [verdict, probs, latency, stream_plot])

if __name__ == "__main__":
    app.launch()
