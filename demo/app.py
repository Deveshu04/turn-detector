"""Gradio demo: is the speaker done, or just pausing?

Record or upload speech (Hinglish welcome!), get P(turn complete), latency,
and a "streaming view" showing how the decision evolves as the clip plays —
watch the probability dip on fillers (matlab, umm...) and mid-thought pauses.

Ships whichever models it finds next to the app (or in ../models): the accurate
Whisper-tiny one and, when it exists, the distilled TinyMelNet — the dropdown
only offers models that are actually present, so the demo runs unchanged with
just one of them.

Torch-free: onnxruntime + numpy only. MODEL_PATH env overrides the default.
"""

import json
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

# (dropdown label, onnx filename, metrics filename). Order = dropdown order,
# first present entry is the default.
MODEL_SPECS = [
    ("accurate (Whisper-tiny, 8.5 MB)", "model_int8.onnx", "metrics.json"),
    ("fast (TinyMelNet, 1.3 MB)", "model_tinymel_int8.onnx", "metrics_tinymel.json"),
]

ENV_MODEL = os.environ.get("MODEL_PATH")
SEARCH_DIRS = ([Path(ENV_MODEL).parent] if ENV_MODEL else []) + [
    ROOT, ROOT.parent / "models",
]


def find_model(filename: str) -> Path | None:
    return next((d / filename for d in SEARCH_DIRS if (d / filename).exists()), None)


def tuned_threshold(metrics_path: Path) -> float:
    """Deployment threshold: the int8 decision-matched value when present
    (quantization shifts the operating point), else the fp32-tuned one."""
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text())
            return round(float(
                m.get("int8_threshold_decision_matched", m["threshold"])), 2)
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return 0.5


MODELS: dict[str, dict] = {}
for label, filename, metrics_name in MODEL_SPECS:
    # MODEL_PATH pins the first (accurate) entry; the rest are discovered
    path = (Path(ENV_MODEL) if ENV_MODEL and filename == MODEL_SPECS[0][1]
            else find_model(filename))
    if path is None or not path.exists():
        continue
    MODELS[label] = {"path": str(path),
                     "threshold": tuned_threshold(path.parent / metrics_name)}

if not MODELS:
    raise FileNotFoundError(
        "No ONNX model found. Set MODEL_PATH or place model_int8.onnx in demo/ "
        "or models/."
    )

DEFAULT_MODEL = next(iter(MODELS))
DEFAULT_THRESHOLD = MODELS[DEFAULT_MODEL]["threshold"]
MODEL_PATH = MODELS[DEFAULT_MODEL]["path"]

# one session per model, built on first use: loading both up front would cost
# ~10 MB of ONNX for a model the visitor may never select
_detectors: dict[str, TurnDetector] = {}


def get_detector(label: str) -> TurnDetector:
    if label not in MODELS:
        label = DEFAULT_MODEL
    if label not in _detectors:
        _detectors[label] = TurnDetector(MODELS[label]["path"], num_threads=2)
    return _detectors[label]


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


def analyze(audio: tuple[int, np.ndarray] | None, threshold: float,
            model_label: str = DEFAULT_MODEL):
    if audio is None:
        return "—", None, "Record or upload a clip first.", None
    sr, wav = audio
    wav = to_16k_mono(sr, wav)
    if len(wav) < SR // 4:
        return "—", None, "Clip too short (need ≥0.25 s).", None

    detector = get_detector(model_label)
    detector.threshold = threshold
    result = detector.predict(wav)
    verdict = "✅ Done speaking" if result["is_complete"] else "⏳ Still speaking…"
    confidences = {
        "complete (done speaking)": result["prob_complete"],
        "incomplete (just pausing)": 1 - result["prob_complete"],
    }
    model_file = Path(MODELS.get(model_label, MODELS[DEFAULT_MODEL])["path"])
    latency = (
        f"**Latency** (CPU): mel {result['mel_ms']:.1f} ms + "
        f"model {result['model_ms']:.1f} ms = **{result['total_ms']:.1f} ms** · "
        f"{model_file.name} {model_file.stat().st_size / 1e6:.1f} MB"
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
            model_sel = gr.Dropdown(
                choices=list(MODELS), value=DEFAULT_MODEL, label="Model",
                info="both are int8 ONNX on CPU; the fast one is distilled "
                     "from the accurate one",
                interactive=len(MODELS) > 1,
            )
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

    def on_model_change(label):
        # each model has its own validation-tuned threshold; carrying the other
        # model's cutoff across would misrepresent both
        return gr.update(value=MODELS[label]["threshold"])

    model_sel.change(on_model_change, model_sel, threshold)

    inputs = [audio_in, threshold, model_sel]
    outputs = [verdict, probs, latency, stream_plot]
    btn.click(analyze, inputs, outputs)
    audio_in.stop_recording(analyze, inputs, outputs)

if __name__ == "__main__":
    app.launch()
