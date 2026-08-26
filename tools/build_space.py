"""Assemble a deployable Hugging Face Space folder at space/ (build artifact).

Usage:  python -m tools.build_space

Copies the Gradio demo, the torch-free inference package (+ bundled mel
filters), the int8 ONNX model and the example clips into space/, then writes
requirements.txt and a README.md carrying HF Spaces front matter.

Idempotent: space/ is wiped and recreated on every run, so it is a pure build
artifact (gitignored). Deploy with e.g.

    hf upload <user>/<space-name> space/ . --repo-type=space

The layout mirrors what demo/app.py already probes for: app.py at the root, a
sibling turn_detector/ package (added to sys.path), model_int8.onnx next to
app.py and examples/*.flac — so no source edits are needed.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
SPACE = ROOT / "space"

# (source, destination-relative-to-space/)
FILES = [
    ("demo/app.py", "app.py"),
    ("src/turn_detector/__init__.py", "turn_detector/__init__.py"),
    ("src/turn_detector/common.py", "turn_detector/common.py"),
    ("src/turn_detector/infer.py", "turn_detector/infer.py"),
    ("src/turn_detector/mel_filters.npz", "turn_detector/mel_filters.npz"),
    ("models/model_int8.onnx", "model_int8.onnx"),
]
EXAMPLES_GLOB = ("demo/examples", "*.flac", "examples")

# No torch, no transformers: turn_detector/mel_filters.npz is resolved by
# infer.NumpyLogMel before it would ever fall back to transformers.audio_utils.
REQUIREMENTS = """\
gradio>=4.44
numpy>=1.26
onnxruntime>=1.18
soxr>=0.5
matplotlib>=3.9
soundfile>=0.12
"""

README = """\
---
title: Tiny Hinglish Turn Detector
emoji: 🎙️
colorFrom: indigo
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Done speaking, or just pausing? Hinglish turn detection
---

# 🎙️ Tiny Hinglish Turn Detector

Given the last 8 seconds of speech, this model decides whether the speaker is
**done talking** or **just pausing** — the call a voice agent has to make before
it starts replying. It is tuned for **Indian Hinglish**: code-switching, filler
words (*haan, matlab, accha, toh*), trailing conjunctions (*aur…*) and
mid-thought pauses.

## Usage

1. **Record** with the microphone or **upload** a clip (or click one of the
   examples). Any sample rate / channel count works — it is resampled to 16 kHz
   mono.
2. Hit **Detect turn**. Recordings also fire automatically when you stop.
3. **Decision threshold** slides the complete/incomplete cutoff; lower values
   make the agent more eager to answer, higher values make it wait longer.

You get back the verdict, P(complete) vs P(incomplete), the CPU latency split
(mel extraction + model) and a **streaming simulation** — P(complete)
re-evaluated every 0.24 s as the clip "plays in", so you can watch the
probability dip on a filler or a mid-thought pause and recover once the sentence
actually lands.

Things worth trying: end a sentence cleanly, then say the same sentence but
trail off with *"…matlab"* or *"…aur"*. A good turn detector should hold the
floor for you in the second case even though both clips end in silence.

## Under the hood

- Whisper-Tiny encoder (8 s window) + attention pooling + MLP head → one logit.
- Exported to **int8 ONNX** and run on CPU via onnxruntime.
- **Torch-free and transformers-free**: mel filterbank ships as
  `turn_detector/mel_filters.npz`, the STFT is plain numpy.
- `MODEL_PATH` env var overrides the bundled model.

This Space is a build artifact — regenerate it from the project repo with
`python -m tools.build_space`.
"""


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    missing = [s for s, _ in FILES if not (ROOT / s).exists()]
    if missing:
        raise FileNotFoundError(
            "missing source file(s): " + ", ".join(missing)
            + "\n(train/export first — models/model_int8.onnx comes from the "
              "Kaggle run, see KAGGLE_RUNBOOK.md)"
        )

    if SPACE.exists():
        shutil.rmtree(SPACE)
    SPACE.mkdir(parents=True)

    written: list[Path] = []
    for src, dst in FILES:
        copy(ROOT / src, SPACE / dst)
        written.append(SPACE / dst)

    ex_dir, pattern, ex_dst = EXAMPLES_GLOB
    examples = sorted((ROOT / ex_dir).glob(pattern)) if (ROOT / ex_dir).exists() else []
    for src in examples:
        copy(src, SPACE / ex_dst / src.name)
        written.append(SPACE / ex_dst / src.name)
    if not examples:
        print(f"warning: no {pattern} under {ex_dir}/ — Space ships without examples")

    for name, text in (("requirements.txt", REQUIREMENTS), ("README.md", README)):
        (SPACE / name).write_text(text, encoding="utf-8")
        written.append(SPACE / name)

    print(f"space/ rebuilt at {SPACE}")
    for path in written:
        print(f"  {path.relative_to(SPACE).as_posix():<38} "
              f"{path.stat().st_size / 1024:8.1f} KB")
    total = sum(p.stat().st_size for p in written)
    print(f"  {'total':<38} {total / 1e6:8.2f} MB  ({len(written)} files)")
    print("\ndeploy:  hf upload <user>/<space> space/ . --repo-type=space")


if __name__ == "__main__":
    main()
