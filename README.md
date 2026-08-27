# turn-detector — tiny audio turn detection for Hinglish

**Is the user done speaking, or just pausing?** This is a 7.9M-parameter
audio-only classifier that takes the last 8 seconds of speech and returns
P(turn complete) — the call a voice agent has to make before it starts talking.
It is built for **Indian Hinglish**: code-switched Hindi/English in both
Devanagari and romanized script, Indian-accented prosody, filler words (*haan,
matlab, accha, yaar*) and trailing conjunctions (*…aur*, *…kyunki*). A
Whisper-Tiny baseline trained only on real English and Hindi data scores **63.1%
on held-out Hinglish (n=225)** and gets *complete* Hinglish sentences wrong 84%
of the time (n=75); adding 2,157 synthetic code-switched training clips takes that to
**95.1% (+32.0 points)** with **zero regression** on English (0.938 → 0.938,
n=7,820), Hindi (0.931 → 0.932, n=1,284) or the real-human slice (0.946 →
0.947, n=5,367). The shipped model is an **8.5 MB int8 ONNX** that runs on CPU
with numpy and onnxruntime — no torch, no transformers.

The result worth reading twice is the ablation. Dropping pause augmentation
produces a model that is *better* on every static table (96.4% Hinglish vs
95.1%) and **fires on 44.5% of still-speaking clips after a one-second pause**,
against 5.1% for the shipped model. Static test sets cannot see this;
production users feel it immediately. Full write-up in
**[`experiments/REPORT.md`](experiments/REPORT.md)**.

## Results

Accuracy at each run's tuned threshold, fp32, on the official
smart-turn v3.2 test split plus a template-disjoint synthetic Hinglish split.

| slice | n | E1 baseline | **E2 shipped** | E3 tiny-from-scratch |
|---|---:|---|---|---|
| overall | 9,329 | 0.930 | **0.938** | 0.871 |
| english | 7,820 | 0.938 | **0.938** | 0.868 |
| hindi | 1,284 | 0.931 | **0.932** | 0.884 |
| **hinglish** | 225 | 0.631 | **0.951** | 0.871 |
| filler | 2,381 | 0.912 | **0.914** | 0.856 |
| human_audio | 5,367 | 0.946 | **0.947** | 0.877 |
| — AUC, hinglish | 225 | 0.612 | **0.986** | 0.949 |
| — AUC, overall | 9,329 | 0.981 | **0.983** | 0.944 |
| params | | 7,885,953 | 7,885,953 | 507,265 |
| int8 ONNX | | 8.50 MB | 8.50 MB | 1.28 MB |
| tuned threshold | | 0.35 | **0.63** | 0.53 |

- **E1** `e1_baseline` — real English + Hindi only, no augmentation.
- **E2** `e2_hinglish_aug` — E1 data + synthetic Hinglish + pause/silence/noise/speed augmentation. **This is the shipped model.**
- **E3** `e3_tinymel_scratch` — same data as E2, 507k-param from-scratch mel-CNN + BiGRU.
- **E4** `e4_no_pause_aug` — E2 minus pause augmentation; see below.

Full tables including E4: [`experiments/RESULTS.md`](experiments/RESULTS.md).

### Silence stress test — why static accuracy is not enough

Append silence to the 225 Hinglish test clips and re-run. Nothing about what was
said changed, so no verdict should change. E4 scores *higher* than E2 on the
static Hinglish slice (0.964 vs 0.951) and falls apart here.

| | **E2 (with pause aug)** | E4 (no pause aug) |
|---|---|---|
| static hinglish accuracy, fp32 (n=225) | 0.951 | **0.964** |
| decisions flipped by +0.5 s silence | **3.1%** (7/225) | 11.6% (26/225) |
| decisions flipped by +1.0 s silence | **8.4%** (19/225) | 28.9% (65/225) |
| early fires at +0.5 s (incomplete → complete) | **1.5%** (2/137) | 16.8% (23/137) |
| early fires at +1.0 s (incomplete → complete) | **5.1%** (7/137) | 44.5% (61/137) |

An early fire is the agent interrupting the user. At a one-second pause E4 does
it on 61 of the 137 still-speaking clips; E2 does it on 7. Two augmentations
buy this: `pause_cut` (truncate a *complete* utterance mid-speech, append
silence, flip the label to incomplete) and `trailing_silence` (pad any clip,
keep the label). Together they break the "long silence ⇒ done" correlation that
a curated corpus otherwise hands the model for free.

Source: [`experiments/silence_stress_test.json`](experiments/silence_stress_test.json).

## Quickstart

```bash
# install (Python 3.12)
uv sync

# tests — 31 of them, ~25 s
uv run python -m pytest -q

# Gradio demo on localhost: record or upload a clip, watch the streaming
# probability curve dip on a filler and recover when the sentence lands
uv run python demo/app.py
```

Inference from Python needs only numpy and onnxruntime:

```python
import numpy as np, soundfile as sf
from turn_detector.infer import TurnDetector

detector = TurnDetector("models/model_int8.onnx", threshold=0.63)
wav, sr = sf.read("demo/examples/incomplete_trailing_aur.flac")  # 16 kHz mono
print(detector.predict(np.asarray(wav, dtype=np.float32)))
# {'prob_complete': ..., 'is_complete': False, 'mel_ms': ..., 'model_ms': ..., 'total_ms': ...}
```

Latency, int8 ONNX on the dev laptop (onnxruntime CPU, p50 over 100 iterations,
includes numpy mel extraction): **E2 ≈ 91 ms** single-thread, **E3 ≈ 14 ms**.
These are laptop numbers and vary ~35% between sessions with thermal state;
server CPUs are several times faster — pipecat's smart-turn v3 reports ~12 ms
for the same encoder class. See [`experiments/REPORT.md`](experiments/REPORT.md) §6.

Training runs on Kaggle (free T4, ~10–16 min per experiment) — see
[`KAGGLE_RUNBOOK.md`](KAGGLE_RUNBOOK.md).

## Repo map

```
src/turn_detector/     library — the single source of truth
  common.py            audio constants + right-aligned 8 s windowing (torch-free)
  features.py          Whisper-exact log-mel, torch (parity-tested vs HF)
  infer.py             torch-free numpy mel + ONNX runner + benchmark()
  model.py             WhisperTinyTurn (7.9M) and TinyMelNet (507k)
  augment.py           pause_cut, trailing_silence, noise, speed
  dataset.py           manifest loading + label-flip-aware balanced sampler
  train.py             training loop, slice metrics, threshold tuning, ONNX export
  config.py            E1-E4 experiment definitions
synth/                 synthetic Hinglish corpus + edge-tts pipeline
  sentences.py         89 bilingual slot templates, fillers, conjunctions
  corpus.py            template expansion -> corpus_plan.jsonl
  tts_generate.py      edge-tts rendering with word-boundary timestamps
  package_kaggle.py    template-level split (no leakage) -> manifest.parquet
tools/
  build_notebooks.py   generates the Kaggle notebooks from the tested library
  push_kaggle.py       Kaggle CLI driver (prep / train / status / pull / resume)
  aggregate_results.py run_*/metrics.json -> experiments/RESULTS.md
  error_analysis.py    worst errors, per-kind accuracy, threshold sweep, plots
  build_space.py       assembles a deployable torch-free HF Space at space/
notebooks/kaggle/      01_data_prep.ipynb, 02_train.ipynb (generated)
experiments/           REPORT.md, RESULTS.md, silence_stress_test.json, run_*/
models/                shipped E2 artifacts (fp32 + int8 ONNX, metrics.json)
demo/                  Gradio app + example clips
docs/MODEL_CARD.md     HF model card
tests/                 31 tests incl. HF feature-extractor parity
```

## Links

- **HF Space (demo):** TODO — deploy with `python -m tools.build_space` then
  `hf upload <user>/<space> space/ . --repo-type=space`
- **HF Hub (model):** TODO — upload `models/model_int8.onnx`,
  `models/model_fp32.onnx`, `models/metrics.json` and `docs/MODEL_CARD.md`
- **Synthetic Hinglish dataset:** TODO — `synth/output/hinglish-synth.zip`
  (2,469 clips / 2.07 h) currently lives as a Kaggle dataset
- Experimental report: [`experiments/REPORT.md`](experiments/REPORT.md)
- Model card: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)
- Training runbook: [`KAGGLE_RUNBOOK.md`](KAGGLE_RUNBOOK.md)

## Attribution

Approach informed by [pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn)
(BSD-2-Clause); all pipeline code here is written from scratch.

Training and evaluation data: **pipecat-ai smart-turn-data v3.2**
([train](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train),
[test](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test)) —
English and Hindi subsets only. The dataset aggregates several upstream corpora
under their own licenses; see the dataset card, whose per-source terms govern
redistribution and commercial use of anything trained on it.

Base model: [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny)
(MIT). Synthetic Hinglish audio generated with
[edge-tts](https://github.com/rany2/edge-tts) using four Indian Microsoft neural
voices (`hi-IN-SwaraNeural`, `hi-IN-MadhurNeural`, `en-IN-NeerjaNeural`,
`en-IN-PrabhatNeural`) — subject to Microsoft's terms for the Edge read-aloud
service. Project code is MIT.
