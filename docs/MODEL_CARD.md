---
language:
  - en
  - hi
license: mit
library_name: onnxruntime
pipeline_tag: audio-classification
tags:
  - audio-classification
  - turn-detection
  - hinglish
  - onnx
  - voice-agents
  - code-switching
base_model: openai/whisper-tiny
datasets:
  - pipecat-ai/smart-turn-data-v3.2-train
metrics:
  - accuracy
  - roc_auc
---

# turn-detector — tiny Hinglish turn detection

Given the last 8 seconds of a user's speech, this model returns
**P(turn complete)**: is the speaker done, or just pausing? It is an
audio-only binary classifier — no ASR, no text — built for **Indian Hinglish**
(code-switched Hindi/English, Devanagari and romanized, Indian-accented prosody,
filler words and trailing conjunctions).

- **Architecture:** `openai/whisper-tiny` encoder, positional embeddings sliced
  1500 → 400 to pin an 8 s window, plus learned-query attention pooling and a
  2-layer MLP head. **7,885,953 parameters.**
- **Artifact:** `model_int8.onnx`, **8.5 MB** (dynamic uint8 weight
  quantization; fp32 ONNX is 31.58 MB).
- **Input:** log-mel `(1, 80, 800)` — Whisper's exact feature math over 128,000
  samples of 16 kHz mono, **right-aligned** (last 8 s kept, shorter clips
  left-padded).
- **Output:** one logit; `sigmoid(logit)` = P(turn complete).
- **Decision threshold: 0.50 for `model_int8.onnx`, 0.63 for `model_fp32.onnx`.**
  The int8 0.50 is a calibrated value that happens to land on the framework
  default, not the default itself — see [Thresholds](#thresholds) below for why
  the two files differ.
- **Runtime:** numpy + onnxruntime only. No torch, no transformers — the mel
  filterbank ships as a bundled `.npz`.

### Two variants ship

| file | model | params | int8 size | threshold | overall acc | hinglish acc |
|---|---|---:|---:|---|---|---|
| `model_int8.onnx` | WhisperTinyTurn (`e2_hinglish_aug`) | 7,885,953 | 8.50 MB | **0.50** | 0.938 | **0.951** |
| `model_tinymel_int8.onnx` | TinyMelNet, distilled (`e5_distill`) | 507,265 | 1.28 MB | **0.57** | 0.896 | 0.871 |

Accuracies are fp32 on the full test split (n=9,329 / n=225); the threshold
column is the one to use with the int8 file listed.

The distilled variant is the same 8 s log-mel input and the same single-logit
output, so it is a drop-in swap: a stride-2 Conv1d stem, three depthwise-separable
Conv1d blocks and a BiGRU(128) in place of the Whisper encoder, trained against
the frozen Whisper model's soft outputs (α = 0.3, T = 2.0). It runs in **~14 ms**
against ~91 ms on the same laptop CPU and costs 4.2 accuracy points overall —
notably more on Hinglish (8.0 points), so prefer the accurate model whenever the
budget allows. Its metrics live in `metrics_tinymel.json`.

## Intended use

**VAD-gated end-of-turn detection in a voice agent.** The intended wiring is:
a voice-activity detector fires when speech energy drops; the agent then hands
the last 8 seconds of audio to this model and uses P(turn complete) to decide
whether to start responding or keep listening. The model answers "did the
thought finish", which is the question VAD cannot answer.

It is designed to hold the floor through mid-thought pauses, fillers (*haan,
matlab, accha, yaar, umm*) and dangling conjunctions (*…aur*, *…kyunki*,
*…lekin*), and to correctly release on complete Hinglish sentences — including
ones ending in a post-verb adverb (*"…bahut zyada hai **aaj**."*), which is the
construction an English-trained baseline gets catastrophically wrong.

**Out of scope:** speaker diarization; detecting whether a turn is a question;
languages other than English, Hindi and Hinglish; deciding *what* to reply;
running without a VAD in front of it (the model expects to be asked at a pause,
not on a rolling basis over active speech); and any safety-relevant or
high-stakes decision.

## How to use

```python
import numpy as np
import soundfile as sf
from turn_detector.infer import TurnDetector

detector = TurnDetector("model_int8.onnx", threshold=0.50, num_threads=1)
# smaller/faster, same interface:
# detector = TurnDetector("model_tinymel_int8.onnx", threshold=0.57)

wav, sr = sf.read("clip.flac")          # mono 16 kHz float; resample first if not
result = detector.predict(np.asarray(wav, dtype=np.float32))

print(result["prob_complete"])          # e.g. 0.91
print(result["is_complete"])            # True  -> the agent may start speaking
print(result["total_ms"], result["model_ms"])
```

`predict` handles the 8 s right-aligned windowing internally: pass whatever you
have and it keeps the last 8 seconds, left-padding shorter clips. For a
streaming view of how the decision evolves as audio arrives:

```python
curve = detector.sliding_probs(wav, step_s=0.24)   # [{"t": 0.24, "prob": ...}, ...]
```

The class is torch-free — `numpy` and `onnxruntime` are the only hard
dependencies.

## Thresholds

**A threshold belongs to a file, not to the task.** Use these:

| artifact | threshold | how it was chosen |
|---|---|---|
| `model_fp32.onnx` | **0.63** | accuracy-tuned on the validation split |
| `model_int8.onnx` | **0.50** | *decision-matched* to fp32@0.63 (below) |
| `model_tinymel_int8.onnx` | **0.57** | accuracy-tuned on the validation split |

Dynamic int8 quantization preserves the ranking but shifts the probabilities, so
inheriting the fp32 threshold across the quantization boundary silently moves the
operating point. Scoring 600 held-in synthetic clips with both graphs and picking
the int8 threshold that reproduces the most fp32-at-0.63 verdicts gives **0.50,
agreeing with fp32 on 99.7% of them**. No test labels were used. This is recorded
as `int8_threshold_decision_matched` in `metrics.json`; the reference demo reads
that key when present and falls back to the fp32 `threshold` otherwise.

If you requantize, re-derive the threshold for your own artifact rather than
copying any number here. Lowering it makes the agent more eager to reply; raising
it makes it wait longer.

## Metrics

Evaluated on the official `pipecat-ai/smart-turn-data-v3.2-test` split
(English and Hindi slices) plus a **template-disjoint** synthetic Hinglish test
split. `human_audio` is the real-human, non-synthetic subset of the official
test set. All figures below are **fp32, at threshold 0.63**, unless labelled
int8.

| slice | n | accuracy | AUC |
|---|---:|---|---|
| overall | 9,329 | 0.938 | 0.983 |
| english | 7,820 | 0.938 | 0.983 |
| hindi | 1,284 | 0.932 | 0.984 |
| **hinglish** | 225 | **0.951** | **0.986** |
| filler | 2,381 | 0.914 | 0.971 |
| human_audio | 5,367 | 0.947 | 0.987 |

Against the same architecture trained without synthetic Hinglish data
(`e1_baseline`), on the identical 225-clip Hinglish split:

| | baseline | this model | Δ |
|---|---|---|---|
| hinglish accuracy | 0.631 | **0.951** | **+32.0 pts** |
| hinglish AUC | 0.612 | **0.986** | +0.374 |
| recall on complete turns | 0.148 | **0.898** | +0.750 |
| english accuracy (n=7,820) | 0.938 | 0.938 | 0.000 |
| hindi accuracy (n=1,284) | 0.931 | 0.932 | +0.001 |

### The other runs, same test split (fp32, each at its own tuned threshold)

| slice | n | E2 **this model** | E3 tiny, no teacher | E5 tiny, **distilled** | E6 2.4× data |
|---|---:|---|---|---|---|
| overall | 9,329 | **0.938** | 0.871 | 0.896 | 0.943 |
| english | 7,820 | 0.938 | 0.868 | 0.898 | 0.945 |
| hindi | 1,284 | 0.932 | 0.884 | 0.885 | 0.931 |
| **hinglish** | 225 | **0.951** | 0.871 | 0.871 | 0.938 |
| filler | 2,381 | 0.914 | 0.856 | 0.870 | 0.915 |
| human_audio | 5,367 | 0.947 | 0.877 | 0.904 | 0.954 |
| AUC, hinglish | 225 | **0.986** | 0.949 | 0.962 | 0.963 |
| AUC, overall | 9,329 | 0.983 | 0.944 | 0.959 | 0.986 |

Two results worth stating explicitly, because they explain what ships:

- **E5 (`model_tinymel_int8.onnx`) is E3 with a teacher.** Distilling this model's
  frozen checkpoint into the 507k-parameter architecture buys **+2.5 points
  overall** (0.871 → 0.896) and **+3.0 on English** at identical size, and lifts
  Hinglish AUC 0.949 → 0.962. E5 also trains on 2.4× the rows E3 saw, so the two
  effects are not separated.
- **E6 trains on 111,509 rows** (English uncapped plus a 21-language tail, 2.4×)
  and beats this model nearly everywhere — overall, English, `human_audio` — while
  **losing on Hinglish** (0.938 vs 0.951; AUC 0.963 vs 0.986). The synthetic
  Hinglish corpus stayed the same size, so its share of training fell from 4.6%
  to 1.9% and its effect was diluted. Since Hinglish is the target domain, E6 is
  not shipped and E2 is also what E5 is distilled from. Note that E6's
  multilingual tail is **train-only** and the v3.2 test split is English+Hindi,
  so nothing in this table measures E6 on those 21 languages.

### Per-kind breakdown, Hinglish split (int8, threshold 0.63)

| kind | label | n | accuracy |
|---|---|---:|---|
| `full` (complete sentence) | complete | 75 | 0.907 |
| `midfiller_full` (complete, filler mid-utterance) | complete | 13 | 0.846 |
| `cut` (truncated mid-speech) | incomplete | 107 | 0.991 |
| `tail_conj` (ends on a dangling conjunction) | incomplete | 15 | 0.867 |
| `tail_filler` (ends on a trailing filler) | incomplete | 15 | 1.000 |
| all | mixed | 225 | 0.947 |

This breakdown predates the int8 threshold calibration above and was scored at
0.63 on the int8 file. It is left as measured; at the decision-matched 0.50 the
int8 verdicts track fp32, whose Hinglish accuracy is 0.951.

### Robustness to trailing silence

Appending silence must not change the verdict — this is the property that
determines whether an agent interrupts. Measured on the 225 Hinglish clips
(int8), against an otherwise identical model trained without pause augmentation:

| | this model | no-pause-aug ablation |
|---|---|---|
| static hinglish accuracy (fp32) | 0.951 | 0.964 |
| decisions flipped by +0.5 s silence | **3.1%** (7/225) | 11.6% (26/225) |
| decisions flipped by +1.0 s silence | **8.4%** (19/225) | 28.9% (65/225) |
| early fires at +1.0 s (incomplete → complete) | **5.1%** (7/137) | 44.5% (61/137) |

The ablation scores higher on the static benchmark and interrupts the user on
44.5% of still-speaking clips after a one-second pause. Static accuracy alone
does not predict production behaviour for this task.

### int8 vs fp32

| | AUC — fp32, full test (n=9,329) | AUC — int8, subset (n=2,000) |
|---|---|---|
| overall | 0.983 | 0.978 |
| distilled variant | 0.959 | 0.963 |

The int8 figure is measured on a class-balanced stratified 2,000-clip subset
(1,000 per label), so its **accuracy** is not comparable to the full-test
accuracy above; AUC is the comparable quantity and it drops by 0.005 for this
model (and is flat for the distilled one). fp32 ONNX matches torch to
max |Δprob| = 1.19e-07.

Quantization costs **ranking quality almost nothing and calibration quite a lot**.
Scored at the fp32 threshold of 0.63, the int8 subset accuracy is 0.887 — which
looks like a quantization penalty but is not one, since the AUC is intact. At the
decision-matched threshold of 0.50 the int8 model reproduces fp32 verdicts on
99.7% of the calibration clips. Use 0.50 with the int8 file.

### Latency and size

| | this model | distilled variant |
|---|---|---|
| params | 7,885,953 | 507,265 |
| int8 ONNX | 8.50 MB | 1.28 MB |
| fp32 ONNX | 31.58 MB | 2.03 MB |
| p50 total, 1 thread (dev laptop) | ~91 ms (~83 ms model + ~8 ms mel) | ~14 ms (~7 ms model + ~7 ms mel) |
| p50 total, 4 threads (dev laptop) | ~59 ms | ~18 ms |

The mel frontend costs a fixed ~8 ms either way, which is why it dominates the
distilled model's budget and is a rounding error in this one — and why the small
model is *slower* at 4 threads than at 1 (too little work to amortize the thread
handoff).

Laptop measurements vary about 35% between sessions with thermal and power
state; an earlier session on the same machine measured ~126 ms single-threaded.
Server CPUs are several times faster — pipecat's smart-turn v3 reports ~12 ms
for the same encoder class — so treat these as a conservative ceiling.

## Training data

**Real (44,751 train / 2,255 val rows):** the English and Hindi subsets of
[`pipecat-ai/smart-turn-data-v3.2`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train),
streamed from the 41 GB train split (270,946 rows scanned), resampled to 16 kHz
mono and truncated to the last 8 s. English is capped at 17,500 rows per label
so the subset does not inherit the source class skew; all Hindi is kept.
Language codes in the raw data are ISO-639-3 (`eng`, `hin`). Evaluation uses the
dataset's **official** test split: 7,820 English + 1,284 Hindi clips.

**Synthetic Hinglish (2,157 train / 87 val / 225 test clips; 2,469 total,
2.07 h):** generated in this project. 89 hand-written bilingual slot templates
over everyday Indian service-app scenarios expand to 189 sentence instances,
rendered by 4 Indian edge-tts neural voices (`hi-IN-SwaraNeural`,
`hi-IN-MadhurNeural`, `en-IN-NeerjaNeural`, `en-IN-PrabhatNeural`) in both
Devanagari (1,283 clips) and romanized Latin (1,186 clips). Incomplete variants
come from word-boundary audio cuts, trailing conjunctions (*aur, lekin, kyunki,
toh, matlab, par, phir, agar*) and trailing fillers (*umm, matlab, wo kya hai
na, haan toh, actually, basically…*); complete-but-disfluent variants inject a
filler mid-utterance (*matlab, accha, haan, wo, umm, yaar, bas*).

The synthetic split is assigned at **template level**, not sentence level, so
slot variants of a template and all of their cut/tail children stay in one
split. Measured overlap between train and test templates is **0**. An earlier
sentence-level split had 69% template overlap; it was found and fixed before any
model reported here was trained.

## Training procedure

Kaggle T4, ~10.6 minutes, mixed precision. AdamW with a split learning rate
(encoder 1e-5, head 1e-4), weight decay 0.01, 5% warmup then cosine decay,
gradient clipping 1.0, batch size 64, 4 epochs over 46,908 rows, seed 42.
Best validation AUC 0.984.

Augmentation, applied to the training split only:

| augmentation | p | effect on label |
|---|---|---|
| `pause_cut` — truncate a **complete** utterance at a speech-active point (40–85% of its active span) and append 0.2–1.2 s silence | 0.15 | flips to incomplete |
| `trailing_silence` — append 0.2–1.2 s silence to any clip | 0.50 | preserved |
| `noise` — additive Gaussian at 10–30 dB SNR | 0.25 | preserved |
| `speed` — resample at 0.9–1.1× | 0.25 | preserved |

The sampler oversamples complete clips by `1/(1 − p_cut)` so that batches are
still balanced *after* `pause_cut` flips labels on the fly.

Export: `torch.onnx.export` at opset 17 with static batch 1, then
`onnxruntime.quantization.quantize_dynamic` (`QuantType.QUInt8`).

**The distilled variant** (`model_tinymel_int8.onnx`, `e5_distill`) uses the same
augmentation recipe on an expanded prep — 111,509 train rows: English uncapped,
all Hindi, a 21-language tail capped at 850 per (language, label), plus the same
2,157 synthetic Hinglish clips — for 8 epochs (head LR 3e-4), 32.1 minutes on a
T4, best validation AUC 0.963. Its loss blends the hard label with this model's
frozen soft outputs on the identical augmented batch:
`0.3·BCE(student, label) + 0.7·BCE(student/T, σ(teacher/T))` at `T = 2.0`.

## Bias, risks and limitations

- **The main Hinglish evaluation is TTS-synthetic; the real-voice check is
  small.** The 0.951 comes from edge-tts audio. A 30-clip single-speaker
  phone-microphone set (fresh sentences, pre-registered thresholds) measured
  the real-world gap directly: **0.800 accuracy / AUC 0.942**, with 15/16
  incompletes correct and 1/16 early-fires after a 1-second real pause — the
  errors sit almost entirely on complete sentences judged "still speaking"
  (the model waits rather than interrupts). One speaker and n=30 is
  directional evidence only; broader real code-switched evaluation remains
  future work.
- **Four voices, two genders, one country.** All Hinglish training and test
  audio comes from `hi-IN-Swara`, `hi-IN-Madhur`, `en-IN-Neerja` and
  `en-IN-Prabhat`. Accent, age and recording-condition diversity are far below
  production reality, and the model may key on voice-specific end-of-utterance
  prosody. Regional Indian accents, older speakers, children, and non-Indian
  Hindi/English speakers are unrepresented in the Hinglish portion.
- **Narrow domain.** The 89 templates cover delivery, recharge, travel, leave
  approval and traffic. Medical, legal, financial or technical Hinglish is
  untested.
- **Thresholds are per file (0.50 int8 / 0.63 fp32 / 0.57 distilled).** The tuned
  threshold varies widely across otherwise-identical training runs (0.35–0.63),
  so it is a property of the specific exported file, not of the task, and
  quantization shifts it again. Retune on your own validation data against the
  exact `.onnx` you serve, especially after any requantization. The
  decision-matching calibration behind the int8 0.50 was measured on **600
  held-in synthetic clips for this model only** — it says nothing about agreement
  on real human audio, and it was not derived for the distilled variant.
- **The distilled variant inherits this model's biases by construction.** 70% of
  its training signal is this model's soft outputs, so every prior encoded here —
  including everything learned from four TTS voices and 89 templates — transfers
  to it. Its numbers are not independent evidence about those priors. A student
  cannot correct a teacher's systematic error; it can only fail to reach it,
  which is what its flat Hinglish accuracy (0.871, against the teacher's 0.951)
  looks like in practice. Do not read the distilled model as a second opinion.
- **Single seed (42), one run per configuration.** No error bars; differences
  under ~1.5 points on the 225-clip Hinglish slice should not be treated as
  real.
- **The silence robustness test uses digital zeros**, not room tone. Real pauses
  carry breath and mic self-noise.
- **Not measured:** interaction with any specific VAD firing policy; p99
  latency; behaviour under repeated streaming calls as a pause lengthens;
  server-hardware latency.
- **Dataset licensing.** smart-turn-data v3.2 aggregates several upstream
  corpora under their own licenses. Those per-source terms — not this card's
  MIT line — govern redistribution and commercial use of models trained on it.

## Citation and attribution

Approach informed by [pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn)
(BSD-2-Clause); the pipeline code in this project is written from scratch.
Base model: [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny)
(MIT). Synthetic audio: [edge-tts](https://github.com/rany2/edge-tts) with
Microsoft neural voices, subject to Microsoft's terms for the Edge read-aloud
service.

Full experimental report, including the ablation that motivates the pause
augmentation: `experiments/REPORT.md` in the project repository.
