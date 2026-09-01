# Tiny audio turn detection for Hinglish: experimental report

## 1. Goal and framing

A voice agent has to answer one question every few hundred milliseconds: *has the
user finished their turn, or are they just pausing?* Voice-activity detection
answers a different question: it tells you speech stopped, not that the thought
finished. The gap between those two is where agents interrupt people.

This project trains a small audio-only classifier for that decision. It consumes
the last 8 seconds of the user's speech and emits a single probability,
P(turn complete). It is meant to sit behind a VAD: the VAD fires on a silence
threshold, the model decides whether that silence is a turn boundary or a
mid-thought pause. The window is the *last* 8 seconds, and it is short on
purpose: the evidence for the decision sits at the end of the utterance (the
terminal pitch contour, final-syllable lengthening, whether the last word can
close a clause), and audio from further back does not change whether this
sentence just landed.

The specific target is **Indian Hinglish**: code-switched Hindi/English with
Devanagari and romanized text, Indian-accented prosody, and the filler
vocabulary that comes with it (*haan, matlab, accha, yaar, umm, toh*). Two
properties matter more than headline accuracy:

1. **Trailing-silence invariance.** Appending silence to a clip must not change
   the verdict. Silence is exactly what the model sees at the moment it is
   asked, so a model that reads silence as "done" will cut people off.
2. **Filler and trailing-conjunction sensitivity.** "*Mumbai mein traffic bahut
   zyada kyunki…*" ends in silence and must still read as incomplete.

Six experiments (E1–E6) isolate what actually buys those properties: E1–E4 are
the core ablations, E6 scales the training data 2.4×, and E5 distills the shipped
model into a 507k-parameter student.

## 2. Data preparation

### 2.1 Real subset: pipecat smart-turn v3.2

The English and Hindi rows of
[`pipecat-ai/smart-turn-data-v3.2`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train)
are streamed from the 41 GB train split and the official test split, resampled
to 16 kHz mono, right-truncated to the last 8 s, and written as FLAC plus a
`manifest.parquet` (`notebooks/kaggle/01_data_prep.ipynb`, generated from
`tools/build_notebooks.py`). The counts below are the **E1–E4 prep**; E5 and E6
use a larger pass over the same source, described in §2.1a.

| quantity | value |
|---|---|
| train-split rows scanned | 270,946 |
| English train rows kept per label | capped at 17,500 (35,000 train-source rows) |
| Hindi train rows | all kept, no cap |
| real train rows after 5% val holdout | 44,751 |
| real val rows | 2,255 |
| official test clips | 7,820 English + 1,284 Hindi = 9,104 |
| total real clips written | 56,110 (7.0 GB FLAC) |

Two details cost real debugging time and are worth recording. First, v3.2 tags
language with **ISO-639-3** codes (`eng`, `hin`), not `en`/`hi`; filtering on the
two-letter codes silently keeps zero rows. The prep code normalizes once at the
boundary and every downstream consumer keys on the long names
(`tests/test_notebooks.py::test_prep_filters_iso639_3_codes` pins this).
Second, the English cap is applied per label, so the kept subset does not inherit
the source distribution's class skew.

The validation split is a 5% hash of the train-source ids; the **test split is
the dataset's own official test split**, never a resample of train.

### 2.1a Expanded prep: the E5/E6 superset

E5 and E6 run on a second, larger pass over the same source (same notebook, wider
caps). English is effectively uncapped (33,000/label, which keeps all of it),
Hindi is still uncapped, and a **multilingual tail** is added: every other
language in v3.2, capped at **850 per (language, label)**, train-only.

| quantity | E1–E4 prep | E5/E6 prep |
|---|---:|---:|
| train-split rows scanned | 270,946 | 270,946 |
| real clips kept (train-source) | 47,006 | 113,148 |
| English train rows | 33,349 | 62,250 |
| Hindi train rows | 11,402 | 11,402 |
| other languages (21, train-only) | 0 | 35,700 |
| real train rows after 5% val holdout | 44,751 | 109,352 |
| + synthetic Hinglish train clips | 2,157 | 2,157 |
| **total train rows** | **46,908** | **111,509** |
| total clips written | 56,110 (7.0 GB) | 122,252 / 217.6 h (15.5 GB) |

The per-language rows are derived from the prep kernel's split × language × label
counts; the totals in bold are the values the training run itself printed.

The multilingual tail is **train-only on purpose**: val stays English+Hindi so
best-checkpoint selection is comparable across all six runs, and the v3.2 test
split is English+Hindi anyway. `train.py` still reports a `multilingual_other`
slice, which is `n=0` on the test set by construction, reported rather than
hidden (§8).

One infrastructure note worth recording: the prep writes its ~122k clips
into **64 ZIP shards** rather than loose files, because Kaggle's kernel-output
publishing **silently fails past ~100k files**: the version reports success and
ships an 845-byte empty `_output_.zip`. The train notebook extracts the shards to
`/tmp/prep_audio`; `manifest.parquet` still comes from the mount. Documented in
`KAGGLE_RUNBOOK.md`.

### 2.2 Synthetic Hinglish pipeline

Neither the real corpus nor any public corpus we found contains labelled
code-switched Hinglish turn boundaries, so the Hinglish slice is synthesized
(`synth/`).

- **Corpus** (`synth/sentences.py`, `synth/corpus.py`): 89 hand-written slot
  templates over everyday Indian service-app domains (delivery, recharge,
  travel, leave approval, traffic), each written in both Devanagari and
  romanized Latin. Slot expansion yields **189 concrete sentence instances**.
- **Rendering** (`synth/tts_generate.py`): 4 Indian edge-tts neural voices
  (`hi-IN-SwaraNeural`, `hi-IN-MadhurNeural`, `en-IN-NeerjaNeural`,
  `en-IN-PrabhatNeural`), with word-boundary timestamps captured alongside the
  audio.
- **Incomplete variants** are built three ways: **`cut`** truncates a complete
  utterance at a real word boundary (using the TTS boundary marks, so the cut
  never lands mid-phoneme); **`tail_conj`** appends a dangling conjunction
  (*aur, lekin, kyunki, toh, matlab, par, phir, agar*); **`tail_filler`**
  appends a trailing filler (*umm, matlab, wo kya hai na, haan toh, actually,
  basically, aisa hai ki, kya bolte hain usko*). **`midfiller_full`** is a
  complete sentence with a filler injected mid-utterance (*matlab, accha, haan,
  wo, umm, yaar, bas*): a complete turn that superficially looks disfluent.

Resulting corpus (`synth/output/manifest.parquet`): **2,469 clips / 2.07 hours**,
1,283 Devanagari and 1,186 romanized.

| kind | label | n |
|---|---|---:|
| `cut` | incomplete | 999 |
| `full` | complete | 945 |
| `tail_conj` | incomplete | 189 |
| `tail_filler` | incomplete | 189 |
| `midfiller_full` | complete | 147 |

| split | templates | clips | % complete |
|---|---:|---:|---|
| train | 72 | 2,157 | 45.0% |
| val | 6 | 87 | 37.9% |
| test | 11 | 225 | 39.1% |

The 225-clip test split breaks down as 107 `cut`, 75 `full`, 15 `tail_conj`,
15 `tail_filler`, 13 `midfiller_full`: 88 complete and 137 incomplete.

### 2.3 Methodology note: the leakage fix

The first version of the split hashed the full `sentence_id`
(`s{template:03d}_{slot_combo}`). Because slot variants of one template differ
only in a filled slot ("मुंबई में traffic…" vs "पुणे में traffic…"), that put
near-identical wording on both sides of the split: **69% of test templates also
appeared in train**. The `cut` and `tail_*` children of a sentence inherited the
same problem.

The fix (`synth/package_kaggle.py::split_of`) hashes only the `s###` template
prefix, so every slot variant of a template, and all of its cut/tail children,
land in the same split. Measured on the shipped manifest, **train ∩ test
templates = 0**, train ∩ val = 0, val ∩ test = 0. This was corrected before any
model in this report was trained; every Hinglish number below comes from the
template-disjoint split. The stale, pre-fix analysis artifact still sitting in
`models/analysis/worst_errors.md` (433 clips, AUC 0.400) is from the old
pipeline and should be ignored.

Train is additionally rebalanced by dropping surplus `cut` rows whenever the
incomplete class would exceed 55%.

## 3. Architecture and training

**WhisperTinyTurn** (E1, E2, E4, E6) is the `openai/whisper-tiny` encoder with its
positional embedding table sliced from 1500 to **400** positions, which pins the
input to an 8 s window (128,000 samples → 800 mel frames → 400 encoder positions
after Whisper's stride-2 conv stem). On top: learned-query attention pooling
over time, then LayerNorm → Linear(384→256) → GELU → Dropout(0.1) →
Linear(256→1). **7,885,953 parameters.**

The encoder is pretrained rather than learned from scratch because Whisper
already carries multilingual acoustic and prosodic features, Hindi and
Indian-accented English included, and carries them at roughly 8M parameters, so
the task reduces to reading a pooled representation instead of learning speech
itself; E3 is the control that prices that pretraining at 6.7 accuracy points
(§5.4). Pooling is a **learned-query attention** rather than a mean because the
frames that decide the question are the last ones: a mean spreads the evidence
uniformly over all 400 positions, including the trailing silence the model is
called on, while an attention query can weight the end of the window.

**TinyMelNet** (E3, E5) is a from-scratch control with no pretrained weights: a
stride-2 Conv1d stem plus three depthwise-separable Conv1d blocks over the mel
frames, a BiGRU(128), and the same pooling and head. **507,265 parameters.**

Both take log-mel `(B, 80, 800)`. The frontend is Whisper's exact feature math
(STFT n_fft=400, hop=160, periodic Hann, 80 slaney mel filters, log10, 8 dB
dynamic-range clamp, `(x+4)/4` scaling), implemented twice: a torch version for
training and a numpy twin for inference.
`tests/test_features.py::test_logmel_matches_hf_feature_extractor` checks the
torch version against `transformers.WhisperFeatureExtractor`, and
`test_numpy_logmel_matches_torch` checks the numpy twin against the torch one,
which is what lets the Space ship without torch or transformers. Windows are
**right-aligned**: the last 8 s are kept and shorter clips are left-padded, which
matches inference, where the decision point is always the present moment.

Training: Kaggle T4, AMP, AdamW, encoder LR 1e-5 / head LR 1e-4 (E3/E5 head
3e-4), weight decay 0.01, 5% warmup then cosine decay, grad clip 1.0, batch 64,
seed 42. E1/E2/E4/E6 run 4 epochs; E3/E5 run 8 (TinyMelNet is smaller and trains
from scratch).

**E5 adds knowledge distillation.** The student is the same 507,265-parameter
TinyMelNet; the teacher is E2's `ckpt_best.pt`, frozen and in eval mode, scoring
the *identical augmented* mel batch the student sees, so the soft target already
accounts for the `pause_cut`/noise/speed edit applied to that clip. The loss
(`train.py::batch_loss`) is

```
α · BCE(student_logit, hard_label) + (1 − α) · BCE(student_logit / T, σ(teacher_logit / T))
```

with **α = 0.3, T = 2.0**: the teacher's soft targets carry 0.7 of the weight
and the hard labels 0.3. `load_teacher` raises rather than defaulting to `None`
when `kd_teacher` is set: silently training a "distilled" student against no
teacher would burn a GPU session and produce a run whose name lies about it.

Training is driven by a **step counter rather than an epoch loop**, drawing with
replacement from a weighted sampler, because that makes the resume state small
enough to be trivial: a step number plus model and optimizer state is the whole
of it, so a session killed at step 3,200 restarts at step 3,200 with no epoch
boundary or shuffle order to reconstruct (§9). Sampling is therefore step-based
with a **balanced sampler that compensates for label flips**:
`pause_cut` converts a complete clip into an incomplete one on the fly,
so drawing 50/50 from the manifest would yield batches that are only
`0.5 × (1 − p_cut)` complete. The sampler oversamples completes by
`1/(1 − p_cut)` so post-augmentation batches are actually balanced
(`dataset.py::balanced_sampler`, pinned by
`tests/test_dataset.py::test_balanced_sampler_offsets_pause_cut`).

The decision threshold is tuned on the validation split (maximizing accuracy)
and then applied unchanged to the test split. It is not re-tuned per slice.

## 4. Experiment matrix

| run | arch | Hinglish synth | `pause_cut` | `trailing_silence` | `noise` | `speed` | epochs | train rows | T4 min |
|---|---|---|---|---|---|---|---:|---:|---:|
| E1 `e1_baseline` | Whisper | no | 0 | 0 | 0 | 0 | 4 | 44,751 | 9.6 |
| E2 `e2_hinglish_aug` | Whisper | yes | 0.15 | 0.50 | 0.25 | 0.25 | 4 | 46,908 | 10.6 |
| E3 `e3_tinymel_scratch` | TinyMelNet | yes | 0.15 | 0.50 | 0.25 | 0.25 | 8 | 46,908 | 16.1 |
| E4 `e4_no_pause_aug` | Whisper | yes | 0 | 0 | 0.25 | 0.25 | 4 | 46,908 | 12.1 |
| E5 `e5_distill` † | TinyMelNet | yes | 0.15 | 0.50 | 0.25 | 0.25 | 8 | 111,509 | 32.1 |
| E6 `e6_full_data` | Whisper | yes | 0.15 | 0.50 | 0.25 | 0.25 | 4 | 111,509 | 21.4 |

† E5 additionally distills from E2's frozen checkpoint (α = 0.3, T = 2.0, §3).
E5 and E6 both train on the expanded prep of §2.1a, 2.4× the rows of E1–E4.

`pause_cut` truncates a **complete** utterance at a speech-active point in the
40–85% band of its active span and appends 0.2–1.2 s of silence; the label flips
to incomplete. `trailing_silence` appends 0.2–1.2 s of silence to any clip and
**preserves** the label. E4 is E2 with exactly those two removed, and it is the
ablation that carries Finding 3.

### Accuracy at the tuned threshold (fp32, official test split)

| slice | n | E1 | E2 | E3 | E4 | E5 | E6 |
|---|---:|---|---|---|---|---|---|
| overall | 9,329 | 0.930 | 0.938 | 0.871 | 0.937 | 0.896 | **0.943** |
| english | 7,820 | 0.938 | 0.938 | 0.868 | 0.937 | 0.898 | **0.945** |
| hindi | 1,284 | 0.931 | **0.932** | 0.884 | 0.928 | 0.885 | 0.931 |
| hinglish | 225 | 0.631 | 0.951 | 0.871 | **0.964** | 0.871 | 0.938 |
| filler | 2,381 | 0.912 | 0.914 | 0.856 | 0.905 | 0.870 | **0.915** |
| human_audio | 5,367 | 0.946 | 0.947 | 0.877 | 0.946 | 0.904 | **0.954** |

### AUC (fp32, official test split)

| slice | E1 | E2 | E3 | E4 | E5 | E6 |
|---|---|---|---|---|---|---|
| overall | 0.981 | 0.983 | 0.944 | 0.984 | 0.959 | **0.986** |
| english | 0.984 | 0.983 | 0.941 | 0.983 | 0.960 | **0.987** |
| hindi | **0.985** | 0.984 | 0.962 | 0.983 | 0.960 | **0.985** |
| hinglish | 0.612 | 0.986 | 0.949 | **0.993** | 0.962 | 0.963 |
| filler | 0.971 | 0.971 | 0.940 | 0.970 | 0.947 | **0.977** |
| human_audio | 0.988 | 0.987 | 0.948 | 0.988 | 0.964 | **0.990** |

`multilingual_other` is `n=0` on this test split for every run (§2.1a), so it is
omitted here.

### Footprint and threshold

| | E1 | E2 | E3 | E4 | E5 | E6 |
|---|---|---|---|---|---|---|
| params | 7,885,953 | 7,885,953 | 507,265 | 7,885,953 | 507,265 | 7,885,953 |
| fp32 ONNX | 31.58 MB | 31.58 MB | 2.03 MB | 31.58 MB | 2.03 MB | 31.58 MB |
| int8 ONNX | 8.50 MB | 8.50 MB | 1.28 MB | 8.50 MB | 1.28 MB | 8.50 MB |
| best val AUC | 0.986 | 0.984 | 0.947 | 0.984 | 0.963 | 0.985 |
| tuned threshold | 0.35 | **0.63** | 0.53 | 0.35 | **0.57** | 0.54 |

**E2 remains the shipped headline model** (`models/model_int8.onnx`); E6 beats
it everywhere except the target domain (§5.5). **E5 is the shipped fast model**
(`models/model_tinymel_int8.onnx`), replacing E3 (§5.6).

## 5. Findings

### 5.1 The baseline has a 63.1% Hinglish hole, and it is one failure family

E1, trained on 44,751 real English and Hindi clips with no augmentation, is a
perfectly respectable turn detector on the data it saw: 93.8% on English
(n=7,820), 93.1% on Hindi (n=1,284), AUC 0.984 and 0.985. On the 225 held-out
Hinglish clips it collapses to **63.1% accuracy and AUC 0.612**, barely above
the 60.9% you get by predicting "incomplete" for everything, and an AUC that says
the ranking is close to uninformative.

The per-kind breakdown (int8 model, threshold 0.35,
`experiments/run_e1_baseline/analysis/worst_errors.md`) shows the failure is
one-sided, not diffuse:

| kind | label | n | E1 accuracy | mean P(complete) |
|---|---|---:|---|---|
| `full` | complete | 75 | **0.160** | 0.135 |
| `midfiller_full` | complete | 13 | **0.077** | 0.083 |
| `cut` | incomplete | 107 | 0.963 | 0.043 |
| `tail_conj` | incomplete | 15 | 0.800 | 0.193 |
| `tail_filler` | incomplete | 15 | 1.000 | 0.037 |

Recall on complete Hinglish turns is **0.148** at precision 0.650. E1 has not
learned "Hinglish is hard"; it has learned to say *incomplete* to nearly all
Hinglish, which is right 137 times out of 225 by construction.

The worst errors are a single coherent linguistic family: **complete sentences
whose final constituent is a post-verb adverb.** The top 10 ranked by
|prob − label| are all instances of the same construction, in both scripts:

| text | prob | label |
|---|---|---|
| लखनऊ में traffic बहुत ज्यादा है **आज**। | 0.000 | complete |
| मुंबई में traffic बहुत ज्यादा है **आज**। | 0.000 | complete |
| पुणे में traffic बहुत ज्यादा है **आज**। | 0.000 | complete |
| Indore mein traffic bahut zyada hai **aaj**. | 0.000 | complete |
| Kolkata mein traffic bahut zyada hai **aaj**. | 0.000 | complete |

Hindi is verb-final, so an English-trained prior expects the sentence to end at
the verb (*hai*). A trailing adverb (*aaj*, "today") after the verb is completely
ordinary in colloquial Hindi and Hinglish, but it reads to E1 as an interrupted
continuation, and E1 assigns those clips P(complete) ≈ 0.000, not 0.4. That is
not uncertainty; it is a confidently wrong prior about where a sentence is
allowed to end. The same error rate holds for the romanized and the Devanagari
renditions of the identical sentence, which rules out a script-specific artifact.

### 5.2 Synthetic Hinglish closes the gap by 32 points with no regression elsewhere

E2 adds 2,157 synthetic Hinglish training clips (4.6% of 46,908 rows) plus the
augmentation recipe. On the same 225 template-disjoint test clips:

| metric (fp32, n=225) | E1 | E2 | Δ |
|---|---|---|---|
| accuracy | 0.631 | **0.951** | **+32.0 pts** |
| AUC | 0.612 | **0.986** | +0.374 |

The failure family is gone. Per-kind (int8, threshold 0.63): `full` 0.907
(n=75, was 0.160), `midfiller_full` 0.846 (n=13, was 0.077), `cut` 0.991
(n=107), `tail_filler` 1.000 (n=15), `tail_conj` 0.867 (n=15). Recall on
completes goes 0.148 → 0.898 at precision 0.963. The post-verb-adverb sentences
that scored 0.000 now score 0.32–0.73; E2's residual Hinglish errors sit near
the threshold rather than confidently inverted, which is the behaviour you want
from a model you are about to threshold-tune.

Crucially, nothing else moves backwards:

| slice | n | E1 | E2 |
|---|---:|---|---|
| english | 7,820 | 0.938 | 0.938 |
| hindi | 1,284 | 0.931 | 0.932 |
| filler | 2,381 | 0.912 | 0.914 |
| human_audio | 5,367 | 0.946 | 0.947 |
| overall | 9,329 | 0.930 | 0.938 |

English AUC drifts 0.984 → 0.983 and Hindi 0.985 → 0.984, differences of one in
the third decimal on n=7,820 and n=1,284, i.e. noise. A 4.6% synthetic injection
is not diluting the real-data signal.

Two guards make this credible rather than a leakage artifact. The Hinglish test
templates are disjoint from train (§2.3, measured: 0 overlap), so E2 is not
recalling memorized wording. And `human_audio`, the real-human, non-synthetic
slice of the official test set (n=5,367), moves 0.946 → 0.947, so training on
TTS audio has not pushed the model toward a synthetic-speech shortcut.

### 5.3 The centerpiece: identical static metrics, opposite production behaviour

E4 is E2 minus `pause_cut` and `trailing_silence`. On the static test set it is
not merely competitive: **it looks better**:

| slice | n | E2 | E4 |
|---|---:|---|---|
| hinglish acc | 225 | 0.951 | **0.964** |
| hinglish AUC | 225 | 0.986 | **0.993** |
| overall acc | 9,329 | **0.938** | 0.937 |
| overall AUC | 9,329 | 0.983 | **0.984** |

On the Hinglish slice E4 makes 8 errors to E2's 11 out of 225. Overall, the
0.001 gap is about 11 clips out of 9,329. On every table in §4 an evaluator
would call these two models equivalent, and would probably prefer E4.

**They are not equivalent.** The stress test
(`experiments/silence_stress_test.json`) takes the 225 local Hinglish test clips,
appends 0.5 s and then 1.0 s of digital silence to each (a label-preserving edit
that changes nothing about what was said), and re-runs both int8 models at their
own tuned thresholds. A well-behaved turn detector returns the same verdict on
all 225.

| | E2 (`pause_cut` + `trailing_silence`) | E4 (no pause aug) |
|---|---|---|
| base accuracy, no padding (int8, n=225) | 0.947 | 0.942 |
| decisions flipped by +0.5 s silence | **7 / 225 = 3.1%** | 26 / 225 = **11.6%** |
| decisions flipped by +1.0 s silence | **19 / 225 = 8.4%** | 65 / 225 = **28.9%** |
| early fires at +0.5 s (incomplete → complete) | **2 / 137 = 1.5%** | 23 / 137 = **16.8%** |
| early fires at +1.0 s (incomplete → complete) | **7 / 137 = 5.1%** | 61 / 137 = **44.5%** |

At a one-second pause, **E4 declares the turn over on 61 of the 137 clips where
the speaker is still mid-thought**, or 44.5%. E2 does it on 7, an 8.7× reduction.
E4's flips are also almost entirely early fires (61 of its 65 flips at 1.0 s),
which is the expensive direction: an early fire is the agent talking over the
user, while a late fire is only added latency.

The failure is easy to state. Without pause augmentation, "long trailing
silence" and "turn complete" are perfectly correlated in the training data,
because in a curated corpus every complete utterance was recorded to the end and
every incomplete one was cut. The model learns the correlation instead of the
linguistics, and the correlation is precisely the one that breaks in production.
`pause_cut` destroys it by producing *incomplete* clips that end in silence;
`trailing_silence` destroys the converse by padding both classes without touching
labels.

**Why static test sets cannot see this.** Every clip in the test set carries its
recording's natural amount of trailing silence, which is short and — critically —
correlated with the label in the same way the training data is. A model that has
learned "silence ⇒ done" is *rewarded* by that test set: it scores 0.964 on
Hinglish, the best of all six runs. The bias is invisible because the test
distribution shares it. Only an intervention that breaks the correlation (hold
the content fixed, vary the silence) separates the two models, and it separates
them by a factor of nearly nine on the metric that actually matters.

**Why trailing-silence invariance is the production property.** In deployment the
model is not called on a curated clip; it is called by a VAD the moment speech
energy drops, and then again, and again, while the user thinks. The input at
every one of those calls is "everything said so far, followed by silence, and the
silence keeps growing." A model whose verdict is a function of silence duration
will, given enough thinking time, fire on every single utterance; the only
question is how long the user gets. E4 gives them under a second. That is the
entire failure mode users describe as "it keeps interrupting me," and no number
in §4 predicts it.

E2's 0.013 deficit on the static Hinglish slice is the price of this robustness,
and it is a good trade: 3 extra errors out of 225 on a static benchmark, in
exchange for 54 fewer interruptions out of 137 at a one-second pause.

### 5.4 Secondary: the from-scratch control

E3 uses the identical data and augmentation as E2 with a 507,265-parameter model
(15.5× smaller) and no pretrained weights. It reaches 0.871 overall (AUC
0.944) against E2's 0.938 (AUC 0.983), and 0.871 on Hinglish (AUC 0.949, n=225).
Its validation AUC was still climbing at the end of 8 epochs (0.9465 → 0.9467
over the last epoch, up from 0.887 after the first), so this is a floor rather
than a converged ceiling. The honest reading: Whisper's pretrained multilingual
representation is worth roughly 6.7 accuracy points here, and the small model is
a real option when the parameter budget is the binding constraint (§6). E5 closes
most of that gap without changing the architecture (§5.6).

### 5.5 Scaling the data 2.4× helps everywhere except the target domain

E6 is E2's recipe (same architecture, same augmentation, same 4 epochs, same
seed) on the expanded prep of §2.1a: 111,509 train rows against 46,908, with
English uncapped and a 21-language tail. It costs twice the GPU time (21.4 min
vs 10.6) and it works, on the general case:

| slice | n | E2 | E6 | Δ acc | E2 AUC | E6 AUC |
|---|---:|---|---|---|---|---|
| overall | 9,329 | 0.938 | **0.943** | +0.5 pts | 0.983 | **0.986** |
| english | 7,820 | 0.938 | **0.945** | +0.7 pts | 0.983 | **0.987** |
| human_audio | 5,367 | 0.947 | **0.954** | +0.7 pts | 0.987 | **0.990** |
| filler | 2,381 | 0.914 | **0.915** | +0.1 pts | 0.971 | **0.977** |
| hindi | 1,284 | **0.932** | 0.931 | −0.1 pts | 0.984 | **0.985** |
| **hinglish** | 225 | **0.951** | 0.938 | **−1.3 pts** | **0.986** | 0.963 |

Overall accuracy moves by about 48 clips out of 9,329, English by about 52 out of
7,820, and the real-human slice, the closest thing here to a production proxy,
gains 0.7 points with AUC 0.987 → 0.990. Every one of those is a genuine
improvement, and Hindi holding flat while English rises suggests the multilingual
tail is not crowding out the languages we report on.

**And the target domain went backwards.** Hinglish accuracy falls 0.951 → 0.938
(11 errors → 14 out of 225) and Hinglish AUC falls 0.986 → 0.963. On n=225 the
three extra errors are inside seed noise; the **0.022 AUC drop is the number to
read**, because AUC is threshold-free. 0.963 is the worst Hinglish ranking of any
Whisper run trained on synthetic Hinglish (E2 0.986, E4 0.993); E6 is closer
here to the 507k-parameter students (E3 0.949, E5 0.962) than to its own
architecture's other runs.

The mechanism is straightforward and worth stating plainly: **the synthetic
Hinglish training set did not grow.** It is the same 2,157 clips in both runs, so
its share of the batch fell from **4.6% to 1.9%**. §5.2's finding was that a 4.6%
synthetic injection buys +32 points on Hinglish without diluting the real-data
signal; §5.5 is the same trade seen from the other end: hold the injection fixed,
grow everything around it, and the injection's influence is diluted away. The
model got better at the average clip and worse at the clip we built it for.

**Consequence for shipping: E2 stays.** The brief is Hinglish turn detection, so a
model that is 0.5 points better overall and 1.3 points plus 0.022 AUC worse on
Hinglish is not an upgrade; it is a different product. E6 is recorded here as a
data-scaling study, and it points at the obvious next experiment: rerun it with
the synthetic Hinglish corpus scaled proportionally (roughly 5,300 clips) so the
4.6% share survives contact with 111k rows. That is a synthesis-cost question,
not a modelling one.

E6 also has an infrastructure lesson attached, in §2.1a: its prep is where the
~100k-file Kaggle publishing failure surfaced, and the fix (64 ZIP shards)
is what makes the expanded dataset reproducible at all.

### 5.6 Distillation recovers most of the small model's gap at identical size

E5 is TinyMelNet again, the same 507,265 parameters and the same 1.28 MB int8
artifact as E3, trained with E2's frozen checkpoint as a teacher (α = 0.3,
T = 2.0, §3):

| slice | n | E3 | E5 | Δ acc | E3 AUC | E5 AUC |
|---|---:|---|---|---|---|---|
| overall | 9,329 | 0.871 | **0.896** | **+2.5 pts** | 0.944 | **0.959** |
| english | 7,820 | 0.868 | **0.898** | **+3.0 pts** | 0.941 | **0.960** |
| human_audio | 5,367 | 0.877 | **0.904** | +2.7 pts | 0.948 | **0.964** |
| filler | 2,381 | 0.856 | **0.870** | +1.4 pts | 0.940 | **0.947** |
| hindi | 1,284 | 0.884 | **0.885** | +0.1 pts | **0.962** | 0.960 |
| hinglish | 225 | 0.871 | 0.871 | 0.0 pts | 0.949 | **0.962** |

Best validation AUC goes 0.947 → 0.963, and the int8 export follows: 0.9045
accuracy / 0.9632 AUC on the 2,000-clip subset against E3's 0.876 / 0.949. The
gap to the E2 teacher narrows from **6.7 points to 4.2** overall, at 15.5× fewer
parameters and 1/6.6 the int8 file size.

**Two honest qualifications.** First, E5 also trains on the expanded prep
(111,509 rows), so the +2.5 points is *distillation and 2.4× data together*, not
distillation alone. §5.5 shows the extra data is worth roughly +0.5 points
overall for the Whisper model, so most of E5's gain is plausibly the teacher,
but separating them needs a third run (E3's recipe on the expanded prep, no
teacher) that was not budgeted. Second, **Hinglish accuracy did not move**: 0.871
in both runs, despite a teacher scoring 0.951 there. The AUC did improve
(0.949 → 0.962), so the student's *ranking* of Hinglish clips got better while its
thresholded accuracy did not, consistent with §5.5's dilution acting on E5 too
(same 1.9% synthetic share) and with 507k parameters simply not having the
capacity to absorb the teacher's Hinglish behaviour.

**Why E2 and not E6 as the teacher.** E6 is the stronger model on paper: better
overall, better English, better `human_audio`. It is also 0.022 AUC worse on
Hinglish, and 70% of the student's loss is the teacher's soft targets, so a
student distilled from E6 would be trained to reproduce precisely the ranking E6
does worst. Distilling the target-domain specialist rather than the
general-purpose leader is the choice that follows from the brief; it is the same
decision as §5.5's, applied one level down. Untested, and worth an hour of T4
time to check: E6 may still be the better teacher for the non-Hinglish slices.

E5 replaces E3 as the shipped small model (`models/model_tinymel_int8.onnx`, the
demo's **fast** dropdown entry). E3 remains in the report as the no-teacher
control that makes E5's number meaningful.

### 5.7 Real-voice spot check: the errors point the right way

Every Hinglish number above comes from TTS audio, so the author recorded a
small real set in their own voice: 30 clips (14 complete / 16 incomplete)
spoken into a phone microphone, using fresh sentences disjoint from the
training templates,
covering the same five categories as the synthetic corpus (complete,
mid-sentence filler, trailing conjunction, trailing filler, mid-thought stop).
Scored at each model's pre-registered threshold; nothing was tuned on these clips.

| model | acc | AUC | incompletes correct | early-fires after +1.0 s real pause |
|---|---|---|---|---|
| E2 (shipped) | 0.800 | 0.942 | 15/16 | **1/16** |
| E5 (fast) | 0.767 | 0.888 | 13/16 | 2/16 |
| E4 (no pause aug) | 0.767 | 0.924 | 15/16 | 1/16 |

The 15-point drop from the synthetic 95.1% is honest domain shift (one
speaker, phone mic, natural pacing). What matters is its structure: E2's
incomplete detection barely moves (trailing fillers 5/5, mid-thought stops 6/6)
and it early-fires once in sixteen 1-second real pauses, so the
interrupt-the-user failure stays rare on real speech. Five of its six errors
are complete sentences judged "still speaking" (two confidently, at p≈0.01),
i.e. the model errs toward waiting, which costs response latency rather than a
barge-in. n=30 and a single speaker make this directional evidence, not a
benchmark; per-clip numbers are in `experiments/real_voice_eval.json`, and the
raw audio deliberately stays out of the repository. E4's pause weakness does
not reproduce at n=16 (1/16, vs 44.5% on the n=137 synthetic test); small-n
noise cuts both ways, and the synthetic stress test remains the controlled
evidence for the pause-augmentation claim.

## 6. Latency and size

Measured with `turn_detector.infer.benchmark`: int8 ONNX, onnxruntime
CPUExecutionProvider, 8 s window, median over 100 iterations × 3 repeats.
`total` includes numpy mel extraction; `model` is the ONNX session alone.

| model | params | int8 size | total p50, 1 thread | model p50, 1 thread | total p50, 4 threads |
|---|---:|---:|---:|---:|---:|
| E2 WhisperTinyTurn | 7,885,953 | 8.50 MB | 91 ms | 83 ms | 59 ms |
| E3 / E5 TinyMelNet | 507,265 | 1.28 MB | 14 ms | 7 ms | 18 ms |

E5 and E3 are the **same architecture and the same exported graph shape**, with
only the weights differing, so E5 inherits E3's footprint exactly: 507,265
parameters, 2.03 MB fp32, 1.28 MB int8, and the same measured latency. The
TinyMelNet row is reported once for both.

**These are laptop numbers and should be read as a conservative ceiling, not a
server figure.** Two measurement sessions on the same Windows dev laptop
disagreed by roughly 35%: an earlier session recorded E2 at ~126 ms total /
~113 ms model single-threaded and ~127 ms at 4 threads, and E3 at 19.1 ms total /
8.2 ms model. Consumer laptops throttle on thermal and power state, and the
4-thread figure in particular is sensitive to background load; E3 is slower at
4 threads than at 1 because the model is too small to amortize the thread
handoff. For calibration, pipecat's smart-turn v3 reports **~12 ms** for the same
encoder class on server CPU; a datacenter core is several times faster than this
machine, so E2 on a server should land in the low tens of milliseconds.

The mel frontend costs a fixed ~8 ms in both cases, which is why it dominates
E3's budget (7 ms model + ~7 ms mel) and is a rounding error in E2's.

**Choosing between them.** E2 at 8.5 MB and tens of milliseconds per call is the
right default: the accuracy gap to the small model is still real (0.938 vs 0.896
overall, 0.951 vs 0.871 Hinglish), and turn detection runs once per VAD pause,
not per audio frame. The small model earns its place when it must fit on-device
alongside ASR and TTS, or when the CPU budget is genuinely single-digit
milliseconds, and distillation makes that trade materially cheaper than it was:
E5 pays 4.2 accuracy points for a 6.6× smaller file and a ~6× faster call, where
E3 paid 6.7 for the same footprint. Both shipped artifacts sit in `models/` and
the demo exposes them as **accurate** (E2) and **fast** (E5).

## 7. int8 quantization

Export is a two-step in `train.py::export_onnx`: `torch.onnx.export` at opset 17
with a **static batch of 1** (turn detection is inherently one window at a time,
and dynamic batch breaks the bidirectional-GRU reshape on export), then
`onnxruntime.quantization.quantize_dynamic` with `QuantType.QUInt8` on weights.

fp32 ONNX is checked against torch on every export: max |Δprob| = **1.19e-07**
for all four Whisper runs (E6 included) and 3.58e-07 / 7.75e-07 for TinyMelNet
(E3 / E5), so the graph conversion itself is lossless to float precision.

| | fp32 ONNX | int8 ONNX | ratio |
|---|---|---|---|
| WhisperTinyTurn (E1/E2/E4/E6) | 31.58 MB | 8.50 MB | 3.7× |
| TinyMelNet (E3/E5) | 2.03 MB | 1.28 MB | 1.6× |

**Reading the int8 numbers correctly.** The int8 evaluation is *not* the full
test set. It is a stratified 2,000-clip subset (1,000 per label, seeded) scored
with the fp32-tuned threshold, chosen to keep the CPU eval bounded. Because it is
class-balanced and the full test set is not, its accuracy is not comparable to
the §4 accuracy table. **AUC is the comparable quantity**, and it barely moves:

| run | AUC, fp32, full test (n=9,329) | AUC, int8, subset (n=2,000) | acc, int8 subset, fp32 threshold |
|---|---|---|---|
| E1 | 0.981 | 0.976 | 0.879 |
| E2 | 0.983 | 0.978 | 0.887 |
| E3 | 0.944 | 0.949 | 0.876 |
| E4 | 0.984 | 0.975 | 0.878 |
| E5 | 0.959 | 0.963 | 0.904 |
| E6 | 0.986 | 0.979 | 0.856 |

Ranking quality survives dynamic quantization: −0.005 for the Whisper runs (E6:
−0.007) and +0.005/+0.004 for TinyMelNet (within subset noise). What quantization
moves is **calibration**, and the threshold does not travel with it. E6 is the
clearest case: AUC holds at 0.979 while subset accuracy at the fp32 threshold
drops to 0.856, the lowest in the table for the strongest model in the report.
Separability is intact; the operating point has shifted underneath it.

### Decision-matched thresholds for the shipped int8 artifact

E2 showed the same symptom more mildly (0.887 subset accuracy at the fp32
threshold of 0.63 against an int8 AUC of 0.978), and an earlier draft of this
report read that as a quantization accuracy cost. It is not. It is a calibration
shift, and it can be corrected without touching the weights.

The correction is a **decision-matching** sweep rather than an accuracy sweep:
take clips the model was trained on (600 held-in synthetic clips, *not* test
data, and no test labels are consulted at any point), score them with both the
fp32 and the int8 graph, and pick the int8 threshold that reproduces the most
fp32-at-0.63 decisions. The answer is **0.50**, which agrees with fp32 on
**99.7% of the 600 clips**. Recorded in `models/metrics.json` as
`int8_threshold_decision_matched` alongside `int8_threshold_note`.

This matters because it changes what "use threshold 0.63" means depending on
which file you load. The demo and the Space now read
`int8_threshold_decision_matched` when present and fall back to the fp32-tuned
`threshold` otherwise (`demo/app.py::tuned_threshold`), so the shipped int8
Whisper model defaults to **0.50** and the shipped int8 TinyMelNet (E5, which has
no decision-matched entry) defaults to its own tuned **0.57**.

**Recommendation: recalibrate per exported artifact.** Never inherit a threshold
across a quantization boundary. The per-run fp32 thresholds already vary widely
between otherwise-identical models (E1 0.35, E2 0.63, E3 0.53, E4 0.35, E5 0.57,
E6 0.54), which shows the operating point is a property of the trained artifact
and not a constant of the task; requantization moves it again.
`tools/error_analysis.py --run <dir> --threshold <t>` sweeps and plots it.

## 8. Limitations and future work

- **The Hinglish evaluation is entirely TTS-synthetic.** Every number in §5.1,
  §5.2 and §5.3 comes from edge-tts audio. There are no real human code-switched
  recordings in this evaluation, and synthetic speech has cleaner boundaries,
  more regular prosody, and no background noise, backchannels or overlapping
  talk. The +32 point improvement is real on this benchmark; whether it transfers
  at full magnitude to real Hinglish speakers is unverified. The most valuable
  next step is a few hundred real recordings, even unbalanced ones, as a check
  set. The `human_audio` slice (n=5,367, 0.946 → 0.947) shows no synthetic-speech
  shortcut developing, which is reassuring but is an English/Hindi measurement,
  not a Hinglish one.
- **Four TTS voices**, two Hindi and two Indian-English. Accent, age, gender and
  recording-condition diversity are all far below what a production system meets,
  and a model can overfit to a specific voice's end-of-utterance prosody in ways
  this test set cannot detect.
- **89 templates over a narrow domain.** The corpus covers everyday Indian
  service-app scenarios. Template-level splitting prevents wording leakage but
  cannot create genuine domain diversity; performance on, say, medical or legal
  Hinglish is unmeasured.
- **Laptop latency.** §6 numbers vary ~35% between sessions on the same machine
  and have not been measured on server hardware, under concurrent load, or at
  p99, which is what a real-time budget is actually set by.
- **Single seed (42), one run per configuration.** No error bars. The E2/E4
  static-metric gaps (0.013 on n=225, 0.001 on n=9,329) are well inside what seed
  variance would plausibly produce, which is exactly why §5.3 rests on the
  stress test and not on those gaps. The same caution applies to §5.5: E2 vs E6
  on Hinglish is 3 clips out of 225, and the claim rests on the threshold-free
  AUC drop (0.986 → 0.963) rather than on those 3 clips.
- **E6's multilingual gains are inferred, not measured.** The 21-language tail is
  train-only and the v3.2 test split is English+Hindi, so the
  `multilingual_other` test slice is **n=0**: there is no row anywhere in this
  report that measures E6 on Arabic, Chinese or any of the other 19 languages.
  The English, Hindi and `human_audio` improvements are real and measured; any
  statement that E6 is "more multilingual" is an inference from training
  composition, and this report does not make it.
- **The distilled student inherits its teacher's biases.** E5 is trained to
  reproduce E2's soft outputs at 0.7 of its loss weight, so every prior E2 holds
  (including whatever it learned from four TTS voices and 89 templates) is
  transferred by construction, and none of E5's numbers are independent evidence
  about those priors. A student cannot correct a teacher's systematic error; it
  can only fail to reach it. E5's Hinglish accuracy staying flat at 0.871 while
  its AUC improved (§5.6) is the visible edge of that ceiling.
- **E5's gain is not cleanly attributed.** It changes two things against E3 at
  once (the teacher and 2.4× the training rows), and the report has no
  no-teacher run on the expanded prep to separate them (§5.6).
- **The int8 decision-matched threshold is validated on 600 synthetic clips**
  (§7), all held-in and all TTS. 99.7% decision agreement with fp32 is a strong
  result on that set, but it says nothing about agreement on real human audio or
  at other operating points, and it was measured for E2's artifact only.
- **The stress test uses digital silence**, i.e. exact zeros. Real pauses contain
  room tone, breath and mic self-noise. Zeros are the cleanest probe of the
  correlation under test, but a room-tone version would be a stronger claim, and
  E2's `noise_p=0.25` augmentation only partially covers it.
- **Dataset licensing.** pipecat smart-turn-data v3.2 aggregates several upstream
  corpora under their own licenses. The per-source terms on the dataset card
  govern any redistribution or commercial use of models trained on it: a genuine
  constraint on shipping the weights, not a formality.
- **Not yet measured:** streaming behaviour under repeated calls as silence grows
  (the demo simulates it, but it is not quantified); interaction with a specific
  VAD's firing policy; the accuracy/latency curve for TinyMelNet trained to
  actual convergence rather than 8 epochs; the silence stress test (§5.3) for E5
  and E6, which was run for E2 and E4 only; and E6 rerun with the synthetic
  Hinglish corpus scaled to hold its 4.6% share, which §5.5 identifies as the
  experiment that would settle the dilution finding.

## 9. Compute budget: the whole matrix on 1.7 free GPU hours

Every result in this report was produced on a free Kaggle account and a laptop.
The six training runs cost **101.9 T4 minutes** in total, **1.70 hours**
(`train_minutes` in each `experiments/run_*/metrics.json`), against a free
allocation of roughly 30 GPU hours per week: **under 6% of one week's quota**,
counting two short failed sessions that produced nothing. That ceiling is not a
footnote to the engineering, it is most of the reason the engineering looks the
way it does.

| run | T4 min | what the run bought |
|---|---:|---|
| E1 `e1_baseline` | 9.6 | the 63.1% Hinglish hole and its failure family (§5.1) |
| E2 `e2_hinglish_aug` | 10.6 | the shipped model, +32.0 Hinglish points (§5.2) |
| E3 `e3_tinymel_scratch` | 16.1 | the from-scratch 507k control (§5.4) |
| E4 `e4_no_pause_aug` | 12.1 | the pause-augmentation ablation, the report's centerpiece (§5.3) |
| E5 `e5_distill` | 32.1 | the shipped fast model (§5.6) |
| E6 `e6_full_data` | 21.4 | the 2.4× data-scaling study (§5.5) |
| **total** | **101.9** | **1.70 h** |

### The GPU trains, and does nothing else

The accelerator was attached for exactly one job. Every other stage of the
project runs on CPU, either in a Kaggle session with no accelerator (which does
not draw on the GPU quota) or on the dev laptop.

| stage | hardware |
|---|---|
| scan and filter the 41 GB v3.2 train split, two passes (§2.1, §2.1a) | CPU-only Kaggle session, no accelerator |
| Hinglish TTS synthesis, 2,469 clips (§2.2) | laptop, `edge-tts` over the network |
| six training runs | Kaggle T4, 101.9 min total |
| int8 quantization and fp32 ONNX parity checks (§7) | laptop CPU |
| threshold calibration and decision matching (§7) | laptop CPU |
| error analysis and per-kind breakdowns (§5.1, §5.2) | laptop CPU |
| latency benchmarks (§6) | laptop CPU |
| silence stress test (§5.3) and real-voice eval (§5.7) | laptop CPU |

The 41 GB source is streamed and filtered row by row and **never stored in
full**: what lands in `/kaggle/working` is only the resampled, truncated,
right-aligned 8 s clips that survive the language filter and the per-label caps.
The int8 evaluation subset (2,000 clips, §7) is stratified rather than complete
for the same reason: it is a CPU eval, and bounding it keeps the analysis loop
fast enough to iterate on.

### A killed session publishes nothing, so runs end themselves

Kaggle kills a commit run at the 12 h wall and publishes **no output at all**,
not a partial checkpoint and not a log. A run that merely gets unlucky would
therefore cost its entire GPU time and return zero artifacts. Three mechanisms
prevent that:

- **Wall-clock budget.** `train()` takes `time_budget_minutes`
  (`TIME_BUDGET_MIN = 630`, i.e. 10.5 h). At the first checkpoint past the
  budget it saves `ckpt_last.pt`, prints `TIME BUDGET REACHED at step X/Y`, and
  returns without the final eval or ONNX export. The version completes normally,
  so its output is published and the run is resumable.
- **Atomic, frequent checkpoints.** Checkpoints are written every 500 steps
  through a sibling temp file plus `os.replace`, so a kill mid-write cannot
  leave a truncated file behind. At most 500 steps are ever lost.
- **Checkpoints that leave the session.** A kernel cannot read its own previous
  output, so `push_kaggle` round-trips `run_<exp>/` out of the finished kernel
  and back in as the `turn-detect-ckpt` dataset. That path serves resume and
  serves E5: the frozen E2 teacher reaches the student's session the same way.

`tests/test_train_smoke.py::test_time_budget_stops_early_and_leaves_a_resumable_checkpoint`
pins the whole loop on CPU, including the assertion that the atomic write left
no `.tmp` debris and that a second call continues past the step the first
stopped at.

### Fail fast, in seconds rather than sessions

A guard at the top of a notebook costs nothing. The same problem discovered at
minute 40 costs a GPU session. Three sit in the generated train notebook
(`tools/build_notebooks.py`):

- **Mount resolver.** Kaggle has used both a flat (`/kaggle/input/<slug>`) and a
  nested (`/kaggle/input/{datasets,notebooks}/<user>/<slug>`) mount layout, and a
  source kernel's output takes minutes to publish after that kernel completes, so
  a path that worked yesterday can be missing today. `resolve_mount` walks
  `/kaggle/input` looking for a marker file (`manifest.parquet`, or
  `ckpt_last.pt` for a resume) instead of trusting a hardcoded path, and on zero
  or multiple hits it raises with a listing of what is actually mounted plus the
  hint to wait and re-push.
- **GPU capability check.** Kaggle's torch build ships no `sm_60` kernels, so a
  session that lands on the P100 dies with "no kernel image" only once the first
  batch reaches the GPU. The notebook reads
  `torch.cuda.get_device_capability(0)` before touching the data and raises if it
  is below sm_70; `push_kaggle` pins `machine_shape: NvidiaTeslaT4` so it should
  never fire.
- **Hard resume validation.** A `RESUME_FROM` with no `ckpt_last.pt`, or with a
  checkpoint whose `cfg_hash` does not match the experiment config, raises. The
  earlier behaviour was a silent restart from step 0 while the log still said
  "resuming", which spends a full session producing a run that is not the run
  that was asked for. `config_hash()` deliberately ignores `notes` and
  `checkpoint_every_steps` so cosmetic edits do not invalidate a good checkpoint.

### Output limits are a design constraint, not an afterthought

The expanded prep writes **122,252 clips / 15.5 GB** into a `/kaggle/working`
whose limit is about **19.6 GB**, and the final prep cell prints the running
total so a composition change that drifts toward the ceiling is visible before a
training job is pushed at a silently truncated dataset. The file *count* turned
out to be the tighter limit: kernel-output publishing silently fails past roughly
**100k loose files**, reporting success and shipping an 845-byte empty
`_output_.zip`. Packing the audio into **64 ZIP shards** (§2.1a) is what makes
the E5/E6 dataset publishable at all, and the prep asserts that every manifest
row has a shard entry before it finishes.

### No GPU minute spent on untested code

Both Kaggle notebooks are **generated** from `src/turn_detector/` by
`tools/build_notebooks.py` rather than maintained by hand, so the code that runs
on the T4 is the code the test suite exercises, and `tests/test_notebooks.py`
enforces that the two cannot drift. Before any push, `tests/test_train_smoke.py`
runs the exact production path end to end on CPU against a tiny synthetic
dataset: train, validate, checkpoint, resume, test-slice metrics, fp32 ONNX
export, int8 quantization, and the fp32-versus-torch parity assertion, plus a
distillation variant that checks a run claiming to distil refuses to start
without a teacher. Every configuration in `config.py` therefore reaches the GPU
having already run to completion on a laptop.

## 10. Reproduction

Everything below runs from the repo root with `.venv/Scripts/python.exe`
(Python 3.12; `uv sync` builds the environment).

| step | command | output |
|---|---|---|
| tests (39, ~1 min) | `python -m pytest -q` | feature parity vs HF, augmentation, sampler, metrics, notebook invariants, train + distillation smoke |
| Hinglish corpus plan | `python -m synth.corpus` | `synth/output/corpus_plan.jsonl` |
| Hinglish TTS render | `python -m synth.tts_generate` | `synth/output/audio/`, `boundaries/`, `manifest.jsonl` |
| package + split | `python -m synth.package_kaggle` | `manifest.parquet` (template-level split), `hinglish-synth.zip` |
| data prep + training | see **`KAGGLE_RUNBOOK.md`** | `experiments/run_<exp>/` |
| aggregate tables | `python -m tools.aggregate_results` | `experiments/RESULTS.md` |
| error analysis | `python -m tools.error_analysis --run experiments/run_e2_hinglish_aug` | `analysis/worst_errors.md`, `prob_curves.png`, `threshold_sweep.png` |
| build HF Space | `python -m tools.build_space` | `space/` (torch-free, deployable) |
| local demo | `python demo/app.py` | Gradio UI on localhost |

`KAGGLE_RUNBOOK.md` covers the two Kaggle notebooks (`01_data_prep`, `02_train`),
which are **generated from the tested library** by
`python -m tools.build_notebooks`, so the notebook and `src/turn_detector/` never
drift, and `tests/test_notebooks.py` enforces it. It also documents the
`push_kaggle` CLI (`prep` / `train <exp>` / `status` / `pull` / `--resume`), the
10.5 h time budget that keeps a killed 12 h session recoverable, and the
config-hash check that makes a mismatched resume fail loudly instead of silently
restarting from step 0. Why each of those exists is §9.

The silence stress test in §5.3 is recorded in
`experiments/silence_stress_test.json`. To re-derive it: load the 225 rows of
`synth/output/manifest.parquet` where `split == "test"`, score each clip with
`TurnDetector(run/model_int8.onnx, threshold=<that run's threshold>)`, re-score
with 0.5 s and 1.0 s of zeros appended, and count the decisions that change and,
among those, the ones going incomplete → complete.

Per-run configs and full metrics live in `experiments/run_*/metrics.json`;
`src/turn_detector/config.py` is the single source of truth for E1–E6, including
the distillation knobs (`kd_teacher`, `kd_alpha`, `kd_temperature`). E5 is the
one run that needs an extra input: `train(..., teacher_dir=<dir with
e2_hinglish_aug/ckpt_best.pt>)`, staged on Kaggle as the `turn-detect-ckpt`
dataset.
