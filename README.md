# turn-detector — Tiny Audio Turn Detection for Hinglish

A tiny (~8M param), fast (<30ms CPU) audio-based **turn detection** model: given the last 8 seconds of a user's speech, decide whether they are **done speaking** or **just pausing** — with specific robustness for **Indian Hinglish, filler words (haan, matlab, accha, toh...), and mid-thought pauses**.

> Status: work in progress. See `experiments/STATE.json` for live progress and `KAGGLE_RUNBOOK.md` for how training runs are executed.

## Approach (summary)

- **Backbone:** Whisper-Tiny encoder (multilingual, 4 layers, d=384) truncated to an 8s input window, + attention pooling + small MLP head → P(turn complete). Exported to int8 ONNX (~8 MB).
- **From-scratch comparison:** ~1.5M-param mel-CNN + BiGRU model trained on identical data.
- **Data:** English + Hindi subsets of [pipecat smart-turn-data-v3.2](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train), plus ~6k **synthetic code-switched Hinglish** clips generated with Indian neural TTS voices from a curated corpus with filler words, trailing conjunctions, and word-boundary audio cuts.
- **Key augmentation:** pause-injection — cutting complete utterances at speech-active points and appending silence teaches the model that *silence ≠ done*.
- **Evaluation:** official smart-turn v3.2 test split (EN/HI slices), held-out Hinglish set, filler-heavy slices, plus CPU latency benchmarks.

Results tables, experiment matrix (E1–E4), and analysis land here after training.

## Attribution

Approach informed by [pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn) (BSD-2-Clause); all pipeline code here is written from scratch. Dataset: pipecat-ai smart-turn-data v3.2 (see dataset card for licenses).
