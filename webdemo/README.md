---
title: Hinglish Turn Detector
emoji: 🎙️
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: On-device turn detection for Hinglish speech, in the browser
---

# Hinglish Turn Detector

Done speaking, or just pausing? This page answers that question for Hinglish speech
without sending audio anywhere. Pick one of the labelled examples, upload a clip, or
record from your microphone. The audio is resampled to 16 kHz mono in the browser,
right aligned into an 8 s window, and run through an int8 ONNX classifier with
onnxruntime-web. You get P(complete), a verdict against an adjustable threshold, the
measured inference latency, and a streaming curve showing how the probability evolves
as the utterance grows. Two models are available: the accurate Whisper-Tiny encoder
head and a much smaller distilled TinyMelNet. Everything, including the runtime and
the weights, is served from this Space.

Training code, data pipeline, and evaluation live in the GitHub repo:
https://github.com/Deveshu04/turn-detector

Weights: https://huggingface.co/deveshu/hinglish-turn-detector
