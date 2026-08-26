import numpy as np
import torch

from turn_detector.features import (
    N_SAMPLES, LogMel, right_align,
)


def test_right_align_pads_left():
    wav = np.ones(1000, dtype=np.float32)
    out = right_align(wav)
    assert len(out) == N_SAMPLES
    assert out[-1000:].sum() == 1000 and out[:1000].sum() == 0


def test_right_align_takes_tail():
    wav = np.arange(N_SAMPLES + 500, dtype=np.float32)
    out = right_align(wav)
    assert len(out) == N_SAMPLES
    assert out[0] == 500 and out[-1] == N_SAMPLES + 499


def test_logmel_shape():
    mel = LogMel()(torch.randn(2, N_SAMPLES))
    assert mel.shape == (2, 80, 800)
    assert torch.isfinite(mel).all()


def test_logmel_matches_hf_feature_extractor():
    from transformers import WhisperFeatureExtractor
    rng = np.random.default_rng(0)
    wav = (0.2 * rng.normal(0, 1, N_SAMPLES)).astype(np.float32)
    fe = WhisperFeatureExtractor()
    hf = fe(wav, sampling_rate=16000, padding="max_length",
            max_length=N_SAMPLES, return_tensors="np")["input_features"][0]
    ours = LogMel()(torch.from_numpy(wav)).numpy()[0]
    assert hf.shape == ours.shape == (80, 800)
    assert np.abs(hf - ours).max() < 1e-4


def test_numpy_logmel_matches_torch():
    from turn_detector.infer import NumpyLogMel
    rng = np.random.default_rng(1)
    wav = (0.2 * rng.normal(0, 1, N_SAMPLES)).astype(np.float32)
    ours = LogMel()(torch.from_numpy(wav)).numpy()[0]
    np_mel = NumpyLogMel()(wav)
    assert np_mel.shape == (80, 800)
    assert np.abs(np_mel - ours).max() < 1e-4
