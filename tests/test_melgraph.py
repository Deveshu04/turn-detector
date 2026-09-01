from pathlib import Path

import numpy as np
import pytest
import torch

from turn_detector.common import N_SAMPLES, right_align
from turn_detector.features import LogMel
from turn_detector.melgraph import FullTurnModel, MelDFT

EXAMPLES = Path(__file__).parent.parent / "demo" / "examples"


def _max_diff(wav: np.ndarray) -> float:
    x = torch.from_numpy(wav)
    return float((MelDFT()(x) - LogMel()(x)).abs().max())


def test_meldft_shape():
    mel = MelDFT()(torch.randn(2, N_SAMPLES))
    assert mel.shape == (2, 80, 800)
    assert torch.isfinite(mel).all()


def test_meldft_matches_logmel_on_noise():
    rng = np.random.default_rng(0)
    wav = (0.2 * rng.normal(0, 1, N_SAMPLES)).astype(np.float32)
    diff = _max_diff(wav)
    print(f"\nMelDFT vs LogMel, random audio: max abs diff = {diff:.3e}")
    assert diff < 1e-3


def test_meldft_matches_logmel_on_real_clip():
    clips = sorted(EXAMPLES.glob("*.flac"))
    if not clips:
        pytest.skip(f"no demo clips in {EXAMPLES}")
    import soundfile as sf
    worst = 0.0
    for path in clips:
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        assert sr == 16000
        diff = _max_diff(right_align(wav))
        print(f"MelDFT vs LogMel, {path.name}: max abs diff = {diff:.3e}")
        worst = max(worst, diff)
    assert worst < 1e-3


def test_full_turn_model_wraps_mel_and_model():
    class Head(torch.nn.Module):
        def forward(self, mel):
            return mel.mean(dim=(1, 2))

    wav = torch.randn(1, N_SAMPLES)
    full = FullTurnModel(Head())
    out = full(wav)
    assert out.shape == (1,)
    assert torch.allclose(out, LogMel()(wav).mean(dim=(1, 2)), atol=1e-4)
