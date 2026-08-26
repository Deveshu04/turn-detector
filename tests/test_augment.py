import numpy as np

from turn_detector import augment
from tests.conftest import SR, make_clip


def test_trailing_silence():
    rng = np.random.default_rng(0)
    wav = make_clip(rng, 2.0)
    out = augment.trailing_silence(wav, rng)
    assert len(out) > len(wav)
    assert np.array_equal(out[: len(wav)], wav)
    assert np.abs(out[len(wav):]).max() == 0


def test_pause_cut_ends_in_silence_and_removes_speech():
    rng = np.random.default_rng(1)
    wav = make_clip(rng, 3.0)
    out = augment.pause_cut(wav, rng)
    assert out is not None
    tail = out[-int(0.15 * SR):]
    assert np.abs(tail).max() == 0            # appended silence
    assert len(out) < len(wav) + int(1.3 * SR)
    # the cut removed real signal from the end of the original
    assert np.abs(wav[len(out) - int(1.2 * SR):]).max() > 0.05


def test_pause_cut_rejects_silence():
    rng = np.random.default_rng(2)
    assert augment.pause_cut(np.zeros(SR * 2, dtype=np.float32), rng) is None


def test_add_noise_snr():
    rng = np.random.default_rng(3)
    wav = make_clip(rng, 2.0)
    out = augment.add_noise(wav, rng, snr_lo=20, snr_hi=20)
    noise = out - wav
    snr = 10 * np.log10((wav ** 2).mean() / (noise ** 2).mean())
    assert 18 < snr < 22


def test_speed_perturb_changes_length():
    rng = np.random.default_rng(4)
    wav = make_clip(rng, 2.0)
    out = augment.speed_perturb(wav, rng, lo=1.1, hi=1.1)
    assert abs(len(out) - len(wav) / 1.1) < 3
