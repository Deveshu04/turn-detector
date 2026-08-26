"""Waveform-level augmentations for turn detection.

The two augmentations that teach "silence != done":
  - trailing_silence: appended to ANY clip without changing its label — a
    finished speaker followed by silence is still finished, an unfinished one
    is still unfinished.
  - pause_cut: truncate a COMPLETE utterance at a speech-active point and
    append silence -> a genuine "paused mid-thought" example (label flips to
    incomplete). Applied on the fly in the dataset.

All functions take/return float32 mono @16kHz and are deterministic given rng.
"""

import numpy as np

SR = 16000


def trailing_silence(wav: np.ndarray, rng: np.random.Generator,
                     min_s: float = 0.2, max_s: float = 1.2) -> np.ndarray:
    n = int(rng.uniform(min_s, max_s) * SR)
    return np.concatenate([wav, np.zeros(n, dtype=np.float32)])


def _energy_envelope(wav: np.ndarray, frame: int = 400, hop: int = 160):
    n_frames = max(1, (len(wav) - frame) // hop + 1)
    idx = np.arange(n_frames)[:, None] * hop + np.arange(frame)[None, :]
    return np.sqrt((wav[idx.clip(max=len(wav) - 1)] ** 2).mean(axis=1)), hop


def pause_cut(wav: np.ndarray, rng: np.random.Generator,
              lo: float = 0.4, hi: float = 0.85,
              min_keep_s: float = 0.6, min_removed_s: float = 0.4):
    """Cut a complete utterance mid-speech. Returns wav or None if impossible."""
    env, hop = _energy_envelope(wav)
    thresh = max(env.max() * 0.15, 1e-4)
    active = np.nonzero(env > thresh)[0]
    if len(active) < 10:
        return None
    span_start, span_end = active[0], active[-1]
    span = span_end - span_start
    if span < 20:
        return None
    # candidate frames inside [lo, hi] of the active span that are speech-active
    frame_lo = span_start + int(span * lo)
    frame_hi = span_start + int(span * hi)
    candidates = active[(active >= frame_lo) & (active <= frame_hi)]
    if len(candidates) == 0:
        return None
    cut_frame = int(rng.choice(candidates))
    cut = cut_frame * hop
    if cut < min_keep_s * SR or len(wav) - cut < min_removed_s * SR:
        return None
    out = wav[:cut]
    return trailing_silence(out, rng)


def add_noise(wav: np.ndarray, rng: np.random.Generator,
              snr_lo: float = 10.0, snr_hi: float = 30.0) -> np.ndarray:
    rms = np.sqrt((wav ** 2).mean())
    if rms < 1e-5:
        return wav
    snr = rng.uniform(snr_lo, snr_hi)
    noise_rms = rms / (10 ** (snr / 20))
    return (wav + rng.normal(0, noise_rms, len(wav))).astype(np.float32)


def speed_perturb(wav: np.ndarray, rng: np.random.Generator,
                  lo: float = 0.9, hi: float = 1.1) -> np.ndarray:
    factor = rng.uniform(lo, hi)
    n_out = int(len(wav) / factor)
    x_old = np.arange(len(wav))
    x_new = np.linspace(0, len(wav) - 1, n_out)
    return np.interp(x_new, x_old, wav).astype(np.float32)
