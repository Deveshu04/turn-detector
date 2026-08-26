import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SR = 16000


def make_clip(rng: np.random.Generator, seconds: float) -> np.ndarray:
    """Speech-like burst: modulated tones with silence tails."""
    t = np.arange(int(seconds * SR)) / SR
    f0 = rng.uniform(120, 260)
    wav = 0.3 * np.sin(2 * np.pi * f0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3.1 * t))
    wav += 0.05 * rng.normal(0, 1, len(t))
    fade = int(0.02 * SR)
    wav[:fade] *= np.linspace(0, 1, fade)
    wav[-fade:] *= np.linspace(1, 0, fade)
    return wav.astype(np.float32)


@pytest.fixture
def tiny_dataset(tmp_path):
    """16 fake clips + manifest.parquet covering all splits and slice columns."""
    rng = np.random.default_rng(7)
    audio_root = tmp_path
    (audio_root / "audio").mkdir()
    rows = []
    langs = ["english", "hindi", "hinglish"]
    for i in range(16):
        wav = make_clip(rng, rng.uniform(1.0, 3.5))
        path = f"audio/clip{i:02d}.flac"
        sf.write(audio_root / path, wav, SR, subtype="PCM_16")
        split = "train" if i < 12 else ("val" if i < 14 else "test")
        rows.append({
            "id": f"clip{i:02d}", "path": path, "label": i % 2,
            "language": langs[i % 3], "split": split,
            "midfiller": bool(i % 4 == 0), "endfiller": bool(i % 5 == 0),
            "synthetic": bool(i % 2), "kind": "full", "source": "test",
        })
    manifest = audio_root / "manifest.parquet"
    pl.DataFrame(rows).write_parquet(manifest)
    return str(manifest), str(audio_root)
