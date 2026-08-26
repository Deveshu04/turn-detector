"""Dataset over FLAC shards + parquet manifests.

Works identically on Kaggle (prep-notebook output + hinglish-synth dataset)
and locally (small subsets, unit tests). Multiple sources are concatenated;
each manifest row needs: id, path, label, language, split — optional:
midfiller, endfiller, synthetic, kind, source.

Augmentation policy (train split only) comes from ExperimentConfig. pause_cut
flips a complete example's label to incomplete on the fly; the sampler
compensates so the effective batch balance stays ~50/50.
"""

from pathlib import Path

import numpy as np
import polars as pl
import soundfile as sf
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from turn_detector import augment
from turn_detector.config import ExperimentConfig
from turn_detector.features import N_SAMPLES, right_align

OPTIONAL_COLS = {
    "midfiller": False, "endfiller": False, "synthetic": False,
    "kind": "", "source": "", "language": "",
}


def load_manifests(sources: list[tuple[str, str]], split: str) -> pl.DataFrame:
    """sources: [(manifest_parquet_path, audio_root), ...] -> unified frame."""
    frames = []
    for manifest_path, audio_root in sources:
        df = pl.read_parquet(manifest_path).filter(pl.col("split") == split)
        for col, default in OPTIONAL_COLS.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default).alias(col))
            else:
                df = df.with_columns(pl.col(col).fill_null(default))
        df = df.with_columns(pl.lit(str(audio_root)).alias("audio_root"))
        frames.append(df.select(
            "id", "path", "label", "language", "split",
            "midfiller", "endfiller", "synthetic", "kind", "source", "audio_root",
        ))
    return pl.concat(frames)


class TurnDataset(Dataset):
    def __init__(self, manifest: pl.DataFrame, cfg: ExperimentConfig,
                 train: bool, seed_offset: int = 0):
        self.rows = manifest.to_dicts()
        self.cfg = cfg
        self.train = train
        self.seed_offset = seed_offset
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        wav, sr = sf.read(Path(row["audio_root"]) / row["path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        label = int(row["label"])

        if self.train:
            rng = np.random.default_rng(
                (self.cfg.seed + self.seed_offset) * 1_000_003
                + self.epoch * 101 + i
            )
            c = self.cfg
            if label == 1 and rng.random() < c.pause_cut_p:
                cut = augment.pause_cut(wav, rng)
                if cut is not None:
                    wav, label = cut, 0
            if rng.random() < c.trailing_silence_p:
                wav = augment.trailing_silence(wav, rng)
            if rng.random() < c.speed_p:
                wav = augment.speed_perturb(wav, rng)
            if rng.random() < c.noise_p:
                wav = augment.add_noise(wav, rng)

        wav = right_align(wav, N_SAMPLES)
        return torch.from_numpy(wav), torch.tensor(label, dtype=torch.float32), i

    def balanced_sampler(self, num_samples: int | None = None) -> WeightedRandomSampler:
        """50/50 sampler; completes oversampled to offset pause_cut label flips."""
        labels = np.array([r["label"] for r in self.rows])
        n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
        # fraction of drawn completes that stay complete after pause_cut
        keep = 1.0 - self.cfg.pause_cut_p if self.train else 1.0
        target_pos_draw = 0.5 / keep if keep > 0 else 0.5
        target_pos_draw = min(target_pos_draw, 0.75)
        w_pos = target_pos_draw / max(n_pos, 1)
        w_neg = (1 - target_pos_draw) / max(n_neg, 1)
        weights = np.where(labels == 1, w_pos, w_neg)
        return WeightedRandomSampler(
            torch.from_numpy(weights).double(),
            num_samples=num_samples or len(self.rows),
            replacement=True,
        )
