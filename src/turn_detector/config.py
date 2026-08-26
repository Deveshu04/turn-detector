"""Experiment configurations E1-E4."""

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class ExperimentConfig:
    name: str
    arch: str = "whisper"              # whisper | tinymel
    use_hinglish_synth: bool = False

    # augmentation probabilities (train only)
    pause_cut_p: float = 0.0           # complete -> cut mid-speech, label flips to 0
    trailing_silence_p: float = 0.0
    noise_p: float = 0.0
    speed_p: float = 0.0

    # optimization
    epochs: int = 4
    batch_size: int = 64
    lr_encoder: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    seed: int = 42

    # bookkeeping
    checkpoint_every_steps: int = 500
    notes: str = ""

    def config_hash(self) -> str:
        """Hash of the fields that change training maths.

        Cosmetic/operational fields (`notes`, `checkpoint_every_steps`) are
        excluded so editing a comment or checkpoint cadence does not invalidate
        an in-flight run's resumable checkpoint.
        """
        d = asdict(self)
        d.pop("notes", None)
        d.pop("checkpoint_every_steps", None)
        return hashlib.sha1(
            json.dumps(d, sort_keys=True).encode()
        ).hexdigest()[:10]


EXPERIMENTS = {
    "e1_baseline": ExperimentConfig(
        name="e1_baseline",
        notes="EN+HI real data only, no augmentation, WhisperTinyTurn",
    ),
    "e2_hinglish_aug": ExperimentConfig(
        name="e2_hinglish_aug",
        use_hinglish_synth=True,
        pause_cut_p=0.15,
        trailing_silence_p=0.5,
        noise_p=0.25,
        speed_p=0.25,
        notes="headline model: +hinglish synth, +pause/silence/noise/speed aug",
    ),
    "e3_tinymel_scratch": ExperimentConfig(
        name="e3_tinymel_scratch",
        arch="tinymel",
        use_hinglish_synth=True,
        pause_cut_p=0.15,
        trailing_silence_p=0.5,
        noise_p=0.25,
        speed_p=0.25,
        epochs=8,
        lr_head=3e-4,
        notes="from-scratch ~1M param model, same data/aug as e2",
    ),
    "e4_no_pause_aug": ExperimentConfig(
        name="e4_no_pause_aug",
        use_hinglish_synth=True,
        pause_cut_p=0.0,
        trailing_silence_p=0.0,
        noise_p=0.25,
        speed_p=0.25,
        notes="ablation: e2 minus pause_cut and trailing_silence",
    ),
}
