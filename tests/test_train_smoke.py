"""End-to-end CPU smoke of the exact path the Kaggle notebook runs:
train -> validate -> checkpoint -> test slices -> ONNX fp32 -> int8 -> parity.
Uses TinyMelNet to stay fast; the Whisper arch is covered by shape tests.
"""

import json

import torch

from turn_detector.config import ExperimentConfig
from turn_detector.train import train


def test_train_smoke(tiny_dataset, tmp_path):
    manifest, root = tiny_dataset
    cfg = ExperimentConfig(
        name="smoke", arch="tinymel", epochs=2, batch_size=4,
        pause_cut_p=0.2, trailing_silence_p=0.5,
        checkpoint_every_steps=2, seed=0,
    )
    sources = {k: [(manifest, root)] for k in ("train", "val", "test")}
    out = tmp_path / "run"
    metrics = train(cfg, sources, str(out), device="cpu",
                    steps_per_epoch=3, num_workers=0)

    assert (out / "ckpt_last.pt").exists()
    assert (out / "ckpt_best.pt").exists()
    assert (out / "model_fp32.onnx").exists()
    assert (out / "model_int8.onnx").exists()
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["test"]["overall"]["n"] == 2
    assert "int8_subset" in saved
    assert saved["int8_subset"]["fp32_parity_max_dprob"] < 1e-3

    # resume path: rerun with same config resumes at final step, retrains nothing
    metrics2 = train(cfg, sources, str(out), device="cpu",
                     steps_per_epoch=3, num_workers=0)
    assert metrics2["experiment"] == "smoke"


def test_time_budget_stops_early_and_leaves_a_resumable_checkpoint(
        tiny_dataset, tmp_path):
    """A Kaggle commit run killed by the 12 h wall publishes nothing, so the
    run must end itself: save ckpt_last, skip the final eval/export, and say so.
    """
    manifest, root = tiny_dataset
    cfg = ExperimentConfig(
        name="budget", arch="tinymel", epochs=4, batch_size=4,
        checkpoint_every_steps=2, seed=0,
    )
    sources = {k: [(manifest, root)] for k in ("train", "val", "test")}
    out = tmp_path / "run_budget"

    metrics = train(cfg, sources, str(out), device="cpu", steps_per_epoch=3,
                    num_workers=0, time_budget_minutes=0)

    assert metrics["status"] == "time_budget_reached"
    assert metrics["experiment"] == "budget"
    assert 0 < metrics["step"] < metrics["total_steps"] == 12
    # stopped before the final eval/export
    assert not (out / "metrics.json").exists()
    assert not (out / "model_fp32.onnx").exists()
    # but left something to resume from, with a matching hash
    assert (out / "ckpt_last.pt").exists()
    assert not list(out.glob("*.tmp"))            # atomic write left no debris
    head = torch.load(out / "ckpt_last.pt", map_location="cpu",
                      weights_only=False)
    assert head["cfg_hash"] == cfg.config_hash()
    assert head["step"] == metrics["step"]

    # the next run continues from that step instead of restarting at 0
    # (a fresh start would stop at the same step it did the first time)
    again = train(cfg, sources, str(out), device="cpu", steps_per_epoch=3,
                  num_workers=0, time_budget_minutes=0)
    assert again["status"] == "time_budget_reached"
    assert again["step"] > metrics["step"]


def test_config_hash_ignores_cosmetic_fields():
    base = ExperimentConfig(name="x")
    assert base.config_hash() == ExperimentConfig(
        name="x", notes="totally different note",
        checkpoint_every_steps=17).config_hash()
    assert base.config_hash() != ExperimentConfig(name="x", seed=1).config_hash()
