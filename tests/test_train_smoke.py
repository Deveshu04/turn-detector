"""End-to-end CPU smoke of the exact path the Kaggle notebook runs:
train -> validate -> checkpoint -> test slices -> ONNX fp32 -> int8 -> parity.
Uses TinyMelNet to stay fast; the Whisper arch is covered by shape tests.
"""

import json

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
