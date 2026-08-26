import numpy as np
import torch

from turn_detector.config import EXPERIMENTS, ExperimentConfig
from turn_detector.dataset import TurnDataset, load_manifests
from turn_detector.features import N_SAMPLES


def test_load_and_getitem(tiny_dataset):
    manifest, root = tiny_dataset
    df = load_manifests([(manifest, root)], "train")
    assert df.height == 12
    ds = TurnDataset(df, EXPERIMENTS["e1_baseline"], train=False)
    wav, label, idx = ds[0]
    assert wav.shape == (N_SAMPLES,) and wav.dtype == torch.float32
    assert label.item() in (0.0, 1.0)


def test_augmented_getitem_deterministic(tiny_dataset):
    manifest, root = tiny_dataset
    df = load_manifests([(manifest, root)], "train")
    cfg = EXPERIMENTS["e2_hinglish_aug"]
    ds = TurnDataset(df, cfg, train=True)
    w1, l1, _ = ds[2]
    w2, l2, _ = ds[2]
    assert torch.equal(w1, w2) and l1 == l2      # same epoch -> same aug
    ds.set_epoch(1)
    w3, _, _ = ds[2]
    assert not torch.equal(w1, w3)               # new epoch -> new aug


def test_balanced_sampler_offsets_pause_cut(tiny_dataset):
    manifest, root = tiny_dataset
    df = load_manifests([(manifest, root)], "train")
    cfg = ExperimentConfig(name="t", pause_cut_p=0.15)
    ds = TurnDataset(df, cfg, train=True)
    sampler = ds.balanced_sampler(num_samples=4000)
    labels = np.array([ds.rows[i]["label"] for i in sampler])
    drawn_pos = labels.mean()
    assert 0.54 < drawn_pos < 0.64               # ~0.588 target draw rate
