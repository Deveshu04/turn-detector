"""rank_auc must use midranks, i.e. agree with the pairwise definition of AUC
even when scores tie (which they do constantly after int8 quantisation and
wherever a sigmoid saturates to 0.0/1.0).
"""

import numpy as np
import pytest

from turn_detector.train import rank_auc


def brute_force_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC by definition: P(pos > neg) + 0.5 * P(pos == neg), over all pairs."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    wins = sum(float(p > n) + 0.5 * float(p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def tied_sample(seed: int, n: int = 120):
    """Random labels/scores coarsely binned so ~every score is part of a tie."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    scores = np.round(rng.random(n) + 0.35 * labels, 1)   # 1 decimal -> heavy ties
    return labels, scores


def test_matches_brute_force_pairwise_auc_with_ties():
    for seed in range(12):
        labels, scores = tied_sample(seed)
        # the sample must actually contain ties, else this proves nothing
        assert len(np.unique(scores)) < len(scores)
        assert rank_auc(labels, scores) == pytest.approx(
            brute_force_auc(labels, scores), abs=1e-12)


def test_matches_brute_force_on_continuous_scores():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 80)
    scores = rng.random(80)
    assert rank_auc(labels, scores) == pytest.approx(
        brute_force_auc(labels, scores), abs=1e-12)


def test_permutation_invariant_on_tie_heavy_data():
    labels, scores = tied_sample(3)
    ref = rank_auc(labels, scores)
    rng = np.random.default_rng(11)
    for _ in range(10):
        perm = rng.permutation(len(labels))
        assert rank_auc(labels[perm], scores[perm]) == pytest.approx(ref, abs=1e-12)


def test_all_scores_identical_is_half():
    labels = np.array([0, 1, 0, 1, 1, 0])
    scores = np.full(6, 0.7)
    assert rank_auc(labels, scores) == pytest.approx(0.5)


def test_single_class_is_nan():
    assert np.isnan(rank_auc(np.ones(5), np.arange(5.0)))
    assert np.isnan(rank_auc(np.zeros(5), np.arange(5.0)))
