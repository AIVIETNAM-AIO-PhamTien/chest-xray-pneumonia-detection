"""Tests for the v6 checkpoint-selection hierarchy and its metrics."""

import numpy as np
import pytest

from src.evaluation.selection import (
    better_checkpoint,
    exact_threshold_at_sensitivity,
    normalized_partial_auc,
    specificity_at_sensitivity,
)


@pytest.fixture
def separable():
    """Scores where the classes overlap only slightly.

    Returns:
        Tuple of (labels, probabilities).
    """
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(200, int), np.ones(200, int)]
    probs = np.r_[rng.beta(2, 8, 200), rng.beta(8, 2, 200)]
    return labels, probs


def test_threshold_delivers_the_requested_sensitivity(separable):
    """The chosen threshold must actually hold the constraint."""
    labels, probs = separable
    threshold = exact_threshold_at_sensitivity(labels, probs, 0.97)
    assert (probs[labels == 1] >= threshold).mean() >= 0.97


def test_threshold_is_the_highest_that_qualifies(separable):
    """Any higher observed score would break the constraint.

    Anything lower would give away specificity for nothing.
    """
    labels, probs = separable
    threshold = exact_threshold_at_sensitivity(labels, probs, 0.97)
    higher = probs[probs > threshold]
    if len(higher):
        assert (probs[labels == 1] >= higher.min()).mean() < 0.97


def test_threshold_search_is_exact_not_subsampled():
    """A grid of a few hundred points can miss the one threshold that matters.

    Two positives sit just above a cluster of negatives; only a candidate
    drawn from the observed values separates them.
    """
    labels = np.r_[np.zeros(500, int), np.ones(100, int)]
    negatives = np.linspace(0.0, 0.60, 500)
    positives = np.r_[np.full(97, 0.9), [0.601, 0.6011, 0.6012]]
    probs = np.r_[negatives, positives]
    threshold = exact_threshold_at_sensitivity(labels, probs, 0.97)
    assert threshold in set(probs)
    assert (probs[labels == 1] >= threshold).mean() >= 0.97
    specificity, _ = specificity_at_sensitivity(labels, probs, 0.97)
    assert specificity == 1.0


def test_unreachable_sensitivity_returns_floor():
    """Asking for more sensitivity than exists must not raise."""
    labels = np.array([0, 1, 1])
    probs = np.array([0.9, 0.1, 0.2])
    assert exact_threshold_at_sensitivity(labels, probs, 1.01) == 0.0


def test_partial_auc_is_one_for_a_perfect_ranking():
    """No overlap means the restricted region is perfect too."""
    labels = np.r_[np.zeros(100, int), np.ones(100, int)]
    probs = np.r_[np.linspace(0.0, 0.4, 100), np.linspace(0.6, 1.0, 100)]
    assert normalized_partial_auc(labels, probs, 0.97) == pytest.approx(1.0, abs=1e-6)


def test_partial_auc_is_near_zero_for_random_scores():
    """Chance ranking buys almost no specificity at 97% sensitivity.

    The metric is mean specificity, not a rescaled area, so the chance value
    is about 1.5% rather than 0.5.
    """
    rng = np.random.default_rng(1)
    labels = rng.binomial(1, 0.5, 6000)
    probs = rng.uniform(size=6000)
    assert normalized_partial_auc(labels, probs, 0.97) == pytest.approx(0.015, abs=0.02)


def test_partial_auc_agrees_with_specificity_at_the_target():
    """It should sit near the discrete specificity it generalises."""
    rng = np.random.default_rng(5)
    labels = np.r_[np.zeros(600, int), np.ones(600, int)]
    probs = np.r_[rng.beta(2, 6, 600), rng.beta(6, 2, 600)]
    discrete, _ = specificity_at_sensitivity(labels, probs, 0.97)
    assert normalized_partial_auc(labels, probs, 0.97) == pytest.approx(
        discrete, abs=0.12)


def test_partial_auc_sees_local_damage_global_auc_misses():
    """The point of the metric: react where the operating point sits.

    A handful of negatives pushed into the range of the lowest-scoring
    positives costs almost nothing globally, because they still rank below
    most positives. It costs a great deal at 97% sensitivity, which is exactly
    where the threshold has to sit.
    """
    from sklearn.metrics import roc_auc_score
    labels = np.r_[np.zeros(400, int), np.ones(400, int)]
    probs = np.r_[np.linspace(0.0, 0.5, 400), np.linspace(0.5, 1.0, 400)]
    damaged = probs.copy()
    damaged[:8] = np.linspace(0.502, 0.512, 8)

    auc_drop = roc_auc_score(labels, probs) - roc_auc_score(labels, damaged)
    pauc_drop = (normalized_partial_auc(labels, probs, 0.97)
                 - normalized_partial_auc(labels, damaged, 0.97))
    assert auc_drop < 0.001
    assert pauc_drop > auc_drop * 10


def test_partial_auc_is_nan_for_a_single_class():
    """One class present means there is no ROC to restrict."""
    assert np.isnan(normalized_partial_auc(np.zeros(10, int),
                                           np.linspace(0, 1, 10)))


def _entry(specificity, partial_auc, nll):
    return {"specificity": specificity, "partial_auc": partial_auc, "nll": nll}


def test_specificity_decides_outside_the_tie_band():
    """A clear specificity gain wins regardless of the other metrics."""
    replace, reason = better_checkpoint(_entry(0.90, 0.10, 9.0),
                                        _entry(0.80, 0.99, 0.01))
    assert replace and reason == "specificity"


def test_specificity_loss_outside_the_band_is_rejected():
    """And a clear loss is refused however good the rest looks."""
    replace, reason = better_checkpoint(_entry(0.80, 0.99, 0.01),
                                        _entry(0.90, 0.10, 9.0))
    assert not replace and reason == "specificity"


def test_partial_auc_breaks_a_specificity_tie():
    """Inside the band the high-sensitivity region decides, not log-loss.

    This is the case Stage B kept hitting: one group changes sides, which is
    inside the band, and the old rule fell straight through to NLL.
    """
    replace, reason = better_checkpoint(_entry(0.9959, 0.95, 0.20),
                                        _entry(0.9918, 0.90, 0.10))
    assert replace and reason == "partial_auc"


def test_nll_breaks_a_tie_in_both_specificity_and_partial_auc():
    """When neither leading metric separates, calibration decides."""
    replace, reason = better_checkpoint(_entry(0.9959, 0.9005, 0.05),
                                        _entry(0.9918, 0.9000, 0.10))
    assert replace and reason == "nll"


def test_a_full_tie_keeps_the_earlier_epoch():
    """Nothing to choose between them means no reason to move."""
    replace, reason = better_checkpoint(_entry(0.99, 0.90, 0.10),
                                        _entry(0.99, 0.90, 0.10))
    assert not replace and reason == "earlier_epoch"


def test_missing_partial_auc_falls_through_to_nll():
    """A degenerate region must not block selection."""
    replace, reason = better_checkpoint(
        {"specificity": 0.99, "partial_auc": float("nan"), "nll": 0.05},
        {"specificity": 0.99, "partial_auc": float("nan"), "nll": 0.10})
    assert replace and reason == "nll"


def test_first_epoch_is_always_taken():
    """There is nothing to compare against yet."""
    replace, reason = better_checkpoint(_entry(0.5, 0.5, 1.0), None)
    assert replace and reason == "first"
