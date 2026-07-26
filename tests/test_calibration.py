"""Tests for post-hoc calibration and its invariance guarantees."""

import numpy as np
import pytest

from src.evaluation.calibration import (
    CALIBRATORS,
    brier,
    calibration_curve_fit,
    ece_adaptive,
    ece_fixed,
    log_loss,
    monotonic_invariance_check,
    threshold_at_sensitivity,
)


@pytest.fixture
def scores():
    """Miscalibrated but informative scores with a known positive rate.

    Returns:
        Tuple of (labels, probabilities).
    """
    rng = np.random.default_rng(0)
    labels = rng.binomial(1, 0.4, 2000)
    latent = rng.normal(labels * 2.0 - 1.0, 1.0)
    # Divide by a temperature below 1 to make the scores over-confident.
    probs = 1.0 / (1.0 + np.exp(-latent / 0.5))
    return labels, probs


@pytest.mark.parametrize("name", sorted(CALIBRATORS))
def test_calibrators_return_valid_probabilities(name, scores):
    """Every calibrator must map into the open unit interval."""
    labels, probs = scores
    out = CALIBRATORS[name](labels, probs)(probs)
    assert out.shape == probs.shape
    assert np.isfinite(out).all()
    assert (out > 0).all() and (out < 1).all()


@pytest.mark.parametrize("name", ["temperature", "intercept", "platt", "beta"])
def test_parametric_calibrators_preserve_ranking(name, scores):
    """Parametric calibrators are monotone, so AUC must not move.

    This is the property the whole post-ensemble analysis rests on: these maps
    can change probabilities but never reorder cases.
    """
    labels, probs = scores
    out = CALIBRATORS[name](labels, probs)(probs)
    order_before = np.argsort(probs, kind="stable")
    order_after = np.argsort(out, kind="stable")
    assert np.array_equal(order_before, order_after)


@pytest.mark.parametrize("name", ["temperature", "intercept", "platt", "beta"])
def test_reselected_threshold_leaves_decisions_unchanged(name, scores):
    """Monotone calibration plus threshold reselection is a no-op on labels.

    A specificity gain reported from that setup would be an implementation
    error, so this pins the invariant rather than the calibrators.
    """
    labels, probs = scores
    calibrator = CALIBRATORS[name](labels, probs)
    assert monotonic_invariance_check(labels, probs, calibrator)


def test_calibration_improves_proper_scores(scores):
    """Fitting on the same data must not make Brier or log loss worse."""
    labels, probs = scores
    for name in ("temperature", "platt", "beta"):
        out = CALIBRATORS[name](labels, probs)(probs)
        assert brier(labels, out) <= brier(labels, probs) + 1e-9
        assert log_loss(labels, out) <= log_loss(labels, probs) + 1e-9


def test_perfectly_calibrated_scores_have_near_zero_ece():
    """ECE must be small when probabilities match observed frequencies."""
    rng = np.random.default_rng(1)
    probs = rng.uniform(0.02, 0.98, 60000)
    labels = rng.binomial(1, probs)
    assert ece_fixed(labels, probs) < 0.02
    assert ece_adaptive(labels, probs) < 0.02


def test_calibration_slope_is_one_when_already_calibrated():
    """A calibrated model should show slope near 1 and intercept near 0."""
    rng = np.random.default_rng(2)
    probs = rng.uniform(0.02, 0.98, 40000)
    labels = rng.binomial(1, probs)
    fit = calibration_curve_fit(labels, probs)
    assert fit["calibration_slope"] == pytest.approx(1.0, abs=0.12)
    assert fit["calibration_intercept"] == pytest.approx(0.0, abs=0.12)


def test_threshold_meets_the_sensitivity_target(scores):
    """The chosen threshold must actually deliver the requested sensitivity."""
    labels, probs = scores
    threshold = threshold_at_sensitivity(labels, probs, 0.97)
    assert (probs[labels == 1] >= threshold).mean() >= 0.97


def test_threshold_is_the_highest_one_that_qualifies(scores):
    """Any higher observed score would break the sensitivity constraint."""
    labels, probs = scores
    threshold = threshold_at_sensitivity(labels, probs, 0.97)
    higher = [c for c in np.unique(probs) if c > threshold]
    if higher:
        assert (probs[labels == 1] >= min(higher)).mean() < 0.97


def test_unreachable_sensitivity_target_returns_floor():
    """Asking for more sensitivity than exists must not raise."""
    labels = np.array([0, 1, 1])
    probs = np.array([0.9, 0.1, 0.2])
    assert threshold_at_sensitivity(labels, probs, 1.01) == 0.0
