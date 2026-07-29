"""Tests for label-shift estimation and prior correction."""

import numpy as np
import pytest

from src.evaluation.label_shift import (
    bbse_prior,
    class_conditional_shift,
    em_prior,
    prior_correct,
)


def _simulate(n, prior, separation=2.0, seed=0):
    """Draw calibrated scores from a two-Gaussian model at a given prior.

    Args:
        n: Sample size.
        prior: Positive-class rate.
        separation: Distance between class means in latent space.
        seed: Random seed.

    Returns:
        Tuple of (labels, calibrated probabilities).
    """
    rng = np.random.default_rng(seed)
    labels = rng.binomial(1, prior, n)
    latent = rng.normal(labels * separation, 1.0)
    # Posterior under equal-variance Gaussians with this prior.
    odds = (prior / (1 - prior)) * np.exp(separation * latent - separation**2 / 2)
    return labels, odds / (1 + odds)


def test_prior_correction_is_an_identity_when_priors_match():
    """Correcting to the prior the scores already carry changes nothing."""
    _, probs = _simulate(500, 0.5)
    assert np.allclose(prior_correct(probs, 0.5, 0.5), probs, atol=1e-9)


def test_prior_correction_moves_the_mean_score_toward_the_target():
    """Lowering the prior must lower predicted probabilities."""
    _, probs = _simulate(2000, 0.75)
    lowered = prior_correct(probs, 0.75, 0.4)
    assert lowered.mean() < probs.mean()
    assert (lowered <= probs + 1e-12).all()


def test_prior_correction_is_monotone():
    """Reweighting must not reorder cases."""
    _, probs = _simulate(500, 0.7)
    corrected = prior_correct(probs, 0.7, 0.3)
    order = np.argsort(probs, kind="stable")
    assert np.array_equal(order, np.argsort(corrected, kind="stable"))


def test_em_recovers_a_known_target_prior():
    """Under true label shift, EM should find the target prior."""
    source_prior, target_prior = 0.75, 0.40
    _, target_probs = _simulate(20000, target_prior, seed=1)
    # Express target scores in source-prior terms, which is what a model
    # trained at the source prior would emit.
    as_source = prior_correct(target_probs, target_prior, source_prior)
    result = em_prior(as_source, source_prior)
    assert result["em_converged"]
    assert result["em_target_prior"] == pytest.approx(target_prior, abs=0.03)


def test_em_returns_the_source_prior_when_nothing_shifted():
    """No shift means no correction."""
    source_prior = 0.62
    _, probs = _simulate(20000, source_prior, seed=2)
    result = em_prior(probs, source_prior)
    assert result["em_target_prior"] == pytest.approx(source_prior, abs=0.03)


def test_bbse_recovers_a_known_target_prior():
    """BBSE should agree with EM on simulated label shift."""
    source_prior, target_prior = 0.75, 0.40
    source_labels, source_probs = _simulate(20000, source_prior, seed=3)
    target_labels, target_probs = _simulate(20000, target_prior, seed=4)
    as_source = prior_correct(target_probs, target_prior, source_prior)
    result = bbse_prior(source_labels, source_probs, as_source, 0.5)
    assert result["bbse_solvable"]
    assert result["bbse_target_prior"] == pytest.approx(target_prior, abs=0.05)


def test_bbse_flags_a_degenerate_confusion_matrix():
    """A predictor with no signal must be reported as unsolvable, not guessed."""
    labels = np.array([0, 1] * 50)
    probs = np.full(100, 0.9)
    result = bbse_prior(labels, probs, probs, 0.5)
    assert result["bbse_solvable"] is False


def test_class_conditional_shift_is_small_under_pure_label_shift():
    """Only the mix changed, so within-class distances should be tiny."""
    source_labels, source_probs = _simulate(8000, 0.75, seed=5)
    target_labels, target_probs = _simulate(8000, 0.40, seed=6)
    as_source = prior_correct(target_probs, 0.40, 0.75)
    out = class_conditional_shift(source_labels, source_probs, target_labels, as_source)
    assert out["ks_normal"] < 0.1
    assert out["ks_pneumonia"] < 0.1


def test_class_conditional_shift_detects_a_genuine_covariate_change():
    """When the classes themselves separate differently, the test must fire."""
    source_labels, source_probs = _simulate(8000, 0.5, separation=2.0, seed=7)
    target_labels, target_probs = _simulate(8000, 0.5, separation=0.6, seed=8)
    out = class_conditional_shift(
        source_labels, source_probs, target_labels, target_probs
    )
    assert out["ks_pneumonia"] > 0.2
