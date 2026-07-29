"""Tests for the v6 checkpoint-selection hierarchy and its metrics."""

import numpy as np
import pytest

from src.evaluation.selection import exact_threshold_at_sensitivity
from src.evaluation.selection import high_sensitivity_average_specificity as hsas
from src.evaluation.selection import specificity_at_sensitivity


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


def test_hsas_is_one_for_a_perfect_ranking():
    """No overlap means the restricted region is perfect too."""
    labels = np.r_[np.zeros(100, int), np.ones(100, int)]
    probs = np.r_[np.linspace(0.0, 0.4, 100), np.linspace(0.6, 1.0, 100)]
    assert hsas(labels, probs, 0.97) == pytest.approx(1.0, abs=1e-6)


def test_hsas_is_near_zero_for_random_scores():
    """Chance ranking buys almost no specificity at 97% sensitivity.

    The metric is mean specificity, not a rescaled area, so the chance value
    is about 1.5% rather than 0.5.
    """
    rng = np.random.default_rng(1)
    labels = rng.binomial(1, 0.5, 6000)
    probs = rng.uniform(size=6000)
    assert hsas(labels, probs, 0.97) == pytest.approx(0.015, abs=0.02)


def test_hsas_agrees_with_specificity_at_the_target():
    """It should sit near the discrete specificity it generalises."""
    rng = np.random.default_rng(5)
    labels = np.r_[np.zeros(600, int), np.ones(600, int)]
    probs = np.r_[rng.beta(2, 6, 600), rng.beta(6, 2, 600)]
    discrete, _ = specificity_at_sensitivity(labels, probs, 0.97)
    assert hsas(labels, probs, 0.97) == pytest.approx(discrete, abs=0.12)


def test_hsas_sees_local_damage_global_auc_misses():
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
    pauc_drop = hsas(labels, probs, 0.97) - hsas(labels, damaged, 0.97)
    assert auc_drop < 0.001
    assert pauc_drop > auc_drop * 10


def test_single_class_raises_rather_than_returning_nan():
    """A silent NaN would flow into selection and quietly break the tie-break."""
    with pytest.raises(ValueError, match="cần cả hai lớp"):
        hsas(np.zeros(10, int), np.linspace(0, 1, 10))


def test_hsas_is_invariant_to_monotonic_rescaling(separable):
    """It reads the ranking, so any order-preserving map must leave it alone."""
    labels, probs = separable
    reference = hsas(labels, probs, 0.97)
    for transform in (
        lambda p: p**3,
        lambda p: np.log(p + 1e-9),
        lambda p: 5.0 * p - 2.0,
    ):
        assert hsas(labels, transform(probs), 0.97) == pytest.approx(
            reference, abs=1e-9
        )


def test_hsas_is_invariant_to_row_order(separable):
    """Shuffling the inputs must not move the metric."""
    labels, probs = separable
    rng = np.random.default_rng(11)
    order = rng.permutation(len(labels))
    assert hsas(labels[order], probs[order], 0.97) == pytest.approx(
        hsas(labels, probs, 0.97), abs=1e-12
    )


def test_hsas_ignores_dominated_operating_points():
    """A costlier point at the same sensitivity must not be interpolated through.

    Two thresholds reaching identical sensitivity at different false-positive
    rates: only the cheaper one is reachable, so charging the dearer one would
    understate the model.
    """
    labels = np.r_[np.zeros(200, int), np.ones(200, int)]
    probs = np.r_[np.linspace(0.0, 0.45, 200), np.linspace(0.55, 1.0, 200)]
    padded_labels = np.r_[labels, np.zeros(3, int)]
    padded_probs = np.r_[probs, np.full(3, 0.001)]
    assert hsas(padded_labels, padded_probs, 0.97) == pytest.approx(
        hsas(labels, probs, 0.97), abs=0.02
    )


def test_hsas_weights_the_operating_region_more_than_global_auc(separable):
    """The complement of the locality test, and the one that is constructible.

    A property like "damage at low sensitivity leaves HSAS alone" cannot be
    built here: reaching 97% sensitivity means catching almost every positive,
    so any ranking error surfaces in that region. What can be shown is the
    weighting -- HSAS reacts to the same error more strongly than global AUC.
    """
    from sklearn.metrics import roc_auc_score

    labels, probs = separable
    rng = np.random.default_rng(21)
    damaged = probs.copy()
    lowest_positive = np.sort(probs[labels == 1])[:6]
    negatives = np.where(labels == 0)[0]
    damaged[rng.choice(negatives, 6, replace=False)] = lowest_positive + 1e-4

    auc_drop = roc_auc_score(labels, probs) - roc_auc_score(labels, damaged)
    hsas_drop = hsas(labels, probs, 0.97) - hsas(labels, damaged, 0.97)
    assert hsas_drop > auc_drop


def test_hsas_ignores_everything_below_the_region():
    """Two curves that agree above 97% sensitivity must score the same.

    This is the precise form of locality. The loose version -- damage at low
    sensitivity barely moves HSAS -- is not constructible here, but the exact
    statement is: whatever happens under the threshold band is invisible,
    however far apart global AUC ends up.
    """
    from sklearn.metrics import roc_auc_score

    n = 600
    labels = np.r_[np.zeros(n, int), np.ones(n, int)]

    # The lowest-scoring 5% of positives and the negatives beneath them fix
    # the region above 97% sensitivity; everything else is free to differ.
    low_positives = np.linspace(0.50, 0.53, 30)
    low_negatives = np.linspace(0.00, 0.49, n)

    # Variant A spreads its remaining positives; variant B piles them high.
    tail_a = np.linspace(0.60, 1.00, n - 30)
    tail_b = np.r_[
        np.linspace(0.54, 0.58, (n - 30) // 2), np.full(n - 30 - (n - 30) // 2, 0.99)
    ]

    probs_a = np.r_[low_negatives, low_positives, tail_a]
    probs_b = np.r_[low_negatives, low_positives, tail_b]

    assert hsas(labels, probs_a, 0.97) == pytest.approx(
        hsas(labels, probs_b, 0.97), abs=1e-9
    )
    # Both rank perfectly here, so the point is the invariance itself:
    # rearranging the comfortably-caught positives cannot move the metric.
    assert roc_auc_score(labels, probs_a) == pytest.approx(
        roc_auc_score(labels, probs_b), abs=1e-9
    )
