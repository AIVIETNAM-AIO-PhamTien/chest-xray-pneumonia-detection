"""Label-shift estimation and prior correction for binary scores.

Label shift assumes the class balance moves between source and target while
the class-conditional score distributions stay put: ``p_s(s|y) = p_t(s|y)``.
When that holds, a shift in prior is correctable from unlabelled target data
alone. When it does not, prior correction cannot repair the model and the
estimators below will quietly mislead, so :func:`class_conditional_shift`
exists to test the assumption before anything is corrected.
"""

from typing import Dict

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

EPS = 1e-9


def _logit(p: np.ndarray) -> np.ndarray:
    """Log-odds, clipped away from the asymptotes.

    Args:
        p: Probabilities.

    Returns:
        Finite log-odds.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function.

    Args:
        z: Real-valued scores.

    Returns:
        Probabilities.
    """
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def prior_correct(
    probs: np.ndarray, source_prior: float, target_prior: float
) -> np.ndarray:
    """Shift scores from a source prior to a target prior.

    Adds the difference of prior log-odds, which is the exact correction when
    the class-conditional densities are unchanged.

    Args:
        probs: Source-calibrated probabilities of the positive class.
        source_prior: Positive-class rate the scores were produced under.
        target_prior: Positive-class rate to move them to.

    Returns:
        Reweighted probabilities.
    """
    shift = _logit(np.array([target_prior]))[0] - _logit(np.array([source_prior]))[0]
    return _sigmoid(_logit(probs) + shift)


def em_prior(
    probs: np.ndarray,
    source_prior: float,
    max_iterations: int = 500,
    tolerance: float = 1e-9,
) -> Dict[str, float]:
    """Estimate the target prior by expectation maximisation, without labels.

    The Saerens-Latinne-Decaestecker procedure: reweight the source-calibrated
    scores under a candidate prior, average them to get a better prior, repeat.

    Args:
        probs: Source-calibrated probabilities on the target sample.
        source_prior: Positive-class rate of the source.
        max_iterations: Iteration cap.
        tolerance: Convergence tolerance on the prior.

    Returns:
        Mapping with the estimate, iteration count and convergence flag.
    """
    probs = np.asarray(probs, dtype=float)
    prior = float(source_prior)
    converged, step = False, 0
    for step in range(1, max_iterations + 1):
        adjusted = prior_correct(probs, source_prior, prior)
        updated = float(np.mean(adjusted))
        updated = min(max(updated, EPS), 1 - EPS)
        if abs(updated - prior) < tolerance:
            prior, converged = updated, True
            break
        prior = updated
    return {
        "em_target_prior": prior,
        "em_iterations": step,
        "em_converged": bool(converged),
    }


def bbse_prior(
    source_labels: np.ndarray,
    source_probs: np.ndarray,
    target_probs: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Estimate the target prior from hard predictions, without target labels.

    Black-box shift estimation: the source confusion matrix says how often each
    true class produces each prediction, so the observed target prediction rates
    can be inverted back to target class rates. Uses only hard decisions, which
    makes it insensitive to how well the scores are calibrated -- a useful
    cross-check on the EM estimate, which is not.

    Args:
        source_labels: Binary labels on the source.
        source_probs: Source probabilities.
        target_probs: Target probabilities.
        threshold: Decision threshold applied to both.

    Returns:
        Mapping with the estimate and whether the linear system was solvable.
    """
    source_preds = (np.asarray(source_probs) >= threshold).astype(int)
    source_labels = np.asarray(source_labels).astype(int)
    joint = np.zeros((2, 2))
    for predicted in (0, 1):
        for actual in (0, 1):
            joint[predicted, actual] = np.mean(
                (source_preds == predicted) & (source_labels == actual)
            )
    target_rates = np.array(
        [
            np.mean((np.asarray(target_probs) >= threshold).astype(int) == k)
            for k in (0, 1)
        ]
    )
    if abs(np.linalg.det(joint)) < 1e-12:
        return {"bbse_target_prior": np.nan, "bbse_solvable": False}
    weights = np.linalg.solve(joint, target_rates)
    source_rates = np.array([np.mean(source_labels == k) for k in (0, 1)])
    estimate = float(np.clip(weights[1] * source_rates[1], 0.0, 1.0))
    return {"bbse_target_prior": estimate, "bbse_solvable": True}


def class_conditional_shift(
    source_labels: np.ndarray,
    source_probs: np.ndarray,
    target_labels: np.ndarray,
    target_probs: np.ndarray,
) -> Dict[str, float]:
    """Test whether score distributions within each class survive the move.

    This decides whether prior correction is even the right tool. Under label
    shift these distances should be small; large values mean the classes
    themselves look different at the target and no reweighting will fix that.

    Args:
        source_labels: Binary labels on the source.
        source_probs: Source probabilities.
        target_labels: Binary labels on the target.
        target_probs: Target probabilities.

    Returns:
        Mapping of per-class KS statistic, p-value and Wasserstein distance,
        computed on log-odds so the crowded tails are not compressed.
    """
    out: Dict[str, float] = {}
    source_logit, target_logit = _logit(source_probs), _logit(target_probs)
    for label, name in ((0, "normal"), (1, "pneumonia")):
        a = source_logit[np.asarray(source_labels) == label]
        b = target_logit[np.asarray(target_labels) == label]
        statistic, pvalue = ks_2samp(a, b)
        out[f"ks_{name}"] = float(statistic)
        out[f"ks_p_{name}"] = float(pvalue)
        out[f"wasserstein_logit_{name}"] = float(wasserstein_distance(a, b))
        out[f"median_logit_source_{name}"] = float(np.median(a))
        out[f"median_logit_target_{name}"] = float(np.median(b))
    return out
