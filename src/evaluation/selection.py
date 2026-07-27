"""Checkpoint selection at a fixed sensitivity, with a finer tie-break.

Stage B showed why the earlier rule was not enough. With one to six false
positives per validation fold, group specificity moves in steps of about 0.004,
so almost every real difference lands inside the 0.005 band that was meant to
absorb noise. The rule then fell through to log-loss, which measures
probability quality rather than the ranking near the operating point.

Version 6 inserts HSAS@97 between the two: the mean specificity the model holds
across sensitivities from 97% to 100%. It reads the same part of the curve the
threshold sits on, but continuously, so it can separate epochs that a
whole-case metric cannot.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import roc_curve

#: Specificity differences below this are treated as ties.
SPECIFICITY_TIE = 0.005
#: HSAS differences below this are treated as ties.
HSAS_TIE = 0.002


def exact_threshold_at_sensitivity(labels: np.ndarray, probs: np.ndarray,
                                   target: float = 0.97) -> float:
    """Highest observed score that still meets a minimum sensitivity.

    Every distinct probability is considered rather than a fixed grid. At group
    level there are only a few thousand of them, and subsampling can miss the
    one threshold that separates two cases.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities of the positive class.
        target: Minimum sensitivity to hold.

    Returns:
        The selected threshold, or 0.0 when the target cannot be met.
    """
    labels, probs = np.asarray(labels), np.asarray(probs, dtype=float)
    positive = probs[labels == 1]
    if not len(positive):
        return 0.5
    candidates = np.unique(probs)
    feasible = candidates[[(positive >= c).mean() >= target for c in candidates]]
    return float(feasible.max()) if len(feasible) else 0.0


def specificity_at_sensitivity(labels: np.ndarray, probs: np.ndarray,
                               target: float = 0.97):
    """Best specificity reachable while holding a minimum sensitivity.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities.
        target: Minimum sensitivity to hold.

    Returns:
        Tuple of (specificity, threshold).
    """
    threshold = exact_threshold_at_sensitivity(labels, probs, target)
    negative = np.asarray(probs, dtype=float)[np.asarray(labels) == 0]
    if not len(negative):
        return 0.0, threshold
    return float((negative < threshold).mean()), threshold


def high_sensitivity_average_specificity(
        labels: np.ndarray, probs: np.ndarray,
        min_sensitivity: float = 0.97) -> float:
    """Mean specificity across the sensitivity range the model operates in.

    Reported as HSAS@97. Deliberately not called partial AUC: a reader would
    reasonably assume the McClish normalisation, and this is a different
    quantity on a different scale.

    Integrates specificity over sensitivity from ``min_sensitivity`` to 1 and
    divides by the range, so the result reads as "the specificity this model
    averages while holding sensitivity at or above the target".

    The obvious alternative, McClish-style normalisation by the region's own
    width, is wrong for this purpose: dividing by the width cancels exactly the
    quantity of interest. A model that pays far more false positives to reach
    97% sensitivity traces a wider region of the same shape and scores the
    same, which is the opposite of what checkpoint selection needs.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities.
        min_sensitivity: Lower bound of the sensitivity range.

    Returns:
        Mean specificity over the range, in [0, 1].

    Raises:
        ValueError: If either class is absent, since there is no ROC to
            restrict and a silent NaN would propagate into selection.
    """
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        raise ValueError("HSAS cần cả hai lớp; chỉ thấy "
                         f"{np.unique(labels).tolist()}")
    fpr, tpr, _ = roc_curve(labels, np.asarray(probs, dtype=float))
    # An ROC can hold several points at one sensitivity; the reachable
    # operating point is the cheapest of them. Interpolating through the
    # others would charge false positives the model never had to pay.
    order = np.lexsort((fpr, tpr))
    tpr, fpr = tpr[order], fpr[order]
    keep = np.r_[True, np.diff(tpr) > 0]
    tpr, fpr = tpr[keep], fpr[keep]

    # Integrate along sensitivity, so the result is in specificity units.
    grid = np.linspace(min_sensitivity, 1.0, 512)
    specificity = 1.0 - np.interp(grid, tpr, fpr)
    return float(np.trapezoid(specificity, grid) / (1.0 - min_sensitivity))


def better_checkpoint(candidate: Dict[str, float],
                      incumbent: Optional[Dict[str, float]],
                      specificity_tie: float = SPECIFICITY_TIE,
                      hsas_tie: float = HSAS_TIE):
    """Decide whether an epoch should replace the one currently held.

    Specificity at the sensitivity target decides first, since it is the
    endpoint. Within the tie band, HSAS@97 over the same region decides, then
    log-loss, then the earlier epoch. Reporting the reason makes the
    hierarchy auditable after the fact rather than inferred from numbers.

    Args:
        candidate: Metrics for the current epoch.
        incumbent: Metrics for the held checkpoint, or None.
        specificity_tie: Band within which specificities count as equal.
        hsas_tie: Band within which HSAS values count as equal.

    Returns:
        Tuple of (replace, reason).
    """
    if incumbent is None:
        return True, "first"

    gap = candidate["specificity"] - incumbent["specificity"]
    if gap > specificity_tie:
        return True, "specificity"
    if gap < -specificity_tie:
        return False, "specificity"

    a, b = candidate.get("hsas"), incumbent.get("hsas")
    if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
        difference = a - b
        if difference > hsas_tie:
            return True, "hsas"
        if difference < -hsas_tie:
            return False, "hsas"

    if candidate["nll"] < incumbent["nll"] - 1e-9:
        return True, "nll"
    return False, "earlier_epoch"


#: Kept so existing callers keep working; new output should say hsas_97.
normalized_partial_auc = high_sensitivity_average_specificity
