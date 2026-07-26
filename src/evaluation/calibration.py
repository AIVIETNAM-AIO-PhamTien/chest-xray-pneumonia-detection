"""Post-hoc calibration fitted on source predictions only.

Every calibrator here is fitted on out-of-fold development predictions and
applied unchanged to the benchmark. None of them ever sees a benchmark label.

A note that decides how the results must be read. Temperature, intercept-only,
Platt and beta calibration are all strictly monotonic in the score. If such a
calibrator is applied after the folds are averaged, and the operating threshold
is then reselected on the same out-of-fold ranking under the same sensitivity
constraint, the resulting classifications cannot change: ``f(p) >= f(t)`` holds
exactly when ``p >= t``. Any reported specificity gain in that setup is a bug,
not a finding, which is why :func:`monotonic_invariance_check` exists.

Calibration can only move decisions when it is applied *before* averaging,
because the mean of calibrated scores is not the calibrated mean.
"""

from typing import Callable, Dict

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from sklearn.isotonic import IsotonicRegression

EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    """Log-odds of a probability, clipped away from the asymptotes.

    Args:
        p: Probabilities in [0, 1].

    Returns:
        Log-odds, finite everywhere.
    """
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

    Args:
        z: Real-valued scores.

    Returns:
        Probabilities in (0, 1).
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _nll(labels: np.ndarray, probs: np.ndarray) -> float:
    """Mean negative log-likelihood.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities of the positive class.

    Returns:
        Mean NLL in nats.
    """
    probs = np.clip(probs, EPS, 1.0 - EPS)
    return float(-np.mean(labels * np.log(probs)
                          + (1 - labels) * np.log(1.0 - probs)))


def fit_identity(labels: np.ndarray, probs: np.ndarray) -> Callable:
    """Return the uncalibrated scores unchanged, as the control condition.

    Args:
        labels: Unused; present so every fitter shares one signature.
        probs: Unused.

    Returns:
        The identity transform.
    """
    return lambda p: np.asarray(p, dtype=float)


def fit_temperature(labels: np.ndarray, probs: np.ndarray) -> Callable:
    """Fit a single temperature dividing the log-odds.

    Sharpens or softens the score without shifting it, so it corrects
    over-confidence but cannot correct a systematic offset.

    Args:
        labels: Binary labels.
        probs: Uncalibrated probabilities.

    Returns:
        Calibrator mapping probabilities to probabilities.
    """
    z = _logit(probs)
    result = minimize_scalar(lambda t: _nll(labels, _sigmoid(z / t)),
                             bounds=(0.02, 100.0), method="bounded")
    temperature = float(result.x)
    return lambda p: _sigmoid(_logit(p) / temperature)


def fit_intercept(labels: np.ndarray, probs: np.ndarray) -> Callable:
    """Fit an additive shift on the log-odds, holding the slope at one.

    This is calibration-in-the-large: it asks whether the model is uniformly
    too eager or too reluctant, which is the failure mode a prevalence change
    between development and deployment produces.

    Args:
        labels: Binary labels.
        probs: Uncalibrated probabilities.

    Returns:
        Calibrator mapping probabilities to probabilities.
    """
    z = _logit(probs)
    result = minimize_scalar(lambda b: _nll(labels, _sigmoid(z + b)),
                            bounds=(-25.0, 25.0), method="bounded")
    bias = float(result.x)
    return lambda p: _sigmoid(_logit(p) + bias)


def fit_platt(labels: np.ndarray, probs: np.ndarray) -> Callable:
    """Fit slope and intercept on the log-odds.

    Args:
        labels: Binary labels.
        probs: Uncalibrated probabilities.

    Returns:
        Calibrator mapping probabilities to probabilities.
    """
    z = _logit(probs)
    result = minimize(lambda w: _nll(labels, _sigmoid(w[0] * z + w[1])),
                      x0=np.array([1.0, 0.0]), method="Nelder-Mead",
                      options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10})
    slope, bias = float(result.x[0]), float(result.x[1])
    return lambda p: _sigmoid(slope * _logit(p) + bias)


def fit_beta(labels: np.ndarray, probs: np.ndarray) -> Callable:
    """Fit beta calibration, which allows the two tails to move differently.

    Platt scaling applies one slope to the whole range. Beta calibration uses
    ``log p`` and ``log(1 - p)`` separately, so it can tighten the confident
    negatives without equally tightening the confident positives -- the shape
    a class-weighted loss tends to produce.

    Args:
        labels: Binary labels.
        probs: Uncalibrated probabilities.

    Returns:
        Calibrator mapping probabilities to probabilities.
    """
    p = np.clip(np.asarray(probs, dtype=float), EPS, 1.0 - EPS)
    features = np.column_stack([np.log(p), -np.log(1.0 - p)])

    def objective(w):
        return _nll(labels, _sigmoid(features @ w[:2] + w[2]))

    result = minimize(objective, x0=np.array([1.0, 1.0, 0.0]),
                      method="Nelder-Mead",
                      options={"maxiter": 6000, "xatol": 1e-8, "fatol": 1e-10})
    a, b, c = (float(v) for v in result.x)

    def calibrate(q):
        q = np.clip(np.asarray(q, dtype=float), EPS, 1.0 - EPS)
        return _sigmoid(a * np.log(q) - b * np.log(1.0 - q) + c)

    return calibrate


def fit_isotonic(labels: np.ndarray, probs: np.ndarray) -> Callable:
    """Fit a non-parametric monotone map, as a sensitivity analysis only.

    Out-of-fold scores here are close to separable, so isotonic regression can
    interpolate the development set almost exactly. It is reported for
    comparison, not as a candidate.

    Args:
        labels: Binary labels.
        probs: Uncalibrated probabilities.

    Returns:
        Calibrator mapping probabilities to probabilities.
    """
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(np.asarray(probs, dtype=float), np.asarray(labels, dtype=float))
    return lambda p: np.clip(model.predict(np.asarray(p, dtype=float)),
                             EPS, 1.0 - EPS)


CALIBRATORS: Dict[str, Callable] = {
    "raw": fit_identity,
    "temperature": fit_temperature,
    "intercept": fit_intercept,
    "platt": fit_platt,
    "beta": fit_beta,
    "isotonic": fit_isotonic,
}

#: Isotonic is reported but excluded from the primary family; see fit_isotonic.
PRIMARY_CALIBRATORS = ("raw", "temperature", "intercept", "platt", "beta")


def brier(labels: np.ndarray, probs: np.ndarray) -> float:
    """Mean squared error between probability and outcome.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities.

    Returns:
        Brier score; lower is better.
    """
    return float(np.mean((np.asarray(probs, dtype=float)
                          - np.asarray(labels, dtype=float)) ** 2))


def log_loss(labels: np.ndarray, probs: np.ndarray) -> float:
    """Mean negative log-likelihood.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities.

    Returns:
        Log loss in nats; lower is better.
    """
    return _nll(np.asarray(labels, dtype=float), np.asarray(probs, dtype=float))


def ece_fixed(labels: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error over equal-width probability bins.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities.
        bins: Number of equal-width bins across [0, 1].

    Returns:
        Weighted mean gap between confidence and accuracy.
    """
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if mask.sum():
            total += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(total)


def ece_adaptive(labels: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error over equal-count bins.

    Equal-width bins are nearly empty in the middle when scores pile up at the
    ends, which is exactly this model's shape, so the fixed-bin number is
    dominated by two crowded bins. Equal-count bins spread the weight.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities.
        bins: Number of quantile bins.

    Returns:
        Weighted mean gap between confidence and accuracy.
    """
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(probs)
    total = 0.0
    for chunk in np.array_split(order, min(bins, len(order))):
        if len(chunk):
            total += (len(chunk) / len(probs)) * abs(
                probs[chunk].mean() - labels[chunk].mean())
    return float(total)


def calibration_curve_fit(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Regress outcomes on log-odds to get calibration intercept and slope.

    Slope below one means the scores are more extreme than the outcomes
    justify; a non-zero intercept means a systematic offset. Together they say
    *how* a model is miscalibrated, which a single ECE number does not.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities.

    Returns:
        Mapping with ``calibration_intercept`` and ``calibration_slope``.
    """
    z = _logit(probs)
    labels = np.asarray(labels, dtype=float)
    result = minimize(lambda w: _nll(labels, _sigmoid(w[0] * z + w[1])),
                      x0=np.array([1.0, 0.0]), method="Nelder-Mead",
                      options={"maxiter": 4000})
    return {"calibration_slope": float(result.x[0]),
            "calibration_intercept": float(result.x[1])}


def threshold_at_sensitivity(labels: np.ndarray, probs: np.ndarray,
                             target: float = 0.97) -> float:
    """Highest threshold still meeting a minimum sensitivity.

    Chooses among observed scores rather than a grid, so the returned value is
    always attainable and never falls inside a tie.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities.
        target: Minimum sensitivity to hold.

    Returns:
        The selected threshold, or 0.0 if the target is unreachable.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=float)
    positive = probs[labels == 1]
    if not len(positive):
        return 0.5
    feasible = [c for c in np.unique(probs) if (positive >= c).mean() >= target]
    return float(max(feasible)) if feasible else 0.0


def monotonic_invariance_check(labels: np.ndarray, probs: np.ndarray,
                               calibrator: Callable,
                               target: float = 0.97) -> bool:
    """Verify a monotonic calibrator leaves reselected decisions untouched.

    Guards the reading of every post-ensemble result: if this fails, a
    reported gain came from the code, not from calibration.

    Args:
        labels: Binary labels used to reselect the threshold.
        probs: Uncalibrated probabilities.
        calibrator: Fitted probability-to-probability map.
        target: Sensitivity constraint used for both selections.

    Returns:
        True when both threshold choices induce identical predictions.
    """
    raw_threshold = threshold_at_sensitivity(labels, probs, target)
    calibrated = calibrator(probs)
    cal_threshold = threshold_at_sensitivity(labels, calibrated, target)
    return bool(np.array_equal(probs >= raw_threshold,
                               calibrated >= cal_threshold))
