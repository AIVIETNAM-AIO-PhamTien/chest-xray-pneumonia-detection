"""Calibration audit on frozen predictions; no network is retrained.

Splits the specificity failure into three parts that need different remedies:
discrimination lost under shift, an operating point that does not transfer, and
probabilities that do not mean what they say.

Two placements are compared because they are not interchangeable. Calibrating
after the folds are averaged cannot change any decision once the threshold is
reselected, since the map is monotone in the score. Calibrating each fold
before averaging can, because the mean of calibrated scores is not the
calibrated mean. Only the second is a candidate for recovering specificity;
the first is included as the control that proves the machinery is sound.

Run from the directory holding the frozen prediction files:
    python3 scripts/calibration_audit.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.calibration import (  # noqa: E402
    CALIBRATORS,
    PRIMARY_CALIBRATORS,
    brier,
    calibration_curve_fit,
    ece_adaptive,
    ece_fixed,
    log_loss,
    monotonic_invariance_check,
    threshold_at_sensitivity,
)

RESULTS = Path("notebooks/results_v4")
FIGURES = RESULTS / "figures"
CONFIGS = ["resnet18", "stretch_nhe", "augment_manh", "stretch_manh"]
LABEL = {"resnet18": "letterbox+nhẹ", "stretch_nhe": "stretch+nhẹ",
         "augment_manh": "letterbox+mạnh", "stretch_manh": "stretch+mạnh"}
TARGET_SENSITIVITY = 0.97
BOOTSTRAP_REPS = 2000
SEED = 42


def to_group(group_ids, labels, probs):
    """Collapse image scores to one score per filename-derived group.

    Args:
        group_ids: Group key per image.
        labels: Binary label per image.
        probs: Probability per image.

    Returns:
        Tuple of (group ids, group labels, group probabilities), sorted by id.
    """
    frame = pd.DataFrame({"g": group_ids, "y": labels, "p": probs})
    rolled = frame.groupby("g").agg(y=("y", "first"), p=("p", "mean")).sort_index()
    return rolled.index.to_numpy(), rolled["y"].to_numpy(), rolled["p"].to_numpy()


def score_block(labels, probs, threshold):
    """Every reported metric for one score vector at one threshold.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities.
        threshold: Decision threshold.

    Returns:
        Mapping of metric name to value.
    """
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
        "brier": brier(labels, probs),
        "log_loss": log_loss(labels, probs),
        "ece_fixed": ece_fixed(labels, probs),
        "ece_adaptive": ece_adaptive(labels, probs),
        **calibration_curve_fit(labels, probs),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def load_predictions():
    """Read frozen out-of-fold and per-fold benchmark predictions.

    Returns:
        Tuple of (out-of-fold frame, benchmark per-fold frame).
    """
    oof = []
    for name in CONFIGS:
        for fold in range(5):
            path = RESULTS / f"validation_predictions_{name}_fold{fold}.csv"
            frame = pd.read_csv(path, usecols=["group_id", "class_id", "p_pneumonia"])
            oof.append(frame.assign(experiment=name, fold=fold))
    benchmark = pd.read_csv(RESULTS / "predictions_benchmark_per_fold.csv")
    return pd.concat(oof, ignore_index=True), benchmark


def cross_fitted_oof(frame, method):
    """Calibrate each fold using calibrators fitted on the other folds.

    Fitting and evaluating a calibrator on the same scores flatters it, and
    the threshold chosen from such scores is optimistic. Holding out the fold
    being transformed removes both.

    Args:
        frame: Out-of-fold predictions for one configuration.
        method: Key into CALIBRATORS.

    Returns:
        The frame with a ``p_cal`` column added.
    """
    out = frame.copy()
    out["p_cal"] = np.nan
    for fold in sorted(out["fold"].unique()):
        held_out = out["fold"] == fold
        fitted = CALIBRATORS[method](out.loc[~held_out, "class_id"].to_numpy(),
                                     out.loc[~held_out, "p_pneumonia"].to_numpy())
        out.loc[held_out, "p_cal"] = fitted(out.loc[held_out, "p_pneumonia"].to_numpy())
    return out


def benchmark_post_ensemble(oof, bench, method):
    """Average the folds, then calibrate once with a pooled calibrator.

    Args:
        oof: Out-of-fold predictions for one configuration.
        bench: Per-fold benchmark predictions for one configuration.
        method: Key into CALIBRATORS.

    Returns:
        Tuple of (group ids, group labels, group probabilities).
    """
    fitted = CALIBRATORS[method](oof["class_id"].to_numpy(),
                                 oof["p_pneumonia"].to_numpy())
    ensemble = (bench.groupby(["group_id", "class_id"], as_index=False)
                .agg(p=("p_pneumonia", "mean")))
    return to_group(ensemble["group_id"], ensemble["class_id"],
                    fitted(ensemble["p"].to_numpy()))


def benchmark_pre_ensemble(oof, bench, method):
    """Calibrate each fold with its own calibrator, then average.

    Each fold's calibrator is fitted on that fold's held-out validation
    scores, which the fold's model never trained on. The benchmark is never
    involved in fitting.

    Args:
        oof: Out-of-fold predictions for one configuration.
        bench: Per-fold benchmark predictions for one configuration.
        method: Key into CALIBRATORS.

    Returns:
        Tuple of (group ids, group labels, group probabilities).
    """
    parts = []
    for fold in sorted(bench["fold"].unique()):
        source = oof[oof["fold"] == fold]
        fitted = CALIBRATORS[method](source["class_id"].to_numpy(),
                                     source["p_pneumonia"].to_numpy())
        rows = bench[bench["fold"] == fold]
        parts.append(rows.assign(p_cal=fitted(rows["p_pneumonia"].to_numpy())))
    stacked = pd.concat(parts, ignore_index=True)
    averaged = (stacked.groupby(["group_id", "class_id"], as_index=False)
                .agg(p=("p_cal", "mean")))
    return to_group(averaged["group_id"], averaged["class_id"],
                    averaged["p"].to_numpy())


def paired_bootstrap(labels, baseline, variant, threshold_a, threshold_b,
                     reps=BOOTSTRAP_REPS, seed=SEED):
    """Confidence intervals for paired metric differences on the same groups.

    Args:
        labels: Binary group labels.
        baseline: Baseline group probabilities.
        variant: Calibrated group probabilities.
        threshold_a: Threshold applied to the baseline.
        threshold_b: Threshold applied to the variant.
        reps: Bootstrap replicates.
        seed: Random seed.

    Returns:
        Mapping of metric name to (low, high) percentile bounds.
    """
    rng = np.random.default_rng(seed)
    negative, positive = np.where(labels == 0)[0], np.where(labels == 1)[0]
    gathered = {"specificity": [], "sensitivity": [], "brier": [], "log_loss": []}
    for _ in range(reps):
        index = np.concatenate([rng.choice(negative, len(negative), True),
                                rng.choice(positive, len(positive), True)])
        y = labels[index]
        a, b = baseline[index], variant[index]
        gathered["specificity"].append(
            (b[y == 0] < threshold_b).mean() - (a[y == 0] < threshold_a).mean())
        gathered["sensitivity"].append(
            (b[y == 1] >= threshold_b).mean() - (a[y == 1] >= threshold_a).mean())
        gathered["brier"].append(brier(y, b) - brier(y, a))
        gathered["log_loss"].append(log_loss(y, b) - log_loss(y, a))
    return {k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in gathered.items()}


def oracle_specificity(labels, probs, target=TARGET_SENSITIVITY):
    """Best specificity reachable on these scores at the sensitivity target.

    A diagnostic ceiling only: it is read off the benchmark labels and can
    never be used to pick an operating point.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities.
        target: Sensitivity constraint.

    Returns:
        The ceiling specificity.
    """
    threshold = threshold_at_sensitivity(labels, probs, target)
    return float((probs[labels == 0] < threshold).mean())


def main():
    oof_all, bench_all = load_predictions()
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("KIỂM TRA BẤT BIẾN: calibrator đơn điệu sau ensemble + chọn lại ngưỡng")
    print("phải KHÔNG đổi quyết định nào. Nếu fail, mọi số post-ensemble đều sai.\n")
    for name in CONFIGS:
        oof = oof_all[oof_all["experiment"] == name]
        _, y, p = to_group(oof["group_id"], oof["class_id"], oof["p_pneumonia"])
        flags = []
        for method in ("temperature", "intercept", "platt", "beta"):
            fitted = CALIBRATORS[method](oof["class_id"].to_numpy(),
                                         oof["p_pneumonia"].to_numpy())
            flags.append(monotonic_invariance_check(y, p, fitted, TARGET_SENSITIVITY))
        print(f"  {LABEL[name]:<16} {'ĐẠT' if all(flags) else 'HỎNG'}")

    metric_rows, operating_rows, decomposition_rows, comparison_rows = [], [], [], []

    for name in CONFIGS:
        oof = oof_all[oof_all["experiment"] == name].reset_index(drop=True)
        bench = bench_all[bench_all["experiment"] == name].reset_index(drop=True)

        raw_oof = cross_fitted_oof(oof, "raw")
        _, y_oof, p_oof_raw = to_group(raw_oof["group_id"], raw_oof["class_id"],
                                       raw_oof["p_pneumonia"])
        locked = threshold_at_sensitivity(y_oof, p_oof_raw, TARGET_SENSITIVITY)

        _, y_bench, p_bench_raw = benchmark_post_ensemble(oof, bench, "raw")
        raw_block = score_block(y_bench, p_bench_raw, locked)
        ceiling = oracle_specificity(y_bench, p_bench_raw)

        for method in CALIBRATORS:
            cal_oof = cross_fitted_oof(oof, method)
            _, _, p_oof_cal = to_group(cal_oof["group_id"], cal_oof["class_id"],
                                       cal_oof["p_cal"])
            oof_block = score_block(y_oof, p_oof_cal,
                                    threshold_at_sensitivity(y_oof, p_oof_cal,
                                                             TARGET_SENSITIVITY))
            metric_rows.append({"experiment": name, "calibrator": method,
                                "split": "oof_cross_fitted", "position": "n/a",
                                **oof_block})

            for position, builder in (("post_ensemble", benchmark_post_ensemble),
                                      ("pre_ensemble", benchmark_pre_ensemble)):
                _, y_b, p_b = builder(oof, bench, method)
                selected = threshold_at_sensitivity(y_oof, p_oof_cal, TARGET_SENSITIVITY)
                metric_rows.append({"experiment": name, "calibrator": method,
                                    "split": "known_benchmark", "position": position,
                                    **score_block(y_b, p_b, selected)})

                points = {"A_raw_oof_threshold": locked,
                          "B_calibrated_oof_threshold": selected,
                          "C_calibrated_half": 0.5}
                for point, threshold in points.items():
                    block = score_block(y_b, p_b, threshold)
                    operating_rows.append({"experiment": name, "calibrator": method,
                                           "position": position,
                                           "operating_point": point, **block})
                    if point == "B_calibrated_oof_threshold":
                        gap = ceiling - raw_block["specificity"]
                        recovered = block["specificity"] - raw_block["specificity"]
                        ci = paired_bootstrap(y_bench, p_bench_raw, p_b, locked, threshold)
                        decomposition_rows.append({
                            "experiment": name, "calibrator": method,
                            "position": position,
                            "oracle_specificity": ceiling,
                            "raw_locked_specificity": raw_block["specificity"],
                            "calibrated_specificity": block["specificity"],
                            "threshold_recovery": recovered,
                            "remaining_oracle_gap": ceiling - block["specificity"],
                            "recovery_ratio": recovered / gap if abs(gap) > 1e-12 else np.nan,
                            "sensitivity": block["sensitivity"],
                            "spec_ci_low": ci["specificity"][0],
                            "spec_ci_high": ci["specificity"][1],
                            "sens_ci_low": ci["sensitivity"][0],
                            "sens_ci_high": ci["sensitivity"][1],
                            "brier_ci_low": ci["brier"][0],
                            "brier_ci_high": ci["brier"][1]})

        for method in PRIMARY_CALIBRATORS:
            _, _, post = benchmark_post_ensemble(oof, bench, method)
            _, _, pre = benchmark_pre_ensemble(oof, bench, method)
            cal_oof = cross_fitted_oof(oof, method)
            _, _, p_oof_cal = to_group(cal_oof["group_id"], cal_oof["class_id"],
                                       cal_oof["p_cal"])
            selected = threshold_at_sensitivity(y_oof, p_oof_cal, TARGET_SENSITIVITY)
            comparison_rows.append({
                "experiment": name, "calibrator": method,
                "post_specificity": (post[y_bench == 0] < selected).mean(),
                "pre_specificity": (pre[y_bench == 0] < selected).mean(),
                "post_auc": roc_auc_score(y_bench, post),
                "pre_auc": roc_auc_score(y_bench, pre),
                "post_brier": brier(y_bench, post),
                "pre_brier": brier(y_bench, pre),
                "rank_correlation": pd.Series(post).corr(pd.Series(pre), method="spearman")})

    for rows, filename in ((metric_rows, "results_calibration_metrics.csv"),
                           (operating_rows, "results_calibration_operating_points.csv"),
                           (decomposition_rows, "results_calibration_gap_decomposition.csv"),
                           (comparison_rows, "results_pre_vs_post_ensemble_calibration.csv")):
        pd.DataFrame(rows).to_csv(RESULTS / filename, index=False)
        print(f"→ {filename}")


if __name__ == "__main__":
    main()
