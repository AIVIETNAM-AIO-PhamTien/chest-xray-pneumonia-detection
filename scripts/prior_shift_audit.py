"""Phase 2A.0 — how much of the benchmark shift is just a change in prevalence.

The benchmark's calibration intercept sits far below the development set's.
Part of that is expected: the two splits do not contain the same proportion of
pneumonia, and a model trained at one prevalence emits scores tuned to it.
This asks how much of the observed gap that accounts for, and whether prior
correction is even the right instrument.

Prior correction only works under label shift, meaning the class-conditional
score distributions are unchanged and only their mixture moved. That is an
assumption, not a given, so it is tested before any correction is reported.

Any correction using the true benchmark prevalence is an oracle: it consumes
target labels and cannot be deployed. It is reported as a ceiling only.

    python3 scripts/prior_shift_audit.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.calibration import (  # noqa: E402
    brier, calibration_curve_fit, ece_adaptive, log_loss,
    threshold_at_sensitivity)
from src.evaluation.label_shift import (  # noqa: E402
    bbse_prior, class_conditional_shift, em_prior, prior_correct)

RESULTS = Path("notebooks/results_v4")
REPORTS = Path("reports")
CONFIGS = ["resnet18", "stretch_nhe", "augment_manh", "stretch_manh"]
LABEL = {"resnet18": "letterbox+nhẹ", "stretch_nhe": "stretch+nhẹ",
         "augment_manh": "letterbox+mạnh", "stretch_manh": "stretch+mạnh"}
TARGET_SENSITIVITY = 0.97


def to_group(group_ids, labels, probs):
    """Collapse image scores to one score per filename-derived group.

    Args:
        group_ids: Group key per image.
        labels: Binary label per image.
        probs: Probability per image.

    Returns:
        Tuple of (labels, probabilities) ordered by group id.
    """
    frame = pd.DataFrame({"g": group_ids, "y": labels, "p": probs})
    rolled = frame.groupby("g").agg(y=("y", "first"), p=("p", "mean")).sort_index()
    return rolled["y"].to_numpy(), rolled["p"].to_numpy()


def operating_metrics(labels, probs, threshold):
    """Metrics that a prior shift could plausibly move.

    Args:
        labels: Binary labels.
        probs: Probabilities.
        threshold: Decision threshold.

    Returns:
        Mapping of metric name to value.
    """
    preds = (probs >= threshold).astype(int)
    negative, positive = labels == 0, labels == 1
    return {
        "threshold": float(threshold),
        "specificity": float((preds[negative] == 0).mean()),
        "sensitivity": float((preds[positive] == 1).mean()),
        "brier": brier(labels, probs),
        "log_loss": log_loss(labels, probs),
        "ece_adaptive": ece_adaptive(labels, probs),
        **calibration_curve_fit(labels, probs),
    }


def load(name):
    """Read out-of-fold and ensemble benchmark predictions for one config.

    Args:
        name: Experiment name.

    Returns:
        Tuple of (oof frame, benchmark frame).
    """
    oof = pd.concat(
        [pd.read_csv(RESULTS / f"validation_predictions_{name}_fold{k}.csv",
                     usecols=["group_id", "class_id", "p_pneumonia"])
         for k in range(5)], ignore_index=True)
    bench = pd.read_csv(RESULTS / f"predictions_known_benchmark_{name}_groups.csv")
    return oof, bench


def main():
    manifest = pd.read_csv(RESULTS / "manifest_fold0.csv")
    development = manifest[manifest["split_original"].isin(["train", "val"])]
    benchmark = manifest[manifest["split_original"] == "test"]

    prevalence = {}
    for level, dev, ben in (
            ("image", development["class_id"], benchmark["class_id"]),
            ("filename_group",
             development.groupby("group_id")["class_id"].first(),
             benchmark.groupby("group_id")["class_id"].first())):
        prevalence[level] = {
            "source_prior": float((dev == 1).mean()), "source_n": int(len(dev)),
            "target_prior": float((ben == 1).mean()), "target_n": int(len(ben))}

    print("PREVALENCE\n")
    print(f"{'mức':<16} {'nguồn':>18} {'đích':>18} {'Δ log-odds':>12}")
    print("-" * 68)
    for level, values in prevalence.items():
        source, target = values["source_prior"], values["target_prior"]
        delta = np.log(target / (1 - target)) - np.log(source / (1 - source))
        values["delta_prior_logodds"] = float(delta)
        print(f"{level:<16} {source:>9.4f} (n={values['source_n']:>5}) "
              f"{target:>9.4f} (n={values['target_n']:>4}) {delta:>12.3f}")

    prior_rows, assumption_rows = [], []
    source_prior = prevalence["filename_group"]["source_prior"]
    true_target = prevalence["filename_group"]["target_prior"]

    for name in CONFIGS:
        oof, bench = load(name)
        y_oof, p_oof = to_group(oof["group_id"], oof["class_id"], oof["p_pneumonia"])
        y_ben = bench["label"].to_numpy()
        p_ben = bench["p_pneumonia"].to_numpy()
        locked = threshold_at_sensitivity(y_oof, p_oof, TARGET_SENSITIVITY)

        assumption_rows.append({
            "experiment": name,
            **class_conditional_shift(y_oof, p_oof, y_ben, p_ben)})

        estimates = {"none": source_prior,
                     "oracle_true_target": true_target,
                     **{"em": em_prior(p_ben, source_prior)["em_target_prior"]},
                     **{"bbse": bbse_prior(y_oof, p_oof, p_ben,
                                           locked)["bbse_target_prior"]}}
        for method, estimate in estimates.items():
            if not np.isfinite(estimate):
                continue
            corrected = (p_ben if method == "none"
                         else prior_correct(p_ben, source_prior, estimate))
            block = operating_metrics(y_ben, corrected, locked)
            prior_rows.append({"experiment": name, "correction": method,
                               "estimated_target_prior": float(estimate),
                               "true_target_prior": true_target,
                               "prior_error": float(estimate - true_target),
                               "oracle_only": method == "oracle_true_target",
                               **block})

    priors = pd.DataFrame(prior_rows)
    assumptions = pd.DataFrame(assumption_rows)
    priors.to_csv(RESULTS / "results_prior_shift.csv", index=False)
    assumptions.to_csv(RESULTS / "results_label_shift_assumptions.csv", index=False)

    print("\n\nƯỚC LƯỢNG PREVALENCE ĐÍCH (mức group; thật = "
          f"{true_target:.4f})\n")
    print(f"{'cấu hình':<16} {'EM':>10} {'BBSE':>10} {'sai số EM':>11} {'sai số BBSE':>13}")
    print("-" * 64)
    for name in CONFIGS:
        rows = priors[priors["experiment"] == name].set_index("correction")
        em = rows.loc["em", "estimated_target_prior"] if "em" in rows.index else np.nan
        bb = rows.loc["bbse", "estimated_target_prior"] if "bbse" in rows.index else np.nan
        print(f"{LABEL[name]:<16} {em:>10.4f} {bb:>10.4f} "
              f"{em - true_target:>+11.4f} {bb - true_target:>+13.4f}")

    print("\n\nGIẢ ĐỊNH LABEL SHIFT: p(score|class) có ổn định không?")
    print("(khoảng cách tính trên log-odds; nhỏ = giả định đứng vững)\n")
    print(f"{'cấu hình':<16} {'KS NORMAL':>10} {'KS PNEU':>9} "
          f"{'W1 NORMAL':>11} {'W1 PNEU':>9}")
    print("-" * 60)
    for _, row in assumptions.iterrows():
        print(f"{LABEL[row['experiment']]:<16} {row['ks_normal']:>10.3f} "
              f"{row['ks_pneumonia']:>9.3f} {row['wasserstein_logit_normal']:>11.2f} "
              f"{row['wasserstein_logit_pneumonia']:>9.2f}")

    print("\n\nHIỆU CHỈNH PRIOR THAY ĐỔI ĐƯỢC GÌ (ngưỡng khóa từ OOF)\n")
    print(f"{'cấu hình':<16} {'hiệu chỉnh':<20} {'đặc hiệu':>9} {'độ nhạy':>8} "
          f"{'Brier':>8} {'intercept':>10}")
    print("-" * 78)
    for name in CONFIGS:
        for _, row in priors[priors["experiment"] == name].iterrows():
            flag = " (oracle)" if row["oracle_only"] else ""
            print(f"{LABEL[name]:<16} {row['correction'] + flag:<20} "
                  f"{row['specificity']:>9.3f} {row['sensitivity']:>8.3f} "
                  f"{row['brier']:>8.4f} {row['calibration_intercept']:>10.3f}")

    REPORTS.mkdir(exist_ok=True)
    print("\n→ results_prior_shift.csv")
    print("→ results_label_shift_assumptions.csv")


if __name__ == "__main__":
    main()
