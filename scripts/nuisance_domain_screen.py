"""Phase 2A.1 — which acquisition features move between the two sets of normals.

The score shift is concentrated in the negative class: benchmark normals score
very differently from development normals, while pneumonia cases barely move.
This asks whether that is visible in the raw images, using descriptors that
know nothing about pathology.

The primary contrast is development normals against benchmark normals. The same
contrast inside pneumonia is the negative control: a feature that moves equally
in both classes is a global acquisition difference, not an explanation for a
failure confined to normals.

The old class contrast, normal against pneumonia, is kept as a second control.
It is what the aspect-ratio work already answered, so it is reported to place
each feature rather than to lead anywhere.

    python3 scripts/nuisance_domain_screen.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.nuisance_features import ALL_FEATURES, FAMILIES  # noqa: E402

RESULTS = Path("notebooks/results_v4")
CONFIGS = ["resnet18", "stretch_nhe", "augment_manh", "stretch_manh"]
LABEL = {"resnet18": "letterbox+nhẹ", "stretch_nhe": "stretch+nhẹ",
         "augment_manh": "letterbox+mạnh", "stretch_manh": "stretch+mạnh"}
BOOTSTRAP_REPS = 2000
SEED = 42
#: Pre-registered screening gate. A feature advances to matching only if the
#: domain contrast inside normals clears one of these.
GATE_SMD, GATE_AUC = 0.5, 0.70


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, order preserved.

    Args:
        pvalues: Raw p-values.

    Returns:
        Adjusted p-values in the input order.
    """
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    m = len(values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def benjamini_hochberg(pvalues):
    """Benjamini-Hochberg adjusted p-values, order preserved.

    Args:
        pvalues: Raw p-values.

    Returns:
        Adjusted p-values in the input order.
    """
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    m = len(values)
    adjusted = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        index = order[rank]
        running = min(running, m * values[index] / (rank + 1))
        adjusted[index] = running
    return adjusted


def smd(a, b):
    """Standardized mean difference between two samples.

    Args:
        a: First sample.
        b: Second sample.

    Returns:
        Difference in means over the pooled standard deviation.
    """
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 1e-12 else 0.0


def contrast(a, b, reps=BOOTSTRAP_REPS, seed=SEED):
    """Every distance measure for one two-sample comparison.

    Args:
        a: Values from the first group.
        b: Values from the second group.
        reps: Bootstrap replicates for the SMD interval.
        seed: Random seed.

    Returns:
        Mapping of statistic name to value.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    statistic, pvalue = ks_2samp(a, b)
    labels = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    values = np.concatenate([a, b])
    auc = roc_auc_score(labels, values)

    rng = np.random.default_rng(seed)
    draws = [smd(rng.choice(a, len(a), True), rng.choice(b, len(b), True))
             for _ in range(reps)]
    low, high = np.percentile(draws, [2.5, 97.5])

    spread = np.std(values, ddof=1)
    return {"smd": smd(b, a), "smd_ci_low": float(min(-low, -high)),
            "smd_ci_high": float(max(-low, -high)),
            "ks": float(statistic), "ks_p": float(pvalue),
            "auc": float(max(auc, 1 - auc)),
            "wasserstein_standardised": float(
                wasserstein_distance(a, b) / spread) if spread > 1e-12 else 0.0,
            "n_a": int(len(a)), "n_b": int(len(b))}


def group_level(features):
    """Reduce image features to one row per filename-derived group.

    Groups are the inference unit, and a group with many images would
    otherwise be counted many times.

    Args:
        features: Image-level feature table.

    Returns:
        Group-level table with the median of each feature.
    """
    keys = ["group_id", "class_id", "split_original"]
    aggregated = (features.groupby(keys, as_index=False)[ALL_FEATURES]
                  .median())
    return aggregated


def main():
    features = pd.read_csv(RESULTS / "nuisance_feature_manifest.csv")
    grouped = group_level(features)
    grouped["domain"] = np.where(grouped["split_original"] == "test",
                                 "benchmark", "development")

    development = grouped[grouped["domain"] == "development"]
    benchmark = grouped[grouped["domain"] == "benchmark"]
    dev_normal = development[development["class_id"] == 0]
    ben_normal = benchmark[benchmark["class_id"] == 0]
    dev_pneu = development[development["class_id"] == 1]
    ben_pneu = benchmark[benchmark["class_id"] == 1]

    print(f"group: development {len(development):,} "
          f"({len(dev_normal):,} NORMAL) | benchmark {len(benchmark):,} "
          f"({len(ben_normal):,} NORMAL)\n")

    rows = []
    for feature in ALL_FEATURES:
        family = next(k for k, v in FAMILIES.items() if feature in v)
        primary = contrast(dev_normal[feature], ben_normal[feature])
        control = contrast(dev_pneu[feature], ben_pneu[feature])
        class_dev = contrast(development[development["class_id"] == 0][feature],
                             development[development["class_id"] == 1][feature])
        class_ben = contrast(benchmark[benchmark["class_id"] == 0][feature],
                             benchmark[benchmark["class_id"] == 1][feature])
        rows.append({
            "feature": feature, "family": family,
            **{f"normal_domain_{k}": v for k, v in primary.items()},
            **{f"pneumonia_domain_{k}": v for k, v in control.items()},
            "class_auc_development": class_dev["auc"],
            "class_auc_benchmark": class_ben["auc"],
            "differential_smd": abs(primary["smd"]) - abs(control["smd"]),
            "differential_auc": primary["auc"] - control["auc"]})

    frame = pd.DataFrame(rows)
    frame["normal_domain_ks_p_holm"] = holm(frame["normal_domain_ks_p"])
    frame["normal_domain_ks_p_bh"] = benjamini_hochberg(frame["normal_domain_ks_p"])
    frame["pneumonia_domain_ks_p_bh"] = benjamini_hochberg(
        frame["pneumonia_domain_ks_p"])
    frame["passes_gate"] = (
        ((frame["normal_domain_smd"].abs() >= GATE_SMD)
         | (frame["normal_domain_auc"] >= GATE_AUC))
        & (frame["normal_domain_ks_p_holm"] < 0.05))

    frame = frame.sort_values("normal_domain_auc", ascending=False)
    frame.to_csv(RESULTS / "results_normal_domain_shift.csv", index=False)

    print("DỊCH CHUYỂN NORMAL nguồn → đích, so với PNEUMONIA làm đối chứng âm")
    print("(sắp theo domain AUC trong NORMAL; * = qua cổng sàng lọc)\n")
    print(f"{'feature':<26} {'họ':<11} {'NORMAL':>16} {'PNEUMONIA':>16} "
          f"{'chênh':>7} {'lớp dev':>8}")
    print(f"{'':<26} {'':<11} {'SMD    AUC':>16} {'SMD    AUC':>16} "
          f"{'AUC':>7} {'AUC':>8}")
    print("-" * 92)
    for _, row in frame.head(18).iterrows():
        mark = "*" if row["passes_gate"] else " "
        print(f"{mark}{row['feature']:<25} {row['family']:<11} "
              f"{row['normal_domain_smd']:>+7.2f} {row['normal_domain_auc']:>7.3f} "
              f"{row['pneumonia_domain_smd']:>+7.2f} {row['pneumonia_domain_auc']:>7.3f} "
              f"{row['differential_auc']:>+7.3f} {row['class_auc_development']:>8.3f}")

    passed = frame[frame["passes_gate"]]
    print(f"\n{len(passed)}/{len(frame)} feature qua cổng "
          f"(|SMD| >= {GATE_SMD} hoặc AUC >= {GATE_AUC}, kèm Holm p < 0.05)")

    print("\nỨng viên mạnh nhất mỗi họ:")
    for family in FAMILIES:
        block = passed[passed["family"] == family]
        if len(block):
            best = block.iloc[0]
            print(f"  {family:<12} {best['feature']:<26} "
                  f"AUC {best['normal_domain_auc']:.3f}  "
                  f"chênh so PNEU {best['differential_auc']:+.3f}")
        else:
            print(f"  {family:<12} không có feature nào qua cổng")

    print("\n→ results_normal_domain_shift.csv")

    # --- Tầng 2: liên hệ với score và lỗi trong riêng benchmark NORMAL -------
    error_rows = []
    lookup = grouped.set_index("group_id")
    for name in CONFIGS:
        predictions = pd.read_csv(
            RESULTS / f"predictions_known_benchmark_{name}_groups.csv")
        predictions = predictions[predictions["label"] == 0].copy()
        joined = predictions.join(lookup[ALL_FEATURES], on="group_id")
        logit = np.log(np.clip(joined["p_pneumonia"], 1e-6, 1 - 1e-6)
                       / (1 - np.clip(joined["p_pneumonia"], 1e-6, 1 - 1e-6)))
        false_positive = joined["pred"].to_numpy() == 1

        for feature in ALL_FEATURES:
            values = joined[feature].to_numpy()
            rho, p_rho = spearmanr(values, logit)
            fp_values, tn_values = values[false_positive], values[~false_positive]
            if len(fp_values) >= 5 and len(tn_values) >= 5:
                pairwise = fp_values[:, None] - tn_values[None, :]
                delta = float((pairwise > 0).mean() - (pairwise < 0).mean())
            else:
                delta = np.nan
            error_rows.append({
                "experiment": name, "feature": feature,
                "family": next(k for k, v in FAMILIES.items() if feature in v),
                "spearman_rho": float(rho), "spearman_p": float(p_rho),
                "cliffs_delta_fp_vs_tn": delta,
                "median_shift_fp_vs_tn": float(np.median(fp_values)
                                               - np.median(tn_values))
                if len(fp_values) and len(tn_values) else np.nan,
                "n_fp": int(false_positive.sum()),
                "n_tn": int((~false_positive).sum())})

    errors = pd.DataFrame(error_rows)
    errors["spearman_p_bh"] = np.concatenate([
        benjamini_hochberg(errors[errors["experiment"] == name]["spearman_p"])
        for name in CONFIGS])
    errors.to_csv(RESULTS / "results_nuisance_error_association.csv", index=False)

    print("\n\nLIÊN HỆ VỚI SCORE trong riêng benchmark NORMAL")
    print("(|rho| trung bình qua 4 cấu hình, chỉ feature qua cổng)\n")
    summary = (errors[errors["feature"].isin(passed["feature"])]
               .groupby("feature")
               .agg(mean_abs_rho=("spearman_rho", lambda s: s.abs().mean()),
                    mean_delta=("cliffs_delta_fp_vs_tn", "mean"),
                    n_significant=("spearman_p_bh", lambda s: (s < 0.05).sum()))
               .sort_values("mean_abs_rho", ascending=False))
    print(f"{'feature':<26} {'|rho| TB':>9} {'delta TB':>9} {'số cấu hình p<0.05':>19}")
    print("-" * 66)
    for feature, row in summary.head(12).iterrows():
        print(f"{feature:<26} {row['mean_abs_rho']:>9.3f} "
              f"{row['mean_delta']:>+9.3f} {int(row['n_significant']):>15}/4")

    print("\n→ results_nuisance_error_association.csv")


if __name__ == "__main__":
    main()
