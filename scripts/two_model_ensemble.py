"""Combine the two frozen models without training anything.

ResNet18 and DenseNet121 disagree on sixteen normal groups: twelve the second
gets right and the first does not, four the other way. That disagreement is
what an ensemble can exploit, and it costs no GPU time to find out.

Rank averaging is computed and reported but excluded from selection, and the
reason is knowable without looking at any benchmark result. A percentile rank
is defined relative to the sample it was computed in, so a rank threshold only
transfers to a new sample whose class balance matches. Development runs at
66.8% pneumonia and the benchmark at 47.4%, which means a rank cut placed to
flag 65% of development groups flags 65% of benchmark groups too, where only
47.4% are positive. The mismatch is arithmetic, not empirical.

Probability averaging carries its own risk, that the more confident model
dominates for reasons unrelated to being more correct, but a probability
threshold at least means the same thing in both samples.

The threshold is chosen on pooled out-of-fold predictions. The benchmark sees
one locked configuration.

    python3 scripts/two_model_ensemble.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

V4, V5 = Path("notebooks/results_v4"), Path("notebooks/results_v5")
MEMBERS = {
    "resnet18_b1": (V4, "stretch_manh"),
    "densenet121": (V5, "densenet121_robust"),
}
TARGET_SENSITIVITY = 0.97
BOOTSTRAP = 10000
SEED = 42


def oof_groups(root, name):
    """Pooled out-of-fold predictions, one row per filename-derived group.

    Args:
        root: Directory holding the per-fold validation files.
        name: Experiment name used in the filenames.

    Returns:
        Frame indexed by group with label and probability.
    """
    parts = [pd.read_csv(root / f"validation_predictions_{name}_fold{k}.csv",
                         usecols=["group_id", "class_id", "p_pneumonia"])
             for k in range(5)]
    pooled = pd.concat(parts, ignore_index=True)
    return (pooled.groupby("group_id")
            .agg(label=("class_id", "first"), p=("p_pneumonia", "mean"))
            .sort_index())


def benchmark_groups(root, name):
    """Benchmark predictions for one model, indexed by group.

    Args:
        root: Directory holding the benchmark file.
        name: Experiment name used in the filename.

    Returns:
        Frame indexed by group with label and probability.
    """
    frame = pd.read_csv(root / f"predictions_known_benchmark_{name}_groups.csv")
    return (frame.set_index("group_id")[["label", "p_pneumonia"]]
            .rename(columns={"p_pneumonia": "p"}).sort_index())


def to_rank(values):
    """Within-model percentile rank, so two scales become comparable.

    Args:
        values: Probabilities from one model.

    Returns:
        Ranks scaled to (0, 1).
    """
    return rankdata(values) / (len(values) + 1.0)


def threshold_at_sensitivity(labels, scores, target=TARGET_SENSITIVITY):
    """Highest threshold still meeting a minimum sensitivity.

    Args:
        labels: Binary labels.
        scores: Model scores.
        target: Minimum sensitivity to hold.

    Returns:
        The selected threshold.
    """
    positive = scores[labels == 1]
    feasible = [c for c in np.unique(scores) if (positive >= c).mean() >= target]
    return float(max(feasible)) if feasible else 0.0


def partial_auc(labels, scores, min_sensitivity=TARGET_SENSITIVITY):
    """Area under the ROC restricted to the high-sensitivity region, normalised.

    Global AUC integrates over the whole curve and barely moves when a few
    dozen normals are reordered near the operating point. This looks only where
    the model is actually used.

    Args:
        labels: Binary labels.
        scores: Model scores.
        min_sensitivity: Lower bound of the region of interest.

    Returns:
        Partial AUC rescaled so 1.0 is perfect within the region.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    mask = tpr >= min_sensitivity
    if mask.sum() < 2:
        return float("nan")
    x, y = fpr[mask], tpr[mask]
    width = x.max() - x.min()
    if width < 1e-12:
        return float("nan")
    area = np.trapezoid(y - min_sensitivity, x)
    return float(area / (width * (1.0 - min_sensitivity)))


def score_block(labels, scores, threshold):
    """Metrics reported for every candidate.

    Args:
        labels: Binary labels.
        scores: Model scores.
        threshold: Decision threshold.

    Returns:
        Mapping of metric name to value.
    """
    tn, fp, fn, tp = confusion_matrix(
        labels, (scores >= threshold).astype(int), labels=[0, 1]).ravel()
    return {"auc": roc_auc_score(labels, scores),
            "partial_auc_sens97": partial_auc(labels, scores),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "threshold": float(threshold)}


def main():
    oof = {name: oof_groups(*spec) for name, spec in MEMBERS.items()}
    bench = {name: benchmark_groups(*spec) for name, spec in MEMBERS.items()}
    names = list(MEMBERS)

    index = oof[names[0]].index
    assert all(oof[n].index.equals(index) for n in names), "OOF group không khớp"
    bench_index = bench[names[0]].index
    assert all(bench[n].index.equals(bench_index) for n in names)
    y_oof = oof[names[0]]["label"].to_numpy()
    y_bench = bench[names[0]]["label"].to_numpy()
    assert all((oof[n]["label"].to_numpy() == y_oof).all() for n in names)

    print(f"OOF {len(y_oof):,} group | benchmark {len(y_bench):,} group\n")

    candidates = {}
    for name in names:
        candidates[name] = (oof[name]["p"].to_numpy(), bench[name]["p"].to_numpy())
    candidates["ens_probability"] = (
        np.mean([oof[n]["p"].to_numpy() for n in names], axis=0),
        np.mean([bench[n]["p"].to_numpy() for n in names], axis=0))
    candidates["ens_rank"] = (
        np.mean([to_rank(oof[n]["p"].to_numpy()) for n in names], axis=0),
        np.mean([to_rank(bench[n]["p"].to_numpy()) for n in names], axis=0))

    print("CHỌN TRÊN OOF — benchmark chưa được nhìn\n")
    print(f"{'ứng viên':<18} {'OOF AUC':>9} {'pAUC≥97':>9} "
          f"{'OOF đặc hiệu':>13} {'ngưỡng':>9}")
    print("-" * 62)
    rows, oof_scores = [], {}
    for name, (scores_oof, _) in candidates.items():
        threshold = threshold_at_sensitivity(y_oof, scores_oof)
        block = score_block(y_oof, scores_oof, threshold)
        oof_scores[name] = block["specificity"]
        rows.append({"candidate": name, "split": "oof", **block})
        print(f"{name:<18} {block['auc']:>9.4f} {block['partial_auc_sens97']:>9.4f} "
              f"{block['specificity']:>13.4f} {threshold:>9.4f}")

    # Rank averaging is reported for completeness and excluded from selection.
    # See the module docstring: a rank threshold cannot cross a prevalence
    # change, and this one crosses 66.8% to 47.4%.
    eligible = ["ens_probability"]
    excluded = {"ens_rank": "ngưỡng rank không chuyển được qua thay đổi tỉ lệ lớp"}
    for name, reason in excluded.items():
        print(f"\n   loại {name}: {reason}")
        print(f"   (OOF của nó là {oof_scores[name]:.4f}, cao hơn — nhưng lý do loại"
              f" là phương pháp, không phải hiệu năng)")
    chosen = max(eligible, key=lambda n: oof_scores[n])
    locked_threshold = threshold_at_sensitivity(y_oof, candidates[chosen][0])
    print(f"\n=> khóa: {chosen}, ngưỡng {locked_threshold:.4f}")

    print("\n\nÁP LÊN KNOWN BENCHMARK\n")
    print(f"{'ứng viên':<18} {'AUC':>8} {'pAUC≥97':>9} {'độ nhạy':>9} "
          f"{'đặc hiệu':>10} {'FP':>4}")
    print("-" * 64)
    for name, (scores_oof, scores_bench) in candidates.items():
        threshold = (locked_threshold if name == chosen
                     else threshold_at_sensitivity(y_oof, scores_oof))
        block = score_block(y_bench, scores_bench, threshold)
        rows.append({"candidate": name, "split": "known_benchmark", **block})
        mark = "  <- đã khóa" if name == chosen else ""
        print(f"{name:<18} {block['auc']:>8.4f} "
              f"{block['partial_auc_sens97']:>9.4f} {block['sensitivity']:>9.4f} "
              f"{block['specificity']:>10.4f} {block['fp']:>4}{mark}")

    frame = pd.DataFrame(rows)
    frame["locked"] = frame["candidate"] == chosen
    frame.to_csv(V5 / "results_two_model_ensemble.csv", index=False)

    # Paired comparison against the stronger single model.
    reference = max(names, key=lambda n: oof_scores[n])
    ref_threshold = threshold_at_sensitivity(y_oof, candidates[reference][0])
    normal = y_bench == 0
    pa = (candidates[reference][1] >= ref_threshold).astype(int)[normal]
    pb = (candidates[chosen][1] >= locked_threshold).astype(int)[normal]
    fixed = int(((pa == 1) & (pb == 0)).sum())
    broken = int(((pa == 0) & (pb == 1)).sum())

    rng = np.random.default_rng(SEED)
    sa, sb = (pa == 0).astype(float), (pb == 0).astype(float)
    draws = rng.integers(0, len(sa), (BOOTSTRAP, len(sa)))
    delta = sb[draws].mean(1) - sa[draws].mean(1)
    low, high = np.percentile(delta, [2.5, 97.5])

    print(f"\n\nGHÉP CẶP: {chosen} so với {reference} (mô hình đơn tốt nhất theo OOF)\n")
    print(f"  {reference} báo nhầm → ensemble đúng : {fixed:>3}")
    print(f"  {reference} đúng → ensemble báo nhầm : {broken:>3}")
    print(f"  ròng                                  : {fixed - broken:+3d}")
    if fixed + broken:
        print(f"  McNemar exact                         : "
              f"p = {binomtest(fixed, fixed + broken, 0.5).pvalue:.4f}")
    print(f"  Δđặc hiệu KTC 95%                     : [{low:+.4f}, {high:+.4f}]")

    final = frame[(frame["candidate"] == chosen)
                  & (frame["split"] == "known_benchmark")].iloc[0]
    print(f"\n{'=' * 58}")
    print(f"  độ đặc hiệu {final['specificity']:.4f}   "
          f"độ nhạy {final['sensitivity']:.4f}   FP {int(final['fp'])}")
    passes = final["specificity"] >= 0.82 and final["sensitivity"] >= 0.97
    print(f"  mục tiêu tối thiểu 0.82 tại độ nhạy ≥0.97: "
          f"{'ĐẠT' if passes else 'CHƯA ĐẠT'}")
    print("=" * 58)
    print("\n→ results_two_model_ensemble.csv")


if __name__ == "__main__":
    main()
