"""Recompute every reported number from the saved predictions, once.

Metrics accumulated across nine notebooks written over several weeks, and the
definitions moved underneath them: the high-sensitivity metric was McClish
partial AUC before it became HSAS@97, and threshold search walked a grid before
it walked every observed score. Copying figures forward would mix those
generations in one table.

So nothing is copied. Every classification value here is computed from raw
group predictions through one implementation. Grad-CAM figures and their
spatial summaries are generated separately by ``build_report_figures.py``.

    python3 scripts/build_final_results.py
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.selection import (  # noqa: E402
    exact_threshold_at_sensitivity, high_sensitivity_average_specificity)

ROOT = Path("notebooks")
FINAL = Path("artifacts/final")
TARGET_SENSITIVITY = 0.97
BOOTSTRAP = 10000
SEED = 42

#: Every model whose predictions survive, and where they live.
SOURCES = {
    "resnet18_b1": (ROOT / "results_v4", "stretch_manh"),
    "densenet121_v5": (ROOT / "results_v5", "densenet121_robust"),
    "deit_small": (ROOT / "results_v6", "deit_small"),
}


def sha256_file(path):
    """Hash a file without loading a large checkpoint into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_input_manifest():
    """Describe the exact predictions/checkpoints behind the final package.

    Returns:
        Tuple of the input-manifest frame, completeness flag and issue list.
    """
    v4, v5, v6 = (ROOT / "results_v4", ROOT / "results_v5", ROOT / "results_v6")
    specs = [
        ("resnet18_checkpoint", sorted(v4.glob("stretch_manh_fold*.pth")), 5),
        (
            "densenet121_checkpoint",
            sorted(v5.glob("densenet121_robust_fold*.pth")),
            5,
        ),
        (
            "resnet18_oof_image_predictions",
            sorted(v4.glob("validation_predictions_stretch_manh_fold*.csv")),
            5,
        ),
        (
            "densenet121_oof_image_predictions",
            sorted(
                v5.glob(
                    "validation_predictions_densenet121_robust_fold*.csv"
                )
            ),
            5,
        ),
        (
            "deit_oof_image_predictions",
            sorted(v6.glob("validation_predictions_deit_small_fold*.csv")),
            5,
        ),
        (
            "resnet18_benchmark_image_predictions",
            [v4 / "predictions_known_benchmark_stretch_manh_images.csv"],
            1,
        ),
        (
            "densenet121_benchmark_image_predictions",
            [
                v5
                / "predictions_known_benchmark_densenet121_robust_images.csv"
            ],
            1,
        ),
        (
            "resnet18_benchmark_group_predictions",
            [v4 / "predictions_known_benchmark_stretch_manh_groups.csv"],
            1,
        ),
        (
            "densenet121_oof_group_predictions",
            [v5 / "predictions_oof_densenet121_robust_groups.csv"],
            1,
        ),
        (
            "densenet121_benchmark_group_predictions",
            [v5 / "predictions_known_benchmark_densenet121_robust_groups.csv"],
            1,
        ),
        (
            "deit_benchmark_group_predictions",
            [v6 / "predictions_known_benchmark_deit_small_groups.csv"],
            1,
        ),
        (
            "final_config",
            [FINAL / "configs" / "final_model_config.json"],
            1,
        ),
    ]
    rows = []
    issues = []
    for role, candidates, expected_count in specs:
        paths = [path for path in candidates if path.is_file()]
        if len(paths) != expected_count:
            issues.append(
                f"{role}: expected {expected_count}, found {len(paths)}"
            )
        for path in paths:
            rows.append(
                {
                    "role": role,
                    "path": path.as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    columns = ["role", "path", "sha256", "size_bytes"]
    return pd.DataFrame(rows, columns=columns), not issues, issues


def pooled_oof(directory, name):
    """One out-of-fold row per group, pooling per-fold files when needed.

    Args:
        directory: Directory holding that run's outputs.
        name: Experiment name as it appears in filenames.

    Returns:
        Frame indexed by group with label and probability.
    """
    path = directory / f"predictions_oof_{name}_groups.csv"
    if path.exists():
        frame = pd.read_csv(path)[["group_id", "label", "p_pneumonia"]]
    else:
        parts = [pd.read_csv(directory / f"validation_predictions_{name}_fold{k}.csv",
                             usecols=["group_id", "class_id", "p_pneumonia"])
                 for k in range(5)]
        frame = (pd.concat(parts, ignore_index=True)
                 .groupby("group_id", as_index=False)
                 .agg(label=("class_id", "first"),
                      p_pneumonia=("p_pneumonia", "mean")))
    return frame.set_index("group_id").sort_index()


def benchmark(directory, name):
    """Benchmark predictions for one model, indexed by group.

    Args:
        directory: Directory holding that run's outputs.
        name: Experiment name as it appears in filenames.

    Returns:
        Frame indexed by group with label and probability.
    """
    frame = pd.read_csv(directory / f"predictions_known_benchmark_{name}_groups.csv")
    return frame.set_index("group_id")[["label", "p_pneumonia"]].sort_index()


def metrics(labels, probs, threshold):
    """Every reported metric for one score vector at one threshold.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities.
        threshold: Decision threshold.

    Returns:
        Mapping of metric name to value.
    """
    predicted = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    return {
        "threshold": float(threshold),
        "auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
        "hsas_97": high_sensitivity_average_specificity(labels, probs),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": precision_score(labels, predicted, zero_division=0),
        "f1": f1_score(labels, predicted, zero_division=0),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "nll": float(-np.mean(labels * np.log(clipped)
                              + (1 - labels) * np.log(1 - clipped))),
        "brier": float(np.mean((probs - labels) ** 2)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def paired_tests(labels, baseline, variant, threshold_a, threshold_b,
                 reps=BOOTSTRAP, seed=SEED):
    """Transition counts, McNemar and paired intervals on the same groups.

    Args:
        labels: Binary group labels.
        baseline: Baseline probabilities.
        variant: Variant probabilities.
        threshold_a: Threshold for the baseline.
        threshold_b: Threshold for the variant.
        reps: Bootstrap replicates.
        seed: Random seed.

    Returns:
        Mapping of statistic name to value.
    """
    negative = labels == 0
    a = (baseline >= threshold_a).astype(int)[negative]
    b = (variant >= threshold_b).astype(int)[negative]
    fixed = int(((a == 1) & (b == 0)).sum())
    broken = int(((a == 0) & (b == 1)).sum())

    rng = np.random.default_rng(seed)
    correct_a, correct_b = (a == 0).astype(float), (b == 0).astype(float)
    index = rng.integers(0, len(correct_a), (reps, len(correct_a)))
    spec_delta = correct_b[index].mean(1) - correct_a[index].mean(1)

    auc_delta, hsas_delta = [], []
    for _ in range(reps // 5):
        draw = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[draw])) < 2:
            continue
        auc_delta.append(roc_auc_score(labels[draw], variant[draw])
                         - roc_auc_score(labels[draw], baseline[draw]))
        hsas_delta.append(
            high_sensitivity_average_specificity(labels[draw], variant[draw])
            - high_sensitivity_average_specificity(labels[draw], baseline[draw]))

    positive = ~negative
    sens_a = (baseline >= threshold_a).astype(int)[positive].mean()
    sens_b = (variant >= threshold_b).astype(int)[positive].mean()
    return {
        "n_normal_groups": int(negative.sum()),
        "fp_baseline": int(a.sum()), "fp_variant": int(b.sum()),
        "fixed_by_variant": fixed, "broken_by_variant": broken,
        "net_fp_reduction": fixed - broken,
        "mcnemar_exact_p": (binomtest(fixed, fixed + broken, 0.5).pvalue
                            if fixed + broken else 1.0),
        "spec_delta": float(correct_b.mean() - correct_a.mean()),
        "spec_ci_low": float(np.percentile(spec_delta, 2.5)),
        "spec_ci_high": float(np.percentile(spec_delta, 97.5)),
        "auc_ci_low": float(np.percentile(auc_delta, 2.5)),
        "auc_ci_high": float(np.percentile(auc_delta, 97.5)),
        "hsas_ci_low": float(np.percentile(hsas_delta, 2.5)),
        "hsas_ci_high": float(np.percentile(hsas_delta, 97.5)),
        "sensitivity_baseline": float(sens_a),
        "sensitivity_variant": float(sens_b),
        "sensitivity_delta": float(sens_b - sens_a),
        "sensitivity_non_inferior": bool(sens_b - sens_a >= -0.01),
    }


def main():
    for folder in ("configs", "predictions", "results", "provenance", "reports"):
        (FINAL / folder).mkdir(parents=True, exist_ok=True)

    oof = {name: pooled_oof(*spec) for name, spec in SOURCES.items()}
    bench = {name: benchmark(*spec) for name, spec in SOURCES.items()}
    reference = oof["resnet18_b1"].index
    for name in SOURCES:
        assert oof[name].index.equals(reference), f"{name}: OOF group lệch"
    bench_index = bench["resnet18_b1"].index
    for name in SOURCES:
        assert bench[name].index.equals(bench_index), f"{name}: benchmark lệch"

    y_oof = oof["resnet18_b1"]["label"].to_numpy()
    y_bench = bench["resnet18_b1"]["label"].to_numpy()
    print(f"OOF {len(y_oof):,} group | benchmark {len(y_bench):,} group\n")

    # The final model, and the two rank variants kept for the record.
    candidates = dict(oof)
    candidates["final_rd_ensemble"] = pd.DataFrame({
        "label": y_oof,
        "p_pneumonia": (oof["resnet18_b1"]["p_pneumonia"].to_numpy()
                        + oof["densenet121_v5"]["p_pneumonia"].to_numpy()) / 2},
        index=reference)
    bench_candidates = dict(bench)
    bench_candidates["final_rd_ensemble"] = pd.DataFrame({
        "label": y_bench,
        "p_pneumonia": (bench["resnet18_b1"]["p_pneumonia"].to_numpy()
                        + bench["densenet121_v5"]["p_pneumonia"].to_numpy()) / 2},
        index=bench_index)

    rows = []
    for name in candidates:
        p_oof = candidates[name]["p_pneumonia"].to_numpy()
        p_bench = bench_candidates[name]["p_pneumonia"].to_numpy()
        threshold = exact_threshold_at_sensitivity(y_oof, p_oof,
                                                   TARGET_SENSITIVITY)
        rows.append({"model": name, "split": "oof",
                     **metrics(y_oof, p_oof, threshold)})
        rows.append({"model": name, "split": "known_benchmark",
                     **metrics(y_bench, p_bench, threshold)})
    ladder = pd.DataFrame(rows)
    ladder.to_csv(FINAL / "results" / "final_model_ladder.csv", index=False)

    print("BẢNG KẾT QUẢ — tính lại toàn bộ từ prediction gốc\n")
    view = ladder[ladder["split"] == "known_benchmark"]
    print(f"{'model':<20} {'AUC':>7} {'HSAS@97':>8} {'độ nhạy':>8} "
          f"{'đặc hiệu':>9} {'FP':>4} {'FN':>3}")
    print("-" * 66)
    for _, row in view.iterrows():
        print(f"{row['model']:<20} {row['auc']:>7.4f} {row['hsas_97']:>8.4f} "
              f"{row['sensitivity']:>8.4f} {row['specificity']:>9.4f} "
              f"{row['fp']:>4} {row['fn']:>3}")

    # --- Paired comparisons against the final model --------------------------
    final_oof = candidates["final_rd_ensemble"]["p_pneumonia"].to_numpy()
    final_bench = bench_candidates["final_rd_ensemble"]["p_pneumonia"].to_numpy()
    final_threshold = exact_threshold_at_sensitivity(y_oof, final_oof,
                                                     TARGET_SENSITIVITY)
    comparisons = []
    for name, role in (("resnet18_b1", "primary"),
                       ("densenet121_v5", "secondary"),
                       ("deit_small", "secondary")):
        p_oof = candidates[name]["p_pneumonia"].to_numpy()
        threshold = exact_threshold_at_sensitivity(y_oof, p_oof,
                                                   TARGET_SENSITIVITY)
        comparisons.append({
            "comparison": f"{name} -> final_rd_ensemble", "role": role,
            **paired_tests(y_bench,
                           bench_candidates[name]["p_pneumonia"].to_numpy(),
                           final_bench, threshold, final_threshold)})
    paired = pd.DataFrame(comparisons)
    paired.to_csv(FINAL / "results" / "final_paired_tests.csv", index=False)

    print("\n\nSO SÁNH GHÉP CẶP với mô hình cuối\n")
    print(f"{'so với':<18} {'sửa':>4} {'phá':>4} {'ròng':>5} {'McNemar p':>10} "
          f"{'Δđặc hiệu KTC 95%':>22}")
    print("-" * 70)
    for _, row in paired.iterrows():
        print(f"{row['comparison'].split(' ->')[0]:<18} "
              f"{row['fixed_by_variant']:>4} {row['broken_by_variant']:>4} "
              f"{row['net_fp_reduction']:>+5} {row['mcnemar_exact_p']:>10.4f} "
              f"  [{row['spec_ci_low']:+.4f}, {row['spec_ci_high']:+.4f}]")

    # --- Locked predictions and configuration --------------------------------
    for label, frame, threshold in (
            ("oof", candidates["final_rd_ensemble"], final_threshold),
            ("benchmark", bench_candidates["final_rd_ensemble"], final_threshold)):
        frame.assign(pred=(frame["p_pneumonia"] >= threshold).astype(int)
                     ).to_csv(
            FINAL / "predictions" / f"predictions_{label}_final_ensemble_groups.csv")

    config = {
        "model": "ResNet18 v4 + DenseNet121 v5, equal probability average",
        "members": {"resnet18": "stretch + strong augmentation + weighted CE",
                    "densenet121": "stretch + strong augmentation + weighted CE"},
        "aggregation": "equal probability mean",
        "threshold": final_threshold,
        "threshold_source": "pooled out-of-fold, exact search over observed scores",
        "threshold_objective": f"maximise specificity at sensitivity >= "
                               f"{TARGET_SENSITIVITY}",
        "primary_unit": "filename-derived group",
        "benchmark_status": "known engineering benchmark, repeatedly inspected; "
                            "not an untouched holdout",
    }
    with open(FINAL / "configs" / "final_model_config.json", "w") as handle:
        json.dump(config, handle, indent=2)

    # --- Provenance ----------------------------------------------------------
    frozen_inputs, frozen_complete, frozen_issues = frozen_input_manifest()
    frozen_inputs.to_csv(
        FINAL / "provenance" / "frozen_input_manifest.csv", index=False
    )

    # Write commit metadata *before* hashing the package.  The previous order
    # hashed the old commit_manifest.json and then replaced it, so the manifest
    # was already stale at the moment it was produced.
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    # Generated files necessarily become dirty before they can be committed.
    # Audit the source tree separately so the flag answers the useful question:
    # was the code/config used for generation itself committed?
    source_status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            ".",
            ":(exclude)artifacts/final/**",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(source_status)
    with open(FINAL / "provenance" / "commit_manifest.json", "w") as handle:
        json.dump({"commit": commit,
                   "working_tree_dirty": dirty,
                   "working_tree_scope":
                   "repository source excluding artifacts/final outputs",
                   "frozen_bundle_complete": frozen_complete,
                   "frozen_bundle_issues": frozen_issues,
                   "sources": {k: str(v[0]) for k, v in SOURCES.items()},
                   "script": "scripts/build_final_results.py"}, handle, indent=2)

    manifest = []
    for path in sorted(FINAL.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.csv":
            manifest.append({
                "path": str(path.relative_to(FINAL)),
                "artifact_type": path.parent.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size})
    pd.DataFrame(manifest).to_csv(
        FINAL / "provenance" / "artifact_manifest.csv", index=False)

    print(f"\n\nngưỡng đã khóa: {final_threshold:.6f}")
    print(f"commit: {commit[:12]}")
    print(f"\n→ {FINAL}/ ({len(manifest)} file, đã băm)")


if __name__ == "__main__":
    main()
