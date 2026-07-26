"""Phase 2A.2 — can acquisition descriptors alone tell the two domains apart?

Fits nuisance-only classifiers separately inside each class. If the normal
classifier separates development from benchmark far better than the pneumonia
classifier does, the domain difference is specific to the class whose
performance collapsed.

Nothing here sees a pixel of lung or a model score. Cross-validation is grouped
so a patient never appears on both sides of a split.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.nuisance_features import ALL_FEATURES  # noqa: E402

RESULTS = Path("notebooks/results_v4")
CONFIGS = ["resnet18", "stretch_nhe", "augment_manh", "stretch_manh"]
SEED = 42


def cross_validated_auc(X, y, groups, model, splits=5):
    """Out-of-fold AUC under grouped stratified cross-validation.

    Args:
        X: Feature matrix.
        y: Binary target.
        groups: Grouping key preventing leakage across folds.
        model: Estimator factory returning a fresh estimator.
        splits: Number of folds.

    Returns:
        Tuple of (out-of-fold AUC, per-fold AUCs).
    """
    predictions = np.zeros(len(y))
    per_fold = []
    splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True,
                                    random_state=SEED)
    for train_index, test_index in splitter.split(X, y, groups):
        estimator = model()
        estimator.fit(X[train_index], y[train_index])
        scores = estimator.predict_proba(X[test_index])[:, 1]
        predictions[test_index] = scores
        per_fold.append(roc_auc_score(y[test_index], scores))
    return roc_auc_score(y, predictions), per_fold


def main():
    features = pd.read_csv(RESULTS / "nuisance_feature_manifest.csv")
    grouped = (features.groupby(["group_id", "class_id", "split_original"],
                                as_index=False)[ALL_FEATURES].median())
    grouped["is_benchmark"] = (grouped["split_original"] == "test").astype(int)

    factories = {
        "logistic": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        "boosting": lambda: HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, random_state=SEED),
    }

    rows = []
    print("PHÂN LOẠI MIỀN CHỈ TỪ ĐẶC TRƯNG THU NHẬN (không thấy phổi)\n")
    print(f"{'lớp':<12} {'mô hình':<10} {'AUC OOF':>9} {'theo fold':>28}")
    print("-" * 62)
    for class_id, class_name in ((0, "NORMAL"), (1, "PNEUMONIA")):
        block = grouped[grouped["class_id"] == class_id]
        X = block[ALL_FEATURES].to_numpy()
        y = block["is_benchmark"].to_numpy()
        groups = block["group_id"].to_numpy()
        for model_name, factory in factories.items():
            auc, per_fold = cross_validated_auc(X, y, groups, factory)
            spread = " ".join(f"{v:.3f}" for v in per_fold)
            print(f"{class_name:<12} {model_name:<10} {auc:>9.4f}   {spread}")
            rows.append({"target": "domain", "class": class_name,
                         "model": model_name, "auc_oof": auc,
                         "n": int(len(y)), "n_positive": int(y.sum())})

    print("\n\nDỰ ĐOÁN FALSE POSITIVE trong riêng benchmark NORMAL")
    print("(chỉ 225 group, cross-validation theo group)\n")
    print(f"{'cấu hình':<16} {'mô hình':<10} {'AUC OOF':>9}")
    print("-" * 40)
    benchmark_normal = grouped[(grouped["class_id"] == 0)
                               & (grouped["is_benchmark"] == 1)]
    for name in CONFIGS:
        predictions = pd.read_csv(
            RESULTS / f"predictions_known_benchmark_{name}_groups.csv")
        predictions = predictions[predictions["label"] == 0].set_index("group_id")
        target = predictions.loc[benchmark_normal["group_id"], "pred"].to_numpy()
        X = benchmark_normal[ALL_FEATURES].to_numpy()
        groups = benchmark_normal["group_id"].to_numpy()
        for model_name, factory in factories.items():
            auc, _ = cross_validated_auc(X, target, groups, factory)
            print(f"{name:<16} {model_name:<10} {auc:>9.4f}")
            rows.append({"target": "false_positive", "class": name,
                         "model": model_name, "auc_oof": auc,
                         "n": int(len(target)), "n_positive": int(target.sum())})

    pd.DataFrame(rows).to_csv(
        RESULTS / "results_multivariable_domain_models.csv", index=False)
    print("\n→ results_multivariable_domain_models.csv")


if __name__ == "__main__":
    main()
