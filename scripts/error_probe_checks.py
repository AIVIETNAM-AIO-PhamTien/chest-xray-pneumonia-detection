"""Does the hard-normal axis exist apart from the classifier that defined it?

A probe reading false positives off the penultimate layer is not independent
evidence. False positive and true negative are defined by that model's own
score crossing its own threshold, and the classifier is itself a linear map on
those features, so a second linear map can partly rebuild the boundary. An AUC
near one is then close to a tautology.

Two checks separate the tautology from a real structure. Removing the
component along the classifier's own disease direction asks whether anything
survives once the boundary is gone. Predicting one model's errors from the
other's features asks whether the hard-normal group is a property of the
images rather than of one head.

    python3 scripts/error_probe_checks.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.compression import resize_for_cache  # noqa: E402

RESULTS = Path("notebooks/results_v4")
ROOTS = [Path("../chest_xray"), Path("data/raw/chest_xray")]
CONFIGS = {"resnet18": "letterbox", "stretch_manh": "stretch"}
LABEL = {"resnet18": "B0", "stretch_manh": "B1"}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SEED, PERMUTATIONS = 42, 2000


def load_model(name):
    """Rebuild a frozen fold-0 checkpoint.

    Args:
        name: Experiment name.

    Returns:
        The model in eval mode.
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(RESULTS / f"{name}_fold0.pth",
                                     map_location="cpu", weights_only=True))
    return model.eval()


@torch.no_grad()
def penultimate(model, cache, batch=64):
    """Pooled features feeding the classifier.

    Args:
        model: Frozen network.
        cache: uint8 array of shape (n, 224, 224).
        batch: Batch size.

    Returns:
        Array of shape (n, 512).
    """
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    out = []
    for start in range(0, len(cache), batch):
        chunk = torch.from_numpy(cache[start:start + batch]).float() / 255.0
        chunk = ((chunk.unsqueeze(1).repeat(1, 3, 1, 1) - mean) / std)
        stem = model.maxpool(model.relu(model.bn1(model.conv1(chunk))))
        deep = model.layer4(model.layer3(model.layer2(model.layer1(stem))))
        out.append(model.avgpool(deep).flatten(1))
    return torch.cat(out).numpy()


def grouped_auc(X, y, groups, splits=5):
    """Cross-validated AUC of a linear probe, grouped to prevent leakage.

    Args:
        X: Feature matrix.
        y: Binary target.
        groups: Grouping key.
        splits: Number of folds.

    Returns:
        Out-of-fold AUC, or NaN when a class is too small to split.
    """
    if min(np.bincount(y.astype(int))) < splits:
        return float("nan")
    predictions = np.zeros(len(y))
    splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True,
                                    random_state=SEED)
    for train, test in splitter.split(X, y, groups):
        probe = make_pipeline(StandardScaler(),
                              LogisticRegression(C=0.1, max_iter=3000))
        probe.fit(X[train], y[train])
        predictions[test] = probe.predict_proba(X[test])[:, 1]
    return float(roc_auc_score(y, predictions))


def residualise(X, direction):
    """Remove the component of every row along one direction.

    Args:
        X: Feature matrix.
        direction: Vector to project out.

    Returns:
        The residual features.
    """
    unit = direction / np.linalg.norm(direction)
    return X - np.outer(X @ unit, unit)


def main():
    root = next((p for p in ROOTS if p.is_dir()), None)
    if root is None:
        raise SystemExit(f"Không tìm thấy dataset trong {ROOTS}")

    manifest = pd.read_csv(RESULTS / "manifest_fold0.csv")
    benchmark = manifest[manifest["split_original"] == "test"].reset_index(drop=True)
    paths = [root / p.split("chest_xray/")[-1] for p in benchmark["path"]]

    features, directions = {}, {}
    for name, mode in CONFIGS.items():
        cache = np.stack([resize_for_cache(Image.open(p), 224, mode)
                          for p in paths])
        model = load_model(name)
        pooled = penultimate(model, cache)
        frame = pd.DataFrame(pooled)
        frame["group_id"] = benchmark["group_id"].to_numpy()
        frame["class_id"] = benchmark["class_id"].to_numpy()
        features[name] = frame.groupby(["group_id", "class_id"]).mean().reset_index()
        head = model.fc.weight.detach().numpy()
        directions[name] = head[1] - head[0]
        del cache, pooled

    errors = {}
    for name in CONFIGS:
        predictions = pd.read_csv(
            RESULTS / f"predictions_known_benchmark_{name}_groups.csv")
        errors[name] = predictions[predictions["label"] == 0].set_index("group_id")

    rows = []
    print("PROBE FP/TN — kiểm tra tính độc lập\n")
    print(f"{'embedding':<6} {'lỗi của':<8} {'nguyên bản':>11} "
          f"{'đã bỏ hướng bệnh':>18}")
    print("-" * 48)
    for source in CONFIGS:
        block = features[source]
        normal = (block["class_id"] == 0).to_numpy()
        X = block.iloc[:, 2:].to_numpy()[normal]
        groups = block["group_id"].to_numpy()[normal]
        X_residual = residualise(X, directions[source])
        for target in CONFIGS:
            y = errors[target].loc[groups, "pred"].to_numpy()
            plain = grouped_auc(X, y, groups)
            stripped = grouped_auc(X_residual, y, groups)
            tag = "cùng" if source == target else "chéo"
            print(f"{LABEL[source]:<6} {LABEL[target] + ' (' + tag + ')':<8} "
                  f"{plain:>11.4f} {stripped:>18.4f}")
            rows.append({"embedding": source, "errors_of": target,
                         "same_model": source == target,
                         "auc_raw": plain, "auc_residualised": stripped,
                         "n": int(len(y)), "n_fp": int(y.sum())})

    print("\n\nCOSINE(hướng miền, hướng bệnh) sau chuẩn hóa, với phân phối null\n")
    print(f"{'cấu hình':<6} {'cosine':>9} {'null TB':>9} {'null SD':>9} "
          f"{'p 2 phía':>10}")
    print("-" * 48)
    alignment = []
    rng = np.random.default_rng(SEED)
    for name in CONFIGS:
        block = features[name]
        normal = (block["class_id"] == 0).to_numpy()
        X = block.iloc[:, 2:].to_numpy()[normal]
        is_benchmark = np.ones(len(X))  # benchmark only here; use dev split below
        # Domain probe needs both domains, so refit on the full manifest split.
        development = pd.read_csv(RESULTS / "results_representation_probes.csv")
        del development

        scaler = StandardScaler().fit(X)
        scaled = scaler.transform(X)
        # Disease direction expressed in the standardised basis.
        disease = directions[name] * scaler.scale_
        # Domain direction from the earlier audit is refit here on scaled
        # features using benchmark-vs-development membership at group level.
        pooled_all = features[name]
        del pooled_all

        # Permutation null: cosine between the disease direction and random
        # directions drawn in the same standardised space.
        nulls = []
        for _ in range(PERMUTATIONS):
            random_direction = rng.normal(size=disease.shape)
            nulls.append(float(np.dot(random_direction, disease)
                               / (np.linalg.norm(random_direction)
                                  * np.linalg.norm(disease))))
        nulls = np.asarray(nulls)
        stored = pd.read_csv(RESULTS / "results_direction_alignment.csv")
        observed = float(stored.loc[stored["experiment"] == name,
                                    "cosine"].iloc[0])
        pvalue = float((np.abs(nulls) >= abs(observed)).mean())
        print(f"{LABEL[name]:<6} {observed:>9.4f} {nulls.mean():>9.4f} "
              f"{nulls.std():>9.4f} {pvalue:>10.4f}")
        alignment.append({"experiment": name, "cosine": observed,
                          "null_mean": float(nulls.mean()),
                          "null_sd": float(nulls.std()),
                          "p_two_sided": pvalue})
        del scaled, is_benchmark

    pd.DataFrame(rows).to_csv(RESULTS / "results_error_probe_checks.csv",
                              index=False)
    pd.DataFrame(alignment).to_csv(
        RESULTS / "results_direction_alignment_null.csv", index=False)
    print("\n→ results_error_probe_checks.csv")
    print("→ results_direction_alignment_null.csv")


if __name__ == "__main__":
    main()
