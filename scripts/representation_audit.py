"""Phase 2C — what the frozen networks encode, without training anything new.

Two preprocessing interventions failed to remove the acquisition signal from
the model's input. That leaves an open question the pixels cannot answer: does
the network's internal representation carry the domain, and if so, does that
direction sit on top of the direction it uses to call pneumonia?

The last part decides whether adversarial removal is even safe here. If the
nuisance direction and the disease direction are close to parallel, suppressing
one damages the other, and no amount of gradient reversal separates what the
data never separated.

Probes are linear and cross-validated by group. The encoder is never
fine-tuned; a probe that can read something off frozen features says the
information is there, not that the classifier uses it.

    python3 scripts/representation_audit.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.compression import resize_for_cache  # noqa: E402

RESULTS = Path("notebooks/results_v4")
ROOTS = [Path("../chest_xray"), Path("data/raw/chest_xray")]
CONFIGS = {"resnet18": "letterbox", "stretch_manh": "stretch"}
LABEL = {"resnet18": "B0 letterbox+nhẹ", "stretch_manh": "B1 stretch+mạnh"}
LAYERS = ("layer1", "layer2", "layer3", "layer4", "penultimate")
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SEED = 42


def load_model(name):
    """Rebuild a frozen Baseline-v4 checkpoint.

    Args:
        name: Experiment name whose fold-0 checkpoint is loaded.

    Returns:
        The model in eval mode.
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(RESULTS / f"{name}_fold0.pth", map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def embed(model, cache, batch=64):
    """Global-average-pooled activations at several depths.

    Pooling keeps the probe dimensionality manageable and removes position, so
    a probe cannot succeed merely by memorising where things sit.

    Args:
        model: Frozen network.
        cache: uint8 array of shape (n, 224, 224).
        batch: Batch size.

    Returns:
        Mapping of layer name to a (n, channels) array.
    """
    captured, handles = {}, []

    def hook(name):
        def store(_module, _inputs, output):
            captured[name] = output.mean(dim=(2, 3)).cpu()
        return store

    for name in ("layer1", "layer2", "layer3", "layer4"):
        handles.append(getattr(model, name).register_forward_hook(hook(name)))

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    gathered = {name: [] for name in LAYERS}
    for start in range(0, len(cache), batch):
        chunk = torch.from_numpy(cache[start:start + batch]).float() / 255.0
        chunk = chunk.unsqueeze(1).repeat(1, 3, 1, 1)
        chunk = (chunk - mean) / std
        features = model.avgpool(model.layer4(model.layer3(model.layer2(
            model.layer1(model.maxpool(model.relu(model.bn1(
                model.conv1(chunk))))))))).flatten(1)
        gathered["penultimate"].append(features.cpu())
        for name in ("layer1", "layer2", "layer3", "layer4"):
            gathered[name].append(captured[name])
    for handle in handles:
        handle.remove()
    return {name: torch.cat(values).numpy() for name, values in gathered.items()}


def probe_auc(X, y, groups, splits=5):
    """Cross-validated AUC of a linear probe, grouped to prevent leakage.

    Args:
        X: Feature matrix.
        y: Binary target.
        groups: Grouping key.
        splits: Number of folds.

    Returns:
        Out-of-fold AUC.
    """
    predictions = np.zeros(len(y))
    splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True,
                                    random_state=SEED)
    for train, test in splitter.split(X, y, groups):
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=0.1, max_iter=3000))
        model.fit(X[train], y[train])
        predictions[test] = model.predict_proba(X[test])[:, 1]
    return float(roc_auc_score(y, predictions))


def probe_r2(X, y, groups, splits=5):
    """Cross-validated R-squared of a ridge probe.

    Args:
        X: Feature matrix.
        y: Continuous target.
        groups: Grouping key.
        splits: Number of folds.

    Returns:
        Out-of-fold R-squared.
    """
    predictions = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=splits).split(X, y, groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(X[train], y[train])
        predictions[test] = model.predict(X[test])
    return float(r2_score(y, predictions))


def main():
    root = next((p for p in ROOTS if p.is_dir()), None)
    if root is None:
        raise SystemExit(f"Không tìm thấy dataset trong {ROOTS}")

    manifest = pd.read_csv(RESULTS / "manifest_fold0.csv")
    features = pd.read_csv(RESULTS / "nuisance_feature_manifest.csv")
    paths = [root / p.split("chest_xray/")[-1] for p in manifest["path"]]

    rows, alignment_rows = [], []
    for name, mode in CONFIGS.items():
        print(f"--- {LABEL[name]} ---", flush=True)
        cache = np.zeros((len(paths), 224, 224), dtype=np.uint8)
        for position, path in enumerate(paths):
            with Image.open(path) as image:
                cache[position] = resize_for_cache(image, 224, mode)
        model = load_model(name)
        embeddings = embed(model, cache)
        del cache

        frame = manifest[["group_id", "class_id", "split_original"]].copy()
        frame["is_benchmark"] = (frame["split_original"] == "test").astype(int)

        for layer in LAYERS:
            matrix = embeddings[layer]
            pooled = (pd.DataFrame(matrix)
                      .groupby([frame["group_id"], frame["class_id"],
                                frame["is_benchmark"]]).mean().reset_index())
            X = pooled.iloc[:, 3:].to_numpy()
            meta = pooled.iloc[:, :3]

            for class_id, class_name in ((0, "NORMAL"), (1, "PNEUMONIA")):
                mask = (meta["class_id"] == class_id).to_numpy()
                auc = probe_auc(X[mask], meta["is_benchmark"].to_numpy()[mask],
                                meta["group_id"].to_numpy()[mask])
                rows.append({"experiment": name, "layer": layer,
                             "probe": f"domain_in_{class_name}", "score": auc,
                             "metric": "auc", "n": int(mask.sum())})
                print(f"  {layer:<12} domain trong {class_name:<10} "
                      f"AUC {auc:.4f}", flush=True)

        # Nuisance proxies and the false-positive probe, at the last layer only.
        matrix = embeddings["penultimate"]
        pooled = (pd.DataFrame(matrix)
                  .groupby([frame["group_id"], frame["class_id"],
                            frame["is_benchmark"]]).mean().reset_index())
        X = pooled.iloc[:, 3:].to_numpy()
        meta = pooled.iloc[:, :3]
        acquisition = features.assign(
            megapixels=features["width"] * features["height"] / 1e6)
        nuisance = (acquisition.groupby("group_id")[
            ["megapixels", "noise_estimate", "laplacian_variance",
             "file_size_per_pixel"]].median().reindex(meta["group_id"]))
        for column in nuisance.columns:
            score = probe_r2(X, nuisance[column].to_numpy(),
                             meta["group_id"].to_numpy())
            rows.append({"experiment": name, "layer": "penultimate",
                         "probe": f"nuisance_{column}", "score": score,
                         "metric": "r2", "n": len(X)})
            print(f"  penultimate  {column:<24} R2 {score:.4f}", flush=True)

        predictions = pd.read_csv(
            RESULTS / f"predictions_known_benchmark_{name}_groups.csv")
        predictions = predictions[predictions["label"] == 0].set_index("group_id")
        benchmark_normal = ((meta["class_id"] == 0)
                            & (meta["is_benchmark"] == 1)).to_numpy()
        target = predictions.loc[
            meta["group_id"].to_numpy()[benchmark_normal], "pred"].to_numpy()
        auc = probe_auc(X[benchmark_normal], target,
                        meta["group_id"].to_numpy()[benchmark_normal])
        rows.append({"experiment": name, "layer": "penultimate",
                     "probe": "false_positive", "score": auc, "metric": "auc",
                     "n": int(benchmark_normal.sum())})
        print(f"  penultimate  FP vs TN                 AUC {auc:.4f}",
              flush=True)

        # Is the domain direction aligned with the direction used for disease?
        normal = (meta["class_id"] == 0).to_numpy()
        domain_probe = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        domain_probe.fit(X[normal], meta["is_benchmark"].to_numpy()[normal])
        domain_direction = domain_probe[-1].coef_.ravel()
        head = model.fc.weight.detach().numpy()
        disease_direction = head[1] - head[0]
        cosine = float(np.dot(domain_direction, disease_direction)
                       / (np.linalg.norm(domain_direction)
                          * np.linalg.norm(disease_direction)))
        alignment_rows.append({"experiment": name, "cosine": cosine,
                               "abs_cosine": abs(cosine)})
        print(f"  cosine(hướng miền, hướng bệnh) = {cosine:+.4f}\n", flush=True)
        del embeddings

    pd.DataFrame(rows).to_csv(RESULTS / "results_representation_probes.csv",
                              index=False)
    pd.DataFrame(alignment_rows).to_csv(
        RESULTS / "results_direction_alignment.csv", index=False)
    print("→ results_representation_probes.csv")
    print("→ results_direction_alignment.csv")


if __name__ == "__main__":
    main()
