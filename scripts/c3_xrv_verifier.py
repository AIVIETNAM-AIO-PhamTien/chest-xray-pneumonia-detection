"""C3.0 and C3.1 — a second-stage verifier built on chest-X-ray encoders.

The oracle frontier put a floor of 22 false positives on the current ensemble's
ranking, so no threshold or fusion reaches twenty. Getting below that needs the
hard normals reordered, which means a representation the ImageNet-pretrained
CNNs do not have.

TorchXRayVision encoders were trained on chest radiographs from several
sources. This stage extracts their frozen features, fits linear verifiers on
top, and applies them only to the cases a very sensitive Stage-1 gate lets
through. Stage 1 already catches 202 of 203 pneumonia groups; the verifier's
only job is to decide which of the survivors are actually normal.

Nothing is fine-tuned here and the benchmark is never opened. This patch
answers one question: do these features carry signal the CNNs lack?

    python3 scripts/c3_xrv_verifier.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.selection import (  # noqa: E402
    HSAS_TIE, SPECIFICITY_TIE, exact_threshold_at_sensitivity,
    high_sensitivity_average_specificity, specificity_at_sensitivity)

V4 = Path("notebooks/results_v4")
V5 = Path("notebooks/results_v5")
OUT = Path("notebooks/results_c3")
ROOTS = [Path("../chest_xray"), Path("data/raw/chest_xray")]
ENCODERS = {"all": "densenet121-res224-all", "rsna": "densenet121-res224-rsna"}
GATE_SENSITIVITY = 0.995
#: Share of below-gate normals added to the verifier training set.
HARD_NORMAL_SHARE = 0.40
TARGET_SENSITIVITY = 0.97
SEED = 42


def load_encoder(weights):
    """Load a frozen chest-X-ray encoder with no output rescaling.

    op_threshs is populated by default and rescales logits around per-pathology
    operating points, which is meant for reading the published classifier
    rather than for using it as a feature extractor. Left in place it would
    warp the representation before the verifier ever sees it.

    Args:
        weights: TorchXRayVision weight name.

    Returns:
        The model in eval mode.
    """
    model = xrv.models.DenseNet(weights=weights, apply_sigmoid=False)
    model.op_threshs = None
    return model.eval()


def xrv_tensor(path, resizer, cropper):
    """Read one radiograph through the encoder's own preprocessing.

    The ImageNet pipeline used elsewhere in this project is wrong here: these
    weights expect one channel scaled to roughly [-1024, 1024], not three
    channels with ImageNet statistics.

    Args:
        path: Image path.
        resizer: XRayResizer instance.
        cropper: XRayCenterCrop instance.

    Returns:
        Tensor of shape (1, 224, 224).
    """
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("L"), dtype=np.uint8)
    normalised = xrv.datasets.normalize(array, maxval=255)
    return torch.from_numpy(resizer(cropper(normalised[None, ...]))).float()


@torch.no_grad()
def extract(model, paths, resizer, cropper, batch=32):
    """Pooled embeddings and raw pathology logits for a list of images.

    Args:
        model: Frozen encoder.
        paths: Image paths.
        resizer: XRayResizer instance.
        cropper: XRayCenterCrop instance.
        batch: Batch size.

    Returns:
        Tuple of (embeddings, logits) arrays.
    """
    embeddings, logits = [], []
    for start in range(0, len(paths), batch):
        chunk = torch.stack([xrv_tensor(p, resizer, cropper)
                             for p in paths[start:start + batch]])
        features = model.features(chunk)
        if features.dim() == 4:
            features = torch.nn.functional.adaptive_avg_pool2d(
                features, 1).flatten(1)
        embeddings.append(features.numpy())
        logits.append(model(chunk).numpy())
        if (start + batch) % 960 == 0:
            print(f"    {min(start + batch, len(paths)):,}/{len(paths):,}",
                  flush=True)
    return np.concatenate(embeddings), np.concatenate(logits)


def group_features(manifest, embeddings, logits, prefix):
    """Reduce image features to one row per filename-derived group.

    Args:
        manifest: Image-level manifest.
        embeddings: Per-image embeddings.
        logits: Per-image pathology logits.
        prefix: Column prefix identifying the encoder.

    Returns:
        Frame indexed by group.
    """
    frame = pd.DataFrame(embeddings,
                         columns=[f"{prefix}_emb{i}" for i in range(embeddings.shape[1])])
    for index in range(logits.shape[1]):
        frame[f"{prefix}_path{index}"] = logits[:, index]
    frame["group_id"] = manifest["group_id"].to_numpy()

    embedding_columns = [c for c in frame.columns if "_emb" in c]
    pathology_columns = [c for c in frame.columns if "_path" in c]
    pooled = frame.groupby("group_id")[embedding_columns + pathology_columns].mean()
    maxed = frame.groupby("group_id")[pathology_columns].max()
    maxed.columns = [f"{c}_max" for c in maxed.columns]
    return pooled.join(maxed).sort_index()


def stage_one_features(oof, threshold):
    """Everything Stage 1 already knows about each group.

    The verifier should not have to rediscover the screening model's own
    confidence, and disagreement between the two CNNs is itself informative
    about which cases sit near the boundary.

    Args:
        oof: Frame with the two model probabilities per group.
        threshold: The frozen ensemble's out-of-fold threshold.

    Returns:
        Frame of Stage-1 derived features.
    """
    logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7)
                             / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    frame = pd.DataFrame(index=oof.index)
    frame["s1_resnet"] = oof["p_resnet"]
    frame["s1_densenet"] = oof["p_densenet"]
    frame["s1_rd"] = (oof["p_resnet"] + oof["p_densenet"]) / 2
    frame["s1_logit_resnet"] = logit(frame["s1_resnet"])
    frame["s1_logit_densenet"] = logit(frame["s1_densenet"])
    frame["s1_logit_rd"] = logit(frame["s1_rd"])
    frame["s1_disagreement"] = (frame["s1_resnet"] - frame["s1_densenet"]).abs()
    frame["s1_margin"] = frame["s1_logit_rd"] - logit(np.array([threshold]))[0]
    frame["s1_n_images"] = oof["n_images"]
    return frame


def cascade_score(stage_one, verifier, gate):
    """Blend gate and verifier into one monotone score.

    Below the gate the case is settled and compresses into [0, 0.5); above it
    the verifier decides within [0.5, 1]. Keeping them on one axis allows AUC
    and HSAS to be computed over the whole cohort, and constraining the final
    threshold to at least 0.5 stops the verifier overriding the gate.

    Args:
        stage_one: Stage-1 probabilities.
        verifier: Verifier probabilities, used only above the gate.
        gate: Stage-1 gate threshold.

    Returns:
        Cascade scores in [0, 1].
    """
    passed = stage_one >= gate
    score = np.where(passed, 0.5 + 0.5 * verifier,
                     0.5 * stage_one / max(gate, 1e-9))
    return np.clip(score, 0.0, 1.0)


def main():
    root = next((p for p in ROOTS if p.is_dir()), None)
    if root is None:
        raise SystemExit(f"Không tìm thấy dataset trong {ROOTS}")
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(V5 / "manifest_fold0.csv")
    paths = [root / p.split("chest_xray/")[-1] for p in manifest["path"]]

    print("PREFLIGHT\n")
    print(f"  torchxrayvision {xrv.__version__}")
    provenance = {"torchxrayvision": xrv.__version__, "encoders": {}}
    resizer = xrv.datasets.XRayResizer(224)
    cropper = xrv.datasets.XRayCenterCrop()

    features = {}
    for name, weights in ENCODERS.items():
        cache = OUT / f"xrv_{name}_group_features.pkl"
        model = load_encoder(weights)
        digest = hashlib.sha256(
            b"".join(v.numpy().tobytes()
                     for _, v in sorted(model.state_dict().items()))
        ).hexdigest()[:16]
        provenance["encoders"][name] = {
            "weights": weights, "sha256": digest,
            "pathologies": len(model.pathologies),
            "op_threshs_disabled": model.op_threshs is None,
            "parameters_millions": sum(p.numel()
                                       for p in model.parameters()) / 1e6}
        print(f"  {weights}: sha256 {digest}, {len(model.pathologies)} pathology, "
              f"op_threshs tắt = {model.op_threshs is None}")

        if cache.exists():
            features[name] = pd.read_pickle(cache)
            print(f"    dùng lại cache {cache.name}")
            continue

        probe = xrv_tensor(paths[0], resizer, cropper)
        assert probe.shape == (1, 224, 224), f"shape lạ: {tuple(probe.shape)}"
        assert torch.isfinite(probe).all()
        assert -1100 <= probe.min() <= probe.max() <= 1100, (
            f"khoảng pixel lạ: [{probe.min():.0f}, {probe.max():.0f}]")
        print(f"    tiền xử lý XRV: shape {tuple(probe.shape)}, "
              f"khoảng [{probe.min():.0f}, {probe.max():.0f}]")

        print(f"    trích {len(paths):,} ảnh ...", flush=True)
        embeddings, logits = extract(model, paths, resizer, cropper)
        table = group_features(manifest, embeddings, logits, name)
        table.to_pickle(cache)
        features[name] = table
        print(f"    {table.shape[0]:,} group × {table.shape[1]} feature "
              f"→ {cache.name}")
        del model, embeddings, logits

    # --- Stage-1 predictions, frozen -----------------------------------------
    def oof(root_dir, name):
        path = root_dir / f"predictions_oof_{name}_groups.csv"
        if path.exists():
            return pd.read_csv(path)[["group_id", "label", "p_pneumonia"]]
        parts = [pd.read_csv(root_dir / f"validation_predictions_{name}_fold{k}.csv",
                             usecols=["group_id", "class_id", "p_pneumonia"])
                 for k in range(5)]
        pooled = pd.concat(parts, ignore_index=True)
        return (pooled.groupby("group_id", as_index=False)
                .agg(label=("class_id", "first"),
                     p_pneumonia=("p_pneumonia", "mean")))

    resnet = oof(V4, "stretch_manh").rename(columns={"p_pneumonia": "p_resnet"})
    densenet = oof(V5, "densenet121_robust").rename(
        columns={"p_pneumonia": "p_densenet"})
    stage1 = resnet.merge(densenet, on=["group_id", "label"], how="inner")
    counts = manifest.groupby("group_id").size().rename("n_images")
    stage1 = stage1.join(counts, on="group_id").set_index("group_id").sort_index()

    development = stage1.index.intersection(features["all"].index)
    stage1 = stage1.loc[development]
    y = stage1["label"].to_numpy()
    p_rd = ((stage1["p_resnet"] + stage1["p_densenet"]) / 2).to_numpy()
    rd_threshold = exact_threshold_at_sensitivity(y, p_rd, TARGET_SENSITIVITY)
    rd_specificity, _ = specificity_at_sensitivity(y, p_rd, TARGET_SENSITIVITY)
    rd_hsas = high_sensitivity_average_specificity(y, p_rd, TARGET_SENSITIVITY)
    print(f"\nStage 1 trên OOF: {len(y):,} group, "
          f"spec@sens97 {rd_specificity:.4f}, HSAS {rd_hsas:.4f}")

    s1_features = stage_one_features(stage1, rd_threshold)

    # --- Fold-specific gates and verifier out-of-fold predictions -------------
    folds = [pd.read_csv(V5 / f"manifest_fold{k}.csv", usecols=["group_id", "split"])
             for k in range(5)]
    assignment = pd.Series(-1, index=stage1.index)
    for index, fold in enumerate(folds):
        validation = set(fold.loc[fold["split"] == "val", "group_id"])
        assignment[assignment.index.isin(validation)] = index
    assert (assignment >= 0).all(), "có group không thuộc fold nào"

    candidates = {
        "V1_all": [c for c in features["all"].columns] + list(s1_features.columns),
        "V2_rsna": [c for c in features["rsna"].columns] + list(s1_features.columns),
    }
    # Feature tables cover every group in the dataset; restrict them to the
    # development groups the labels and folds describe.
    combined = (features["all"].join(features["rsna"])
                .loc[development].join(s1_features))
    assert len(combined) == len(y), "feature và nhãn lệch số hàng"

    print("\nCỔNG STAGE 1 THEO TỪNG FOLD (độ nhạy ≥99.5%)\n")
    print(f"{'fold':>4} {'gate':>9} {'PNEU qua':>16} {'NORMAL qua':>16}")
    print("-" * 50)
    gates, gate_rows = {}, []
    for index in range(5):
        outside = assignment != index
        gate = exact_threshold_at_sensitivity(y[outside], p_rd[outside],
                                              GATE_SENSITIVITY)
        gates[index] = gate
        inside = assignment == index
        passed = p_rd[inside] >= gate
        labels = y[inside]
        entry = {"fold": index, "gate": gate,
                 "pneumonia_passed": int((passed & (labels == 1)).sum()),
                 "pneumonia_total": int((labels == 1).sum()),
                 "normal_passed": int((passed & (labels == 0)).sum()),
                 "normal_total": int((labels == 0).sum())}
        gate_rows.append(entry)
        print(f"{index:>4} {gate:>9.5f} "
              f"{entry['pneumonia_passed']:>7}/{entry['pneumonia_total']:<8} "
              f"{entry['normal_passed']:>7}/{entry['normal_total']:<8}")
    pd.DataFrame(gate_rows).to_csv(OUT / "results_c3_gates.csv", index=False)

    print("\nHUẤN LUYỆN VERIFIER (chỉ trên group qua cổng)\n")
    verifier_scores = {}
    for name, columns in list(candidates.items()) + [("V3_dual", None)]:
        scores = np.full(len(y), np.nan)
        for index in range(5):
            gate = gates[index]
            # Training on gated groups alone would give the verifier almost
            # no negatives: Stage 1 is strong out-of-fold, so only 70 of 1,219
            # normals clear the gate across the whole development set, about
            # fourteen per fold. The benchmark behaves differently -- all forty
            # of its false positives clear it -- so a verifier fitted on
            # fourteen negatives would never have seen the population it must
            # judge. The hardest normals below the gate are added to fix that.
            outer = (assignment != index).to_numpy()
            gated = p_rd >= gate
            normal_below = outer & ~gated & (y == 0)
            if normal_below.sum():
                cutoff = np.quantile(p_rd[normal_below], 1.0 - HARD_NORMAL_SHARE)
                normal_below = normal_below & (p_rd >= cutoff)
            train_mask = outer & (gated | normal_below)
            test_mask = (assignment == index).to_numpy() & gated
            if train_mask.sum() < 20 or test_mask.sum() < 1:
                continue

            if name == "V3_dual":
                # PCA inside the training split only; fitting it across all
                # groups would leak the validation fold into the basis.
                blocks = []
                for encoder in ("all", "rsna"):
                    embedding_columns = [c for c in features[encoder].columns
                                         if "_emb" in c]
                    pca = PCA(n_components=128, random_state=SEED)
                    pca.fit(combined.loc[train_mask, embedding_columns])
                    blocks.append(pd.DataFrame(
                        pca.transform(combined[embedding_columns]),
                        index=combined.index,
                        columns=[f"{encoder}_pc{i}" for i in range(128)]))
                pathology = [c for c in combined.columns if "_path" in c]
                matrix = pd.concat(blocks + [combined[pathology], s1_features],
                                   axis=1)
            else:
                matrix = combined[columns]

            best, best_nll = None, np.inf
            for regularisation in (0.01, 0.1, 1.0):
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=regularisation, max_iter=4000,
                                       class_weight="balanced"))
                inner = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                             random_state=SEED)
                losses = []
                sub = matrix[train_mask].to_numpy()
                sub_y = y[train_mask]
                groups = np.arange(train_mask.sum())
                for fit_index, score_index in inner.split(sub, sub_y, groups):
                    model.fit(sub[fit_index], sub_y[fit_index])
                    predicted = np.clip(
                        model.predict_proba(sub[score_index])[:, 1], 1e-7, 1 - 1e-7)
                    actual = sub_y[score_index]
                    losses.append(-np.mean(actual * np.log(predicted)
                                           + (1 - actual) * np.log(1 - predicted)))
                if np.mean(losses) < best_nll:
                    best, best_nll = regularisation, float(np.mean(losses))

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=best, max_iter=4000,
                                   class_weight="balanced"))
            model.fit(matrix[train_mask].to_numpy(), y[train_mask])
            scores[test_mask] = model.predict_proba(
                matrix[test_mask].to_numpy())[:, 1]
        verifier_scores[name] = scores

    # --- Cascade evaluation on pooled out-of-fold data ------------------------
    final_gate = exact_threshold_at_sensitivity(y, p_rd, GATE_SENSITIVITY)
    print(f"\ncổng cuối trên OOF gộp: {final_gate:.5f}")

    rows = []
    for name, scores in verifier_scores.items():
        filled = np.where(np.isnan(scores), 0.0, scores)
        cascade = cascade_score(p_rd, filled, final_gate)
        threshold = max(exact_threshold_at_sensitivity(y, cascade,
                                                       TARGET_SENSITIVITY), 0.5)
        predicted = (cascade >= threshold).astype(int)
        negative, positive = y == 0, y == 1
        rows.append({
            "verifier": name,
            "specificity": float((predicted[negative] == 0).mean()),
            "sensitivity": float((predicted[positive] == 1).mean()),
            "hsas_97": high_sensitivity_average_specificity(y, cascade),
            "threshold": threshold,
            "n_gated": int((p_rd >= final_gate).sum())})
    baseline = {"verifier": "R+D (Stage 1)", "specificity": rd_specificity,
                "sensitivity": TARGET_SENSITIVITY, "hsas_97": rd_hsas,
                "threshold": rd_threshold, "n_gated": len(y)}
    table = pd.DataFrame([baseline] + rows)
    table.to_csv(OUT / "results_c3_frozen_verifiers.csv", index=False)

    print("\nKẾT QUẢ OOF — benchmark chưa được đọc\n")
    print(f"{'verifier':<16} {'spec@sens97':>12} {'HSAS@97':>9} {'ngưỡng':>9}")
    print("-" * 52)
    for _, row in table.iterrows():
        print(f"{row['verifier']:<16} {row['specificity']:>12.4f} "
              f"{row['hsas_97']:>9.4f} {row['threshold']:>9.4f}")

    gain_specificity = table.iloc[1:]["specificity"] - rd_specificity
    gain_hsas = table.iloc[1:]["hsas_97"] - rd_hsas
    single = table.iloc[1:3]
    passes = ((single["specificity"] - rd_specificity >= 0.005)
              | (single["hsas_97"] - rd_hsas >= 0.01))
    print(f"\nso với Stage 1: Δspec {gain_specificity.max():+.4f}, "
          f"ΔHSAS {gain_hsas.max():+.4f}")
    print(f"\n=> fine-tune từng phần: "
          f"{'GO' if passes.any() else 'NO-GO'} "
          f"(cần Δspec ≥0.005 hoặc ΔHSAS ≥0.01 ở V1 hoặc V2)")

    provenance.update({"gate_sensitivity": GATE_SENSITIVITY,
                       "final_gate": final_gate,
                       "rd_threshold": rd_threshold,
                       "benchmark_loaded": False})
    with open(OUT / "feature_manifest.json", "w") as handle:
        json.dump(provenance, handle, indent=2)
    print("\nBenchmark chưa được đọc ở bước này.")


if __name__ == "__main__":
    main()
