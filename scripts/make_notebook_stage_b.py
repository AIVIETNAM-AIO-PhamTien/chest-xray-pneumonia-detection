"""Build the Stage B fine-tuning notebook from the executed v5 notebook.

v5 supplies the data handling, splits, preprocessing and metrics. Stage B keeps
all of it and replaces only the training loop, because it starts from frozen
checkpoints rather than from ImageNet and weights a subset of the normals more
heavily.
"""

import json
import re
from pathlib import Path

SOURCE = Path("notebooks/legacy/model-improvement-pneumonia.ipynb")
TARGET = Path("notebooks/legacy/stage_b_hard_negative.ipynb")

source = json.load(open(SOURCE))
original = {i: "".join(c["source"]) for i, c in enumerate(source["cells"])}


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "outputs": [],
            "execution_count": None,
            "source": text.strip().splitlines(keepends=True)}


def patch_config(text):
    """Swap the architecture ladder for Stage B's fine-tuning knobs."""
    block = text[text.index("# Stage A1 chỉ chạy MỘT kiến trúc"):
                 text.index("CLASSES = (")]
    replacement = '''# Stage B khởi tạo từ checkpoint DenseNet121 của v5, không train từ ImageNet.
# Thay đổi phương pháp DUY NHẤT là trọng số tương đối của nhóm NORMAL khó.
HARD_MULTIPLIER   = 2.0    # nhân vào loss của group NORMAL khó
HARD_FRACTION     = 0.25   # tỉ lệ NORMAL trong training split bị đánh dấu
FINETUNE_EPOCHS   = 3      # cộng thêm epoch 0 = checkpoint gốc
FINETUNE_LR       = 1e-5
FINETUNE_DECAY    = 1e-5

STAGE_B_EXPERIMENT = {
    "name": "densenet_hn2", "arch": "densenet121", "size": 224,
    "aug": "manh", "balancing": "weighted", "resize": "stretch",
    "hoi": "DenseNet121 + trọng số nhóm NORMAL khó",
}
EXPERIMENTS = [STAGE_B_EXPERIMENT]

# Cấu hình nguồn phải khớp, nếu không thì checkpoint không dùng lại được.
EXPECTED_SOURCE = {"arch": "densenet121", "size": 224, "resize": "stretch",
                   "aug": "manh", "balancing": "weighted"}
SOURCE_EXPERIMENT = "densenet121_robust"

# Băm của bảng hard-negative đã khóa ở commit b3036dd. Notebook dựng lại bảng
# từ OOF rồi đối chiếu, nên một thay đổi im lặng ở phía nguồn sẽ lộ ra.
EXPECTED_HARDNESS_HASHES = {
    "fold0": "b6c6a6a3d8891724", "fold1": "25081fc3411338f3",
    "fold2": "068bea3a87521f51", "fold3": "5bcf6bf8fb177992",
    "fold4": "2c7bbc70c79d15c0",
}

# Epoch 0 phải tái hiện đúng dự đoán đã lưu của v5, nếu không thì checkpoint
# đang load không phải checkpoint đã sinh ra kết quả v5.
EPOCH0_MAX_ABS_DIFF = 1e-5
EPOCH0_MIN_CORRELATION = 0.999999

BASELINE_DENSENET = {"auc": 0.9801, "sensitivity": 0.9951,
                     "specificity": 0.8044, "tn": 181, "fp": 44,
                     "fn": 1, "tp": 202}
BASELINE_ENSEMBLE = {"specificity": 0.8222, "fp": 40}

'''
    text = text.replace(block, replacement)
    text = re.sub(r"\n# Mốc cần vượt: B1.*?\n\n", "\n", text, flags=re.S)
    text = re.sub(r"\nALL_EXPERIMENTS = \[.*?\n\]\n", "\n", text, flags=re.S)
    text = re.sub(r"\nSMOKE_EXPERIMENTS = \[.*?\n\]\n", "\n", text, flags=re.S)
    text = re.sub(r"\nBASELINE_B1 = \{.*?\}\n", "\n", text, flags=re.S)
    text = re.sub(r"\nEXPECTED_STAGE_A1 = \{.*?\n\}\n", "\n", text, flags=re.S)
    return text


def patch_dataset(text):
    """Have the loader hand back cache indices too.

    Stage B needs to know which images belong to hard groups, and the cache
    index is the only identifier that survives shuffling. Returning it from the
    dataset is less fragile than reconstructing the order afterwards.
    """
    text = text.replace(
        """    def __getitem__(self, index):
        cache_index, label = self.rows[index]
        return torch.from_numpy(IMAGE_CACHES[self.mode][cache_index]), label""",
        """    def __getitem__(self, index):
        cache_index, label = self.rows[index]
        # Stage B also returns the cache index so the training loop can look up
        # which images sit in a hard group.
        return (cache_index,
                torch.from_numpy(IMAGE_CACHES[self.mode][cache_index]), label)""")
    return text


PARTIAL_AUC = '''

def partial_auc(labels, probs, min_sensitivity=TARGET_SENSITIVITY):
    """ROC area restricted to the high-sensitivity region, rescaled to [0, 1].

    Global AUC integrates the whole curve and barely registers a few dozen
    normals reordered near the operating point, which is exactly where this
    model is used and exactly what Stage B is trying to move. Logged per epoch
    but deliberately not used for checkpoint selection, so hard-negative
    weighting stays the only methodological change.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities.
        min_sensitivity: Lower bound of the region of interest.

    Returns:
        Normalised partial AUC, or NaN when the region is degenerate.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    mask = tpr >= min_sensitivity
    if mask.sum() < 2:
        return float("nan")
    x, y = fpr[mask], tpr[mask]
    width = x.max() - x.min()
    if width < 1e-12:
        return float("nan")
    return float(np.trapezoid(y - min_sensitivity, x)
                 / (width * (1.0 - min_sensitivity)))
'''


def patch_predict(text):
    """Unpack the extra element, drop v5's run_fold, add partial AUC.

    Stage B defines its own run_fold, so leaving v5's in place would keep dead
    code that no longer matches the loader contract.
    """
    text = text.replace("    for images, labels in loader:",
                        "    for _, images, labels in loader:")
    return text[:text.index("def run_fold(")].rstrip() + "\n" + PARTIAL_AUC


def patch_driver(text):
    """Run only the folds this mode asks for, and report fine-tune epochs."""
    text = text.replace("tối đa {EPOCHS} epoch",
                        "tối đa {FINETUNE_EPOCHS} epoch fine-tune")
    text = text.replace("for fold in range(len(FOLDS))]",
                        "for fold in FOLDS_TO_RUN]")
    text = text.replace("{len(EXPERIMENTS)} thí nghiệm × {len(FOLDS)} fold)",
                        "{len(EXPERIMENTS)} thí nghiệm × {len(FOLDS_TO_RUN)} fold)")
    return text


def patch_runtime(text):
    """Stage B has one experiment; drop v5's ladder bookkeeping."""
    text = text.replace(
        "precision_score, recall_score, roc_auc_score)",
        "precision_score, recall_score, roc_auc_score,\n"
        "                             roc_curve)")
    # Replace only the mode block; the lines after it define LOCAL_PROJECT_ROOT
    # and WORK_DIR, which Stage B still needs.
    text = re.sub(
        r'if RUN_MODE == "smoke":.*?raise AssertionError\(\n'
        r'            f"Cấu hình Stage A1 bị lệch: \{_actual\}; '
        r'kỳ vọng \{EXPECTED_STAGE_A1\}"\)\n',
        'if RUN_MODE == "smoke":\n'
        '    # N_FOLDS giữ nguyên 5: đổi nó sẽ đổi luôn cách chia, và dự đoán\n'
        '    # epoch 0 sẽ không còn so được với bản v5 đã lưu.\n'
        '    FINETUNE_EPOCHS, FOLDS_TO_RUN = 1, [0]\n'
        '    if DEVICE.type != "cuda":\n'
        '        BATCH_SIZE = min(BATCH_SIZE, 16)\n'
        'else:\n'
        '    FOLDS_TO_RUN = list(range(N_FOLDS))\n',
        text, flags=re.S)
    text = text.replace('LOG_PATH = WORK_DIR / "train_log_v5.txt"',
                        'LOG_PATH = WORK_DIR / "train_log_stage_b.txt"')
    text = re.sub(r'\n *"epochs": EPOCHS,', '\n    "finetune_epochs": FINETUNE_EPOCHS,'
                  '\n    "hard_multiplier": HARD_MULTIPLIER,'
                  '\n    "hard_fraction": HARD_FRACTION,', text)
    text = re.sub(r'\n *"learning_rate": LR,', '\n    "learning_rate": FINETUNE_LR,', text)
    return text


REUSE_FOLDS = '''
_local = dict(zip(manifest["filename"], manifest["path"]))
_cache_index = dict(zip(manifest["filename"], manifest["cache_index"]))
FOLDS = []
for _index in range(5):
    _saved = pd.read_csv(find_source_file(f"manifest_fold{_index}.csv",
                                          SOURCE_ROOT))
    # Đường dẫn trong manifest v5 trỏ tới mount của Kaggle lúc đó; ánh xạ lại
    # theo tên file để notebook chạy được ở bất kỳ đâu.
    _saved["path"] = _saved["filename"].map(_local)
    if _saved["path"].isna().any():
        _missing = _saved.loc[_saved["path"].isna(), "filename"].head(3).tolist()
        raise RuntimeError(f"manifest v5 có ảnh không tìm thấy: {_missing}")
    _saved["cache_index"] = _saved["filename"].map(_cache_index)
    FOLDS.append(_saved)

log(f"dùng lại {len(FOLDS)} manifest của v5")
for _index, _fold in enumerate(FOLDS):
    _counts = _fold["split"].value_counts()
    _groups = _fold.groupby("split")["group_id"].nunique()
    log(f"  fold {_index}: train {_counts.get('train', 0):,} "
        f"val {_counts.get('val', 0):,} test {_counts.get('test', 0):,}  |  "
        f"group train/val chồng nhau: "
        f"{len(set(_fold[_fold.split == 'train'].group_id) & set(_fold[_fold.split == 'val'].group_id))}")
    assert not (set(_fold[_fold.split == "train"].group_id)
                & set(_fold[_fold.split == "val"].group_id))
'''

PROVENANCE = '''
def find_source_root():
    """Locate the attached v5 notebook output without hardcoding its slug.

    Kaggle names the mount after the source notebook, and that name changes if
    the notebook is renamed or copied. Searching for the checkpoints themselves
    survives that; requiring exactly one file per fold catches the case where
    two versions of the output are attached at once.

    Returns:
        Mapping of fold index to checkpoint path.

    Raises:
        RuntimeError: If the folds are not exactly 0 to 4, or one is ambiguous.
    """
    roots = [Path("/kaggle/input")] if IS_KAGGLE else [
        LOCAL_PROJECT_ROOT / "notebooks/results_v5"]
    found = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".pt", ".pth"}:
                continue
            if SOURCE_EXPERIMENT not in path.stem:
                continue
            match = re.search(r"fold[_-]?([0-4])", path.stem, re.IGNORECASE)
            if match is None:
                continue
            fold = int(match.group(1))
            if fold in found:
                raise RuntimeError(
                    f"Nhiều checkpoint cho fold {fold}: {found[fold]} và {path}. "
                    "Gỡ bớt input để chỉ còn đúng một version của v5.")
            found[fold] = path
    if set(found) != set(range(5)):
        raise RuntimeError(
            f"Cần đủ fold 0-4, chỉ thấy {sorted(found)}. Trên Kaggle hãy "
            "Add Input -> Your Work -> Notebook -> bản v5 chạy THÀNH CÔNG.")
    return dict(sorted(found.items()))


def find_source_file(pattern, root=None):
    """Find one supporting file, scoped to avoid picking up the wrong run.

    Several runs are attached at once and they share filenames, so an
    unscoped search matches v4's resolved_config.json as readily as v5's.
    Files belonging to the checkpoint source are looked up under the directory
    the checkpoints came from; anything else searches every input.

    Args:
        pattern: Glob pattern matched against filenames.
        root: Directory to search, or None to search all inputs.

    Returns:
        The single matching path.

    Raises:
        RuntimeError: If the pattern matches other than exactly one file.
    """
    if root is not None:
        roots = [root]
    else:
        roots = [Path("/kaggle/input")] if IS_KAGGLE else [
            LOCAL_PROJECT_ROOT / "notebooks"]
    hits = sorted({p for r in roots if r.is_dir() for p in r.rglob(pattern)})
    if len(hits) != 1:
        listing = ", ".join(str(h) for h in hits[:4])
        raise RuntimeError(
            f"{pattern}: cần đúng 1 file, thấy {len(hits)}. {listing}")
    return hits[0]


CHECKPOINTS = find_source_root()
SOURCE_ROOT = CHECKPOINTS[0].parent
log(f"nguồn v5: {SOURCE_ROOT}")
log("checkpoint nguồn:")
for fold, path in CHECKPOINTS.items():
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    log(f"  fold {fold}: {path.name}  sha256 {digest}  "
        f"{path.stat().st_size / 1e6:.1f} MB")

source_config = json.loads(
    find_source_file("resolved_config.json", SOURCE_ROOT).read_text())
source_spec = next(s for s in source_config["experiments"]
                   if s["name"] == SOURCE_EXPERIMENT)
actual = {k: source_spec[k] for k in EXPECTED_SOURCE}
if actual != EXPECTED_SOURCE:
    raise RuntimeError(f"Cấu hình nguồn lệch: {actual}; kỳ vọng {EXPECTED_SOURCE}")
log(f"\\ncấu hình nguồn khớp: {actual}")
'''

HARDNESS = '''
def build_hardness(teachers, manifest, fraction=HARD_FRACTION):
    """Rank one fold's training normals by how hard the teachers found them.

    Ranks are computed inside this fold's own training normals. Ranking over a
    shared pool would let groups the fold never trains on move its cutoff, and
    all five folds would then land on the same number.

    Args:
        teachers: Merged out-of-fold predictions, one row per group.
        manifest: This fold's manifest with group_id and split.
        fraction: Portion of normals to mark hard.

    Returns:
        Table of this fold's training normals with hardness and flag.
    """
    training = set(manifest.loc[manifest["split"] == "train", "group_id"])
    block = teachers[teachers["group_id"].isin(training)
                     & (teachers["label"] == 0)].copy()
    if block.empty:
        raise ValueError("Fold không có group NORMAL nào trong training split")
    columns = [c for c in block.columns if c.startswith("p_")]
    for column in columns:
        block[f"rank_{column[2:]}"] = (
            block[column].rank(method="average") / (len(block) + 1.0))
    block["hardness"] = block[[f"rank_{c[2:]}" for c in columns]].mean(axis=1)
    n_hard = int(round(fraction * len(block)))
    cutoff = block["hardness"].nlargest(n_hard).min() if n_hard else np.inf
    block["hard_normal"] = block["hardness"] >= cutoff
    return block.sort_values("hardness", ascending=False).reset_index(drop=True)


def load_teacher(experiment, name):
    """Read one model's pooled out-of-fold group predictions.

    v5 writes a pooled file directly; v4 predates it and only has per-fold
    validation predictions, so those are pooled here. Either way every group
    is scored exactly once, by the fold that held it out.

    Args:
        experiment: Experiment name as it appears in filenames.
        name: Short model name used to prefix its probability column.

    Returns:
        Frame with group_id, label and a prefixed probability column.

    Raises:
        ValueError: If a group appears more than once.
    """
    try:
        frame = pd.read_csv(find_source_file(
            f"predictions_oof_{experiment}_groups.csv"))
        frame = frame[["group_id", "label", "p_pneumonia"]]
    except RuntimeError:
        parts = []
        for fold in range(5):
            path = find_source_file(
                f"validation_predictions_{experiment}_fold{fold}.csv")
            parts.append(pd.read_csv(
                path, usecols=["group_id", "class_id", "p_pneumonia"]))
        pooled = pd.concat(parts, ignore_index=True)
        frame = (pooled.groupby("group_id", as_index=False)
                 .agg(label=("class_id", "first"),
                      p_pneumonia=("p_pneumonia", "mean")))
    if frame["group_id"].duplicated().any():
        raise ValueError(f"{name}: group_id lặp lại")
    return frame.rename(columns={"p_pneumonia": f"p_{name}"})


resnet = load_teacher("stretch_manh", "resnet")
densenet = load_teacher("densenet121_robust", "densenet")
TEACHERS = resnet.merge(densenet, on=["group_id", "label"], how="inner")
if not (len(TEACHERS) == len(resnet) == len(densenet)):
    raise RuntimeError("Hai bảng OOF không cùng tập group")
log(f"teacher: {len(TEACHERS):,} group, "
    f"{int((TEACHERS['label'] == 0).sum()):,} NORMAL")

HARDNESS = {}
rows = []
for index, manifest in enumerate(FOLDS):
    table = build_hardness(TEACHERS, manifest)
    HARDNESS[index] = table
    hard, rest = table[table["hard_normal"]], table[~table["hard_normal"]]

    # Leakage assertions, run every time rather than trusted.
    training = set(manifest.loc[manifest["split"] == "train", "group_id"])
    other = set(manifest.loc[manifest["split"] != "train", "group_id"])
    assert set(table["group_id"]) <= training, f"fold {index}: có group ngoài train"
    assert not (set(table["group_id"]) & other), f"fold {index}: rò rỉ val/test"
    assert (table["label"] == 0).all(), f"fold {index}: có PNEUMONIA bị đánh dấu"
    assert not table["group_id"].duplicated().any(), f"fold {index}: group trùng"

    path = WORK_DIR / f"hard_negative_groups_fold{index}.csv"
    table.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    rows.append({"fold": index, "n_train_normal_groups": len(table),
                 "n_hard_normal_groups": int(len(hard)),
                 "hard_fraction": len(hard) / len(table),
                 "hardness_cutoff": float(table.loc[table["hard_normal"],
                                                    "hardness"].min()),
                 "median_resnet_hard": float(hard["p_resnet"].median()),
                 "median_resnet_other": float(rest["p_resnet"].median()),
                 "median_densenet_hard": float(hard["p_densenet"].median()),
                 "median_densenet_other": float(rest["p_densenet"].median()),
                 "sha256": digest})

hardness_summary = pd.DataFrame(rows)
hardness_summary.to_csv(WORK_DIR / "hard_negative_summary.csv", index=False)
display(hardness_summary.round(4))

if hardness_summary["hardness_cutoff"].nunique() != len(FOLDS):
    raise RuntimeError("Các fold có cutoff giống nhau: rank có thể đã tính "
                       "trên pool chung thay vì trong từng training split.")

mismatched = {f"fold{r.fold}": (r.sha256, EXPECTED_HARDNESS_HASHES[f"fold{r.fold}"])
              for r in hardness_summary.itertuples()
              if len(FOLDS) == 5
              and r.sha256 != EXPECTED_HARDNESS_HASHES[f"fold{r.fold}"]}
if mismatched:
    log("\\nCẢNH BÁO: bảng hardness khác bản đã khóa ở commit b3036dd:")
    for fold, (got, want) in mismatched.items():
        log(f"  {fold}: dựng lại {got}  đã khóa {want}")
    log("  Nếu đây là smoke run với ít fold hơn thì bỏ qua; nếu là full run "
        "thì dữ liệu nguồn đã thay đổi.")
else:
    log("\\nbảng hardness khớp bản đã khóa ở commit b3036dd")
'''

TRAINING = '''
def hard_negative_loss(logits, labels, class_weights, multiplier):
    """Weighted cross-entropy that leans on the flagged normals.

    The denominator is the summed combined weight, not the summed multipliers.
    PyTorch's weighted cross-entropy already normalises by the summed class
    weights rather than by the batch size, so dividing by multipliers alone
    rescales the loss even when every multiplier is one, and rescaling the loss
    rescales the effective learning rate. Emphasis is the intended change; step
    size is not.

    Args:
        logits: Model outputs.
        labels: Target classes.
        class_weights: Per-class weights, or None.
        multiplier: Per-sample multiplier.

    Returns:
        Scalar loss.
    """
    per_sample = F.cross_entropy(logits, labels, weight=class_weights,
                                 reduction="none")
    weight_per_sample = (torch.ones_like(multiplier) if class_weights is None
                         else class_weights[labels])
    combined = weight_per_sample * multiplier
    return (per_sample * multiplier).sum() / combined.sum().clamp_min(1e-12)


def load_source_model(fold_index):
    """Rebuild a frozen v5 checkpoint without touching ImageNet.

    The state dict is the only source of weights, so downloading pretrained
    parameters would be both unnecessary and a way for a different version to
    creep in. strict=True makes any architecture mismatch fail here.

    Args:
        fold_index: Which fold's checkpoint to load.

    Returns:
        The model on DEVICE, in eval mode.
    """
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
    state = torch.load(CHECKPOINTS[fold_index], map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def check_epoch_zero(fold_index, labels, probs, rows):
    """Confirm the loaded checkpoint reproduces the run it came from.

    If this fails the weights are not the ones that produced v5's numbers, and
    every comparison against v5 downstream would be meaningless.

    Args:
        fold_index: Fold being checked.
        labels: Labels predicted over, in manifest order.
        probs: Freshly computed probabilities.
        rows: Manifest rows for this fold's validation split.

    Returns:
        Mapping describing the agreement.

    Raises:
        RuntimeError: If the saved predictions cannot be located or disagree.
    """
    saved = pd.read_csv(find_source_file(
        f"validation_predictions_{SOURCE_EXPERIMENT}_fold{fold_index}.csv",
        SOURCE_ROOT))
    if not np.array_equal(saved["filename"].to_numpy(),
                          rows["filename"].to_numpy()):
        raise RuntimeError(f"fold {fold_index}: thứ tự ảnh khác bản v5")
    if not np.array_equal(saved["class_id"].to_numpy(), labels):
        raise RuntimeError(f"fold {fold_index}: nhãn khác bản v5")
    reference = saved["p_pneumonia"].to_numpy()
    difference = float(np.abs(probs - reference).max())
    correlation = float(np.corrcoef(probs, reference)[0, 1])
    entry = {"fold": fold_index, "n": len(probs), "max_abs_diff": difference,
             "correlation": correlation,
             "passes": difference <= EPOCH0_MAX_ABS_DIFF
                       and correlation >= EPOCH0_MIN_CORRELATION}
    rows.append if False else None
    log(f"  epoch 0 fold {fold_index}: lệch tối đa {difference:.3e}  "
        f"tương quan {correlation:.8f}  "
        f"{'ĐẠT' if entry['passes'] else 'HỎNG'}")
    if not entry["passes"]:
        raise RuntimeError(
            f"fold {fold_index}: epoch 0 không tái hiện được checkpoint v5 "
            f"(lệch {difference:.3e}, tương quan {correlation:.8f})")
    return entry


def run_fold(spec, fold_index, epochs=None):
    """Fine-tune one fold from its frozen checkpoint.

    Args:
        spec: Experiment specification.
        fold_index: Which fold to run.
        epochs: Fine-tuning epochs beyond epoch 0.

    Returns:
        Result mapping in the same shape v5 produced.
    """
    epochs = FINETUNE_EPOCHS if epochs is None else epochs
    log(f"\\n{'=' * 62}\\n{spec['name']}  |  fold {fold_index}\\n{'=' * 62}")
    resize, size = spec["resize"], spec["size"]
    aug = AUG_PRESETS[spec["aug"]]
    set_seed(SEED + fold_index)

    split = FOLDS[fold_index]
    loaders = make_loaders(split, seed=SEED + fold_index, mode=resize)
    model = load_source_model(fold_index)

    val_rows = split[split["split"] == "val"].reset_index(drop=True)
    val_groups = val_rows["group_id"].to_numpy()
    train_rows = split[split["split"] == "train"].reset_index(drop=True)
    hard_groups = set(HARDNESS[fold_index].loc[
        HARDNESS[fold_index]["hard_normal"], "group_id"])
    hard_by_index = dict(zip(train_rows["cache_index"],
                             train_rows["group_id"].isin(hard_groups)))
    log(f"NORMAL khó: {len(hard_groups)} group, "
        f"{int(sum(hard_by_index.values())):,}/{len(train_rows):,} ảnh train")

    weights = (class_weights_from(split) if spec["balancing"] == "weighted"
               else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR,
                                 weight_decay=FINETUNE_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    def evaluate(epoch, train_loss):
        labels, probs = predict(model, loaders["val"], size)
        g_labels, g_probs = group_scores(labels, probs, val_groups)
        specificity, threshold = specificity_at_sensitivity(g_labels, g_probs)
        operating = metrics_at(g_labels, g_probs, threshold)
        (tn, fp), (fn, tp) = operating["confusion_matrix"]
        return {"epoch": epoch, "train_loss": train_loss,
                "specificity": specificity,
                "sensitivity": operating["recall"], "threshold": threshold,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "nll": group_nll(g_labels, g_probs),
                "brier": group_brier(g_labels, g_probs),
                "auc": roc_auc_score(g_labels, g_probs),
                "pr_auc": average_precision_score(g_labels, g_probs),
                "partial_auc_sens97": partial_auc(g_labels, g_probs)}, labels, probs

    # Epoch 0 is the untouched checkpoint and stays eligible: if fine-tuning
    # makes a fold worse, that fold should keep what it already had.
    zero, labels0, probs0 = evaluate(0, float("nan"))
    EPOCH_ZERO_CHECKS.append(check_epoch_zero(fold_index, labels0, probs0,
                                              val_rows))
    best, best_epoch, history = zero, 0, [zero]
    best_state = {k: v.detach().cpu().clone()
                  for k, v in model.state_dict().items()}
    log(f"epoch  0 (gốc)  spec@sens{TARGET_SENSITIVITY:.0%} "
        f"{zero['specificity']:.4f}  AUC {zero['auc']:.4f}  "
        f"NLL {zero['nll']:.4f}  pAUC {zero['partial_auc_sens97']:.4f}")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for indices, images, labels in loaders["train"]:
            inputs = to_model_input(images, size, aug)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
            multiplier = torch.tensor(
                [HARD_MULTIPLIER if hard_by_index.get(int(i), False) else 1.0
                 for i in indices], device=DEVICE, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                loss = hard_negative_loss(model(inputs), labels, weights,
                                          multiplier)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * inputs.size(0)

        current, _, _ = evaluate(epoch, running / len(loaders["train"].dataset))
        history.append(current)
        marker = ""
        if better_checkpoint(current, best):
            best, best_epoch = current, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            marker = "  <- best"
        log(f"epoch {epoch:>2}/{epochs}  loss {current['train_loss']:.4f}  "
            f"spec@sens{TARGET_SENSITIVITY:.0%} {current['specificity']:.4f}  "
            f"sens {current['sensitivity']:.4f}  AUC {current['auc']:.4f}  "
            f"NLL {current['nll']:.4f}  pAUC "
            f"{current['partial_auc_sens97']:.4f}{marker}")

    model.load_state_dict(best_state)
    log(f"giữ checkpoint epoch {best_epoch}"
        f"{' (không đổi so với v5)' if best_epoch == 0 else ''}, "
        f"spec {best['specificity']:.4f}, NLL {best['nll']:.4f}")

    val_labels, val_probs = predict(model, loaders["val"], size)
    tag = f"{spec['name']}_fold{fold_index}"
    torch.save(best_state, WORK_DIR / f"{tag}.pth")
    val_rows.assign(p_pneumonia=val_probs).to_csv(
        WORK_DIR / f"validation_predictions_{tag}.csv", index=False)
    pd.DataFrame(history).to_csv(WORK_DIR / f"epoch_history_{tag}.csv",
                                 index=False)

    result = {"experiment": spec["name"], "resize": resize, "arch": spec["arch"],
              "size": size, "balancing": spec["balancing"], "fold": fold_index,
              "best_epoch": best_epoch, "kept_original": best_epoch == 0,
              "checkpoint": str(WORK_DIR / f"{tag}.pth"),
              "val_labels": val_labels, "val_probs": val_probs,
              "val_groups": val_groups,
              "val": metrics_at(val_labels, val_probs, 0.5), "selection": best}
    del model, loaders, optimizer, scaler, best_state
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()
    return result


EPOCH_ZERO_CHECKS = []
'''

cells = [
    md("""# Stage B — trọng số nhóm NORMAL khó

Khởi tạo từ năm checkpoint DenseNet121 của v5, **không** train lại từ ImageNet.
Thay đổi phương pháp duy nhất là trọng số tương đối của những ảnh NORMAL mà hai
mô hình đã đóng băng đều thấy khó.

---

## Mốc cần vượt

| | Độ đặc hiệu | FP | Độ nhạy |
|---|---:|---:|---:|
| DenseNet121 v5 | 0,8044 | 44 | 0,9951 |
| Ensemble ResNet+DenseNet | 0,8222 | 40 | 0,9951 |

Mục tiêu Stage B: **≥0,82 hoặc tăng ≥0,02 so với DenseNet**, giảm ròng ít nhất
4 ca báo nhầm, độ nhạy giữ ≥0,97.

## Cần gắn vào notebook trên Kaggle

```
Add Data  → paultimothymooney/chest-xray-pneumonia
Add Input → Your Work → Notebook → bản v5 chạy THÀNH CÔNG
Add Input → Your Work → Notebook → bản v4 (cho OOF của ResNet)
```

Notebook tự tìm checkpoint theo tên file, không phụ thuộc slug của mount. Nếu
gắn nhầm hai version của v5 thì nó dừng ngay thay vì chọn bừa.

## Bốn thứ được kiểm trước khi có bất kỳ bước tối ưu nào

1. đúng năm checkpoint, mỗi fold một cái, `strict=True`;
2. cấu hình nguồn đúng là DenseNet121 + stretch + augment mạnh + weighted;
3. bảng hardness dựng lại từ OOF khớp băm đã khóa ở commit `b3036dd`;
4. **epoch 0 tái hiện đúng dự đoán v5 trên cả năm fold** — lệch ≤ 1e-5, tương
   quan ≥ 0,999999.

Điểm 4 quan trọng nhất: nếu trượt thì trọng số đang load không phải trọng số đã
sinh ra kết quả v5, và mọi so sánh phía sau đều vô nghĩa.

> Epoch 0 là ứng viên hợp lệ trong việc chọn checkpoint. Fold nào fine-tune làm
> tệ đi thì giữ nguyên checkpoint cũ."""),
    md("## Cấu hình"),
    code(patch_config(original[2])),
    code(patch_runtime(original[3])),
    md("# 1. Dữ liệu\n\nGiữ nguyên từ v5: cùng manifest, cùng group, cùng fold."),
    code(original[5]),
    code(original[6]),
    code(original[8]),
    code(original[9]),
    code(original[10]),
    md("# 2. Phương pháp"),
    code(original[12]),
    md("## 2.2. Tiền xử lý và augmentation"),
    code(patch_dataset(original[14])),
    md("## 2.3. Chỉ số đánh giá"),
    code(original[16]),
    md("## 2.4. Kiến trúc"),
    code(original[18]),
    md("""## 2.5. Nguồn gốc checkpoint

Trước khi chạm vào dữ liệu, xác định chính xác checkpoint nào đang được dùng và
băm chúng lại."""),
    code(PROVENANCE),
    md("""## 2.5b. Dùng lại đúng cách chia của v5

Không dựng lại fold. `StratifiedGroupKFold` phụ thuộc thứ tự hàng trong
manifest, và thứ tự đó phụ thuộc cách hệ tệp liệt kê ảnh — nên fold dựng lại
trên một máy khác có thể lệch vài chục ảnh so với bản gốc. Lệch một ảnh là đủ
để epoch 0 không còn so được với dự đoán đã lưu.

v5 đã lưu năm manifest. Đọc thẳng chúng loại bỏ hẳn nguồn sai lệch này."""),
    code(REUSE_FOLDS),
    md("""## 2.6. Bảng nhóm NORMAL khó

Dựng lại từ dự đoán out-of-fold của hai mô hình đã đóng băng, rồi đối chiếu băm
với bản đã khóa. Rank tính **trong training split của từng fold** — nếu tính
trên pool chung thì cả năm cutoff sẽ bằng nhau, và notebook dừng."""),
    code(HARDNESS),
    md("""## 2.7. Vòng fine-tune

Mẫu số của loss là tổng trọng số **kết hợp** (trọng số lớp × multiplier), không
phải tổng multiplier. PyTorch đã chuẩn hóa weighted cross-entropy theo tổng
trọng số lớp chứ không theo cỡ batch, nên chia cho riêng multiplier sẽ đổi scale
loss ngay cả khi multiplier toàn 1 — và đổi scale loss là đổi learning rate hiệu
dụng, đúng thứ cần tránh."""),
    code(patch_predict(original[20])),
    code(TRAINING),
    md("# 3. Kết quả"),
    code(patch_driver(original[22])),
    code('''display(pd.DataFrame([
    {"fold": r["fold"], "epoch giữ": r["best_epoch"],
     "giữ bản gốc": r["kept_original"],
     "spec@sens97": round(r["selection"]["specificity"], 4),
     "pAUC≥97": round(r["selection"]["partial_auc_sens97"], 4),
     "NLL": round(r["selection"]["nll"], 4)}
    for r in RUNS]).set_index("fold"))

kept = sum(r["kept_original"] for r in RUNS)
print(f"\\n{kept}/{len(RUNS)} fold giữ nguyên checkpoint v5")
if kept >= 3:
    print("  Từ 3 fold trở lên giữ bản gốc là điều kiện an toàn đã đăng ký để")
    print("  cân nhắc multiplier 1.5 — quyết định trên tín hiệu OOF, không phải")
    print("  trên benchmark.")

display(pd.DataFrame(EPOCH_ZERO_CHECKS).set_index("fold"))'''),
    md("## 3.2. Tổng hợp OOF"),
    code(original[25]),
    md("## 3.3. Khóa ngưỡng"),
    code(original[27]),
    md("""## 3.4. Known benchmark

Chỉ đọc sau khi ngưỡng đã khóa bằng OOF."""),
    code(original[29]),
    md("""## 3.5. So với DenseNet v5 và ensemble

Điều kiện giữ đã thống nhất trước khi chạy: độ nhạy ≥0,97, và độ đặc hiệu ≥0,82
hoặc tăng ≥0,02 so với DenseNet, và giảm ròng ít nhất 4 ca báo nhầm."""),
    code('''group_row = FINAL["group_tuned"]
predictions = (FINAL["group_probs"] >= GROUP_THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(FINAL["group_labels"], predictions,
                                  labels=[0, 1]).ravel()

display(pd.DataFrame([
    {"model": "DenseNet121 v5", **BASELINE_DENSENET},
    {"model": "ensemble R+D", "specificity": BASELINE_ENSEMBLE["specificity"],
     "fp": BASELINE_ENSEMBLE["fp"], "sensitivity": 0.9951},
    {"model": "DenseNet HN2", "auc": group_row["auc"],
     "sensitivity": group_row["recall"],
     "specificity": group_row["specificity"],
     "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
]).round(4))

delta = group_row["specificity"] - BASELINE_DENSENET["specificity"]
avoided = BASELINE_DENSENET["fp"] - int(fp)
print(f"\\nso với DenseNet v5: Δđặc hiệu {delta:+.4f}, tránh {avoided:+d} ca báo nhầm")
print(f"độ nhạy {group_row['recall']:.4f}")

keep = (group_row["recall"] >= 0.97
        and (group_row["specificity"] >= 0.82 or delta >= 0.02)
        and avoided >= 4)
print(f"\\n=> {'GIỮ' if keep else 'KHÔNG GIỮ'} theo tiêu chí đã đặt trước.")
if RUN_MODE == "smoke":
    print("   (smoke run — chỉ kiểm tra pipeline)")'''),
    md("""# 4. Bước tiếp theo

Chưa triển khai ở đây, theo thứ tự đã thống nhất:

1. ensemble E1/E2/E3 từ prediction đã lưu, không cần train;
2. DeiT-Small dưới cùng pipeline;
3. ensemble cuối cùng.

Multiplier 1,5 chỉ chạy nếu điều kiện an toàn phía OOF được kích hoạt, không
phải vì benchmark cho kết quả không như ý."""),
]

notebook = {"cells": cells, "metadata": source.get("metadata", {}),
            "nbformat": 4, "nbformat_minor": 4}
json.dump(notebook, open(TARGET, "w"), ensure_ascii=False, indent=1)

import ast
errors = 0
for index, cell in enumerate(cells):
    if cell["cell_type"] == "code":
        try:
            ast.parse("".join(cell["source"]))
        except SyntaxError as exc:
            print(f"  LỖI cell {index}: {exc}")
            errors += 1
print(f"{TARGET}: {len(cells)} cell "
      f"({sum(1 for c in cells if c['cell_type'] == 'code')} code), "
      f"{errors} lỗi cú pháp")
