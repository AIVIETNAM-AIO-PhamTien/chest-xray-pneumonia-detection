"""Build the DeiT-Small notebook from the executed v5 notebook.

v5 supplies data handling, splits, preprocessing and metrics unchanged, so the
architecture is the only thing that differs from the CNN runs. The training
loop is replaced because a Transformer needs AdamW with warm-up and cosine
decay rather than plain Adam, and because the checkpoint rule gains HSAS@97.
"""

import json
import re
from pathlib import Path

SOURCE = Path("notebooks/legacy/model-improvement-pneumonia.ipynb")
TARGET = Path("notebooks/legacy/model_improvement_deit_v6.ipynb")

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
    """Swap the CNN ladder for DeiT's optimiser and selection settings."""
    block = text[text.index("# Stage A1 chỉ chạy MỘT kiến trúc"):
                 text.index("CLASSES = (")]
    replacement = '''# Stage A2 chạy đúng một Transformer. Mọi thứ khác giữ nguyên như CNN để
# bảng so sánh còn nghĩa: cùng manifest, cùng fold, cùng preprocessing, cùng loss.
MODEL_ID          = "deit_small_patch16_224.fb_in1k"
REQUIRE_PRETRAINED = True   # tải hỏng thì dừng, không rơi về khởi tạo ngẫu nhiên

DEIT_LR           = 5e-5
DEIT_WEIGHT_DECAY = 0.05
DEIT_BATCH        = 16
WARMUP_EPOCHS     = 1
GRAD_CLIP_NORM    = 1.0

# Biên hòa của quy tắc v6, khóa trước khi chạy.
SPECIFICITY_TIE = 0.005
HSAS_TIE        = 0.002

DEIT_EXPERIMENT = {
    "name": "deit_small", "arch": "deit_small", "size": 224,
    "aug": "manh", "balancing": "weighted", "resize": "stretch",
    "hoi": "Transformer dưới đúng pipeline của CNN",
}
EXPERIMENTS = [DEIT_EXPERIMENT]

# Mốc so sánh, mức filename-group trên known benchmark.
COMPARATORS = {
    "resnet18 B1":  {"auc": 0.9779, "hsas": None, "sensitivity": 0.9951,
                     "specificity": 0.7689, "fp": 52},
    "densenet121":  {"auc": 0.9801, "hsas": None, "sensitivity": 0.9951,
                     "specificity": 0.8044, "fp": 44},
    "ens R+D":      {"auc": 0.9792, "hsas": None, "sensitivity": 0.9951,
                     "specificity": 0.8222, "fp": 40},
}

'''
    text = text.replace(block, replacement)
    for pattern in (r"\n# Mốc cần vượt: B1.*?\n\n", r"\nALL_EXPERIMENTS = \[.*?\n\]\n",
                    r"\nSMOKE_EXPERIMENTS = \[.*?\n\]\n", r"\nBASELINE_B1 = \{.*?\}\n",
                    r"\nEXPECTED_STAGE_A1 = \{.*?\n\}\n"):
        text = re.sub(pattern, "\n", text, flags=re.S)
    text = text.replace("EPOCHS        = 12", "EPOCHS        = 12")
    text = text.replace("PATIENCE      = 5", "PATIENCE      = 4")
    text = text.replace("BATCH_SIZE    = 32", "BATCH_SIZE    = 16")
    return text


def patch_runtime(text):
    """One experiment, DeiT logging, and no CNN ladder bookkeeping."""
    text = text.replace(
        "precision_score, recall_score, roc_auc_score)",
        "precision_score, recall_score, roc_auc_score,\n"
        "                             roc_curve)")
    text = re.sub(
        r'if RUN_MODE == "smoke":.*?raise AssertionError\(\n'
        r'            f"Cấu hình Stage A1 bị lệch: \{_actual\}; '
        r'kỳ vọng \{EXPECTED_STAGE_A1\}"\)\n',
        'if RUN_MODE == "smoke":\n'
        '    # N_FOLDS giữ nguyên để cách chia khớp các lần chạy CNN.\n'
        '    EPOCHS, FOLDS_TO_RUN = 1, [0]\n'
        'else:\n'
        '    FOLDS_TO_RUN = list(range(N_FOLDS))\n',
        text, flags=re.S)
    text = text.replace('LOG_PATH = WORK_DIR / "train_log_v5.txt"',
                        'LOG_PATH = WORK_DIR / "train_log_deit.txt"')
    text = re.sub(r'\n *"learning_rate": LR,',
                  '\n    "learning_rate": DEIT_LR,'
                  '\n    "weight_decay_deit": DEIT_WEIGHT_DECAY,'
                  '\n    "warmup_epochs": WARMUP_EPOCHS,'
                  '\n    "grad_clip_norm": GRAD_CLIP_NORM,'
                  '\n    "model_id": MODEL_ID,', text)
    return text


PREFLIGHT = '''
import timm

# Fail here rather than at the first forward pass, and never quietly train a
# randomly initialised Transformer: it would look like a weak architecture
# result instead of a missing download.
available = timm.list_models(f"*{MODEL_ID.split('.')[0]}*", pretrained=True)
if MODEL_ID not in available:
    raise RuntimeError(
        f"{MODEL_ID} không có trong timm {timm.__version__}. "
        f"Gần nhất: {available[:5]}")

set_seed()  # trước create_model, để head 2 lớp khởi tạo tái lập được
try:
    _probe = timm.create_model(MODEL_ID, pretrained=REQUIRE_PRETRAINED,
                               num_classes=len(CLASSES))
except Exception as exc:
    raise RuntimeError(
        "Không tải được pretrained weights. Bật Internet trong Notebook "
        "Settings. Không fallback sang khởi tạo ngẫu nhiên."
    ) from exc

_cfg = _probe.default_cfg
DEIT_PROVENANCE = {
    "timm_version": timm.__version__, "model_id": MODEL_ID,
    "pretrained": REQUIRE_PRETRAINED, "num_classes": len(CLASSES),
    "input_size": list(_cfg["input_size"]),
    "mean": list(_cfg["mean"]), "std": list(_cfg["std"]),
    "parameters_millions": sum(p.numel() for p in _probe.parameters()) / 1e6,
}

# Hash the backbone only. The classification head is freshly initialised for
# two classes, so including it would make the digest change every run.
_backbone = {k: v for k, v in _probe.state_dict().items()
             if not k.startswith("head")}
DEIT_PROVENANCE["backbone_sha256"] = hashlib.sha256(
    b"".join(v.cpu().numpy().tobytes() for _, v in sorted(_backbone.items()))
).hexdigest()[:16]

with open(WORK_DIR / "checkpoint_manifest_deit.json", "w") as handle:
    json.dump(DEIT_PROVENANCE, handle, indent=2)

log(f"model: {MODEL_ID}")
log(f"timm: {timm.__version__} | pretrained: {REQUIRE_PRETRAINED} | "
    f"tham số: {DEIT_PROVENANCE['parameters_millions']:.1f}M")
log(f"backbone sha256: {DEIT_PROVENANCE['backbone_sha256']}")

_out = _probe(torch.randn(2, 3, 224, 224))
assert _out.shape == (2, len(CLASSES)), f"output shape lạ: {tuple(_out.shape)}"
log(f"output shape: {tuple(_out.shape)}")

# The pipeline already normalises with these values; a mismatch would mean the
# cache and the pretrained weights disagree about what the input looks like.
assert np.allclose(_cfg["mean"], IMAGENET_MEAN), "mean của model khác pipeline"
assert np.allclose(_cfg["std"], IMAGENET_STD), "std của model khác pipeline"
log("chuẩn hóa khớp pipeline hiện tại")
del _probe, _out
gc.collect()
'''

SELECTION = '''
def exact_threshold_at_sensitivity(labels, probs, target=TARGET_SENSITIVITY):
    """Highest observed score that still meets a minimum sensitivity.

    Every distinct probability is a candidate. At group level there are only a
    few thousand, and a fixed grid can step straight over the one value that
    separates two cases.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities.
        target: Minimum sensitivity to hold.

    Returns:
        The selected threshold, or 0.0 when the target is unreachable.
    """
    labels, probs = np.asarray(labels), np.asarray(probs, dtype=float)
    positive = probs[labels == 1]
    if not len(positive):
        return 0.5
    candidates = np.unique(probs)
    feasible = candidates[[(positive >= c).mean() >= target for c in candidates]]
    return float(feasible.max()) if len(feasible) else 0.0


def specificity_at_sensitivity(labels, probs, target=TARGET_SENSITIVITY):
    """Best specificity reachable while holding a minimum sensitivity.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities.
        target: Minimum sensitivity to hold.

    Returns:
        Tuple of (specificity, threshold).
    """
    threshold = exact_threshold_at_sensitivity(labels, probs, target)
    negative = np.asarray(probs, dtype=float)[np.asarray(labels) == 0]
    if not len(negative):
        return 0.0, threshold
    return float((negative < threshold).mean()), threshold


def hsas_97(labels, probs, min_sensitivity=TARGET_SENSITIVITY):
    """Mean specificity held across sensitivities from the target to 1.

    Reported as HSAS@97, and deliberately not called partial AUC: the McClish
    normalisation divides by the region's own width, which cancels exactly what
    matters here. A model paying far more false positives to reach 97%
    sensitivity traces a wider region of the same shape and would score the
    same.

    Group specificity moves in steps of about one case, so nearly every real
    difference between epochs lands inside the tie band. This reads the same
    part of the curve continuously and can separate them.

    Args:
        labels: Binary labels.
        probs: Predicted probabilities.
        min_sensitivity: Lower bound of the sensitivity range.

    Returns:
        Mean specificity over the range, in [0, 1].

    Raises:
        ValueError: If either class is missing.
    """
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        raise ValueError(f"HSAS cần cả hai lớp; thấy {np.unique(labels).tolist()}")
    fpr, tpr, _ = roc_curve(labels, np.asarray(probs, dtype=float),
                            drop_intermediate=False)
    # Several thresholds can reach one sensitivity; only the cheapest is a
    # real operating point. Interpolating through the others charges false
    # positives the model never had to pay.
    order = np.lexsort((fpr, tpr))
    tpr, fpr = tpr[order], fpr[order]
    keep = np.r_[True, np.diff(tpr) > 0]
    tpr, fpr = tpr[keep], fpr[keep]
    assert np.all(np.diff(fpr) >= -1e-12), "biên Pareto không đơn điệu"
    grid = np.linspace(min_sensitivity, 1.0, 512)
    return float(np.trapezoid(1.0 - np.interp(grid, tpr, fpr), grid)
                 / (1.0 - min_sensitivity))


def better_checkpoint(candidate, incumbent):
    """Decide whether an epoch replaces the one currently held.

    Specificity decides first because it is the endpoint. Inside its tie band
    HSAS@97 decides, then log-loss, then the earlier epoch. The reason is
    returned so the epoch history records why, rather than leaving it to be
    reconstructed from the numbers.

    Args:
        candidate: Metrics for the current epoch.
        incumbent: Metrics for the held checkpoint, or None.

    Returns:
        Tuple of (replace, reason).
    """
    if incumbent is None:
        return True, "first"
    gap = candidate["specificity"] - incumbent["specificity"]
    if gap > SPECIFICITY_TIE:
        return True, "higher_specificity"
    if gap < -SPECIFICITY_TIE:
        return False, "lower_specificity"

    difference = candidate["hsas_97"] - incumbent["hsas_97"]
    if difference > HSAS_TIE:
        return True, "specificity_tie_higher_hsas"
    if difference < -HSAS_TIE:
        return False, "specificity_tie_lower_hsas"

    if candidate["nll"] < incumbent["nll"] - 1e-9:
        return True, "specificity_hsas_tie_lower_nll"
    return False, "all_tied_keep_earlier"


def group_scores(labels, probs, groups):
    """Collapse image predictions to one score per filename-derived group."""
    frame = pd.DataFrame({"g": groups, "y": labels, "p": probs})
    rolled = frame.groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    return rolled["y"].to_numpy(), rolled["p"].to_numpy()


def group_nll(labels, probs):
    """Unweighted log-loss at group level.

    Unweighted on purpose: class weights distort the probability scale, so a
    weighted loss cannot report whether the model is calibrated.
    """
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def group_brier(labels, probs):
    """Brier score at group level."""
    return float(np.mean((probs - labels) ** 2))
'''

TRAINING = '''
def build_deit():
    """Create a pretrained DeiT-Small with a two-class head.

    Returns:
        The model on DEVICE.
    """
    model = timm.create_model(MODEL_ID, pretrained=REQUIRE_PRETRAINED,
                              num_classes=len(CLASSES))
    return model.to(DEVICE)


def resolve_batch_size(model, size, requested=DEIT_BATCH):
    """Find a physical batch that fits, and accumulate to keep the effective one.

    Halving the batch without accumulating would change the optimisation
    problem, not just the memory footprint, so the two must move together.

    Args:
        model: The model to test.
        size: Input side length.
        requested: Desired effective batch size.

    Returns:
        Tuple of (physical batch, accumulation steps).
    """
    if DEVICE.type != "cuda":
        return requested, 1
    for physical in (requested, requested // 2):
        try:
            torch.cuda.empty_cache()
            probe = torch.zeros(physical, size, size, dtype=torch.uint8)
            with torch.amp.autocast(device_type="cuda", enabled=AMP_ENABLED):
                loss = model(to_model_input(probe, size)).float().sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return physical, requested // physical
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    raise RuntimeError("Không vừa cả batch 16 lẫn 8 trên GPU này")


def run_fold(spec, fold_index, epochs=EPOCHS):
    """Train one fold of DeiT-Small.

    Args:
        spec: Experiment specification.
        fold_index: Which fold to run.
        epochs: Maximum epochs.

    Returns:
        Result mapping matching the shape the CNN runs produced.
    """
    log(f"\\n{'=' * 62}\\n{spec['name']}  |  fold {fold_index}\\n{'=' * 62}")
    resize, size = spec["resize"], spec["size"]
    aug = AUG_PRESETS[spec["aug"]]
    set_seed(SEED + fold_index)

    split = FOLDS[fold_index]
    model = build_deit()
    physical, accumulation = resolve_batch_size(model, size)
    loaders = make_loaders(split, seed=SEED + fold_index, mode=resize,
                           batch_size=physical)
    log(f"{MODEL_ID} | {size}px | resize {resize} | augment {spec['aug']} | "
        f"balancing {spec['balancing']}")
    log(f"AdamW | lr {DEIT_LR:.0e} | wd {DEIT_WEIGHT_DECAY} | "
        f"batch {physical}×{accumulation} = {physical * accumulation}")
    log("quy tắc chọn: spec@sens97 → HSAS@97 → NLL → epoch sớm hơn")

    weights = (class_weights_from(split) if spec["balancing"] == "weighted"
               else None)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEIT_LR,
                                  weight_decay=DEIT_WEIGHT_DECAY)
    steps_per_epoch = max(len(loaders["train"]) // accumulation, 1)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    def lr_scale(step):
        """Linear warm-up then cosine decay, indexed by optimizer updates."""
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    val_rows = split[split["split"] == "val"].reset_index(drop=True)
    val_groups = val_rows["group_id"].to_numpy()

    best, best_epoch, best_state, stale, history = None, 0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, (images, labels) in enumerate(loaders["train"]):
            inputs = to_model_input(images, size, aug)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                loss = criterion(model(inputs), labels) / accumulation
            scaler.scale(loss).backward()
            running += loss.item() * accumulation * inputs.size(0)
            seen += inputs.size(0)

            if (step + 1) % accumulation == 0 or step + 1 == len(loaders["train"]):
                # Unscale before clipping: clipping scaled gradients would
                # apply a threshold that depends on the loss scale.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        labels, probs = predict(model, loaders["val"], size)
        g_labels, g_probs = group_scores(labels, probs, val_groups)
        specificity, threshold = specificity_at_sensitivity(g_labels, g_probs)
        operating = metrics_at(g_labels, g_probs, threshold)
        (tn, fp), (fn, tp) = operating["confusion_matrix"]
        current = {
            "epoch": epoch, "train_loss": running / seen,
            "specificity": specificity, "sensitivity": operating["recall"],
            "hsas_97": hsas_97(g_labels, g_probs), "threshold": threshold,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "nll": group_nll(g_labels, g_probs),
            "brier": group_brier(g_labels, g_probs),
            "auc": roc_auc_score(g_labels, g_probs),
            "pr_auc": average_precision_score(g_labels, g_probs),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        replace, reason = better_checkpoint(current, best)
        current["selected"] = replace
        current["selection_reason"] = reason
        history.append(current)

        if replace:
            best, best_epoch, stale = current, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1

        log(f"epoch {epoch:>2}/{epochs}  loss {current['train_loss']:.4f}  "
            f"spec@sens{TARGET_SENSITIVITY:.0%} {specificity:.4f}  "
            f"HSAS {current['hsas_97']:.4f}  AUC {current['auc']:.4f}  "
            f"NLL {current['nll']:.4f}  lr {current['learning_rate']:.2e}  "
            f"{reason}{'  <- best' if replace else ''}")
        if stale >= PATIENCE:
            log(f"dừng sớm ở epoch {epoch}")
            break

    model.load_state_dict(best_state)
    log(f"giữ checkpoint epoch {best_epoch} ({best['selection_reason']}), "
        f"spec {best['specificity']:.4f}, HSAS {best['hsas_97']:.4f}")

    peak = {}
    if DEVICE.type == "cuda":
        peak = {"peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
        log(f"GPU đỉnh: cấp phát {peak['peak_allocated_gb']:.2f} GB, "
            f"dành riêng {peak['peak_reserved_gb']:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    val_labels, val_probs = predict(model, loaders["val"], size)
    tag = f"{spec['name']}_fold{fold_index}"
    torch.save(best_state, WORK_DIR / f"{tag}.pth")
    val_rows.assign(p_pneumonia=val_probs).to_csv(
        WORK_DIR / f"validation_predictions_{tag}.csv", index=False)
    pd.DataFrame(history).to_csv(WORK_DIR / f"epoch_history_{tag}.csv",
                                 index=False)

    result = {"experiment": spec["name"], "resize": resize, "arch": spec["arch"],
              "size": size, "balancing": spec["balancing"], "fold": fold_index,
              "best_epoch": best_epoch, "checkpoint": str(WORK_DIR / f"{tag}.pth"),
              "physical_batch": physical, "accumulation": accumulation,
              "val_labels": val_labels, "val_probs": val_probs,
              "val_groups": val_groups,
              "val": metrics_at(val_labels, val_probs, 0.5),
              "selection": best, **peak}
    del model, loaders, optimizer, scheduler, scaler, best_state
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()
    return result
'''

cells = [
    md("""# Stage A2 — DeiT-Small

Một Transformer, dưới đúng pipeline đã dùng cho hai CNN. Cùng manifest, cùng
fold, cùng preprocessing, cùng loss — chỉ kiến trúc và lịch optimizer thay đổi.

---

## Mốc so sánh (mức filename-group, known benchmark)

| | AUC | Độ nhạy | Độ đặc hiệu | FP |
|---|---:|---:|---:|---:|
| ResNet18 B1 | 0,9779 | 0,9951 | 0,7689 | 52 |
| DenseNet121 v5 | 0,9801 | 0,9951 | 0,8044 | 44 |
| Ensemble R+D | 0,9792 | 0,9951 | **0,8222** | 40 |

Mục tiêu: **≥0,85 độ đặc hiệu** ở độ nhạy ≥0,97, tức FP ≤33 trên 225 group
NORMAL — cần giảm thêm 7 ca so với ensemble hiện tại.

## Quy tắc chọn checkpoint v6

```
1. độ đặc hiệu @ độ nhạy ≥97%     (endpoint chính)
2. hòa nếu |Δ| < 0,005
3. HSAS@97 cao hơn
4. hòa nếu |Δ| < 0,002
5. NLL không trọng số thấp hơn
6. epoch sớm hơn
```

**HSAS@97** là độ đặc hiệu trung bình mô hình giữ được khi độ nhạy nằm trong
dải 97–100%. Đây **không phải** partial AUC theo chuẩn McClish: phép chuẩn hóa
đó chia cho bề rộng vùng, và bề rộng chính là thứ cần đo. Một mô hình phải trả
nhiều false positive hơn để đạt 97% độ nhạy sẽ vẽ vùng rộng hơn nhưng cùng hình
dạng, và cho điểm y hệt.

Stage B cho thấy vì sao cần bậc này: độ đặc hiệu mức group nhảy theo từng ca
(~0,004), nên gần như mọi khác biệt thật đều rơi vào biên hòa 0,005.

> Đây là quy tắc **v6**. DenseNet v5 dùng NLL làm tie-break đầu tiên, nên so
> sánh DenseNet với DeiT **không phải** một ablation kiến trúc thuần túy.

## Yêu cầu

Bật **Internet** trong Notebook Settings — DeiT cần tải pretrained weights.
Không có nhánh offline, và notebook **dừng** nếu tải hỏng thay vì lặng lẽ
huấn luyện một Transformer khởi tạo ngẫu nhiên."""),
    md("## Cấu hình"),
    code(patch_config(original[2])),
    code(patch_runtime(original[3])),
    md("# 1. Dữ liệu\n\nGiữ nguyên từ v5."),
    code(original[5]),
    code(original[6]),
    code(original[8]),
    code(original[9]),
    code(original[10]),
    md("# 2. Phương pháp"),
    code(original[12]),
    md("## 2.2. Tiền xử lý và augmentation"),
    code(original[14]),
    md("## 2.3. Chỉ số đánh giá"),
    code(original[16]),
    md("""## 2.4. Kiểm tra model trước khi chạy

Xác nhận model ID tồn tại, weights tải được, output đúng hình dạng, và chuẩn
hóa của model khớp cái pipeline đang dùng. Hash riêng backbone — head hai lớp
khởi tạo mới mỗi lần nên đưa vào sẽ làm digest đổi liên tục."""),
    code(PREFLIGHT),
    md("""## 2.5. Hàm dùng chung từ v5"""),
    code(original[20][:original[20].index("def run_fold(")].rstrip()),
    md("""## 2.6. Chỉ số và quy tắc chọn checkpoint v6

Định nghĩa sau các hàm của v5 nên chúng ghi đè bản cũ: v5 chọn checkpoint bằng
`spec → NLL`, v6 chèn HSAS@97 vào giữa và trả về cả lý do."""),
    code(SELECTION),
    md("""## 2.7. Vòng huấn luyện

Warm-up rồi cosine decay, tính theo **lần cập nhật optimizer** chứ không theo
mini-batch vật lý — nếu batch 16 không vừa GPU thì batch 8 với accumulation 2
vẫn cho đúng một lịch learning rate.

Gradient được unscale trước khi clip. Clip trên gradient còn scale sẽ áp một
ngưỡng phụ thuộc loss scale, tức là một ngưỡng khác ở mỗi bước."""),
    code(TRAINING),
    md("# 3. Kết quả"),
    code(original[22].replace("for fold in range(len(FOLDS))]",
                              "for fold in FOLDS_TO_RUN]")
         .replace("{len(EXPERIMENTS)} thí nghiệm × {len(FOLDS)} fold)",
                  "{len(EXPERIMENTS)} thí nghiệm × {len(FOLDS_TO_RUN)} fold)")),
    code('''display(pd.DataFrame([
    {"fold": r["fold"], "epoch": r["best_epoch"],
     "lý do": r["selection"]["selection_reason"],
     "spec@sens97": round(r["selection"]["specificity"], 4),
     "HSAS@97": round(r["selection"]["hsas_97"], 4),
     "NLL": round(r["selection"]["nll"], 4),
     "batch": f"{r['physical_batch']}×{r['accumulation']}"}
    for r in RUNS]).set_index("fold"))

if any("peak_allocated_gb" in r for r in RUNS):
    print(f"\\nGPU đỉnh qua các fold: "
          f"cấp phát {max(r.get('peak_allocated_gb', 0) for r in RUNS):.2f} GB, "
          f"dành riêng {max(r.get('peak_reserved_gb', 0) for r in RUNS):.2f} GB")

print("\\nBenchmark chưa được đọc ở bước này.")'''),
    md("## 3.2. Tổng hợp OOF và khóa ngưỡng"),
    code('''oof_labels = np.concatenate([r["val_labels"] for r in RUNS])
oof_probs = np.concatenate([r["val_probs"] for r in RUNS])
oof_groups = np.concatenate([r["val_groups"] for r in RUNS])
G_LABELS, G_PROBS = group_scores(oof_labels, oof_probs, oof_groups)

OOF_THRESHOLD = exact_threshold_at_sensitivity(G_LABELS, G_PROBS)
OOF_SPECIFICITY, _ = specificity_at_sensitivity(G_LABELS, G_PROBS)
OOF_HSAS = hsas_97(G_LABELS, G_PROBS)

pd.DataFrame([{"experiment": "deit_small", "unit": "filename_group",
               "n_groups": len(G_LABELS), "threshold": OOF_THRESHOLD,
               "specificity_at_sens97": OOF_SPECIFICITY, "hsas_97": OOF_HSAS,
               "auc": roc_auc_score(G_LABELS, G_PROBS),
               "pr_auc": average_precision_score(G_LABELS, G_PROBS),
               "nll": group_nll(G_LABELS, G_PROBS),
               "brier": group_brier(G_LABELS, G_PROBS)}]
             ).to_csv(WORK_DIR / "results_deit_oof.csv", index=False)

print(f"OOF gộp: {len(G_LABELS):,} group")
print(f"  ngưỡng khóa        : {OOF_THRESHOLD:.6f}")
print(f"  spec@sens97        : {OOF_SPECIFICITY:.4f}")
print(f"  HSAS@97            : {OOF_HSAS:.4f}")

# Cổng một chiều: benchmark chỉ được đọc sau dòng này.
OOF_THRESHOLD_LOCKED = True
print("\\nOOF_THRESHOLD_LOCKED = True")'''),
    md("""## 3.3. Known benchmark

Chỉ chạy sau khi ngưỡng đã khóa bằng OOF."""),
    code('''assert OOF_THRESHOLD_LOCKED, "Chưa khóa ngưỡng OOF — không được đọc benchmark"

test_rows = FOLDS[0][FOLDS[0]["split"] == "test"].reset_index(drop=True)
test_loader = make_loader(FOLDS[0], "test", seed=SEED,
                          mode=DEIT_EXPERIMENT["resize"])

stack, test_labels = [], None
for run in sorted(RUNS, key=lambda r: r["fold"]):
    model = build_deit()
    model.load_state_dict(torch.load(run["checkpoint"], map_location="cpu",
                                     weights_only=True))
    model.eval()
    labels, probs = predict(model, test_loader, run["size"])
    test_labels = labels if test_labels is None else test_labels
    assert np.array_equal(test_labels, labels)
    stack.append(probs)
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

test_probs = np.mean(stack, axis=0)
B_LABELS, B_PROBS = group_scores(test_labels, test_probs,
                                 test_rows["group_id"].to_numpy())
block = metrics_at(B_LABELS, B_PROBS, OOF_THRESHOLD)
(tn, fp), (fn, tp) = block["confusion_matrix"]

test_rows.assign(p_pneumonia=test_probs,
                 pred=(test_probs >= OOF_THRESHOLD).astype(int)).to_csv(
    WORK_DIR / "predictions_known_benchmark_deit_small_images.csv", index=False)
pd.DataFrame({"group_id": np.unique(test_rows["group_id"]),
              "label": B_LABELS, "p_pneumonia": B_PROBS,
              "pred": (B_PROBS >= OOF_THRESHOLD).astype(int)}).to_csv(
    WORK_DIR / "predictions_known_benchmark_deit_small_groups.csv", index=False)

rows = [{"model": name, **stats} for name, stats in COMPARATORS.items()]
rows.append({"model": "deit_small", "auc": block["auc"],
             "hsas": hsas_97(B_LABELS, B_PROBS),
             "sensitivity": block["recall"], "specificity": block["specificity"],
             "fp": int(fp)})
comparison = pd.DataFrame(rows)
comparison.to_csv(WORK_DIR / "results_deit_small.csv", index=False)
display(comparison.round(4))

print(f"\\nTN {tn}  FP {fp}  FN {fn}  TP {tp}")
print(f"khoảng cách OOF→benchmark, độ đặc hiệu: "
      f"{OOF_SPECIFICITY - block['specificity']:+.4f}")

keep = (block["recall"] >= 0.97
        and (block["specificity"] >= 0.82
             or hsas_97(B_LABELS, B_PROBS) >= max(0.8766, 0.8596) + 0.01))
print(f"\\n=> {'GIỮ' if keep else 'KHÔNG GIỮ'} DeiT làm mô hình đơn theo tiêu chí "
      f"đã đặt trước.")
print("   (DeiT yếu hơn vẫn có thể giữ làm thành viên ensemble nếu nó sửa được "
      "những ca khác.)")
if RUN_MODE == "smoke":
    print("   smoke run — chỉ kiểm tra pipeline")'''),
    md("""# 4. Bước tiếp theo

Ensemble cuối cùng, chỉ trung bình xác suất, chọn thành viên và ngưỡng hoàn
toàn bằng OOF:

```
ResNet + DenseNet          (hiện 0,8222)
ResNet + DeiT
DenseNet + DeiT
ResNet + DenseNet + DeiT
```

Không dùng trung bình rank, không đưa mô hình Stage B vào (nó trùng DenseNet),
không tối ưu trọng số trên benchmark."""),
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
      f"({sum(1 for c in cells if c['cell_type'] == 'code')} code), {errors} lỗi")
