"""Build the C2 notebook: B1's recipe, only the checkpoint rule changed.

Half the locked ensemble still uses the v4 rule, which selected on validation
AUC — the metric later shown to be saturated and to transfer poorly. C2 asks
whether ResNet18 was simply keeping the wrong epoch.

For that question to be answerable, everything else has to stay identical to
B1: optimizer, scheduler, learning rate, epoch budget, patience, batch size,
augmentation, loss, folds and seeds. Only the rule that picks which epoch to
keep is replaced.
"""

import json
import re
from pathlib import Path

SOURCE = Path("notebooks/baseline_kaggle.ipynb")
TARGET = Path("notebooks/c2_resnet_v6.ipynb")

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
    """One experiment reproducing B1, plus the v6 tie bands."""
    block = text[text.index("# BỐN DÒNG ĐẦU"):text.index("EXPERIMENTS = ALL_EXPERIMENTS")]
    replacement = '''# C2 tái hiện đúng công thức B1 (stretch_manh của v4). Thay đổi phương pháp
# DUY NHẤT là quy tắc chọn checkpoint; mọi siêu tham số khác giữ nguyên.
ALL_EXPERIMENTS = [
    {"name": "resnet18_v6", "arch": "resnet18", "size": 224, "aug": "manh",
     "balancing": "weighted", "resize": "stretch",
     "hoi": "B1 với quy tắc chọn checkpoint v6"},
]
SMOKE_EXPERIMENTS = [dict(ALL_EXPERIMENTS[0], name="smoke_resnet_v6")]

# Biên hòa của quy tắc v6, khóa trước khi chạy.
SPECIFICITY_TIE = 0.005
HSAS_TIE        = 0.002

# Công thức B1 phải khớp, nếu không thì đây không còn là ablation về checkpoint.
EXPECTED_B1 = {"arch": "resnet18", "size": 224, "resize": "stretch",
               "aug": "manh", "balancing": "weighted"}

'''
    text = text.replace(block, replacement)
    text = re.sub(r"\nFACTORIAL_NAMES = \[.*?\]\n", "\n", text, flags=re.S)
    return text


def patch_runtime(text):
    """One experiment; keep B1's budget untouched."""
    text = text.replace(
        "precision_score, recall_score, roc_auc_score)",
        "precision_score, recall_score, roc_auc_score,\n"
        "                             roc_curve)")
    text = re.sub(
        r'if RUN_MODE == "smoke":.*?(?=\nif platform\.system)',
        'if RUN_MODE == "smoke":\n'
        '    # N_FOLDS giữ nguyên để cách chia khớp B1.\n'
        '    EPOCHS, FOLDS_TO_RUN = 1, [0]\n'
        '    EXPERIMENTS = SMOKE_EXPERIMENTS\n'
        'else:\n'
        '    EXPERIMENTS = ALL_EXPERIMENTS\n'
        '    FOLDS_TO_RUN = list(range(N_FOLDS))\n'
        '    _actual = {k: EXPERIMENTS[0][k] for k in EXPECTED_B1}\n'
        '    if _actual != EXPECTED_B1:\n'
        '        raise AssertionError(\n'
        '            f"Công thức lệch B1: {_actual}; kỳ vọng {EXPECTED_B1}")\n\n',
        text, flags=re.S)
    text = text.replace('LOG_PATH = WORK_DIR / "train_log.txt"',
                        'LOG_PATH = WORK_DIR / "train_log_c2.txt"')
    text = re.sub(r'\n *"factorial_names": FACTORIAL_NAMES,', "", text)
    return text


SELECTION = '''
def exact_threshold_at_sensitivity(labels, probs, target=TARGET_SENSITIVITY):
    """Highest observed score that still meets a minimum sensitivity.

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

    Not partial AUC: the McClish normalisation divides by the region's own
    width, which cancels the quantity that matters. A model paying more false
    positives to reach 97% sensitivity traces a wider region of the same shape
    and would score identically.

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
    order = np.lexsort((fpr, tpr))
    tpr, fpr = tpr[order], fpr[order]
    keep = np.r_[True, np.diff(tpr) > 0]
    tpr, fpr = tpr[keep], fpr[keep]
    grid = np.linspace(min_sensitivity, 1.0, 512)
    return float(np.trapezoid(1.0 - np.interp(grid, tpr, fpr), grid)
                 / (1.0 - min_sensitivity))


def better_checkpoint(candidate, incumbent):
    """The v6 hierarchy, with the reason recorded.

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
    """Unweighted log-loss at group level."""
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def group_brier(labels, probs):
    """Brier score at group level."""
    return float(np.mean((probs - labels) ** 2))
'''

TRAINING = '''
def run_fold(spec, fold_index, epochs=EPOCHS):
    """Train one fold with B1's recipe and the v6 checkpoint rule.

    The optimizer, scheduler, learning rate, epoch budget and patience are
    exactly B1's. In particular the scheduler still follows validation AUC, as
    it did in v4: changing it would make this something other than a
    checkpoint-selection ablation.

    Args:
        spec: Experiment specification.
        fold_index: Which fold to run.
        epochs: Maximum epochs.

    Returns:
        Result mapping in the shape the other runs produced.
    """
    log(f"\\n{'=' * 62}\\n{spec['name']}  |  fold {fold_index}\\n{'=' * 62}")
    resize, size = spec["resize"], spec["size"]
    aug = AUG_PRESETS[spec["aug"]]
    set_seed(SEED + fold_index)

    split = FOLDS[fold_index]
    loaders = make_loaders(split, seed=SEED + fold_index, mode=resize)
    model = build_model(spec["arch"], pretrained=True)

    weights = (class_weights_from(split) if spec["balancing"] == "weighted"
               else None)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    # Unchanged from B1, including following AUC: the scheduler is part of the
    # recipe being held fixed, not part of the selection rule being tested.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    val_rows = split[split["split"] == "val"].reset_index(drop=True)
    val_groups = val_rows["group_id"].to_numpy()
    log(f"{spec['arch']} | {size}px | resize {resize} | augment {spec['aug']} | "
        f"balancing {spec['balancing']} | lr {LR:.0e} | epochs {epochs}")
    log("chọn checkpoint: spec@sens97 → HSAS@97 → NLL → epoch sớm hơn")

    best, best_epoch, best_state, stale, history = None, 0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for images, labels in loaders["train"]:
            inputs = to_model_input(images, size, aug)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                loss = criterion(model(inputs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * inputs.size(0)

        labels, probs = predict(model, loaders["val"], size)
        g_labels, g_probs = group_scores(labels, probs, val_groups)
        specificity, threshold = specificity_at_sensitivity(g_labels, g_probs)
        operating = metrics_at(g_labels, g_probs, threshold)
        (tn, fp), (fn, tp) = operating["confusion_matrix"]
        auc = roc_auc_score(g_labels, g_probs)
        scheduler.step(auc)

        current = {
            "epoch": epoch, "train_loss": running / len(loaders["train"].dataset),
            "specificity": specificity, "sensitivity": operating["recall"],
            "hsas_97": hsas_97(g_labels, g_probs), "threshold": threshold,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "nll": group_nll(g_labels, g_probs),
            "brier": group_brier(g_labels, g_probs), "auc": auc,
            "pr_auc": average_precision_score(g_labels, g_probs),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        replace, reason = better_checkpoint(current, best)
        current["selected"] = replace
        current["selection_reason"] = reason
        history.append(current)

        # Every epoch's group predictions are kept, so the rule can be audited
        # afterwards -- and an alternative rule compared -- without retraining.
        pd.DataFrame({"group_id": np.unique(val_groups), "label": g_labels,
                      "p_pneumonia": g_probs}).to_csv(
            WORK_DIR / f"epoch_predictions_{spec['name']}_fold{fold_index}"
                       f"_epoch{epoch:02d}.csv", index=False)

        if replace:
            best, best_epoch, stale = current, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1

        log(f"epoch {epoch:>2}/{epochs}  loss {current['train_loss']:.4f}  "
            f"spec@sens{TARGET_SENSITIVITY:.0%} {specificity:.4f}  "
            f"HSAS {current['hsas_97']:.4f}  AUC {auc:.4f}  "
            f"NLL {current['nll']:.4f}  {reason}"
            f"{'  <- best' if replace else ''}")
        if stale >= PATIENCE:
            log(f"dừng sớm ở epoch {epoch}")
            break

    model.load_state_dict(best_state)
    best_auc_epoch = max(history, key=lambda h: h["auc"])["epoch"]
    log(f"giữ checkpoint epoch {best_epoch} ({best['selection_reason']}); "
        f"quy tắc AUC cũ sẽ chọn epoch {best_auc_epoch}"
        f"{' — trùng nhau' if best_epoch == best_auc_epoch else ' — KHÁC'}")

    val_labels, val_probs = predict(model, loaders["val"], size)
    tag = f"{spec['name']}_fold{fold_index}"
    torch.save(best_state, WORK_DIR / f"{tag}.pth")
    val_rows.assign(p_pneumonia=val_probs).to_csv(
        WORK_DIR / f"validation_predictions_{tag}.csv", index=False)
    pd.DataFrame(history).to_csv(WORK_DIR / f"epoch_history_{tag}.csv",
                                 index=False)

    result = {"experiment": spec["name"], "resize": resize, "arch": spec["arch"],
              "size": size, "balancing": spec["balancing"], "fold": fold_index,
              "best_epoch": best_epoch, "best_auc_epoch": best_auc_epoch,
              "checkpoint": str(WORK_DIR / f"{tag}.pth"),
              "val_labels": val_labels, "val_probs": val_probs,
              "val_groups": val_groups,
              "val": metrics_at(val_labels, val_probs, 0.5), "selection": best}
    del model, loaders, optimizer, scheduler, scaler, best_state
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()
    return result
'''

cells = [
    md("""# C2 — ResNet18 với quy tắc chọn checkpoint v6

Thí nghiệm cuối cùng, và nó vá một lỗ hổng thật: **một nửa ensemble đang khóa
vẫn dùng quy tắc chọn checkpoint cũ**.

| Thành viên | Chọn checkpoint bằng |
|---|---|
| ResNet18 B1 (v4) | **validation AUC** |
| DenseNet121 v5 | spec@sens97 → NLL |

Validation AUC sau đó được chứng minh là bão hòa (biên độ 0,0055 qua mọi epoch,
so với 0,0492 của specificity) và chuyển kém. Nên câu hỏi rất cụ thể:

> ResNet18 có đang giữ nhầm epoch không?

## Đây là ablation về chọn checkpoint, không phải công thức mới

Giữ **nguyên vẹn** từ B1: Adam, lr 1e-4, weight decay 1e-5, 15 epoch,
patience 5, batch 32, ReduceLROnPlateau theo **AUC**, stretch, augment mạnh,
weighted CE, đúng năm fold và seed cũ.

Scheduler vẫn bám AUC là **cố ý** — nó thuộc công thức đang giữ cố định, không
thuộc quy tắc đang thử.

## Thay đổi duy nhất

```
validation AUC
→
spec@sens97 → hòa 0,005 → HSAS@97 → hòa 0,002 → NLL → epoch sớm hơn
```

Mỗi epoch lưu lại prediction mức group, nên sau này kiểm toán được quy tắc và
so sánh với lựa chọn theo AUC mà **không cần train lại**.

## Ba ensemble đăng ký trước

```
E0 = 0,50·ResNet_v4 + 0,50·DenseNet_v5     (đang khóa, 40 FP)
E1 = 0,50·ResNet_v6 + 0,50·DenseNet_v5
E2 = 0,25·ResNet_v4 + 0,25·ResNet_v6 + 0,50·DenseNet_v5
```

E2 giữ tổng trọng số họ ResNet ở 0,5 để hai ResNet không chiếm hai phần ba
ensemble. Không phải trọng số tinh chỉnh trên benchmark.

**E0 nằm trong tập ứng viên.** Nếu nó thắng trên OOF thì benchmark **không được
mở** và mô hình cũ được đóng băng."""),
    md("## Cấu hình"),
    code(patch_config(original[2])),
    code(patch_runtime(original[3])),
    md("# 1. Dữ liệu"),
    code(original[5]),
    code(original[7]),
    code(original[9]),
    code(original[11]),
    code(original[13]),
    md("# 2. Phương pháp"),
    code(original[34]),
    md("## 2.2. Tiền xử lý và augmentation"),
    code(original[36]),
    md("## 2.3. Chỉ số đánh giá"),
    code(original[38]),
    md("## 2.4. Kiến trúc"),
    code(original[40]),
    md("## 2.5. Hàm dùng chung"),
    code(original[42][:original[42].index("def run_fold(")].rstrip()),
    md("""## 2.6. Quy tắc chọn checkpoint v6

Định nghĩa sau các hàm cũ nên chúng ghi đè bản v4."""),
    code(SELECTION),
    md("## 2.7. Vòng huấn luyện"),
    code(TRAINING),
    md("# 3. Kết quả"),
    code(original[44].replace("for fold in range(len(FOLDS))]",
                              "for fold in FOLDS_TO_RUN]")
         .replace("{len(EXPERIMENTS)} thí nghiệm × {len(FOLDS)} fold)",
                  "{len(EXPERIMENTS)} thí nghiệm × {len(FOLDS_TO_RUN)} fold)")),
    code('''display(pd.DataFrame([
    {"fold": r["fold"], "epoch v6": r["best_epoch"],
     "epoch AUC cũ": r["best_auc_epoch"],
     "khác nhau": r["best_epoch"] != r["best_auc_epoch"],
     "lý do": r["selection"]["selection_reason"],
     "spec@sens97": round(r["selection"]["specificity"], 4),
     "HSAS@97": round(r["selection"]["hsas_97"], 4)}
    for r in RUNS]).set_index("fold"))

differing = sum(r["best_epoch"] != r["best_auc_epoch"] for r in RUNS)
print(f"\\n{differing}/{len(RUNS)} fold chọn epoch khác quy tắc AUC cũ")
if differing == 0:
    print("  Quy tắc mới không đổi lựa chọn nào: ResNet18 vốn đã giữ đúng epoch,")
    print("  và C2 là một null control.")'''),
    md("""## 3.2. Ba ensemble, chọn hoàn toàn bằng OOF

E0 là mô hình đang khóa và nằm trong tập ứng viên. Nếu nó thắng, benchmark
không được mở."""),
    code('''ROOT_V4 = next((p for p in [LOCAL_PROJECT_ROOT / "notebooks/results_v4",
                            Path("/kaggle/input")] if p.is_dir()), None)
ROOT_V5 = next((p for p in [LOCAL_PROJECT_ROOT / "notebooks/results_v5",
                            Path("/kaggle/input")] if p.is_dir()), None)


def pooled_oof(root, name):
    """One out-of-fold row per group for a frozen model."""
    hits = sorted(root.rglob(f"predictions_oof_{name}_groups.csv"))
    if hits:
        return pd.read_csv(hits[0])[["group_id", "label", "p_pneumonia"]]
    parts = []
    for fold in range(5):
        path = sorted(root.rglob(
            f"validation_predictions_{name}_fold{fold}.csv"))[0]
        parts.append(pd.read_csv(path, usecols=["group_id", "class_id",
                                                "p_pneumonia"]))
    pooled = pd.concat(parts, ignore_index=True)
    return (pooled.groupby("group_id", as_index=False)
            .agg(label=("class_id", "first"),
                 p_pneumonia=("p_pneumonia", "mean")))


members = {
    "resnet_v4": pooled_oof(ROOT_V4, "stretch_manh"),
    "densenet_v5": pooled_oof(ROOT_V5, "densenet121_robust"),
}
own = pd.concat([pd.DataFrame({"group_id": r["val_groups"],
                               "class_id": r["val_labels"],
                               "p_pneumonia": r["val_probs"]}) for r in RUNS])
members["resnet_v6"] = (own.groupby("group_id", as_index=False)
                        .agg(label=("class_id", "first"),
                             p_pneumonia=("p_pneumonia", "mean")))

table = None
for name, frame in members.items():
    frame = frame.rename(columns={"p_pneumonia": name})
    table = frame if table is None else table.merge(frame,
                                                    on=["group_id", "label"])
table = table.set_index("group_id").sort_index()
y = table["label"].to_numpy()
print(f"{len(y):,} group chung cho cả ba mô hình")

CANDIDATES = {
    "E0 R_v4+D": {"resnet_v4": 0.50, "densenet_v5": 0.50},
    "E1 R_v6+D": {"resnet_v6": 0.50, "densenet_v5": 0.50},
    "E2 R_v4+R_v6+D": {"resnet_v4": 0.25, "resnet_v6": 0.25,
                       "densenet_v5": 0.50},
}
rows = []
for name, weights in CANDIDATES.items():
    score = sum(w * table[m].to_numpy() for m, w in weights.items())
    specificity, threshold = specificity_at_sensitivity(y, score)
    rows.append({"ensemble": name, "n_members": len(weights),
                 "specificity": specificity, "hsas_97": hsas_97(y, score),
                 "nll": group_nll(y, score), "threshold": threshold})
candidates = pd.DataFrame(rows)
display(candidates.round(4))

top = candidates.sort_values("specificity", ascending=False)
tied = top[(top["specificity"] - top.iloc[0]["specificity"]).abs()
           < SPECIFICITY_TIE]
if len(tied) > 1:
    tied = tied.sort_values("hsas_97", ascending=False)
    tied2 = tied[(tied["hsas_97"] - tied.iloc[0]["hsas_97"]).abs() < HSAS_TIE]
    if len(tied2) > 1:
        tied2 = tied2.sort_values(["nll", "n_members"])
    winner = tied2.iloc[0]
else:
    winner = tied.iloc[0]

print(f"\\n=> CHỌN trên OOF: {winner['ensemble']}")
OPEN_BENCHMARK = not winner["ensemble"].startswith("E0")
print(f"   mở benchmark: {'CÓ' if OPEN_BENCHMARK else 'KHÔNG — E0 vẫn thắng, đóng băng mô hình cũ'}")
candidates.to_csv(WORK_DIR / "results_c2_ensembles.csv", index=False)'''),
    md("""# 4. Benchmark — chỉ khi ensemble mới thắng OOF

Nếu E0 thắng thì ô dưới không chạy gì, và mô hình đang khóa được giữ nguyên."""),
    code('''if not OPEN_BENCHMARK:
    print("E0 thắng trên OOF. Benchmark KHÔNG được mở.")
    print("Mô hình kỹ thuật cuối cùng giữ nguyên: ResNet18 v4 + DenseNet121 v5.")
else:
    raise NotImplementedError(
        "Ensemble mới thắng OOF. Khóa thành viên và ngưỡng, băm lựa chọn, "
        "rồi chấm benchmark một lần trong một patch riêng.")'''),
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
