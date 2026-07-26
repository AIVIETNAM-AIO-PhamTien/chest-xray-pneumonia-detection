"""Build notebook v5 from the frozen v4, changing only what Stage A1 needs.

v4 stays untouched. This keeps its data handling, splits, preprocessing,
metrics and evaluation verbatim, drops the forensic sections that are now out
of scope, and replaces two things: the experiment list and the rule that picks
a checkpoint.
"""

import json
import re
from pathlib import Path

SOURCE = Path("notebooks/baseline_kaggle.ipynb")
TARGET = Path("notebooks/model_improvement_v5.ipynb")

# Infrastructure kept verbatim from v4; the EDA and forensic cells are dropped.
KEEP_CODE = [2, 3, 5, 7, 9, 11, 13, 34, 36, 38, 40, 42, 44, 45, 47, 49, 51]


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "outputs": [],
            "execution_count": None,
            "source": text.strip().splitlines(keepends=True)}


source = json.load(open(SOURCE))
original = {i: "".join(c["source"]) for i, c in enumerate(source["cells"])}


def patch_config(text):
    """Point the run at one architecture and one operating-point objective."""
    block = text[text.index("ALL_EXPERIMENTS = ["):
                 text.index("EXPERIMENTS = ALL_EXPERIMENTS")]
    replacement = '''# Một kiến trúc mỗi lần. So sánh chỉ công bằng khi mọi thứ khác giữ nguyên:
# cùng manifest, cùng fold, cùng preprocessing, cùng loss như B1 của v4.
ALL_EXPERIMENTS = [
    {"name": "densenet121_robust", "arch": "densenet121", "size": 224,
     "aug": "manh", "balancing": "weighted", "resize": "stretch",
     "hoi": "DenseNet121 dưới đúng pipeline đã giúp ResNet18"},
]

SMOKE_EXPERIMENTS = [
    {"name": "smoke_densenet", "arch": "densenet121", "size": 224,
     "aug": "manh", "balancing": "weighted", "resize": "stretch",
     "hoi": "kiểm tra pipeline, không dùng để báo cáo"},
]

# Mốc cần vượt: B1 của v4, mức filename-group trên known benchmark.
BASELINE_B1 = {"auc": 0.9779, "sensitivity": 0.9951, "specificity": 0.7689,
               "tn": 173, "fp": 52, "fn": 1, "tp": 202}

'''
    text = text.replace(block, replacement)
    text = text.replace(
        'THRESHOLD_OBJECTIVE = "sensitivity"  # sensitivity | balanced_accuracy',
        'THRESHOLD_OBJECTIVE = "sensitivity"  # sensitivity | balanced_accuracy\n'
        'CHECKPOINT_TIE_MARGIN = 0.005  # chênh độ đặc hiệu dưới mức này coi như bằng nhau')
    text = text.replace("EPOCHS        = 15", "EPOCHS        = 12")
    text = text.replace("PATIENCE      = 5", "PATIENCE      = 3")
    return text


def patch_runtime(text):
    """Drop what the dropped 2x2 section owned; keep everything else."""
    text = re.sub(r"\n *# Smoke phải đi qua cả mục 3\.5.*?\n *FACTORIAL_NAMES = .*?\n",
                  "\n", text, flags=re.S)
    text = re.sub(r'\n *"factorial_names": FACTORIAL_NAMES,', "", text)
    return text


def patch_training(text):
    """Replace run_fold's selection rule; leave the rest of the cell alone."""
    start = text.index("def run_fold(")
    helper = '''def group_scores(labels, probs, groups):
    """Gộp dự đoán về mức filename-derived group."""
    frame = pd.DataFrame({"g": groups, "y": labels, "p": probs})
    rolled = frame.groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    return rolled["y"].to_numpy(), rolled["p"].to_numpy()


def specificity_at_sensitivity(labels, probs, target=TARGET_SENSITIVITY):
    """Độ đặc hiệu cao nhất còn giữ được độ nhạy tối thiểu, kèm ngưỡng.

    Đây là chỉ số dự án đang thực sự cần cải thiện. Validation AUC đã bão hòa ở
    0,999x nên chọn epoch theo nó chỉ là chọn theo nhiễu.
    """
    positive = probs[labels == 1]
    if not len(positive):
        return 0.0, 0.5
    feasible = [c for c in np.unique(probs) if (positive >= c).mean() >= target]
    if not feasible:
        return 0.0, 0.0
    threshold = float(max(feasible))
    return float((probs[labels == 0] < threshold).mean()), threshold


def group_nll(labels, probs):
    """Log-loss không trọng số ở mức group.

    Không dùng loss có trọng số lớp để đánh giá: trọng số làm lệch thang xác
    suất, nên nó không nói được mô hình hiệu chuẩn tốt hay xấu.
    """
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def group_brier(labels, probs):
    """Brier score ở mức group."""
    return float(np.mean((probs - labels) ** 2))


def better_checkpoint(candidate, incumbent, margin=CHECKPOINT_TIE_MARGIN):
    """Quy tắc chọn checkpoint, khóa trước khi chạy.

    Độ đặc hiệu là chỉ số chính. Chênh lệch dưới ``margin`` coi như bằng nhau và
    NLL quyết định, vì độ đặc hiệu ở mức group nhảy bậc rời rạc — một ca đổi
    phía đã là 0,004 — nên chênh lệch nhỏ hơn thế là nhiễu lấy mẫu.

    Args:
        candidate: Chỉ số của epoch hiện tại.
        incumbent: Chỉ số của checkpoint đang giữ, hoặc None.
        margin: Ngưỡng coi hai độ đặc hiệu là bằng nhau.

    Returns:
        True nếu nên thay checkpoint.
    """
    if incumbent is None:
        return True
    gap = candidate["specificity"] - incumbent["specificity"]
    if gap > margin:
        return True
    if gap < -margin:
        return False
    # Hòa về độ đặc hiệu: lấy NLL thấp hơn. Vẫn hòa thì giữ epoch sớm hơn.
    return candidate["nll"] < incumbent["nll"] - 1e-9


'''
    body = '''def run_fold(spec, fold_index, epochs=EPOCHS):
    log(f"\\n{'=' * 62}\\n{spec['name']}  |  fold {fold_index}\\n{'=' * 62}")
    resize = spec.get("resize", RESIZE_MODE)
    log(f"{spec['arch']} | {spec['size']}px | resize {resize} | "
        f"augment {spec['aug']} | balancing {spec['balancing']}")
    set_seed(SEED + fold_index)

    split = FOLDS[fold_index]
    loaders = make_loaders(split, seed=SEED + fold_index, mode=resize)
    try:
        model = build_model(spec["arch"], pretrained=True)
    except Exception as exc:
        raise RuntimeError("Không tải được ImageNet weights. Bật Internet trên Kaggle "
                           "hoặc tải weights vào cache trước khi chạy.") from exc
    size, aug = spec["size"], AUG_PRESETS[spec["aug"]]

    weights = class_weights_from(split) if spec["balancing"] == "weighted" else None
    log("trọng số lớp:", [round(w, 3) for w in weights.tolist()] if weights is not None
        else "không dùng")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Scheduler bám NLL không trọng số: nó phản ánh chất lượng xác suất, còn
    # AUC đã bão hòa và không còn phân biệt được epoch nào tốt hơn.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    val_rows = split[split["split"] == "val"].reset_index(drop=True)
    val_groups = val_rows["group_id"].to_numpy()

    best, best_epoch, best_state, stale, history = None, 0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in loaders["train"]:
            inputs = to_model_input(images, size, aug)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                loss = criterion(model(inputs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(loaders["train"].dataset)
        val_labels, val_probs = predict(model, loaders["val"], size)
        g_labels, g_probs = group_scores(val_labels, val_probs, val_groups)
        specificity, threshold = specificity_at_sensitivity(g_labels, g_probs)
        current = {
            "epoch": epoch, "train_loss": train_loss,
            "specificity": specificity, "threshold": threshold,
            "nll": group_nll(g_labels, g_probs),
            "brier": group_brier(g_labels, g_probs),
            "auc": roc_auc_score(g_labels, g_probs),
            "pr_auc": average_precision_score(g_labels, g_probs),
        }
        history.append(current)
        scheduler.step(current["nll"])

        marker = ""
        if better_checkpoint(current, best):
            best, best_epoch, stale = current, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            marker = "  <- best"
        else:
            stale += 1

        log(f"epoch {epoch:>2}/{epochs}  loss {train_loss:.4f}  "
            f"spec@sens{TARGET_SENSITIVITY:.0%} {specificity:.4f}  "
            f"AUC {current['auc']:.4f}  PR {current['pr_auc']:.4f}  "
            f"NLL {current['nll']:.4f}  Brier {current['brier']:.4f}  "
            f"thr {threshold:.3f}{marker}")
        if stale >= PATIENCE:
            log(f"dừng sớm ở epoch {epoch}")
            break

    model.load_state_dict(best_state)
    log(f"khôi phục checkpoint epoch {best_epoch}, "
        f"spec {best['specificity']:.4f}, NLL {best['nll']:.4f}")
    val_labels, val_probs = predict(model, loaders["val"], size)
    assert np.array_equal(val_labels, val_rows["class_id"].to_numpy())

    tag = f"{spec['name']}_fold{fold_index}"
    checkpoint_path = WORK_DIR / f"{tag}.pth"
    torch.save(best_state, checkpoint_path)
    val_rows.assign(p_pneumonia=val_probs).to_csv(
        WORK_DIR / f"validation_predictions_{tag}.csv", index=False)
    pd.DataFrame(history).to_csv(
        WORK_DIR / f"epoch_history_{tag}.csv", index=False)

    result = {
        "experiment": spec["name"], "resize": resize, "arch": spec["arch"],
        "size": spec["size"], "balancing": spec["balancing"], "fold": fold_index,
        "best_epoch": best_epoch, "checkpoint": str(checkpoint_path),
        "val_labels": val_labels, "val_probs": val_probs, "val_groups": val_groups,
        "val": metrics_at(val_labels, val_probs, 0.5),
        "selection": best,
    }
    del model, loaders, optimizer, scheduler, scaler, best_state
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()
    return result
'''
    return text[:start] + helper + body


def patch_run_table(text):
    return text.replace('"epoch tốt nhất": r["best_epoch"], "val F1": round(r["val"]["f1"], 4),\n'
                        '     "val AUC": round(r["val"]["auc"], 4),',
                        '"epoch tốt nhất": r["best_epoch"],\n'
                        '     "spec@sens97": round(r["selection"]["specificity"], 4),\n'
                        '     "group AUC": round(r["selection"]["auc"], 4),\n'
                        '     "NLL": round(r["selection"]["nll"], 4),')


cells = [
    md("""# Cải thiện mô hình — v5

**Stage A1: DenseNet121 dưới pipeline robust**

Baseline v4 đã đóng băng và không bị notebook này chạm vào. Mọi manifest, fold,
định nghĩa preprocessing và chỉ số đều lấy nguyên từ đó, nên bảng so sánh mới
đặt cạnh bảng cũ được.

---

## Mốc cần vượt

`B1 = ResNet18 + stretch + augment mạnh + weighted CE`, mức filename-group trên
known benchmark:

| | |
|---|---:|
| AUC | 0,9779 |
| độ nhạy | 99,51% |
| **độ đặc hiệu** | **76,89%** |
| TN / FP / FN / TP | 173 / 52 / 1 / 202 |

Mục tiêu: **độ đặc hiệu 82–85%** trong khi giữ độ nhạy ≥97%. Trên 225 group
NORMAL, đó là giảm từ 52 xuống 33–40 ca báo nhầm.

## Thay đổi duy nhất về phương pháp: cách chọn checkpoint

v4 chọn epoch theo validation AUC. Chỉ số đó đã bão hòa ở 0,999x nên nó không
còn phân biệt được epoch nào tốt hơn — chọn theo nó là chọn theo nhiễu.

v5 chọn theo đúng thứ đang cần cải thiện:

```
chính:    độ đặc hiệu ở mức group, tại ngưỡng giữ độ nhạy ≥97%
hòa:      NLL không trọng số ở mức group thấp hơn
vẫn hòa:  giữ epoch sớm hơn
```

Chênh lệch độ đặc hiệu dưới 0,005 coi như hòa: ở mức group, một ca đổi phía đã
là 0,004, nên nhỏ hơn thế là nhiễu lấy mẫu.

Scheduler theo dõi NLL không trọng số. Không dùng loss có trọng số lớp để đánh
giá, vì trọng số làm lệch thang xác suất.

> **Về known benchmark.** Tập test này đã được xem nhiều lần trong các giai đoạn
> trước. Số liệu ở đây dùng để so sánh và cải tiến, **không** phải ước lượng
> khái quát hóa không thiên lệch."""),
    md("## Cấu hình"),
    code(patch_config(original[2])),
    code(patch_runtime(original[3])),
    md("# 1. Dữ liệu\n\nGiữ nguyên từ v4: cùng manifest, cùng group, cùng fold."),
    code(original[5]),
    code(original[7]),
    md("## 1.2. Kiểm tra chất lượng dữ liệu"),
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
    md("""## 2.5. Vòng huấn luyện

Quy tắc chọn checkpoint nằm trong `better_checkpoint`, khóa trước khi chạy.
Mỗi epoch ghi lại đủ AUC, PR-AUC, độ đặc hiệu tại ngưỡng, NLL, Brier và ngưỡng
đã chọn, xuất ra `epoch_history_*.csv` để truy ngược được vì sao một epoch được
giữ."""),
    code(patch_training(original[42])),
    md("# 3. Kết quả"),
    code(original[44]),
    code(patch_run_table(original[45])),
    md("## 3.2. Tổng hợp OOF"),
    code(original[47]),
    md("## 3.3. Khóa cấu hình"),
    code(original[49]),
    md("""## 3.4. Known benchmark

Chỉ chạy sau khi cấu hình đã khóa bằng OOF."""),
    code(original[51]),
    md("""## 3.5. So với B1

Bảng dưới đặt DenseNet121 cạnh mốc B1 của v4. Điều kiện giữ DenseNet121, thống
nhất trước khi chạy:

- độ đặc hiệu tăng ít nhất **0,02** mà độ nhạy không xuống dưới 0,97; **hoặc**
- AUC tăng và độ đặc hiệu không thấp hơn B1 quá 0,01."""),
    code('''group_row = FINAL["group_tuned"]
predictions = (FINAL["group_probs"] >= GROUP_THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(FINAL["group_labels"], predictions,
                                  labels=[0, 1]).ravel()

comparison = pd.DataFrame([
    {"model": "B1 resnet18 (v4)", **BASELINE_B1},
    {"model": f"{BEST} (v5)", "auc": group_row["auc"],
     "sensitivity": group_row["recall"], "specificity": group_row["specificity"],
     "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
])
display(comparison.round(4))

delta_specificity = group_row["specificity"] - BASELINE_B1["specificity"]
print(f"\\nchênh độ đặc hiệu so với B1 : {delta_specificity:+.4f}")
print(f"chênh AUC so với B1        : {group_row['auc'] - BASELINE_B1['auc']:+.4f}")
print(f"ca báo nhầm tránh được     : {BASELINE_B1['fp'] - int(fp):+d}")
print(f"độ nhạy                    : {group_row['recall']:.4f}"
      f"  ({'đạt' if group_row['recall'] >= 0.97 else 'KHÔNG ĐẠT'} ngưỡng 0.97)")

keep = ((delta_specificity >= 0.02 and group_row["recall"] >= 0.97)
        or (group_row["auc"] > BASELINE_B1["auc"] and delta_specificity >= -0.01))
print(f"\\n=> {'GIỮ' if keep else 'KHÔNG GIỮ'} DenseNet121 theo tiêu chí đã đặt trước.")
if RUN_MODE == "smoke":
    print("   (smoke run — con số chỉ để kiểm tra pipeline, không dùng để quyết định)")'''),
    md("""# 4. Bước tiếp theo

Chưa triển khai trong notebook này, theo đúng thứ tự đã thống nhất:

1. DeiT-Small dưới cùng pipeline;
2. hard-negative fine-tuning trên mô hình đơn tốt nhất;
3. ensemble từ prediction đã lưu.

Mỗi bước là một lần chạy riêng, review xong mới sang bước sau."""),
]

notebook = {"cells": cells, "metadata": source.get("metadata", {}),
            "nbformat": 4, "nbformat_minor": 4}
json.dump(notebook, open(TARGET, "w"), ensure_ascii=False, indent=1)

code_cells = [c for c in cells if c["cell_type"] == "code"]
print(f"{TARGET}: {len(cells)} cell ({len(code_cells)} code)")

import ast
for index, cell in enumerate(cells):
    if cell["cell_type"] == "code":
        try:
            ast.parse("".join(cell["source"]))
        except SyntaxError as exc:
            print(f"  LỖI cell {index}: {exc}")
print("cú pháp: OK")
