"""Build the canonical end-to-end research notebook.

The notebook is generated from code so its cell order and narrative can be
reviewed in version control without depending on stale execution state.

Run from the repository root:

    python scripts/make_notebook_research_complete.py
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "chest_xray_research_complete.ipynb"


def markdown(source: str):
    """Create one Markdown cell."""
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    """Create one code cell with no stale output or execution count."""
    return nbf.v4.new_code_cell(source.strip())


def build_notebook():
    """Return the complete notebook object."""
    cells = [
        markdown(r"""
# Phát hiện viêm phổi trên X-quang ngực: nghiên cứu end-to-end

**Notebook chuẩn duy nhất — dữ liệu → 5-fold OOF → ensemble → benchmark → Grad-CAM**

## Tóm tắt

Nghiên cứu phân loại ảnh X-quang ngực trẻ em thành `NORMAL` và `PNEUMONIA`.
Mục tiêu vận hành không phải tối đa hóa accuracy, mà là **giảm báo động giả
trong khi giữ độ nhạy ít nhất 97%**. Mô hình cuối lấy trung bình xác suất của
ResNet18 và DenseNet121, mỗi kiến trúc gồm năm checkpoint từ cross-validation
theo filename-derived group.

Notebook này thay thế các notebook rời rạc trước đây. Mọi biến được tạo theo
thứ tự tuyến tính và được kiểm tra để chạy bằng
**Restart Kernel → Run All**.

### Nguyên tắc nghiên cứu

- Official test được giữ nguyên và được gọi là **known engineering benchmark**
  vì project đã xem nó trong quá trình phát triển; không gọi là external test.
- Train/validation được chia theo group suy từ tên file, không chia ngẫu nhiên
  từng ảnh.
- Checkpoint, cấu hình và ngưỡng đều được khóa từ validation/OOF trước khi đọc
  kết quả benchmark.
- Grad-CAM là attribution hậu kiểm, không phải annotation tổn thương và không
  chứng minh quan hệ nhân quả.
"""),
        markdown(r"""
## Cách chạy

| Chế độ | Mục đích | Hành vi |
|---|---|---|
| `auto` | mặc định | đủ artifact: `reproduce`; CUDA: `full`; còn lại: `smoke` |
| `reproduce` | tái tạo report | kiểm SHA rồi đọc prediction, config và 10 checkpoint |
| `full` | huấn luyện nghiên cứu | 2 kiến trúc × 5 fold, tối đa 15/12 epoch |
| `smoke` | kiểm tra kỹ thuật | 2 kiến trúc × 1 fold × 1 epoch trên tập con |

Notebook điều phối toàn bộ nghiên cứu nhưng tái sử dụng module đã version hóa
trong `src/`; vì vậy phải chạy từ **full checkout của repo**, không chỉ upload
riêng file `.ipynb`. Trên Kaggle, clone/attach repo vào
`/kaggle/working/chest-xray-pneumonia-detection`, Add Data bộ
`paultimothymooney/chest-xray-pneumonia`, rồi bật GPU và Internet. Để chạy
local, đặt dữ liệu cạnh repo tại `../chest_xray` hoặc gán
`DATA_ROOT_OVERRIDE`.

> `reproduce` cần 10 checkpoint `.pth`, 10 CSV prediction OOF, hai CSV
> prediction benchmark, config cuối và frozen-input manifest. Một clone thiếu
> hoặc sai hash bất kỳ thành phần nào sẽ tự chuyển sang `full` hoặc `smoke`
> khi dùng `auto`.
"""),
        code(r"""
# ========================= CẤU HÌNH DUY NHẤT =========================
RUN_MODE = __import__("os").environ.get(
    "CXR_RUN_MODE", "auto"
)                               # auto | reproduce | full | smoke
DATA_ROOT_OVERRIDE = None       # ví dụ: "/content/chest_xray"
SEED = 42
TARGET_SENSITIVITY = 0.97

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
N_FOLDS = 5
PATIENCE = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
VERIFY_HASHES = True

MODEL_SPECS = {
    "resnet18": {"epochs": 15},
    "densenet121": {"epochs": 12},
}

# Smoke chỉ xác nhận đường code, không tạo số dùng trong báo cáo.
SMOKE_MAX_IMAGES_PER_CLASS = {
    "train": 128,
    "val": 64,
    "test": 64,
}
"""),
        code(r'''
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from IPython.display import display
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

warnings.filterwarnings("ignore", category=UserWarning)


def locate_project_root() -> Path:
    """Find the checkout when Jupyter starts in repo or notebooks."""
    starts = [
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd() / "chest-xray-pneumonia-detection",
    ]
    for start in starts:
        if (start / "src").is_dir() and (start / "notebooks").is_dir():
            return start.resolve()
    raise FileNotFoundError(
        "Không tìm thấy full repo chứa src/ và notebooks/. "
        "Trên Kaggle, clone repo vào /kaggle/working trước khi Run All."
    )


PROJECT_ROOT = locate_project_root()
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import find_data_root
from src.evaluation.selection import (
    better_checkpoint,
    exact_threshold_at_sensitivity,
    high_sensitivity_average_specificity,
    specificity_at_sensitivity,
)
from src.explainability import GradCAM, resolve_target_layer
from src.splits import build_manifest
from src.utils import set_seed

set_seed(SEED)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
if platform.system() == "Darwin":
    NUM_WORKERS = 0

FROZEN_V4 = PROJECT_ROOT / "notebooks/results_v4"
FROZEN_V5 = PROJECT_ROOT / "notebooks/results_v5"
FROZEN_RESNET = [
    FROZEN_V4 / f"stretch_manh_fold{fold}.pth" for fold in range(5)
]
FROZEN_DENSENET = [
    FROZEN_V5 / f"densenet121_robust_fold{fold}.pth"
    for fold in range(5)
]
FROZEN_RESNET_OOF = [
    FROZEN_V4 / f"validation_predictions_stretch_manh_fold{fold}.csv"
    for fold in range(5)
]
FROZEN_DENSENET_OOF = [
    FROZEN_V5
    / f"validation_predictions_densenet121_robust_fold{fold}.csv"
    for fold in range(5)
]
FROZEN_RESNET_TEST = (
    FROZEN_V4 / "predictions_known_benchmark_stretch_manh_images.csv"
)
FROZEN_DENSENET_TEST = (
    FROZEN_V5 / "predictions_known_benchmark_densenet121_robust_images.csv"
)
FROZEN_CONFIG = (
    PROJECT_ROOT / "artifacts/final/configs/final_model_config.json"
)
FROZEN_INPUT_MANIFEST = (
    PROJECT_ROOT / "artifacts/final/provenance/frozen_input_manifest.csv"
)
FROZEN_REQUIRED = (
    FROZEN_RESNET
    + FROZEN_DENSENET
    + FROZEN_RESNET_OOF
    + FROZEN_DENSENET_OOF
    + [FROZEN_RESNET_TEST, FROZEN_DENSENET_TEST, FROZEN_CONFIG]
)


def file_sha256(path: Path | str) -> str:
    """Hash an artifact incrementally so checkpoints do not fill RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_bundle() -> tuple[bool, list[str]]:
    """Validate presence, schema, configuration and frozen SHA-256 values."""
    issues = [
        f"thiếu {path.relative_to(PROJECT_ROOT)}"
        for path in FROZEN_REQUIRED
        if not path.is_file()
    ]
    if not FROZEN_INPUT_MANIFEST.is_file():
        issues.append("thiếu provenance/frozen_input_manifest.csv")
    if issues:
        return False, issues

    try:
        frozen_manifest = pd.read_csv(FROZEN_INPUT_MANIFEST)
        required_manifest_columns = {"path", "sha256", "size_bytes"}
        missing_columns = required_manifest_columns - set(
            frozen_manifest.columns
        )
        if missing_columns:
            issues.append(
                f"input manifest thiếu cột {sorted(missing_columns)}"
            )
            return False, issues
        if frozen_manifest["path"].duplicated().any():
            issues.append("input manifest có path trùng")

        indexed = frozen_manifest.set_index("path")
        for path in FROZEN_REQUIRED:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if relative not in indexed.index:
                issues.append(f"input manifest thiếu {relative}")
                continue
            row = indexed.loc[relative]
            if int(row["size_bytes"]) != path.stat().st_size:
                issues.append(f"sai kích thước: {relative}")
            elif str(row["sha256"]) != file_sha256(path):
                issues.append(f"sai SHA-256: {relative}")

        prediction_files = (
            FROZEN_RESNET_OOF
            + FROZEN_DENSENET_OOF
            + [FROZEN_RESNET_TEST, FROZEN_DENSENET_TEST]
        )
        prediction_columns = {
            "filename",
            "class_id",
            "group_id",
            "p_pneumonia",
        }
        for path in prediction_files:
            frame = pd.read_csv(path)
            missing = prediction_columns - set(frame.columns)
            if missing:
                issues.append(
                    f"{path.name} thiếu cột {sorted(missing)}"
                )
                continue
            probabilities = pd.to_numeric(
                frame["p_pneumonia"], errors="coerce"
            )
            if (
                probabilities.isna().any()
                or not probabilities.between(0.0, 1.0).all()
            ):
                issues.append(f"{path.name} có xác suất không hợp lệ")
            if not set(frame["class_id"].unique()).issubset({0, 1}):
                issues.append(f"{path.name} có class_id không hợp lệ")

        config = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
        model_name = str(config.get("model", "")).lower()
        if "resnet18" not in model_name or "densenet121" not in model_name:
            issues.append("config cuối không khớp hai backbone")
        threshold = float(config.get("threshold", -1.0))
        if not 0.0 <= threshold <= 1.0:
            issues.append("config cuối có threshold không hợp lệ")
        if config.get("primary_unit") != "filename-derived group":
            issues.append("config cuối không dùng filename-derived group")
    except (OSError, ValueError, KeyError, TypeError) as error:
        issues.append(f"không đọc được frozen bundle: {error}")

    return not issues, issues


if RUN_MODE in {"auto", "reproduce"}:
    HAS_FROZEN, FROZEN_ISSUES = validate_frozen_bundle()
else:
    HAS_FROZEN, FROZEN_ISSUES = False, []

if RUN_MODE == "auto":
    MODE = "reproduce" if HAS_FROZEN else (
        "full" if DEVICE.type == "cuda" else "smoke"
    )
else:
    MODE = RUN_MODE
if MODE == "reproduce" and not HAS_FROZEN:
    raise FileNotFoundError(
        "reproduce cần đủ 10 checkpoint, 10 CSV OOF, hai CSV benchmark "
        "và provenance hợp lệ. Chi tiết: " + "; ".join(FROZEN_ISSUES[:5])
    )
if MODE not in {"reproduce", "full", "smoke"}:
    raise ValueError(f"RUN_MODE không hợp lệ: {RUN_MODE!r}")

WORK_BASE = (
    Path("/kaggle/working/chest_xray_research_complete")
    if Path("/kaggle/working").is_dir()
    else PROJECT_ROOT / "artifacts/notebook_complete"
)
WORK_DIR = WORK_BASE / MODE
WORK_DIR.mkdir(parents=True, exist_ok=True)

environment = {
    "mode": MODE,
    "seed": SEED,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torchvision": __import__("torchvision").__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "sklearn": sklearn.__version__,
    "device": str(DEVICE),
}
(WORK_DIR / "environment.json").write_text(
    json.dumps(environment, indent=2), encoding="utf-8"
)
display(pd.Series(environment, name="giá trị").to_frame())
'''),
        markdown(r"""
# 1. Dữ liệu, provenance và kiểm tra chất lượng

Bộ Kermany có 5.856 ảnh X-quang ngực trẻ em 1–5 tuổi. Tập validation công bố
chỉ có 16 ảnh, nên không đủ để chọn checkpoint. Notebook gộp original
`train+val` thành development pool và chỉ dùng original `test` làm benchmark
sau khi khóa ngưỡng.

Tên file là thông tin duy nhất để suy group. Với PNEUMONIA, khóa phải chứa cả
subtype `bacteria`/`virus`; nếu chỉ lấy số sau `person`, 170 group giả sẽ bị
coi là trùng giữa các split.
"""),
        code(r"""
search_paths = []
if DATA_ROOT_OVERRIDE:
    search_paths.append(Path(DATA_ROOT_OVERRIDE))
if os.environ.get("CXR_DATA_ROOT"):
    search_paths.append(Path(os.environ["CXR_DATA_ROOT"]))
search_paths += [
    PROJECT_ROOT.parent / "chest_xray",
    PROJECT_ROOT / "data/raw",
    Path("/kaggle/input"),
    Path("/content"),
]
DATA_ROOT = find_data_root(search_paths)
manifest = build_manifest(DATA_ROOT)

if VERIFY_HASHES:
    started = time.time()
    manifest["sha256"] = [file_sha256(path) for path in manifest["path"]]
    hash_summary = {
        "unique_hashes": int(manifest["sha256"].nunique()),
        "duplicate_images": int(len(manifest) - manifest["sha256"].nunique()),
        "cross_original_split_hashes": int(
            (manifest.groupby("sha256")["split_original"].nunique() > 1).sum()
        ),
        "seconds": round(time.time() - started, 1),
    }
else:
    manifest["sha256"] = manifest["path"]
    hash_summary = {"status": "skipped"}

group_overlap = int(
    (manifest.groupby("group_id")["split_original"].nunique() > 1).sum()
)
cohort = (
    manifest.groupby(["split_original", "class_name"])
    .size()
    .unstack(fill_value=0)
    .reindex(["train", "val", "test"])
)

print("DATA_ROOT:", DATA_ROOT)
display(cohort.assign(TOTAL=cohort.sum(axis=1)))
print("Số filename-derived group:", manifest["group_id"].nunique())
print("Group nằm ở nhiều original split:", group_overlap)
print("Hash audit:", hash_summary)
assert len(manifest) == 5856, "Dataset không đúng bản Kermany 5.856 ảnh."
assert group_overlap == 0
assert hash_summary.get("cross_original_split_hashes", 0) == 0
"""),
        markdown(r"""
## 1.1. Phân bố lớp và ảnh mẫu

Accuracy dễ gây hiểu nhầm vì tỉ lệ PNEUMONIA khác nhau giữa development và
benchmark. Vì vậy phần kết quả luôn báo sensitivity, specificity, ROC-AUC,
PR-AUC và ma trận nhầm lẫn; chỉ số chính là specificity tại sensitivity ≥97%.
"""),
        code(r"""
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
cohort.plot(kind="bar", ax=axes[0], color=["#2a78d6", "#eb6834"])
axes[0].set_title("Số ảnh theo split và lớp")
axes[0].set_xlabel("")
axes[0].set_ylabel("số ảnh")
axes[0].tick_params(axis="x", rotation=0)

rng = np.random.default_rng(SEED)
for axis, class_name in zip(axes[1:], ["NORMAL", "PNEUMONIA"]):
    row = manifest[manifest["class_name"] == class_name].iloc[
        rng.integers(0, (manifest["class_name"] == class_name).sum())
    ]
    with Image.open(row["path"]) as image:
        axis.imshow(image.convert("L"), cmap="gray")
    axis.set_title(f"{class_name}\n{row['filename']}")
    axis.axis("off")
plt.tight_layout()
plt.show()
"""),
        markdown(r"""
# 2. Giao thức chia dữ liệu

`StratifiedGroupKFold` tạo năm fold từ original `train+val`; mọi ảnh của cùng
filename-derived group nằm hoàn toàn ở train hoặc validation. Original test
giữ nguyên ở cả năm manifest.

Đơn vị chính khi đánh giá là group. Xác suất group bằng trung bình xác suất các
ảnh trong group; cách này tránh cho group có nhiều ảnh trọng số lớn hơn.
"""),
        code(r'''
def make_group_folds(frame: pd.DataFrame, n_folds: int = 5) -> list[pd.DataFrame]:
    """Create leakage-resistant folds while preserving the original test."""
    pool = frame[frame["split_original"].isin(["train", "val"])].copy()
    test = frame[frame["split_original"] == "test"].copy()
    splitter = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=SEED
    )
    folds = []
    for train_index, val_index in splitter.split(
        pool, pool["class_id"], groups=pool["group_id"]
    ):
        split = frame.copy()
        split["split"] = pd.NA
        split.loc[pool.iloc[train_index].index, "split"] = "train"
        split.loc[pool.iloc[val_index].index, "split"] = "val"
        split.loc[test.index, "split"] = "test"
        leaked_groups = int(
            (split.groupby("group_id")["split"].nunique() > 1).sum()
        )
        leaked_hashes = int(
            (split.groupby("sha256")["split"].nunique() > 1).sum()
        )
        assert leaked_groups == 0 and leaked_hashes == 0
        folds.append(split)
    return folds


FOLDS = make_group_folds(manifest, N_FOLDS)
split_rows = []
for fold_index, split in enumerate(FOLDS):
    split.to_csv(WORK_DIR / f"manifest_fold{fold_index}.csv", index=False)
    for split_name in ("train", "val", "test"):
        subset = split[split["split"] == split_name]
        split_rows.append(
            {
                "fold": fold_index,
                "split": split_name,
                "images": len(subset),
                "groups": subset["group_id"].nunique(),
                "normal": int((subset["class_id"] == 0).sum()),
                "pneumonia": int((subset["class_id"] == 1).sum()),
            }
        )
display(pd.DataFrame(split_rows).set_index(["fold", "split"]))
'''),
        markdown(r"""
# 3. Tiền xử lý và mô hình

- Ảnh được chuyển grayscale rồi nhân thành ba kênh cho backbone ImageNet.
- `stretch` đưa ảnh về 224×224, đúng cấu hình của hai thành viên cuối.
- Train augmentation: lật ngang, affine mạnh (±30°, dịch 10%, scale
  0,8–1,2), và color jitter.
- Validation/test hoàn toàn tất định.
- Loss dùng trọng số lớp nghịch tần suất; không dùng thêm weighted sampler để
  tránh bù mất cân bằng hai lần.

Hai backbone được fine-tune toàn bộ. `pretrained=False` chỉ dùng trong smoke để
không phụ thuộc tải weights qua Internet.
"""),
        code(r'''
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=30, translate=(0.10, 0.10), scale=(0.80, 1.20)
        ),
        transforms.ColorJitter(brightness=0.20, contrast=0.20),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class ManifestXRayDataset(Dataset):
    """Dataset that preserves path/group metadata in a deterministic order."""

    def __init__(self, rows: pd.DataFrame, transform) -> None:
        self.rows = rows.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("L"))
        return tensor, int(row["class_id"])


def make_loader(
    rows: pd.DataFrame,
    train: bool,
    seed: int,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ManifestXRayDataset(rows, TRAIN_TRANSFORM if train else EVAL_TRANSFORM),
        batch_size=batch_size,
        shuffle=train,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        generator=generator,
    )


def build_model(architecture: str, pretrained: bool) -> nn.Module:
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif architecture == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    else:
        raise ValueError(f"Kiến trúc không hỗ trợ: {architecture}")
    return model.to(DEVICE)


display(
    pd.DataFrame(
        [
            {
                "architecture": name,
                "parameters_M": round(
                    sum(
                        parameter.numel()
                        for parameter in build_model(name, False).parameters()
                    )
                    / 1e6,
                    1,
                ),
                "full_epochs": config["epochs"],
            }
            for name, config in MODEL_SPECS.items()
        ]
    ).set_index("architecture")
)
gc.collect()
'''),
        markdown(r"""
# 4. Chỉ số và quy tắc chọn checkpoint

Với mỗi epoch, prediction validation được gộp về mức group. Thứ tự chọn
checkpoint được khóa trước:

1. specificity cao nhất tại sensitivity ≥97%;
2. nếu chênh specificity <0,005, dùng HSAS@97;
3. nếu HSAS chênh <0,002, dùng negative log-likelihood;
4. vẫn hòa thì giữ epoch sớm hơn.

HSAS@97 là specificity trung bình trong dải sensitivity 97–100%, nên nhạy với
đúng vùng ROC mà công cụ sàng lọc vận hành.
"""),
        code(r'''
@torch.no_grad()
def predict_images(model: nn.Module, rows: pd.DataFrame) -> pd.DataFrame:
    """Predict in row order and retain identifiers needed for grouping."""
    loader = make_loader(rows, train=False, seed=SEED)
    probabilities = []
    model.eval()
    for images, _labels in loader:
        logits = model(images.to(DEVICE, non_blocking=True))
        probabilities.extend(
            torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist()
        )
    return rows.reset_index(drop=True).assign(p_pneumonia=probabilities)


def to_groups(image_predictions: pd.DataFrame) -> pd.DataFrame:
    """Average image probabilities within each filename-derived group."""
    return (
        image_predictions.groupby("group_id", as_index=False)
        .agg(
            label=("class_id", "first"),
            p_pneumonia=("p_pneumonia", "mean"),
        )
        .sort_values("group_id")
        .reset_index(drop=True)
    )


def binary_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "hsas_97": float(
            high_sensitivity_average_specificity(
                labels, probabilities, TARGET_SENSITIVITY
            )
        ),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "accuracy": float((tp + tn) / max(len(labels), 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def group_nll(labels, probabilities):
    labels = np.asarray(labels, dtype=float)
    probabilities = np.clip(np.asarray(probabilities), 1e-7, 1 - 1e-7)
    return float(
        -np.mean(
            labels * np.log(probabilities)
            + (1 - labels) * np.log(1 - probabilities)
        )
    )
'''),
        markdown(r"""
# 5. Huấn luyện hoặc nạp artifact

Cell kế tiếp là điểm rẽ duy nhất:

- `reproduce`: nạp prediction/checkpoint đã đóng băng, tái tính mọi metric.
- `full`/`smoke`: huấn luyện theo đúng thứ tự; chỉ sau khi OOF ensemble và
  threshold đã khóa mới dự đoán benchmark.

Không có cell nào cần chạy thủ công ngoài thứ tự.
"""),
        code(r'''
def limit_for_smoke(rows: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Deterministic class-stratified subset used only for technical smoke."""
    maximum = SMOKE_MAX_IMAGES_PER_CLASS[split_name]
    parts = []
    for class_id in (0, 1):
        part = rows[rows["class_id"] == class_id]
        parts.append(part.sample(min(maximum, len(part)), random_state=SEED))
    return pd.concat(parts).sort_values(["class_id", "filename"])


def class_weights(rows: pd.DataFrame) -> torch.Tensor:
    counts = rows["class_id"].value_counts().reindex([0, 1]).to_numpy()
    weights = len(rows) / (2 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def train_one_fold(
    architecture: str,
    fold_index: int,
    split: pd.DataFrame,
    epochs: int,
    pretrained: bool,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    """Train one fold and return checkpoint, OOF predictions and history."""
    set_seed(SEED + fold_index)
    train_rows = split[split["split"] == "train"].copy()
    val_rows = split[split["split"] == "val"].copy()
    if MODE == "smoke":
        train_rows = limit_for_smoke(train_rows, "train")
        val_rows = limit_for_smoke(val_rows, "val")

    train_loader = make_loader(
        train_rows, train=True, seed=SEED + fold_index,
        batch_size=8 if MODE == "smoke" else BATCH_SIZE,
    )
    model = build_model(architecture, pretrained=pretrained)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=2
    )
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=DEVICE.type == "cuda"
        )
    except AttributeError:  # PyTorch 2.2 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    best_state = None
    best = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=DEVICE.type, enabled=DEVICE.type == "cuda"
            ):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)

        val_predictions = predict_images(model, val_rows)
        val_groups = to_groups(val_predictions)
        labels = val_groups["label"].to_numpy()
        probabilities = val_groups["p_pneumonia"].to_numpy()
        specificity, threshold = specificity_at_sensitivity(
            labels, probabilities, TARGET_SENSITIVITY
        )
        current = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_rows),
            "specificity": specificity,
            "threshold": threshold,
            "hsas": high_sensitivity_average_specificity(
                labels, probabilities, TARGET_SENSITIVITY
            ),
            "nll": group_nll(labels, probabilities),
            "auc": roc_auc_score(labels, probabilities),
        }
        replace, reason = better_checkpoint(current, best)
        current["selection_reason"] = reason
        history.append(current)
        scheduler.step(current["nll"])

        marker = ""
        if replace:
            best = current.copy()
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
            marker = " <- best"
        else:
            stale += 1
        print(
            f"{architecture} fold {fold_index} epoch {epoch:02d}/{epochs}: "
            f"loss={current['train_loss']:.4f}, "
            f"spec@97={specificity:.4f}, HSAS={current['hsas']:.4f}, "
            f"AUC={current['auc']:.4f}{marker}"
        )
        if stale >= PATIENCE:
            break

    checkpoint = WORK_DIR / f"{architecture}_fold{fold_index}.pth"
    torch.save(best_state, checkpoint)
    model.load_state_dict(best_state)
    oof_predictions = predict_images(model, val_rows)
    pd.DataFrame(history).to_csv(
        WORK_DIR / f"history_{architecture}_fold{fold_index}.csv", index=False
    )
    del model, optimizer, scheduler, scaler, best_state
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()
    return checkpoint, oof_predictions, pd.DataFrame(history)


def predict_checkpoint(
    architecture: str, checkpoint: Path, rows: pd.DataFrame
) -> pd.DataFrame:
    model = build_model(architecture, pretrained=False)
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    predictions = predict_images(model, rows)
    del model
    gc.collect()
    return predictions
'''),
        code(r"""
def load_frozen_images(files: list[Path]) -> pd.DataFrame:
    if not files:
        raise FileNotFoundError("Danh sách frozen prediction rỗng.")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def load_reproduction_artifacts():
    oof_images = {
        "resnet18": load_frozen_images(FROZEN_RESNET_OOF),
        "densenet121": load_frozen_images(FROZEN_DENSENET_OOF),
    }
    test_images = {
        "resnet18": pd.read_csv(FROZEN_RESNET_TEST),
        "densenet121": pd.read_csv(FROZEN_DENSENET_TEST),
    }
    checkpoints = {
        "resnet18": FROZEN_RESNET,
        "densenet121": FROZEN_DENSENET,
    }
    for architecture, frame in oof_images.items():
        assert len(frame) == 5232
        assert frame["filename"].nunique() == 5232
        assert frame["split"].eq("val").all()
    for architecture, frame in test_images.items():
        assert len(frame) == 624
        assert frame["filename"].nunique() == 624
        assert frame["split"].eq("test").all()
    return oof_images, test_images, checkpoints


if MODE == "reproduce":
    OOF_IMAGES, TEST_IMAGES, CHECKPOINTS = load_reproduction_artifacts()
else:
    fold_count = 1 if MODE == "smoke" else N_FOLDS
    epoch_overrides = {
        name: 1 if MODE == "smoke" else spec["epochs"]
        for name, spec in MODEL_SPECS.items()
    }
    CHECKPOINTS = {name: [] for name in MODEL_SPECS}
    OOF_IMAGES = {name: [] for name in MODEL_SPECS}
    for architecture in MODEL_SPECS:
        for fold_index in range(fold_count):
            checkpoint, predictions, _history = train_one_fold(
                architecture,
                fold_index,
                FOLDS[fold_index],
                epochs=epoch_overrides[architecture],
                pretrained=MODE == "full",
            )
            CHECKPOINTS[architecture].append(checkpoint)
            OOF_IMAGES[architecture].append(predictions)
        OOF_IMAGES[architecture] = pd.concat(
            OOF_IMAGES[architecture], ignore_index=True
        )

    # Khóa OOF trước khi model nhìn benchmark.
    _oof_resnet = to_groups(OOF_IMAGES["resnet18"]).rename(
        columns={"p_pneumonia": "p_resnet"}
    )
    _oof_dense = to_groups(OOF_IMAGES["densenet121"]).rename(
        columns={"p_pneumonia": "p_densenet"}
    )
    _oof_locked = _oof_resnet.merge(
        _oof_dense, on=["group_id", "label"], validate="one_to_one"
    )
    _oof_locked["p_ensemble"] = (
        _oof_locked["p_resnet"] + _oof_locked["p_densenet"]
    ) / 2
    LOCKED_THRESHOLD_BEFORE_BENCHMARK = exact_threshold_at_sensitivity(
        _oof_locked["label"],
        _oof_locked["p_ensemble"],
        TARGET_SENSITIVITY,
    )
    print("Đã khóa ngưỡng OOF trước benchmark:", LOCKED_THRESHOLD_BEFORE_BENCHMARK)

    test_rows = FOLDS[0][FOLDS[0]["split"] == "test"].copy()
    if MODE == "smoke":
        test_rows = limit_for_smoke(test_rows, "test")
    TEST_IMAGES = {}
    for architecture, checkpoints in CHECKPOINTS.items():
        member_predictions = []
        for member_index, checkpoint in enumerate(checkpoints):
            frame = predict_checkpoint(architecture, checkpoint, test_rows)
            member_predictions.append(
                frame[["path", "filename", "class_name", "class_id", "group_id"]]
                .assign(**{f"p_{member_index}": frame["p_pneumonia"]})
            )
        merged = member_predictions[0]
        for frame in member_predictions[1:]:
            merged = merged.merge(
                frame,
                on=["path", "filename", "class_name", "class_id", "group_id"],
                validate="one_to_one",
            )
        probability_columns = [
            column for column in merged if column.startswith("p_")
        ]
        TEST_IMAGES[architecture] = merged.assign(
            p_pneumonia=merged[probability_columns].mean(axis=1)
        )

print(
    f"MODE={MODE}; OOF images: "
    + ", ".join(f"{key}={len(value):,}" for key, value in OOF_IMAGES.items())
)
print(
    "checkpoints: "
    + ", ".join(f"{key}={len(value)}" for key, value in CHECKPOINTS.items())
)
"""),
        markdown(r"""
# 6. Khóa ngưỡng OOF và đánh giá benchmark

Xác suất của hai kiến trúc được gộp riêng ở mức group, sau đó lấy trung bình
đều. Ngưỡng final là ngưỡng quan sát cao nhất vẫn giữ sensitivity ≥97% trên
pooled OOF.

Mỗi mô hình thành viên cũng được chấm bằng ngưỡng OOF riêng để bảng so sánh
không dùng benchmark labels khi chọn điểm vận hành.
"""),
        code(r"""
def merge_model_groups(image_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    resnet = to_groups(image_frames["resnet18"]).rename(
        columns={"p_pneumonia": "p_resnet"}
    )
    densenet = to_groups(image_frames["densenet121"]).rename(
        columns={"p_pneumonia": "p_densenet"}
    )
    merged = resnet.merge(
        densenet, on=["group_id", "label"], validate="one_to_one"
    )
    merged["p_ensemble"] = (
        merged["p_resnet"] + merged["p_densenet"]
    ) / 2
    return merged


OOF_GROUPS = merge_model_groups(OOF_IMAGES)
TEST_GROUPS = merge_model_groups(TEST_IMAGES)

probability_columns = {
    "ResNet18": "p_resnet",
    "DenseNet121": "p_densenet",
    "R+D ensemble": "p_ensemble",
}
THRESHOLDS = {
    name: exact_threshold_at_sensitivity(
        OOF_GROUPS["label"],
        OOF_GROUPS[column],
        TARGET_SENSITIVITY,
    )
    for name, column in probability_columns.items()
}

rows = []
for name, column in probability_columns.items():
    metrics = binary_metrics(
        TEST_GROUPS["label"], TEST_GROUPS[column], THRESHOLDS[name]
    )
    rows.append({"model": name, **metrics})
RESULTS = pd.DataFrame(rows).set_index("model")
display(
    RESULTS[
        [
            "roc_auc",
            "pr_auc",
            "hsas_97",
            "sensitivity",
            "specificity",
            "tn",
            "fp",
            "fn",
            "tp",
            "threshold",
        ]
    ].round(4)
)

FINAL_THRESHOLD = THRESHOLDS["R+D ensemble"]
OOF_GROUPS.assign(
    pred=(OOF_GROUPS["p_ensemble"] >= FINAL_THRESHOLD).astype(int)
).to_csv(WORK_DIR / "predictions_oof_final_groups.csv", index=False)
TEST_GROUPS.assign(
    pred=(TEST_GROUPS["p_ensemble"] >= FINAL_THRESHOLD).astype(int)
).to_csv(WORK_DIR / "predictions_benchmark_final_groups.csv", index=False)
RESULTS.to_csv(WORK_DIR / "results_group_level.csv")

if MODE == "reproduce":
    frozen_config = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    assert abs(
        FINAL_THRESHOLD - float(frozen_config["threshold"])
    ) < 1e-12, "Ngưỡng tính lại không khớp config đã khóa."
    expected = {
        "sensitivity": 202 / 203,
        "specificity": 185 / 225,
        "fp": 40,
        "fn": 1,
    }
    final = RESULTS.loc["R+D ensemble"]
    assert abs(final["sensitivity"] - expected["sensitivity"]) < 1e-12
    assert abs(final["specificity"] - expected["specificity"]) < 1e-12
    assert int(final["fp"]) == expected["fp"]
    assert int(final["fn"]) == expected["fn"]
    print("✓ Khớp artifact cuối: sensitivity 202/203, specificity 185/225.")
"""),
        code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for name, column in probability_columns.items():
    fpr, tpr, _ = roc_curve(TEST_GROUPS["label"], TEST_GROUPS[column])
    axes[0].plot(
        fpr,
        tpr,
        label=f"{name} (AUC={RESULTS.loc[name, 'roc_auc']:.3f})",
    )
axes[0].axhline(TARGET_SENSITIVITY, color="gray", linestyle="--", linewidth=1)
axes[0].set_xlim(0, 0.65)
axes[0].set_ylim(0.90, 1.005)
axes[0].set_xlabel("False-positive rate")
axes[0].set_ylabel("Sensitivity")
axes[0].set_title("ROC trong vùng vận hành")
axes[0].legend(fontsize=8)

final_predictions = (
    TEST_GROUPS["p_ensemble"].to_numpy() >= FINAL_THRESHOLD
).astype(int)
matrix = confusion_matrix(TEST_GROUPS["label"], final_predictions)
image = axes[1].imshow(matrix, cmap="Blues")
for row in range(2):
    for column in range(2):
        axes[1].text(column, row, str(matrix[row, column]), ha="center", va="center")
axes[1].set_xticks([0, 1], ["NORMAL", "PNEUMONIA"])
axes[1].set_yticks([0, 1], ["NORMAL", "PNEUMONIA"])
axes[1].set_xlabel("Dự đoán")
axes[1].set_ylabel("Nhãn thật")
axes[1].set_title(f"Ensemble group-level (t={FINAL_THRESHOLD:.3f})")
fig.colorbar(image, ax=axes[1], fraction=0.046)
plt.tight_layout()
plt.show()
"""),
        markdown(r"""
# 7. Phân tích lỗi

Bảng dưới lấy các group gần ngưỡng nhất trong từng ô TN/FP/FN/TP. Đây là quy
tắc xác định trước và cũng là nguồn ca cho Grad-CAM; không chọn ảnh vì heatmap
trông đẹp.

Nếu group có nhiều ảnh, ảnh đại diện là ảnh có xác suất ensemble gần xác suất
trung bình của group nhất.
"""),
        code(r"""
def merged_test_images() -> pd.DataFrame:
    keys = ["path", "filename", "class_name", "class_id", "group_id"]
    resnet = TEST_IMAGES["resnet18"][keys + ["p_pneumonia"]].rename(
        columns={"p_pneumonia": "p_resnet"}
    )
    dense = TEST_IMAGES["densenet121"][keys + ["p_pneumonia"]].rename(
        columns={"p_pneumonia": "p_densenet"}
    )
    images = resnet.merge(dense, on=keys, validate="one_to_one")
    images["p_ensemble_image"] = (
        images["p_resnet"] + images["p_densenet"]
    ) / 2
    return images


def select_boundary_cases():
    groups = TEST_GROUPS.copy()
    groups["pred"] = (groups["p_ensemble"] >= FINAL_THRESHOLD).astype(int)
    images = merged_test_images()
    definitions = [("TN", 0, 0), ("FP", 0, 1), ("FN", 1, 0), ("TP", 1, 1)]
    selected = []
    for category, label, prediction in definitions:
        candidates = groups[
            (groups["label"] == label) & (groups["pred"] == prediction)
        ].copy()
        if candidates.empty:
            print(f"Smoke không có ô {category}; bỏ qua ô này.")
            continue
        candidates["distance_to_threshold"] = (
            candidates["p_ensemble"] - FINAL_THRESHOLD
        ).abs()
        group = candidates.sort_values(
            ["distance_to_threshold", "group_id"]
        ).iloc[0]
        group_images = images[images["group_id"] == group["group_id"]].copy()
        group_images["distance_to_mean"] = (
            group_images["p_ensemble_image"] - group["p_ensemble"]
        ).abs()
        image = group_images.sort_values(
            ["distance_to_mean", "filename"]
        ).iloc[0]
        selected.append(
            {
                "category": category,
                "group_id": group["group_id"],
                "filename": image["filename"],
                "path": image["path"],
                "class_name": image["class_name"],
                "label": int(label),
                "prediction": int(prediction),
                "p_ensemble_group": float(group["p_ensemble"]),
                "p_ensemble_image": float(image["p_ensemble_image"]),
                "threshold": float(FINAL_THRESHOLD),
            }
        )
    return pd.DataFrame(selected)


XAI_CASES = select_boundary_cases()
display(XAI_CASES)
XAI_CASES.to_csv(WORK_DIR / "gradcam_case_manifest.csv", index=False)
"""),
        markdown(r"""
# 8. Grad-CAM đồng thuận qua các checkpoint

Với logit PNEUMONIA \(s^{(c)}\) và feature map \(A^k\):

\[
\alpha_k^{(c)}=\frac{1}{Z}\sum_i\sum_j
\frac{\partial s^{(c)}}{\partial A_{ij}^k},\qquad
L^{(c)}=\mathrm{ReLU}\left(\sum_k\alpha_k^{(c)}A^k\right).
\]

Notebook luôn giải thích logit PNEUMONIA cho mọi ca. Target layer được khai báo
tường minh:

- ResNet18: `layer4[-1]`
- DenseNet121: `features.denseblock4`

CAM được tính riêng cho từng checkpoint rồi lấy **trung vị trong từng họ kiến
trúc**. Hai hàng không được trộn với nhau: đây là đồng thuận hậu nghiệm của các
thành viên, không phải gradient chính xác của phép trung bình xác suất ensemble.
"""),
        code(r"""
def load_xai_batch(cases: pd.DataFrame):
    originals, tensors = [], []
    for row in cases.itertuples():
        path = Path(row.path)
        if not path.is_file():
            path = DATA_ROOT / "test" / row.class_name / row.filename
        with Image.open(path) as image:
            gray = np.asarray(
                image.convert("L").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
        originals.append(gray)
        tensor = torch.from_numpy(gray).unsqueeze(0).repeat(3, 1, 1)
        tensors.append(tensor)
    batch = torch.stack(tensors)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return originals, ((batch - mean) / std).to(DEVICE)


def consensus_cams(architecture: str, checkpoints: list[Path], inputs):
    member_maps = []
    member_rows = []
    for member_index, checkpoint in enumerate(checkpoints):
        model = build_model(architecture, pretrained=False)
        state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        target_layer = resolve_target_layer(model, architecture)
        with GradCAM(model, target_layer) as explainer:
            maps, logits = explainer(inputs, target_class=1)
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        maps = maps.cpu().numpy()
        member_maps.append(maps)
        for case, probability in zip(XAI_CASES.itertuples(), probabilities):
            member_rows.append(
                {
                    "architecture": architecture,
                    "member": member_index,
                    "checkpoint": checkpoint.name,
                    "category": case.category,
                    "filename": case.filename,
                    "p_pneumonia": float(probability),
                }
            )
        del model, logits
        gc.collect()
    return np.stack(member_maps), member_rows


ORIGINALS, XAI_INPUTS = load_xai_batch(XAI_CASES)
RESNET_CAMS, resnet_member_rows = consensus_cams(
    "resnet18", CHECKPOINTS["resnet18"], XAI_INPUTS
)
DENSENET_CAMS, dense_member_rows = consensus_cams(
    "densenet121", CHECKPOINTS["densenet121"], XAI_INPUTS
)

consensus = {
    "ResNet18 — trung vị member": np.median(RESNET_CAMS, axis=0),
    "DenseNet121 — trung vị member": np.median(DENSENET_CAMS, axis=0),
}
fig, axes = plt.subplots(
    3,
    len(XAI_CASES),
    figsize=(3.1 * len(XAI_CASES), 8.2),
    squeeze=False,
    constrained_layout=True,
)
heatmap = plt.colormaps["jet"]
for column, (case, original) in enumerate(
    zip(XAI_CASES.itertuples(), ORIGINALS)
):
    axes[0, column].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0, column].set_title(
        f"{case.category} | {case.class_name}\n"
        f"$p_g$={case.p_ensemble_group:.3f}, $t$={case.threshold:.3f}"
    )
    for row, (name, maps) in enumerate(consensus.items(), start=1):
        coloured = heatmap(maps[column])[..., :3]
        base = np.repeat(original[..., None], 3, axis=2)
        axes[row, column].imshow(np.clip(0.55 * base + 0.45 * coloured, 0, 1))
        if column == 0:
            axes[row, column].set_ylabel(name)
    for row in range(3):
        axes[row, column].set_xticks([])
        axes[row, column].set_yticks([])
axes[0, 0].set_ylabel("Ảnh model nhìn thấy\n(stretch 224×224)")

XAI_FIGURE = WORK_DIR / "fig_gradcam_consensus.png"
fig.savefig(XAI_FIGURE, dpi=180, bbox_inches="tight")
plt.show()
pd.DataFrame(resnet_member_rows + dense_member_rows).to_csv(
    WORK_DIR / "gradcam_member_predictions.csv", index=False
)
print("Đã lưu:", XAI_FIGURE)
"""),
        markdown(r"""
# 9. Diễn giải, hạn chế và kết luận

## Cách đọc Grad-CAM

- Màu nóng nghĩa là vùng có đóng góp dương lớn hơn đối với logit PNEUMONIA.
- Sự khác nhau giữa hai hàng cho thấy attribution phụ thuộc kiến trúc.
- Không có lung mask, lesion annotation hay đánh giá của bác sĩ; vì vậy không
  được viết “mô hình nhìn đúng vùng bệnh”.
- Ca sát ngưỡng phù hợp để phân tích quyết định nhưng không đại diện cho toàn bộ
  quần thể. Muốn định lượng cần lấy mẫu lớn, khóa protocol và so sánh với mask
  giải phẫu/annotation độc lập.

## Hạn chế nghiên cứu

1. Benchmark đã được xem nhiều lần; metric có thể lạc quan.
2. Group suy từ filename, không phải patient ID chính thức.
3. Dữ liệu một trung tâm, chỉ gồm trẻ 1–5 tuổi; không suy rộng sang người lớn.
4. Weighted cross-entropy làm thang probability chưa được calibration lâm sàng.
5. Không có external validation.
6. Checkpoint chỉ là `state_dict`; muốn triển khai phải đóng gói thêm class
   mapping, preprocessing, threshold và model card.

## Kết luận

Ở artifact đóng băng, ensemble ResNet18 + DenseNet121 giữ 202/203 group
PNEUMONIA và nhận đúng 185/225 group NORMAL, tương ứng sensitivity 99,51% và
specificity 82,22%. Grad-CAM bổ sung một lớp audit cho từng họ backbone nhưng
không thay thế external validation hoặc xác minh lâm sàng.

## Tài liệu nền

1. Kermany et al., “Identifying Medical Diagnoses and Treatable Diseases by
   Image-Based Deep Learning,” *Cell*, 2018.
   https://doi.org/10.1016/j.cell.2018.02.010
2. He et al., “Deep Residual Learning for Image Recognition,” CVPR 2016.
   https://doi.org/10.1109/CVPR.2016.90
3. Huang et al., “Densely Connected Convolutional Networks,” CVPR 2017.
   https://doi.org/10.1109/CVPR.2017.243
4. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via
   Gradient-Based Localization,” ICCV 2017.
   https://doi.org/10.1109/ICCV.2017.74
"""),
        code(r"""
artifact_summary = []
for path in sorted(WORK_DIR.glob("*")):
    if path.is_file():
        artifact_summary.append(
            {
                "file": path.name,
                "size_kb": round(path.stat().st_size / 1024, 1),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            }
        )
display(pd.DataFrame(artifact_summary))

print("\nHoàn tất.")
print("Mode:", MODE)
print("Artifact directory:", WORK_DIR)
print(
    "Nhãn kết quả:",
    "reproduced_filename_grouped_known_benchmark"
    if MODE == "reproduce"
    else ("new_full_run" if MODE == "full" else "smoke_not_for_reporting"),
)
"""),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
        },
    )
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"research-{index:03d}"
    return notebook


def main() -> None:
    """Write the canonical notebook with normalized nbformat metadata."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), OUTPUT)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
