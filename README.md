# Chest X-Ray Pneumonia Detection

Phân loại nhị phân phát hiện viêm phổi (Pneumonia) từ ảnh X-quang ngực.
Baseline hiện tại dùng ResNet18 pretrained ImageNet; repo cũng có `SimpleCNN` để
làm mẫu cho model train-from-scratch. ResNet34 và các kiến trúc khác nằm trong
roadmap, chưa được đăng ký trong code hiện tại.

## Bắt đầu từ đâu

| Muốn gì | Đi đâu |
|---|---|
| **Kết quả cuối, đã đóng băng** | [`artifacts/final/`](artifacts/final/) — model card, báo cáo, hạn chế, băm SHA-256 |
| Tính lại mọi con số từ prediction gốc | `python scripts/build_final_results.py` |
| Chạy baseline trên Kaggle hoặc MacBook | [`notebooks/baseline_kaggle.ipynb`](notebooks/baseline_kaggle.ipynb) — tự nhận CUDA/MPS/CPU, có smoke/full mode |
| Hiểu vì sao dataset này khó hơn vẻ ngoài | [`reports/`](reports/) — audit về đặc trưng thu nhận ảnh |
| Kiểm định ghép cặp giữa các cấu hình | `python scripts/paired_factorial_tests.py` |
| Hiểu dataset trước khi tin số | `python -m scripts.audit_dataset --root-dir <path>` |
| Biết vì sao có hai protocol split | [`src/splits.py`](src/splits.py) và [`docs/README.md`](docs/README.md) |

## Kết quả cuối

**ResNet18 + DenseNet121**, trung bình xác suất, ngưỡng 0,587268 khóa từ
out-of-fold. Đơn vị đánh giá là filename-derived group.

| | |
|---|---:|
| Độ nhạy | 0,9951 (202/203) |
| **Độ đặc hiệu** | **0,8222** (185/225) |
| ROC-AUC | 0,9792 |
| TN / FP / FN / TP | 185 / 40 / 1 / 202 |

So với ResNet18 đơn: từ 52 xuống 40 ca báo nhầm, không tăng ca bỏ sót.
McNemar p = 0,0005.

Đầy đủ ở [`artifacts/final/reports/`](artifacts/final/reports/).

## Ba cảnh báo trước khi dùng lại con số nào

**Tập test không còn nguyên vẹn.** Nó đã được đọc từ lần chạy v2 và định hướng
mọi thiết kế sau đó. Repo gọi nó là *known engineering benchmark*; mọi số trên
nó là ước lượng **lạc quan**.

**Nhãn tương quan mạnh với cách chụp ảnh.** Bảng lượng tử hóa JPEG một mình xác
định được lớp ở 96,6% ảnh development — `train/NORMAL` lưu ở quality 95,5, mọi
nhóm khác ở 75,0. Chồng lấn giữa hai lớp theo vector nhiễu chỉ **0,4%**. Hai can
thiệp tiền xử lý đều không gỡ được dấu vết này.

**Validation nội bộ gần bão hòa.** Biên độ group AUC giữa 7 cấu hình chỉ 0,0014,
trong khi trên test là 0,0290. Ba lần OOF nghiêng về phương án chuyển kém hơn;
cả ba đều bị chặn bởi ràng buộc đặt trước, không phải bởi nhìn benchmark.

Chi tiết ở [`reports/phase2a_revised_report.md`](reports/phase2a_revised_report.md)
và [`artifacts/final/reports/final_limitations.md`](artifacts/final/reports/final_limitations.md).

## Protocol split

Val gốc chỉ 16 ảnh nên không chọn checkpoint trên đó được. Hai protocol dưới đây
đều **giữ nguyên test split gốc** làm holdout; chỉ ranh giới train/val đổi:

| Protocol | Cắt val | Kết quả đo được |
|---|---|---|
| `a_paper_compatible` | mức **ảnh** | 239 bệnh nhân nằm cả hai bên train/val |
| `b_patient_grouped` | mức **bệnh nhân** | 0 — đây là default |

Chênh lệch giữa hai bên đo đúng mức mà split theo ảnh thổi phồng validation metric.
Báo cáo số chính từ protocol B; số protocol A chỉ dùng để so với literature và
phải ghi rõ nhãn.

```bash
python -m src.train --config configs/protocol_b.yaml --root-dir ../chest_xray
python -m src.train --config configs/protocol_a.yaml --root-dir ../chest_xray
```

Mỗi run ghi manifest ra `outputs/logs/<run>_manifest.csv` để truy ngược được đúng
danh sách file đứng sau từng con số.

## Bài toán

- **Input**: ảnh X-quang ngực (grayscale, được convert 3 kênh cho backbone pretrained).
- **Output**: 2 lớp — `NORMAL` và `PNEUMONIA`.
- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle: `paultimothymooney/chest-xray-pneumonia`).

Dataset gốc chỉ có 16 ảnh trong `val/` (8 ảnh mỗi lớp), không đủ để chọn
checkpoint. Notebook canonical gộp original `train+val` thành development pool,
chia lại bằng `StratifiedGroupKFold` theo filename-derived group, và chỉ đọc
test sau khi đã khóa cấu hình bằng OOF validation. Test không tham gia vào việc
chọn checkpoint, chọn cấu hình hay chọn ngưỡng.

Các transform luôn chuyển ảnh đầu vào về grayscale rồi nhân thành 3 kênh để
phù hợp với backbone pretrained, kể cả khi file nguồn đã được lưu ở chế độ RGB.

## Cấu trúc thư mục

```
chest-xray-pneumonia-detection/
├── configs/
│   └── baseline.yaml         # hyperparameters cho baseline
├── data/
│   ├── raw/                  # dataset gốc tải từ Kaggle (gitignored)
│   └── processed/            # dữ liệu đã qua tiền xử lý (gitignored)
├── notebooks/
│   ├── baseline_kaggle.ipynb # notebook canonical, chạy Kaggle/Mac
│   ├── baseline-kaggle-...   # các bản đã chạy kèm output, giữ làm bằng chứng
│   ├── result/               # train_log của từng lần chạy
│   ├── results_v4/           # CSV/JSON của lần chạy đầy đủ mới nhất
│   └── train_baseline.ipynb  # notebook CLI cũ
├── src/
│   ├── config.py              # dataclass Config + load_config từ YAML
│   ├── dataset.py             # build_dataloaders (ImageFolder train/val/test)
│   ├── transforms.py          # augmentation & preprocessing pipeline
│   ├── models/                  # model architectures (registry pattern)
│   ├── train.py                # training loop + CLI entrypoint
│   ├── evaluate.py             # accuracy/precision/recall/f1/confusion matrix
│   └── utils.py                 # set_seed, checkpoint, EarlyStopping
├── models/                     # checkpoint đã train (gitignored)
├── outputs/                    # log training, biểu đồ (gitignored)
└── tests/
    └── test_dataset.py         # smoke test cho transforms
```

## Cài đặt (dev/test local, Windows)

Việc train thực tế chạy trên Kaggle Notebook (GPU miễn phí) — bước cài đặt dưới đây chỉ để viết/test code trên máy trước khi thực hiện train.

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
.\.venv\Scripts\Activate.ps1
```

`setup.ps1` tự tạo virtual environment tại `.venv/` và cài `requirements.txt`.

## Chạy notebook trên Kaggle hoặc MacBook

Chi tiết nằm trong [`notebooks/baseline_kaggle.ipynb`](notebooks/baseline_kaggle.ipynb).

- **Kaggle:** Add Data `paultimothymooney/chest-xray-pneumonia`, bật GPU T4 và
  Internet. `RUN_MODE="auto"` chọn full; nên chạy `smoke` trước rồi mới đổi sang
  `full`.
- **MacBook:** mở notebook từ repo; nó tự tìm dataset sibling `../chest_xray`,
  chọn MPS nếu khả dụng và mặc định chạy smoke. Có thể đặt đường dẫn rõ ràng bằng
  `DATA_ROOT_OVERRIDE` hoặc biến môi trường `CXR_DATA_ROOT`.
- Full run gồm 8 thí nghiệm × 5 fold × tối đa 15 epoch, khoảng 80 phút trên T4.
  Các giai đoạn sau (DenseNet, DeiT, verifier) có notebook riêng, dựng lại bằng
  `scripts/make_notebook_*.py`.
  Dùng **Save Version → Save & Run All**; chạy interactive sẽ bị ngắt vì giới hạn
  idle 20 phút. Mac phù hợp để audit dữ liệu và smoke test.
- Smoke chạy 2 cấu hình nhỏ đi qua cả hai chế độ resize, nên nó kiểm tra được
  đường code mà chỉ full run mới dùng tới.

Output local nằm trong `artifacts/notebook_rerun/`; trên Kaggle nằm trong
`/kaggle/working`. Checkpoint `.pth` không được commit — `train_log.txt` và
`resolved_config.json` đủ để tái tạo.

```bash
python -m pip install -r requirements-notebook.txt
python -m jupyter lab notebooks/baseline_kaggle.ipynb
```

## Config baseline

`configs/baseline.yaml`:

```yaml
seed: 42

data:
  root_dir: "data/raw/chest_xray"
  image_size: 224
  batch_size: 32
  num_workers: 2

model:
  name: "resnet18"
  pretrained: true
  num_classes: 2
  freeze_backbone: false
  params: {}

train:
  epochs: 10
  lr: 0.0001
  weight_decay: 0.00001
  optimizer: "adam"
  device: "cuda"
  scheduler: "none"
  scheduler_params: {}

output:
  checkpoint_dir: "models"
  log_dir: "outputs/logs"
  run_name: null
  wandb_project: "chest-xray-pneumonia"
  wandb_entity: null
  wandb_mode: "online"
```

## Đánh giá

`src/evaluate.py` lấy dự đoán bằng `argmax` và tính accuracy, precision, recall,
F1 cùng confusion matrix trên toàn bộ `DataLoader`. Với layout hai lớp hiện
tại, `ImageFolder` ánh xạ `NORMAL -> 0`, `PNEUMONIA -> 1`; precision, recall và
F1 mặc định được tính cho lớp dương `PNEUMONIA`.

Trong lúc train, repo lưu `model.state_dict()` của epoch có validation F1 tốt
nhất. Đây là file **weights-only**, không chứa optimizer, scheduler, epoch,
config hay class mapping nên chưa hỗ trợ resume training. Sau vòng lặp, code
hiện đánh giá weights của epoch cuối (hoặc epoch kích hoạt early stopping) trên
test set; chưa load lại best weights trước khi test.

## Giới hạn baseline hiện tại

- `set_seed()` seed Python, NumPy và PyTorch nhưng chưa bật chế độ deterministic
  đầy đủ trên GPU, nên chưa tái lập bit-for-bit.
- Mới chạy một seed. Muốn kết luận kiến trúc nào hơn thì cần 3–5 seed và báo cáo
  khoảng dao động.
- Nếu dùng scheduler `plateau` để theo dõi validation F1, cần đặt
  `scheduler_params.mode: "max"`; mặc định của PyTorch là `"min"`.
- `freeze_backbone: true` trên ResNet18 tắt gradient của backbone, nhưng
  BatchNorm running statistics vẫn có thể thay đổi khi model ở train mode.
- `SimpleCNN` chấp nhận nhưng bỏ qua hai cờ `pretrained` và `freeze_backbone`.
- `protocol: "original"` vẫn validate trên 16 ảnh gốc — chỉ giữ để tái hiện
  baseline cũ, không dùng cho số đưa vào report.

## Chuẩn code

- PEP 8, format bằng `black`, sort import bằng `isort`, lint bằng `flake8`.
- Docstring Google style, type hints cho mọi function public.
- Trước khi commit:
  ```bash
  black src/ tests/
  isort src/ tests/
  flake8 src/ tests/
  pytest tests/
  ```

## Roadmap

Sau baseline, sẽ mở rộng thử nghiệm sang:

- **Dữ liệu**: tạo validation split đủ lớn từ train, ưu tiên chia theo patient
  khi có patient ID đáng tin cậy để tránh leakage.
- **Tiền xử lý**: CLAHE, histogram equalization, các cách resize/crop khác nhau,
  chuẩn hoá mean/std riêng của dataset, lung segmentation/cropping.
- **Model**: ResNet34, EfficientNet-B0, DenseNet121, Vision Transformer
  (ViT-Base), so sánh với baseline ResNet18.

Chi tiết đầy đủ xem [`CLAUDE.md`](CLAUDE.md).
