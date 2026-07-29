# Chest X-Ray Pneumonia Detection

Phân loại nhị phân phát hiện viêm phổi (Pneumonia) từ ảnh X-quang ngực.
Pipeline nghiên cứu cuối ghép ResNet18 + DenseNet121 qua 5 fold mỗi kiến trúc;
repo vẫn giữ `SimpleCNN` và CLI ResNet18 để tái hiện baseline lịch sử.

## Bắt đầu từ đâu

| Muốn gì | Đi đâu |
|---|---|
| **Kết quả cuối và provenance** | [`artifacts/final/`](artifacts/final/) — model card, báo cáo, hạn chế, băm SHA-256 |
| Dựng lại hình/XAI rồi khóa số và provenance | `python scripts/build_report_figures.py` → `python scripts/build_final_results.py` |
| **Chạy toàn bộ nghiên cứu + Grad-CAM** | [`notebooks/chest_xray_research_complete.ipynb`](notebooks/chest_xray_research_complete.ipynb) — một luồng `reproduce`/`full`/`smoke` |
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
│   ├── baseline.yaml          # hyperparameters cho baseline lịch sử
│   ├── protocol_a.yaml        # split image-level (a_paper_compatible)
│   └── protocol_b.yaml        # split patient-grouped (b_patient_grouped, default)
├── data/
│   ├── raw/                   # dataset gốc tải từ Kaggle (gitignored)
│   └── processed/             # dữ liệu đã qua tiền xử lý (gitignored)
├── notebooks/
│   ├── chest_xray_research_complete.ipynb # notebook canonical end-to-end
│   ├── legacy/                # các bản notebook lịch sử (v2-v6, stage-b, deit...),
│   │                          # không cần chạy lại — xem README.md ở gốc repo
│   └── results_*/             # CSV/JSON theo từng giai đoạn thí nghiệm (không có
│                              # thư mục nào là "mới nhất" duy nhất; xem mtime/tên)
├── src/
│   ├── config.py              # dataclass Config + load_config từ YAML
│   ├── dataset.py              # pipeline canonical: find_data_root, build_dataloaders
│   │                          # (ImageFolder), compute_class_weights — không đổi số
│   ├── transforms.py           # augmentation & preprocessing của pipeline canonical
│   ├── splits.py                # build_manifest/make_splits (protocol A/B), dùng bởi
│   │                          # cả notebook canonical và src/train.py
│   ├── data/                    # pipeline riêng cho CLI baseline của src/train.py
│   │                          # (CXRDataset, build_loaders, transforms, imbalance) —
│   │                          # xem src/data/README.md
│   ├── evaluation/               # calibration, label_shift, nuisance_features, selection
│   ├── models/                    # model architectures (registry pattern)
│   ├── explainability/             # Grad-CAM dùng chung cho notebook/report
│   ├── train.py                 # training loop CLI baseline (không dùng cho số report)
│   ├── evaluate.py              # accuracy/precision/recall/f1/confusion matrix
│   └── utils.py                  # set_seed, checkpoint, EarlyStopping
├── scripts/                        # audit & phân tích (jpeg_encoding_audit, paired_
│                                 # factorial_tests, build_final_results, ...);
│                                 # make_notebook_{v5,c2,stage_b,deit}.py là generator
│                                 # tạo các notebook lịch sử trong notebooks/legacy/,
│                                 # không cần chạy lại để tái lập report
├── reports/                         # báo cáo theo từng giai đoạn — xem reports/README.md
│                                 # cho thứ tự đọc; đã bị artifacts/final/ thay thế
├── artifacts/final/                 # kết quả cuối, đóng băng + SHA-256 (canonical)
├── docs/                            # tài liệu tổng hợp — xem docs/README.md
├── clinical_review_package/          # gói đọc phim mù — xem clinical_review_package/README.md
├── models/                         # checkpoint đã train (gitignored)
├── outputs/                         # log training, biểu đồ (gitignored)
└── tests/                          # 1 file test theo mỗi module trong src/
```

## Cài đặt (dev/test local, Windows)

Việc train thực tế chạy trên Kaggle Notebook (GPU miễn phí) — bước cài đặt dưới đây chỉ để viết/test code trên máy trước khi thực hiện train.

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
.\.venv\Scripts\Activate.ps1
```

`setup.ps1` tự tạo virtual environment tại `.venv/` và cài `requirements.txt`.

## Chạy notebook hợp nhất trên Kaggle hoặc MacBook

Chi tiết nằm trong
[`notebooks/chest_xray_research_complete.ipynb`](notebooks/chest_xray_research_complete.ipynb).

- **Kaggle:** notebook dùng các module trong `src`, nên phải clone/attach
  **toàn repo** vào `/kaggle/working/chest-xray-pneumonia-detection`, không chỉ
  upload riêng `.ipynb`. Sau đó Add Data
  `paultimothymooney/chest-xray-pneumonia`, bật GPU T4 và Internet. `auto` dùng
  bộ artifact đóng băng nếu có đủ và đúng hash; nếu không, CUDA chọn `full`.
  Nên chạy `CXR_RUN_MODE=smoke` trước khi chạy full.
- **MacBook:** mở notebook từ repo; nó tự tìm dataset sibling `../chest_xray`,
  chọn MPS nếu khả dụng. Khi có đủ bộ artifact local, `auto` chọn `reproduce`;
  clone mới thiếu artifact sẽ chạy `smoke`. Có thể đặt đường dẫn bằng
  `DATA_ROOT_OVERRIDE` hoặc biến môi trường `CXR_DATA_ROOT`.
- Full run gồm ResNet18 + DenseNet121 × 5 fold. Notebook khóa ngưỡng bằng pooled
  OOF trước khi đọc known benchmark, rồi chạy error analysis và Grad-CAM.
- `reproduce` tái tính đúng 202/203 sensitivity và 185/225 specificity từ
  artifact đóng băng, đồng thời kiểm tra bằng assertion.

Output local nằm trong `artifacts/notebook_complete/<mode>/`; trên Kaggle nằm
trong `/kaggle/working/chest_xray_research_complete/<mode>/`. Chế độ
`reproduce` cần 10 checkpoint, prediction OOF/benchmark, config cuối và
frozen-input manifest; notebook kiểm SHA-256 trước khi sử dụng. Checkpoint
`.pth` không được commit. Một clone mới phải train lại hoặc được cung cấp trọn
bộ artifact để tái lập Grad-CAM của mô hình cuối.

```bash
python -m pip install -r requirements-notebook.txt
python -m jupyter lab notebooks/chest_xray_research_complete.ipynb
```

Nếu bắt đầu bằng một Kaggle notebook trống, cell bootstrap tối thiểu là:

```python
!git clone https://github.com/AIVIETNAM-AIO-PhamTien/chest-xray-pneumonia-detection.git /kaggle/working/chest-xray-pneumonia-detection
```

Sau khi clone, chạy notebook chuẩn nằm trong checkout đó; branch/commit phải
chứa cùng generator và module `src/explainability` với bản notebook.

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
config hay class mapping nên chưa hỗ trợ resume training. Epoch đầu luôn tạo
được checkpoint, kể cả khi F1 bằng 0; sau vòng lặp, code nạp lại đúng weights
có validation F1 tốt nhất rồi mới đánh giá test.

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

## Hướng tiếp theo sau pipeline cuối

DenseNet121 và DeiT đã được thử; mô hình cuối hiện ghép ResNet18 + DenseNet121.
Các bước còn giá trị nghiên cứu là:

- xác nhận trên một cohort ngoài thật sự chưa được xem và có metadata bệnh
  nhân đáng tin cậy;
- lặp lại nhiều seed, báo khoảng dao động thay vì chỉ một seed;
- đánh giá calibration, subgroup và lung segmentation/cropping dưới một giao
  thức khóa trước;
- thực hiện đọc phim mù cho các ca báo nhầm trước khi diễn giải lâm sàng.

Chi tiết đầy đủ xem [`CLAUDE.md`](CLAUDE.md).
