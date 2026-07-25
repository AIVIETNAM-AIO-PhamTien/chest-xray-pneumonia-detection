# Chest X-Ray Pneumonia Detection

Phân loại nhị phân phát hiện viêm phổi (Pneumonia) từ ảnh X-quang ngực.
Baseline hiện tại dùng ResNet18 pretrained ImageNet; repo cũng có `SimpleCNN` để
làm mẫu cho model train-from-scratch. ResNet34 và các kiến trúc khác nằm trong
roadmap, chưa được đăng ký trong code hiện tại.

## Bài toán

- **Input**: ảnh X-quang ngực (grayscale, được convert 3 kênh cho backbone pretrained).
- **Output**: 2 lớp — `NORMAL` và `PNEUMONIA`.
- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle: `paultimothymooney/chest-xray-pneumonia`).

Dataset gốc chỉ có 16 ảnh trong `val/` (8 ảnh mỗi lớp), không đủ để đánh giá
tin cậy trong lúc train. Pipeline hiện tại **dùng trực tiếp** ba thư mục
`train/`, `val/`, `test/`; chưa tự chia lại train/val. Vì vậy validation metric
và việc chọn checkpoint của baseline chỉ mang tính sơ bộ. Lớp `PNEUMONIA`
trong tập train cũng nhiều hơn `NORMAL` đáng kể.

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
│   └── train_baseline.ipynb  # notebook chạy trên Colab/Kaggle
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

## Train trên Kaggle

Chi tiết đầy đủ nằm trong [`notebooks/train_baseline.ipynb`](notebooks/train_baseline.ipynb). Tóm tắt các bước:

1. Lấy dataset:
   - **Kaggle Notebook**: thêm dataset qua "+ Add Data", nhớ bật **Internet** trong Notebook Settings nếu cần `git clone`.
2. Clone source code và cài dependencies:
   ```bash
   git clone https://github.com/AIVIETNAM-AIO-PhamTien/chest-xray-pneumonia-detection.git
   cd chest-xray-pneumonia-detection
   pip install -r requirements.txt
   ```
3. Xác định thư mục chứa trực tiếp `train/`, `val/`, `test/`. Đường dẫn mount
   trên Kaggle có thể thay đổi theo cách dataset được thêm; hãy kiểm tra thay vì
   giả định một đường dẫn cố định. Giá trị `data.root_dir` trong config là
   placeholder cho layout local. Có thể giữ nguyên config và truyền override:
   ```python
   !python -m src.train \
       --config configs/baseline.yaml \
       --root-dir "/kaggle/input/.../chest_xray" \
       --wandb-mode disabled
   ```
4. Khi validation F1 tăng và lớn hơn `0`, weights tốt nhất được lưu tại
   `models/baseline_best.pth`.

Cũng có thể chạy trực tiếp qua CLI:

```bash
python -m src.train --config configs/baseline.yaml
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

- Validation dùng 16 ảnh gốc nên metric có phương sai lớn.
- Loss đang là `CrossEntropyLoss()` không có class weight hoặc weighted sampler.
- `set_seed()` seed Python, NumPy và PyTorch nhưng chưa bật chế độ deterministic
  đầy đủ trên GPU.
- Nếu dùng scheduler `plateau` để theo dõi validation F1, cần đặt
  `scheduler_params.mode: "max"`; mặc định của PyTorch là `"min"`.
- `freeze_backbone: true` trên ResNet18 tắt gradient của backbone, nhưng
  BatchNorm running statistics vẫn có thể thay đổi khi model ở train mode.
- `SimpleCNN` chấp nhận nhưng bỏ qua hai cờ `pretrained` và `freeze_backbone`.

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
