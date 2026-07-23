# Chest X-Ray Pneumonia Detection

Phân loại nhị phân phát hiện viêm phổi (Pneumonia) từ ảnh X-quang ngực, sử dụng transfer learning với backbone pretrained ImageNet (ResNet18/ResNet34 làm baseline).

## Bài toán

- **Input**: ảnh X-quang ngực (grayscale, được convert 3 kênh cho backbone pretrained).
- **Output**: 2 lớp — `NORMAL` và `PNEUMONIA`.
- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle: `paultimothymooney/chest-xray-pneumonia`).

Dataset gốc chỉ có 16 ảnh trong `val/`, không đủ để đánh giá tin cậy trong lúc train — pipeline tự chia lại train/val theo tỷ lệ ~85/15 (stratified) từ thư mục `train/` gốc. Lớp `PNEUMONIA` mất cân bằng nhiều hơn `NORMAL` trong tập train.

## Cấu trúc thư mục

```
module2/
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
│   ├── model.py                # build_model (transfer learning backbone)
│   ├── train.py                # training loop + CLI entrypoint
│   ├── evaluate.py             # accuracy/precision/recall/f1/confusion matrix
│   └── utils.py                 # set_seed, checkpoint, EarlyStopping
├── models/                     # checkpoint đã train (gitignored)
├── outputs/                    # log training, biểu đồ (gitignored)
└── tests/
    └── test_dataset.py         # smoke test cho transforms
```

## Cài đặt (dev/test local, Windows)

Việc train thực tế chạy trên Google Colab hoặc Kaggle Notebook (GPU miễn phí) — bước cài đặt dưới đây chỉ để viết/test code trên máy trước khi đưa lên đó.

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
.\.venv\Scripts\Activate.ps1
```

`setup.ps1` tự tạo virtual environment tại `.venv/` và cài `requirements.txt`.

## Train trên Colab / Kaggle

Chi tiết đầy đủ nằm trong [`notebooks/train_baseline.ipynb`](notebooks/train_baseline.ipynb). Tóm tắt các bước:

1. Lấy dataset:
   - **Colab**: upload `kaggle.json`, chạy `kaggle datasets download -d paultimothymooney/chest-xray-pneumonia`.
   - **Kaggle Notebook**: thêm dataset qua "+ Add Data" (có sẵn tại `/kaggle/input/chest-xray-pneumonia`), nhớ bật **Internet** trong Notebook Settings nếu cần `git clone`.
2. Clone source code và cài dependencies:
   ```bash
   git clone https://github.com/AIVIETNAM-AIO-PhamTien/chest-xray-pneumonia-detection.git
   cd chest-xray-pneumonia-detection
   pip install -r requirements.txt
   ```
3. Chỉnh `root_dir` trong `configs/baseline.yaml` cho đúng đường dẫn dataset của môi trường, rồi chạy:
   ```python
   from src.train import run_training
   run_training("configs/baseline.yaml")
   ```
4. Checkpoint tốt nhất theo F1 trên val được lưu tại `models/baseline_best.pth`.

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
  backbone: "resnet18"
  pretrained: true
  num_classes: 2
  freeze_backbone: false

train:
  epochs: 10
  lr: 0.0001
  weight_decay: 0.00001
  optimizer: "adam"
  device: "cuda"

output:
  checkpoint_dir: "models"
  log_dir: "outputs/logs"
```

## Đánh giá

`src/evaluate.py` tính accuracy, precision, recall, F1 và confusion matrix trên một `DataLoader`. Trong lúc train, checkpoint tốt nhất được chọn theo F1 trên tập val; sau khi train xong, model được đánh giá lại trên tập test.

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

- **Tiền xử lý**: CLAHE, histogram equalization, các cách resize/crop khác nhau, chuẩn hoá mean/std riêng của dataset, lung segmentation/cropping.
- **Model**: EfficientNet-B0, DenseNet121, Vision Transformer (ViT-Base), so sánh với baseline ResNet.

Chi tiết đầy đủ xem [`CLAUDE.md`](CLAUDE.md).
