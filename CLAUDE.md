# CLAUDE.md

## Tổng quan project

- **Bài toán**: Phân loại nhị phân (binary classification) phát hiện viêm phổi (Pneumonia) từ ảnh X-quang ngực, 2 lớp: `NORMAL` và `PNEUMONIA`.
- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) trên Kaggle (`paultimothymooney/chest-xray-pneumonia`).
- **Giai đoạn hiện tại**: xây baseline bằng transfer learning (ResNet18/ResNet34 pretrained ImageNet). Sau baseline sẽ mở rộng sang nhiều kỹ thuật tiền xử lý ảnh (CLAHE, histogram equalization, các chiến lược resize/crop, chuẩn hoá khác nhau...) và nhiều kiến trúc model khác (EfficientNet, DenseNet121, Vision Transformer...) để so sánh.
- **Môi trường train**: Google Colab hoặc Kaggle Notebook (dùng GPU miễn phí), không train local.

## Dataset

- Nguồn: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Cấu trúc gốc sau khi giải nén, đặt tại `data/raw/chest_xray/`:
  ```
  chest_xray/
    train/{NORMAL,PNEUMONIA}/
    val/{NORMAL,PNEUMONIA}/
    test/{NORMAL,PNEUMONIA}/
  ```
- **Lưu ý quan trọng**: tập `val/` gốc của dataset này chỉ có 16 ảnh (8 mỗi lớp) — không đủ để đánh giá tin cậy trong lúc training. Nên tự chia lại train/val (ví dụ stratified split ~85/15 từ thư mục `train/`) thay vì dùng `val/` gốc, và ghi rõ trong code/notebook nếu làm vậy.
- Lớp mất cân bằng: `PNEUMONIA` nhiều hơn `NORMAL` đáng kể trong tập train — cân nhắc class weighting hoặc weighted sampler khi tính loss.
- Ảnh X-quang là grayscale; khi dùng backbone pretrained ImageNet (input 3 kênh), cần convert sang 3 kênh (xem `src/transforms.py`).

## Cấu trúc thư mục

```
module2/
├── CLAUDE.md
├── requirements.txt
├── setup.ps1                 # tự tạo .venv + cài requirements (local dev, Windows)
├── .gitignore
├── configs/
│   └── baseline.yaml         # hyperparameters cho baseline
├── data/
│   ├── raw/                  # dataset gốc tải từ Kaggle (gitignored)
│   └── processed/            # dữ liệu đã qua tiền xử lý, nếu cache lại (gitignored)
├── notebooks/
│   └── train_baseline.ipynb  # notebook chạy trên Colab/Kaggle
├── src/
│   ├── __init__.py
│   ├── config.py              # dataclass Config + load_config từ YAML
│   ├── dataset.py             # build_dataloaders (ImageFolder train/val/test)
│   ├── transforms.py          # augmentation & preprocessing pipeline
│   ├── models/                 # model architectures (registry pattern)
│   │   ├── registry.py           # register_model decorator + build_model
│   │   ├── resnet18.py            # transfer-learning ResNet-18
│   │   └── simple_cnn.py          # example from-scratch CNN (template)
│   ├── train.py                # training loop + CLI entrypoint
│   ├── evaluate.py             # accuracy/precision/recall/f1/confusion matrix
│   └── utils.py                 # set_seed, checkpoint, EarlyStopping
├── models/                     # checkpoint đã train (gitignored)
├── outputs/
│   ├── logs/                   # log training (gitignored)
│   └── figures/                # biểu đồ, confusion matrix... (gitignored)
└── tests/
    └── test_dataset.py         # smoke test cho transforms
```

## Setup môi trường local (Windows)

Dùng để dev/test code trên máy trước khi đưa lên Colab/Kaggle (Colab/Kaggle không cần bước này vì đã là môi trường cách ly sẵn).

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Script `setup.ps1` sẽ tự tạo virtual environment tại `.venv/` (nếu chưa có) và cài `requirements.txt` vào đó. Sau khi chạy xong, kích hoạt môi trường cho phiên PowerShell hiện tại bằng:

```powershell
.\.venv\Scripts\Activate.ps1
```

`.venv/` đã được thêm vào `.gitignore`, không commit lên git.

## Chuẩn code Python

- Tuân thủ **PEP 8**. Format bằng `black` (line length mặc định 88), sắp xếp import bằng `isort`, lint bằng `flake8`.
- **Docstring theo Google style** cho mọi module, class, function public — xem ví dụ trong `src/`. Bắt buộc có `Args`, `Returns`, và `Raises` (nếu có) khi function nhận tham số hoặc trả về giá trị.
- Dùng **type hints** cho tham số và giá trị trả về của mọi function.
- Quy ước đặt tên: `snake_case` cho biến/hàm, `PascalCase` cho class, `UPPER_SNAKE_CASE` cho hằng số.
- Không dùng mutable global state; truyền config qua tham số hoặc dataclass (`src/config.py`).
- Mỗi module trong `src/` phải chạy độc lập được (qua CLI nếu có `__main__`) và import được từ notebook.
- Trước khi commit, chạy:
  ```
  black src/ tests/
  isort src/ tests/
  flake8 src/ tests/
  pytest tests/
  ```

## Quy trình train trên Colab / Kaggle

Chi tiết từng bước nằm trong `notebooks/train_baseline.ipynb`. Tóm tắt:

1. Phát hiện môi trường (Colab hay Kaggle) để trỏ đúng đường dẫn dataset.
2. Lấy dataset:
   - **Colab**: upload `kaggle.json` (Kaggle → Account → Create New API Token), chạy `kaggle datasets download -d paultimothymooney/chest-xray-pneumonia`.
   - **Kaggle Notebook**: thêm dataset qua nút "+ Add Data", dữ liệu có sẵn tại `/kaggle/input/chest-xray-pneumonia`.
3. Lấy source code bằng `git clone` từ GitHub:
   ```
   !git clone https://github.com/AIVIETNAM-AIO-PhamTien/chest-xray-pneumonia-detection.git
   %cd chest-xray-pneumonia-detection
   ```
   Trên **Kaggle Notebook**, phải bật **Internet** ở Notebook Settings (sidebar bên phải) trước khi chạy `git clone`, nếu không sẽ báo lỗi kết nối. Colab đã có Internet sẵn.
4. `pip install -r requirements.txt`.
5. Ghi đè `root_dir` trong `configs/baseline.yaml` bằng đường dẫn dataset đúng môi trường, rồi chạy:
   ```python
   from src.train import run_training
   run_training("configs/baseline.yaml")
   ```
6. Checkpoint tốt nhất theo F1 trên val được lưu vào `models/baseline_best.pth`; tải về máy nếu cần (Colab: `files.download(...)`).

## Roadmap thử nghiệm (sau baseline)

- **Tiền xử lý**: CLAHE, histogram equalization, các cách resize/crop khác nhau, chuẩn hoá theo mean/std riêng của dataset thay vì ImageNet, lung segmentation/cropping.
- **Model**: EfficientNet-B0, DenseNet121, Vision Transformer (ViT-Base), so sánh với baseline ResNet.
- Ghi lại kết quả từng thử nghiệm (config + metric) để so sánh công bằng — cân nhắc thêm bảng kết quả tổng hợp hoặc tích hợp tool tracking (vd. Weights & Biases) khi số lượng thử nghiệm tăng lên.

## Ghi chú khác

- Không commit dữ liệu ảnh (`data/raw/`, `data/processed/`) hay checkpoint (`models/*.pth`) — đã được thêm vào `.gitignore`.
- Luôn gọi `set_seed()` trước khi train để kết quả có thể tái lập.
