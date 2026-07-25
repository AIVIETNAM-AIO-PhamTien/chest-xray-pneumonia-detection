# CLAUDE.md

## Tổng quan project

- **Bài toán**: Phân loại nhị phân (binary classification) phát hiện viêm phổi (Pneumonia) từ ảnh X-quang ngực, 2 lớp: `NORMAL` và `PNEUMONIA`.
- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) trên Kaggle (`paultimothymooney/chest-xray-pneumonia`).
- **Giai đoạn hiện tại**: baseline transfer learning bằng ResNet18 pretrained
  ImageNet; repo cũng có SimpleCNN làm template train-from-scratch. ResNet34 và
  các kiến trúc khác chưa được đăng ký, nằm trong roadmap.
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
- **Lưu ý quan trọng**: tập `val/` gốc chỉ có 16 ảnh (8 mỗi lớp). Code hiện
  dùng trực tiếp split này, chưa tự chia lại train/val, nên validation metric và
  chọn checkpoint chỉ mang tính sơ bộ. Khi triển khai split mới, ưu tiên chia
  theo patient nếu có patient ID đáng tin cậy để tránh leakage giữa train/val.
- Lớp mất cân bằng: `PNEUMONIA` nhiều hơn `NORMAL` đáng kể trong tập train.
  Baseline hiện dùng `CrossEntropyLoss()` không weight; class weighting hoặc
  weighted sampler là hạng mục cần thử nghiệm.
- Transform luôn convert mọi ảnh về grayscale rồi tạo 3 kênh cho backbone
  pretrained ImageNet; không giả định mọi file nguồn đều được lưu single-channel
  (xem `src/transforms.py`).

## Cấu trúc thư mục

```
chest-xray-pneumonia-detection/
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
│   ├── dataset.py             # ImageFolder cho split train/val/test có sẵn
│   ├── transforms.py          # augmentation & preprocessing pipeline
│   ├── models/                # model architectures (registry pattern)
│   │   ├── registry.py        # register_model decorator + build_model
│   │   ├── resnet18.py        # transfer-learning ResNet-18
│   │   └── simple_cnn.py      # example from-scratch CNN (template)
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
   - **Kaggle Notebook**: thêm dataset qua nút "+ Add Data". Kiểm tra mount thực
     tế và chọn thư mục chứa trực tiếp `train/`, `val/`, `test/`.
   - **Colab**: notebook hiện chưa có cell upload `kaggle.json`, download và
     giải nén. Cần chuẩn bị dataset thủ công trước khi chạy các cell training.
3. Lấy source code bằng `git clone` từ GitHub:
   ```
   !git clone https://github.com/AIVIETNAM-AIO-PhamTien/chest-xray-pneumonia-detection.git
   %cd chest-xray-pneumonia-detection
   ```
   Trên **Kaggle Notebook**, phải bật **Internet** ở Notebook Settings (sidebar bên phải) trước khi chạy `git clone`, nếu không sẽ báo lỗi kết nối. Colab đã có Internet sẵn.
4. `pip install -r requirements.txt`.
5. Giá trị `data.root_dir` trong config là placeholder. Truyền đường dẫn dataset
   đúng môi trường bằng `--root-dir` như notebook, hoặc cập nhật config trước
   khi gọi `run_training()`.
6. Khi validation F1 tăng và lớn hơn `0`, weights tốt nhất được lưu vào
   `models/baseline_best.pth`. File chỉ chứa `model.state_dict()`, không phải
   checkpoint có thể resume. Sau vòng lặp, test hiện dùng weights của epoch cuối
   hoặc epoch early-stop, chưa tự load lại best weights.

## Roadmap thử nghiệm (sau baseline)

- **Dữ liệu**: tạo validation split đủ lớn, ưu tiên split theo patient khi có ID
  đáng tin cậy.
- **Tiền xử lý**: CLAHE, histogram equalization, các cách resize/crop khác nhau, chuẩn hoá theo mean/std riêng của dataset thay vì ImageNet, lung segmentation/cropping.
- **Model**: ResNet34, EfficientNet-B0, DenseNet121, Vision Transformer
  (ViT-Base), so sánh với baseline ResNet18.
- Ghi lại kết quả từng thử nghiệm (config + metric) để so sánh công bằng — cân nhắc thêm bảng kết quả tổng hợp hoặc tích hợp tool tracking (vd. Weights & Biases) khi số lượng thử nghiệm tăng lên.

## Ghi chú khác

- Không commit dữ liệu ảnh (`data/raw/`, `data/processed/`) hay checkpoint (`models/*.pth`) — đã được thêm vào `.gitignore`.
- Luôn gọi `set_seed()` trước khi train. Seed hiện tại chưa bảo đảm tái lập
  bit-for-bit trên mọi GPU vì code chưa bật deterministic algorithms/cuDNN đầy
  đủ.
- Khi thêm model mới, ngoài `@register_model(...)` cần import module đó trong
  `src/models/__init__.py`; registry hiện chưa auto-discover file.
- Nếu dùng scheduler `plateau` theo validation F1, đặt
  `scheduler_params.mode: "max"`.
