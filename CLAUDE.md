# CLAUDE.md

## Tổng quan project

- **Bài toán**: Phân loại nhị phân (binary classification) phát hiện viêm phổi (Pneumonia) từ ảnh X-quang ngực, 2 lớp: `NORMAL` và `PNEUMONIA`.
- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) trên Kaggle (`paultimothymooney/chest-xray-pneumonia`).
- **Giai đoạn hiện tại**: mô hình cuối là ensemble ResNet18 + DenseNet121,
  trung bình xác suất của 10 checkpoint (5 fold mỗi kiến trúc). Ngưỡng
  group-level `0.587268` được khóa trên pooled OOF; known benchmark đạt
  sensitivity `202/203` và specificity `185/225`.
- **Môi trường train**: Google Colab hoặc Kaggle Notebook (dùng GPU miễn phí), không train local.

## Nguồn chạy chuẩn

- Notebook end-to-end:
  `notebooks/chest_xray_research_complete.ipynb`.
- Ba chế độ: `reproduce` (10 checkpoint, prediction OOF/benchmark, config và
  frozen-input manifest), `full` (2 kiến trúc × 5 fold), và `smoke` (kiểm tra
  kỹ thuật). `auto` chỉ chọn reproduce khi đủ artifact và đúng SHA-256, rồi mới
  xét device.
- Notebook cũ trong `notebooks/` và `source/` chỉ là bằng chứng lịch sử/tham
  khảo; không chạy nối tiếp chúng để tái lập report.
- Grad-CAM dùng `src/explainability/grad_cam.py`, tính riêng từng checkpoint và
  báo trung vị theo từng họ backbone. Không gọi heatmap là annotation tổn
  thương hoặc lời giải thích gradient chính xác của ensemble.
- `src.train` vẫn là CLI baseline/single-holdout; không dùng nó để tạo số cuối
  trong report 5-fold.

## Dataset

- Nguồn: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Cấu trúc gốc sau khi giải nén, đặt tại `data/raw/chest_xray/`:
  ```
  chest_xray/
    train/{NORMAL,PNEUMONIA}/
    val/{NORMAL,PNEUMONIA}/
    test/{NORMAL,PNEUMONIA}/
  ```
- **Lưu ý quan trọng**: tập `val/` gốc chỉ có 16 ảnh (8 mỗi lớp).
  `protocol: "original"` chỉ giữ để tái hiện baseline lịch sử. Notebook chuẩn
  gộp development data rồi chia 5-fold theo filename-derived group; benchmark
  gốc chỉ được đọc sau khi khóa ngưỡng trên pooled OOF.
- Lớp mất cân bằng: `PNEUMONIA` nhiều hơn `NORMAL` đáng kể trong tập train.
  Config baseline và notebook chuẩn đều dùng balanced class weights trong
  `CrossEntropyLoss`.
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
│   ├── baseline.yaml         # hyperparameters cho baseline lịch sử
│   ├── protocol_a.yaml       # split image-level
│   └── protocol_b.yaml       # split patient-grouped (default)
├── data/
│   ├── raw/                  # dataset gốc tải từ Kaggle (gitignored)
│   └── processed/            # dữ liệu đã qua tiền xử lý, nếu cache lại (gitignored)
├── notebooks/
│   ├── chest_xray_research_complete.ipynb # notebook chuẩn end-to-end
│   └── legacy/               # các notebook lịch sử (v2-v6, stage-b, deit...)
├── src/
│   ├── __init__.py
│   ├── config.py              # dataclass Config + load_config từ YAML
│   ├── dataset.py             # pipeline canonical: find_data_root, build_dataloaders
│   │                         # (ImageFolder), compute_class_weights
│   ├── transforms.py          # augmentation & preprocessing của pipeline canonical
│   ├── splits.py               # build_manifest/make_splits (protocol A/B) — dùng bởi
│   │                         # cả notebook canonical và src/train.py
│   ├── data/                   # pipeline riêng cho CLI baseline của src/train.py
│   │                         # (CXRDataset, build_loaders, transforms, imbalance);
│   │                         # xem src/data/README.md — KHÔNG dùng để tạo số report
│   ├── evaluation/              # calibration, label_shift, nuisance_features, selection
│   ├── models/                # model architectures (registry pattern)
│   │   ├── registry.py        # register_model decorator + build_model
│   │   ├── resnet18.py        # transfer-learning ResNet-18
│   │   └── simple_cnn.py      # example from-scratch CNN (template)
│   ├── explainability/          # Grad-CAM dùng chung cho notebook/report
│   ├── train.py                # training loop CLI baseline (không dùng cho số report)
│   ├── evaluate.py             # accuracy/precision/recall/f1/confusion matrix
│   └── utils.py                 # set_seed, checkpoint, EarlyStopping
├── models/                     # checkpoint đã train (gitignored)
├── outputs/
│   ├── logs/                   # log training (gitignored)
│   └── figures/                # biểu đồ, confusion matrix... (gitignored)
└── tests/                       # 1 file test theo mỗi module trong src/
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

## Quy trình CLI baseline cũ trên Colab / Kaggle

Phần dưới chỉ mô tả CLI baseline. Nghiên cứu/report dùng notebook chuẩn nêu ở
đầu file. Chi tiết baseline cũ nằm trong `notebooks/train_baseline.ipynb`.

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
6. Epoch đầu luôn lưu một weights-only checkpoint; các epoch sau ghi đè khi
   validation F1 tăng. Sau vòng lặp, CLI nạp lại checkpoint tốt nhất trước khi
   đánh giá test. File không chứa optimizer/scheduler nên chưa resume được.

## Hướng tiếp theo sau pipeline cuối

- DenseNet121 và DeiT đã được thử; mô hình cuối ghép ResNet18 + DenseNet121.
- Ưu tiên cohort ngoài chưa được xem, nhiều seed, calibration/subgroup audit,
  lung segmentation và đọc phim mù các ca báo nhầm.
- Mọi thử nghiệm mới phải khóa split, metric và ngưỡng trước khi đọc benchmark.

## Ghi chú khác

- Không commit dữ liệu ảnh (`data/raw/`, `data/processed/`) hay checkpoint (`models/*.pth`) — đã được thêm vào `.gitignore`.
- Luôn gọi `set_seed()` trước khi train. Seed hiện tại chưa bảo đảm tái lập
  bit-for-bit trên mọi GPU vì code chưa bật deterministic algorithms/cuDNN đầy
  đủ.
- Khi thêm model mới, ngoài `@register_model(...)` cần import module đó trong
  `src/models/__init__.py`; registry hiện chưa auto-discover file.
- Nếu dùng scheduler `plateau` theo validation F1, đặt
  `scheduler_params.mode: "max"`.
