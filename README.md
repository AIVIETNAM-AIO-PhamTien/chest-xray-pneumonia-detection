# Chest X-ray Image Classification for Pneumonia Detection

Reproduction của paper **"Trustworthy pneumonia detection in chest X-ray imaging through attention-guided deep learning"** (Slimi et al., *Scientific Reports*, 2025) — kiến trúc lai CNN + Bi-GRU + Spiking Neural Network (SNN) với attention, đạt 99.35% accuracy trên [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

Paper: https://doi.org/10.1038/s41598-025-23664-x

## Kiến trúc (theo paper)

| Block | Vai trò | Thư mục code |
|---|---|---|
| Spatial Feature Extraction (Fig.2) | CNN 3 lớp trích đặc trưng không gian | `src/pneumo_snn/models/` |
| Temporal Dynamics Modeling (Fig.3) | Bi-GRU (2 lớp, hidden=256) mô hình hóa chuỗi thời gian | `src/pneumo_snn/models/` |
| Spiking Neural Processing (Fig.4) | LIF neuron (β=0.95), snnTorch, 25 timestep | `src/pneumo_snn/models/` |
| Decision Head (Fig.5) | Fully-connected + dropout, output 2 lớp | `src/pneumo_snn/models/` |
| Attention | Interpretability map (Fig.10), không ảnh hưởng prediction | `src/pneumo_snn/interpretability/` |

Chi tiết mapping code ↔ paper: xem [`docs/paper_mapping.md`](docs/paper_mapping.md).

## Setup

Dùng [uv](https://docs.astral.sh/uv/) để quản lý môi trường:

```bash
uv sync --all-extras --dev
uv run nbstripout --install
uv run pre-commit install
```

Xem quy trình làm việc, quy ước nhánh/commit trong [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Cấu trúc thư mục

```
chest-xray-pneumonia-detection/
├── src/pneumo_snn/       # package chính (data, models, training, robustness, interpretability, utils)
├── configs/               # config YAML cho từng experiment
├── notebooks/              # notebook thử nghiệm nhanh (luôn strip output)
├── scripts/                # entrypoint CLI (train.py, evaluate.py, ...)
├── kaggle/                 # script push kernel lên Kaggle
├── results/runs/           # kết quả từng run, mỗi run 1 thư mục riêng
├── tests/                  # pytest
└── docs/                   # tài liệu, mapping code <-> paper
```

## Train trên Kaggle

Xem hướng dẫn trong `CONTRIBUTING.md` mục "Chạy trên Kaggle" và script `kaggle/push_kernel.sh`.

## Trạng thái

Đây là khung repo ban đầu — logic model/training sẽ được team code dần theo các issue trong GitHub Project board (mỗi issue tương ứng 1 phần của paper: preprocessing, từng block model, ablation study, robustness study, cross-validation, so sánh baseline).
