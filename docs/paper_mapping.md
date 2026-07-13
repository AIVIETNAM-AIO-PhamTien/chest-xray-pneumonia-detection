# Mapping: paper -> code

Paper: Slimi et al., "Trustworthy pneumonia detection in chest X-ray imaging through attention-guided deep learning", Sci. Rep. 15:40029 (2025). https://doi.org/10.1038/s41598-025-23664-x

Mục đích: thống nhất quy ước đặt tên file/thư mục để nhiều người code song song không đụng file nhau. Khi bắt đầu code 1 phần, cập nhật bảng dưới (owner + trạng thái).

## Preprocessing

| Paper | Chi tiết | File đề xuất | Owner | Trạng thái |
|---|---|---|---|---|
| Section "Preprocessing" | Resize 128x128, augmentation (flip, zoom 0.9-1.1, brightness ±20%) | `src/pneumo_snn/data/preprocessing.py` | | todo |
| Section "Preprocessing" | SMOTE oversampling | `src/pneumo_snn/data/balance.py` | | todo |
| Section "Preprocessing" | Dataset loader (train/val/test split 80/20) | `src/pneumo_snn/data/dataset.py` | | todo |

## Model blocks (Table 1, Fig.2-6)

| Paper | Chi tiết | File đề xuất | Owner | Trạng thái |
|---|---|---|---|---|
| Fig.2 - Spatial Feature Extraction | Conv2d 3->32->64->128, BN, LeakyReLU, MaxPool | `src/pneumo_snn/models/spatial_cnn.py` | | todo |
| Fig.3 - Temporal Dynamics Modeling | Bi-GRU, layers=2, hidden=256, dropout=0.3, num_steps=25 | `src/pneumo_snn/models/temporal_gru.py` | | todo |
| Fig.4 - Spiking Neural Processing | LIF (β=0.95), snnTorch, surrogate gradient, num_steps=25 | `src/pneumo_snn/models/spiking.py` | | todo |
| Fig.5 - Decision Head | FC 768->512->128->2, dropout=0.4 | `src/pneumo_snn/models/decision_head.py` | | todo |
| Fig.6 - Full architecture | Ráp toàn bộ pipeline | `src/pneumo_snn/models/hybrid_model.py` | | todo |
| Fig.10 - Attention | Attention map / overlay (post-hoc, không ảnh hưởng prediction) | `src/pneumo_snn/interpretability/attention_map.py` | | todo |

## Training & Evaluation

| Paper | Chi tiết | File đề xuất | Owner | Trạng thái |
|---|---|---|---|---|
| Eq.10-13 | Accuracy/Precision/Recall/F1 | `src/pneumo_snn/training/metrics.py` | | todo |
| "Experimental setup" | Adam, lr=1e-5, wd=1e-5, OneCycleLR, batch=32, 30 epoch, early stop 5 epoch | `src/pneumo_snn/training/loop.py` | | todo |
| "Ablation study" | So sánh có/không GRU | `configs/model/full_snn_gru_cnn.yaml`, `configs/model/ablation_no_gru.yaml` | | todo |
| Table 5 | 5-fold cross-validation | `src/pneumo_snn/training/cross_validation.py` | | todo |

## Robustness (Fig.9)

| Paper | Chi tiết | File đề xuất | Owner | Trạng thái |
|---|---|---|---|---|
| Gaussian blur, salt-and-pepper, speckle noise | 3 mức độ nhiễu mỗi loại | `src/pneumo_snn/robustness/perturbations.py` | | todo |

## So sánh baseline (Table 3, Table 4)

| Paper | Chi tiết | File đề xuất | Owner | Trạng thái |
|---|---|---|---|---|
| Table 3 | So sánh với Xception/InceptionV3/DenseNet121/ResNet50/VGG16 | `scripts/train_baseline.py` | | todo |

## Quy ước

- 1 block = 1 file, tránh 2 người sửa cùng file cùng lúc.
- Khi nhận 1 dòng trong bảng trên, mở PR nhỏ tương ứng, cập nhật cột Owner/Trạng thái trong PR đó luôn.
- Format kết quả run: `results/runs/<username>_<date>_<exp>/metrics.json` với các khóa `accuracy`, `precision`, `recall`, `f1`, `auc` (đúng theo Table 3/4/5 của paper) + copy `config_used.yaml` cùng thư mục.
