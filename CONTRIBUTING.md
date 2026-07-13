# Contributing

Hướng dẫn nhanh cho team (kể cả bạn mới dùng Git).

## Setup lần đầu

```bash
git clone <repo-url>
cd chest-xray-pneumonia-detection
uv sync --all-extras --dev     # tạo .venv + cài toàn bộ dependency đúng version trong uv.lock
uv run nbstripout --install    # bắt buộc, để notebook luôn strip output khi commit
uv run pre-commit install      # bật lint/format tự động trước mỗi commit
```

Chạy code trong venv bằng `uv run <command>` (ví dụ `uv run pytest`, `uv run python scripts/train.py ...`), không cần tự activate venv.

## Quy trình làm việc

1. Tạo issue (hoặc nhận issue có sẵn) theo template "Reproduce a paper table/section" — mỗi issue có 1 owner để tránh trùng việc.
2. Tạo nhánh từ `main`, đặt tên theo mẫu:
   - `feat/<block>-<ten>` — thêm code mới (vd `feat/snn-block-anh`)
   - `fix/<block>-<ten>` — sửa lỗi (vd `fix/gru-shape-binh`)
   - `exp/<ten-thi-nghiem>-<ten>` — chạy thí nghiệm/ablation
3. Code trong `src/pneumo_snn/<module>/` đúng module bạn phụ trách (xem `docs/paper_mapping.md`). Hyperparameter đưa vào `configs/`, không hardcode.
4. Trước khi push:
   ```bash
   uv run ruff check .
   uv run black --check .
   uv run pytest -q
   ```
5. Mở Pull Request vào `main`, điền checklist trong PR template. Cần ít nhất 1 review pass + CI xanh mới được merge. PR được **squash-merge** (không cần biết rebase để xử lý conflict).
6. Nếu PR có kết quả training mới: lưu vào `results/runs/<username>_<date>_<exp>/metrics.json` — **không sửa file kết quả của người khác**, mỗi run 1 thư mục riêng.

## Chạy trên Kaggle

- Đẩy code lên `main`/nhánh trước, sau đó notebook trên Kaggle chỉ cần `pip install` từ repo (theo commit/tag cụ thể) rồi gọi `scripts/train.py` với config tương ứng — không viết logic model trực tiếp trong notebook Kaggle.
- Mỗi experiment = 1 Kaggle kernel riêng, đặt tên `pneumo-<block>-<expname>-v<N>` để tránh 2 người đè kernel của nhau.
- Push kernel: `kaggle/push_kernel.sh <kernel-dir>` (cần `~/.kaggle/kaggle.json`, không commit file này).

## Không làm

- Không commit dataset, checkpoint (`*.pt`, `*.ckpt`), hay `kaggle.json`.
- Không commit notebook có output chưa strip.
- Không sửa tay `results/leaderboard.md` (nếu có về sau) hay file kết quả của người khác.
- Không push thẳng/force-push vào `main`.
