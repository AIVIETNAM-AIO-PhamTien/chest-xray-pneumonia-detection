# Notebooks

Quy tắc:

- Notebook chỉ dùng để thử nghiệm nhanh (EDA, debug, thử ý tưởng). Logic chính thức phải nằm trong `src/pneumo_snn/` và được gọi lại từ notebook, không viết trực tiếp trong notebook.
- **Luôn strip output trước khi commit.** Cài 1 lần sau khi clone:
  ```
  uv run nbstripout --install
  ```
  Từ đó git sẽ tự động xóa output khi commit `.ipynb`. Không tắt hook này.
- Đặt tên notebook có số thứ tự + mục đích, ví dụ `00_eda.ipynb`, `01_train_kaggle.ipynb`.
- Notebook chạy trên Kaggle nên là bản mỏng: cài package từ repo (`pip install -e .` hoặc từ 1 commit/tag cụ thể trên GitHub) rồi gọi hàm/script trong `src/` hoặc `scripts/`.
