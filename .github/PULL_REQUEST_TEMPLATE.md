## Tóm tắt
<!-- Thay đổi gì, cho phần nào của paper (block/section nào)? -->

## Checklist
- [ ] Notebook (nếu có) đã strip output (`uv run nbstripout --install` đã chạy)
- [ ] `uv run pytest -q` pass local
- [ ] `uv run ruff check .` và `uv run black --check .` pass local
- [ ] Nếu thêm/đổi hyperparameter: đã cập nhật file trong `configs/`, không hardcode trong code
- [ ] Nếu có kết quả run mới: đã ghi vào `results/runs/<username>_<date>_<exp>/metrics.json`, không sửa tay file kết quả của người khác
