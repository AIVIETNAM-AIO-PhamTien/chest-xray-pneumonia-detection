# Tài liệu

| File | Nội dung |
|---|---|
| [`handbook_cxr_pneumonia.md`](handbook_cxr_pneumonia.md) | Handbook 18 mục: bối cảnh lĩnh vực, dataset, kiến trúc, tiền xử lý, cách train/eval đúng, các mâu thuẫn trong paper chính, protocol reproduce, roadmap thí nghiệm, checklist review paper |
| [`paper_review_slimi_2025.md`](paper_review_slimi_2025.md) | Review paper Slimi et al. 2025 (CNN + Bi-GRU + SNN), theo template QA |

Cả hai giữ **nguyên văn** để bảo toàn provenance. Mọi cập nhật sau này ghi ở dưới,
không sửa trực tiếp vào thân tài liệu.

---

## Cập nhật audit

### 2026-07-25 — Chốt câu hỏi patient leakage (handbook mục 6.10 và 14.13)

**Câu hỏi bỏ ngỏ:** handbook ghi nhận *"khoảng 170 ID dạng `person<number>` xuất
hiện ở cả train và test gốc"* và đánh dấu đây là **tín hiệu audit, chưa phải kết
luận** — cần xác minh quy ước đặt ID từ nguồn Kermany trước khi tin.

**Kết luận: không có leakage. Split gốc là patient-disjoint.**

Chuỗi bằng chứng, tái chạy được bằng `python -m scripts.audit_dataset --root-dir <path>`:

| Bước | Kết quả |
|---|---|
| Parse toàn bộ 5.856 filename | 0 file không parse được |
| Key ngây thơ `person<N>` | 3.257 nhóm, **170** vắt qua >1 split |
| Cả 170 trường hợp đó | train = `bacteria`, test = `virus`; secondary number interleave **0/170** |
| Dãy ID `bacteria` | 1.437 ID, dải 1..1954, mật độ 0,735 |
| Dãy ID `virus` | 1.216 ID, dải 1..1685, mật độ 0,722 |
| Số `person` được dùng bởi **cả hai** subtype | **979** |
| Key đúng `(subtype, person<N>)` | 4.097 nhóm, **0** vắt qua >1 split |

Hai dãy đều gần liên tục và đều bắt đầu từ 1, đồng thời 979 số bị dùng chung. Nếu
counter là toàn cục thì dãy `virus` phải *né* các ID đã thuộc `bacteria` — thực tế
chúng đụng nhau liên tục. Vậy counter chạy **độc lập theo subtype**:
`person1_bacteria` và `person1_virus` là **hai bệnh nhân khác nhau**.

**Hệ quả:**

1. **Test split gốc dùng làm holdout sạch được.** Không cần dựng grouped holdout
   mới như phương án dự phòng ở mục 15.1.
2. Group key đúng là `(subtype, person<N>)` — cài trong `src/splits.py:parse_group_id`.
3. Vẫn **cần** group-aware split khi cắt validation: 726/4.097 bệnh nhân có nhiều
   hơn một ảnh, cá biệt một người có 30 ảnh. Cắt theo ảnh làm **239 bệnh nhân**
   nằm cả hai bên ranh giới train/val (đo bằng `count_leaked_groups`).
4. Con số 170 nên được ghi lại trong report như một **false positive đã xử lý**,
   kèm cách bác bỏ — chứ không im lặng bỏ đi.

### 2026-07-25 — Đường mount trên Kaggle

Kaggle mount dataset này theo **hai kiểu**, tuỳ cách thêm, cả hai đều gặp thật:

```
/kaggle/input/chest-xray-pneumonia/chest_xray/...
/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray/...
```

Hardcode một kiểu là hỏng ở kiểu kia. Ngoài ra bên trong còn hai thứ phải loại:

- `chest_xray/chest_xray/` — cây lồng chứa **bản sao thứ hai của toàn bộ ảnh**;
  quét đệ quy ngây thơ sẽ đếm 11.712 thay vì 5.856.
- `__MACOSX/chest_xray/train/...` — file resource-fork macOS dạng `._*.jpeg`.
  Đuôi `.jpeg` nhưng **không phải ảnh**, và cây này cũng có đủ `train/NORMAL` +
  `train/PNEUMONIA` nên qua mặt được kiểm tra thư mục mốc.

`find_data_root` trong `src/dataset.py` và trong notebook Kaggle xử lý cả bốn
trường hợp: dò theo thư mục mốc, loại `__MACOSX`, chọn cây nông nhất, lọc `._*`.
