# Reports

Các báo cáo audit theo từng giai đoạn nghiên cứu, chạy trên prediction đã
đóng băng (không huấn luyện lại mạng nào). **Số liệu chính thức của model
cuối** (ensemble ResNet18+DenseNet121) nằm ở
[`../artifacts/final/reports/`](../artifacts/final/reports/) — các file dưới
đây là lịch sử/audit dẫn tới quyết định cuối, không phải nguồn số để trích
dẫn.

## Thứ tự đọc

1. [`prior_shift_report.md`](prior_shift_report.md) — Phase 2A.0: prior shift
   và giả định label shift.
2. [`phase2a_revised_report.md`](phase2a_revised_report.md) — Phase 2A: nguồn
   gốc dịch chuyển trong lớp NORMAL (không có Phase 2B riêng — bị gộp/bỏ
   trong quá trình audit, không phải file bị thiếu).
3. [`phase2c_representation_audit.md`](phase2c_representation_audit.md) —
   Phase 2C: biểu diễn ẩn mã hoá gì, probe tuyến tính theo group.
4. [`calibration_audit_report.md`](calibration_audit_report.md) — calibration
   audit cho baseline v4 (giai đoạn sớm hơn, trước ensemble cuối).

## Track riêng — chưa chạy

[`clinical_review_preregistration.md`](clinical_review_preregistration.md)
không nằm trong chuỗi phase trên. Đây là kế hoạch phân tích **đăng ký trước**
cho gói đọc phim mù ở
[`../clinical_review_package/`](../clinical_review_package/) — vẫn là tài
liệu sống, áp dụng cho lần đọc phim tiếp theo, không bị thay thế bởi
`artifacts/final/`.
