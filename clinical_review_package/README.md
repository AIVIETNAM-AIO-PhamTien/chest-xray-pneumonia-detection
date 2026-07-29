# Gói đọc phim mù (blinded clinical review)

Kế hoạch phân tích đầy đủ (giả thuyết, endpoint, kiểm định thống kê, quy tắc
hòa giải, quy trình khóa) nằm ở
[`../reports/clinical_review_preregistration.md`](../reports/clinical_review_preregistration.md) —
đọc file đó **trước**, tài liệu này chỉ mô tả các file trong thư mục.

## Thứ tự thao tác

1. Đọc [`review_codebook.md`](review_codebook.md) để biết ý nghĩa từng trường.
2. Đọc thử **5 ca luyện tập** trong `practice_cases.csv` (ảnh ở `images/P01`...`P04`)
   trước — các ca này nằm ngoài mọi phân tích, chỉ để làm quen định dạng và
   giao diện đọc phim.
3. Đọc 125 ca thật liệt kê trong `package_manifest_blinded.csv` (ảnh ở
   `images/R0001`...`R0125`), điền vào `review_form_reviewer_A.csv` hoặc
   `_B.csv` tuỳ người đọc. Hai người đọc độc lập, không trao đổi, thứ tự ca
   đã bị xáo ngẫu nhiên.
4. Sau khi **cả hai** đã nộp đủ 125 dòng: theo đúng "Quy trình khóa" trong
   preregistration — kiểm tra đủ mã/không trùng, tính SHA-256 hai form, đóng
   băng file — rồi mới mở `private/unblinding_key.csv`.

`images/` và `private/` bị gitignore (ảnh bệnh nhân + khóa giải mù không đưa
lên git).

## TODO — chưa có trong tài liệu hiện tại

Tiêu chuẩn/kinh nghiệm yêu cầu cho người đọc phim (bác sĩ chẩn đoán hình ảnh?
số năm kinh nghiệm tối thiểu?) chưa được ghi ở đâu trong gói này hay trong
preregistration. Đây là quyết định chuyên môn lâm sàng, cần người phụ trách
domain điền trước khi mời reviewer thật — không tự suy đoán thay.
