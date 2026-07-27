# Sổ mã hóa cho phần đọc phim

Mỗi thư mục trong `images/` là **một ca**. Nếu có nhiều ảnh, chúng thuộc cùng
một người và nên được đọc cùng nhau.

Điền vào `review_form_reviewer_A.csv` hoặc `_B.csv`. Mỗi người đọc độc lập,
không trao đổi cho tới khi cả hai đã nộp.

Chỉ mô tả những gì nhìn thấy trên phim. Không cần đoán vì sao một mô hình có
thể sai — phần đó được phân tích sau khi mở khóa.

## Các trường

**diagnostic_quality** — `adequate` | `limited` | `non-diagnostic`

**positioning_crop** — `acceptable` | `rotation` | `poor_inspiration` |
`crop_issue` | `other`

**pneumonia_compatible_opacity** — `absent` | `indeterminate` | `present`

**other_abnormality** — `none` | `atelectatic_change` |
`interstitial_or_peribronchial_change` | `effusion` | `other`

**overall_examination** — `clearly_normal` | `probably_normal` |
`indeterminate` | `probably_abnormal` | `clearly_abnormal`

**confidence_1_to_5** — 1 là rất không chắc, 5 là rất chắc

**free_text_comment** — tự do, tiếng Việt hoặc tiếng Anh

## Những điều cần biết

Bộ ảnh gồm nhiều loại ca khác nhau, không đồng nhất. Đừng giả định mọi ca đều
bình thường hay đều bất thường.

Thứ tự đã được xáo ngẫu nhiên. Số thứ tự không mang thông tin.

Nếu không kết luận được, chọn `indeterminate` — đó là câu trả lời hợp lệ và
hữu ích, không phải thất bại.
