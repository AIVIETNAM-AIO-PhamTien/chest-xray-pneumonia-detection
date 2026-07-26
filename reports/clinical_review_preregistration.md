# Đăng ký trước — phân tích kết quả đọc phim mù

Khóa **trước** khi bất kỳ form nào được nộp. Mọi endpoint, phép kiểm định và
phân tích phụ dưới đây được chốt tại thời điểm này. Không được sửa sau khi mở
`unblinding_key.csv`.

Nếu cần lệch khỏi tài liệu này, phải ghi rõ là **lệch sau đăng ký** kèm lý do,
và báo cả kết quả theo kế hoạch gốc.

## Giả thuyết

Mô hình chia nhóm ảnh NORMAL ở benchmark thành một phân hoạch ổn định: probe
tuyến tính tách được false positive khỏi true negative ở AUC 0,98, phân hoạch
này sống sót khi bỏ hướng phân loại (0,983) và chuyển được giữa hai pipeline
(0,94–0,95).

Câu hỏi mà máy tính không trả lời được: **các ca đó là gì?**

Ba khả năng loại trừ lẫn nhau một phần:

1. ảnh bình thường thật, mô hình sai vì đặc trưng thu nhận;
2. ảnh kỹ thuật kém hoặc tư thế/crop bất thường;
3. ảnh có bất thường mà nhãn nhị phân `NORMAL` không mô tả.

## Bộ dữ liệu

| Ô | n | Vai trò |
|---|---:|---|
| false positive | 52 | ca chính |
| true negative | 52 | đối chứng, khớp theo số ảnh mỗi ca |
| true positive | 20 | đối chứng ẩn, kiểm tra chất lượng người đọc |
| false negative | 1 | chỉ mô tả định tính |
| luyện tập | 4 | ngoài mọi phân tích |

Cấu hình: `stretch_manh` (B1). Tổng 125 ca phân tích, 151 ảnh.

## Endpoint chính

```
clinically_abnormal = overall_examination ∈ {probably_abnormal, clearly_abnormal}
```

So sánh tỉ lệ giữa **false positive** và **true negative**.

`indeterminate` **không** nằm trong endpoint chính, vì nó có thể phản ánh chất
lượng ảnh hoặc sự dè dặt của người đọc chứ chưa chắc là bất thường thật.

Kiểm định: Fisher exact hai phía. Báo kèm risk difference, risk ratio, odds
ratio, và khoảng tin cậy 95% bootstrap theo ca (10.000 lần).

Với 52 so 52, khoảng tin cậy sẽ rộng. **Ưu tiên báo effect size và KTC hơn kết
luận có/không có ý nghĩa.**

## Endpoint phụ

Mỗi cái so FP với TN, hiệu chỉnh Benjamini-Hochberg trong họ này:

```
not_clearly_normal   = overall_examination ∈ {indeterminate, probably_abnormal,
                                              clearly_abnormal}
opacity_positive     = pneumonia_compatible_opacity ∈ {present, indeterminate}
other_abnormality    ≠ none
quality_limited      = diagnostic_quality ∈ {limited, non-diagnostic}
positioning_issue    ≠ acceptable
low_confidence       = confidence_1_to_5 ≤ 2
```

Điều chỉnh theo `n_images` bằng hồi quy logistic, vì đối chứng chỉ khớp theo
tần suất chứ không ghép cặp một-một.

## Độ đồng thuận giữa hai người đọc

Tính **trước** khi so FP với TN.

| Loại biến | Chỉ số |
|---|---|
| thứ bậc (`overall_examination`, `confidence`) | weighted kappa, Gwet AC2, phần trăm đồng thuận |
| nhị phân/danh mục | Cohen kappa, Gwet AC1, positive/negative agreement |

Gwet AC1/AC2 bắt buộc phải báo: kappa tụt giả tạo khi tỉ lệ lệch, và bộ này dự
kiến lệch mạnh về phía bình thường.

**Cổng chất lượng:** nếu người đọc bỏ sót quá 30% trong 20 ca true positive
(gọi là `clearly_normal` hoặc `probably_normal`), toàn bộ phần đọc phim chỉ
được gọi là *exploratory visual-quality audit*, không được dùng để bàn về nhãn.

## Hòa giải

Chỉ hòa giải khi:

- bất đồng làm đổi endpoint chính;
- một bên `opacity absent`, bên kia `present`;
- một bên đánh giá `non-diagnostic`;
- lệch quá hai bậc trên `overall_examination`.

Người hòa giải cũng **phải mù** với trạng thái FP/TN.

## Ba phân tích độ nhạy, khóa trước

| | Nội dung | Cách gọi |
|---|---|---|
| A | Giữ nguyên nhãn công khai | hiệu năng chính thức |
| B | Loại ca đồng thuận `indeterminate` hoặc `non-diagnostic`, tính lại độ đặc hiệu | post-hoc adjudication |
| C | Coi ca NORMAL công khai được đồng thuận `probably/clearly_abnormal` là không thuộc lớp âm | post-hoc adjudication |

B và C **không** phải hiệu năng benchmark đã sửa. Phải gắn nhãn *post-hoc
adjudication sensitivity analysis* ở mọi chỗ xuất hiện.

Phải tách ba khái niệm, không được gộp:

```
phim bất thường  ≠  có opacity kiểu viêm phổi  ≠  bất thường khác
```

## Phân tích thăm dò sau mở khóa

Không phải kiểm định giả thuyết chính, ghi rõ là thăm dò:

- xác suất viêm phổi theo nhóm lâm sàng;
- điểm probe miền theo nhóm lâm sàng;
- điểm probe hard-NORMAL theo nhóm lâm sàng;
- đặc trưng thu nhận theo nhóm lâm sàng;
- xu hướng Spearman theo bậc `overall_examination`.

## Quy trình khóa

1. Hai người đọc độc lập, thứ tự khác nhau, không trao đổi.
2. Kiểm tra mỗi form đủ 125 mã, không trùng, không thiếu trường bắt buộc.
3. Tính SHA-256 cho cả hai form, ghi lại.
4. Đóng băng file.
5. **Chỉ khi đó** mới mở `private/unblinding_key.csv`.

Không mở khóa khi mới có một người nộp.

## Điều tuyệt đối không làm

Không dùng bất kỳ ca nào đã được đọc để huấn luyện lại, chọn ngưỡng, chọn
checkpoint hay chọn siêu tham số. Benchmark đã bị xem nhiều lần; việc đọc phim
này làm nó kém phù hợp hơn nữa cho vai trò đánh giá.

## Cách đọc kết quả

| Quan sát | Kết luận được phép |
|---|---|
| FP nhiều opacity/bất thường hơn TN | nhãn lớp âm không đồng nhất góp phần tạo nhóm khó |
| FP chủ yếu bình thường nhưng chất lượng/crop kém hơn | dịch chuyển thu nhận là cơ chế chính |
| FP có cả hai | hai cơ chế cùng tồn tại |
| FP và TN không khác biệt | phân hoạch có thật nhưng audit hiện tại chưa giải thích được |
| đồng thuận thấp | mơ hồ nhãn là một phần của vấn đề |
| bỏ sót nhiều true positive | phần đọc phim không đủ tin cậy để bàn về nhãn |
