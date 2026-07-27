# Phase 2C — Biểu diễn ẩn mã hóa gì, và dữ liệu có cho phép gỡ không

Không huấn luyện lại mạng nào. Probe tuyến tính, cross-validation theo group,
encoder đóng băng.

Tái tạo: `python3 scripts/representation_audit.py`

## Kết quả 1 — Miền được mã hóa ở cả hai lớp, mạnh hơn hẳn ở NORMAL

Probe tuyến tính đọc từ activation đã pooling:

| Tầng | B0: domain trong NORMAL | B0: trong PNEUMONIA | B1: NORMAL | B1: PNEUMONIA |
|---|---:|---:|---:|---:|
| layer1 | **0,941** | 0,661 | **0,937** | 0,729 |
| layer2 | **0,954** | 0,694 | **0,950** | 0,705 |
| layer3 | **0,978** | 0,727 | **0,956** | 0,693 |
| layer4 | **0,974** | 0,701 | **0,956** | 0,695 |
| penultimate | **0,974** | 0,701 | **0,956** | 0,695 |

Từ `layer1` trở đi, biểu diễn đã phân biệt được ảnh NORMAL đến từ development
hay benchmark ở AUC 0,94.

Với PNEUMONIA là 0,66–0,73 — thấp hơn nhiều nhưng **vẫn trên mức ngẫu nhiên**.
Nên câu đúng là thông tin miền được mã hóa ở **cả hai lớp**, chỉ mạnh hơn và
bền hơn rõ rệt ở NORMAL. Không được viết "chỉ tồn tại ở NORMAL".

Bằng chứng nằm ở **mức độ bất đối xứng**, và nó khớp với chuỗi đã đo: thống kê
đầu vào → biểu diễn ẩn → điểm số → hành vi báo nhầm.

## Kết quả 2 — Embedding mã hóa đặc trưng thu nhận

R² khi hồi quy từ penultimate embedding:

| Mục tiêu | B0 | B1 |
|---|---:|---:|
| file_size_per_pixel | **0,868** | **0,814** |
| megapixels gốc | **0,844** | **0,838** |
| noise_estimate | 0,661 | 0,613 |
| laplacian_variance | 0,363 | 0,285 |

Mạng giữ lại thông tin về cách ảnh được thu nhận, không chỉ nội dung.

## Kết quả 3 — Một phân hoạch hard-NORMAL ổn định

Probe FP so với TN trong riêng benchmark NORMAL:

| Cấu hình | AUC |
|---|---:|
| B0 | **0,984** |
| B1 | **0,978** |

Cao hơn hẳn mức 0,84–0,89 khi dự đoán từ đặc trưng thu nhận thủ công.

**Cảnh báo về tính độc lập.** FP và TN được định nghĩa bởi chính điểm số và
ngưỡng của mô hình, còn classifier cũng là một ánh xạ tuyến tính trên chính
embedding này — nên một probe tuyến tính khác có thể đang dựng lại đường biên
đã tạo ra nhãn. Hai kiểm tra ở `scripts/error_probe_checks.py` cho thấy không
phải vậy:

| Embedding | Lỗi của | Nguyên bản | Đã bỏ hướng bệnh |
|---|---|---:|---:|
| B0 | B0 | 0,9842 | **0,9832** |
| B0 | B1 (chéo) | 0,9524 | 0,9532 |
| B1 | B0 (chéo) | 0,9438 | 0,9334 |
| B1 | B1 | 0,9777 | **0,9710** |

Bỏ hẳn hướng phân loại gần như không đổi gì, và embedding của mô hình này dự
đoán được lỗi của mô hình kia ở 0,94–0,95.

Kết luận được phép: **một phân hoạch hard-NORMAL ổn định, chia sẻ giữa hai
pipeline ResNet18 và gần như độc lập với hướng phân loại tuyến tính.**

Chưa được phép: "độc lập với mô hình" hay "độc lập với kiến trúc" — B0 và B1
cùng là ResNet18, chỉ khác tiền xử lý và augmentation.

## Kết quả 4 — Hướng miền và hướng bệnh liên kết yếu

Cosine giữa vector probe miền và hiệu hai hàng của lớp phân loại:

| Cấu hình | cosine |
|---|---:|
| B0 | +0,110 |
| B1 | +0,063 |

Đối chiếu với phân phối hoán vị (2.000 lần):

| Cấu hình | cosine | null TB | null SD | p hai phía |
|---|---:|---:|---:|---:|
| B0 | 0,110 | −0,000 | 0,045 | **0,015** |
| B1 | 0,063 | −0,001 | 0,044 | 0,160 |

B0 nhỉnh hơn ngẫu nhiên một chút; B1 **không** phân biệt được với ngẫu nhiên.

Cách viết đúng: hai hướng tuyến tính **liên kết yếu**. Không được viết "hoàn
toàn độc lập" hay "hai không gian con trực giao" — cosine tính trên toạ độ thô
phụ thuộc scaling và regularization, và trực giao của hai hướng tuyến tính
không bảo đảm gỡ được thông tin miền mà không chạm phi tuyến vào thông tin bệnh.

Dù sao, vấn đề quyết định **không phải cosine** mà là thiếu chồng lấn.

## Kết quả 5 — Nhưng dữ liệu không cho phép

Đây là kết quả quyết định.

| Phép đo chồng lấn | Kết quả |
|---|---|
| Giữa hai **lớp** trong development, theo vector nhiễu | **0,4%** mẫu nằm trong vùng chung |
| Giữa hai **miền** trong riêng NORMAL | **0%** — không có vùng chồng lấn |
| Khoảng cách trung vị giữa hai miền (NORMAL) | **13,53** log-odds |

Với 99,6% dữ liệu development, đặc trưng thu nhận **một mình đã xác định nhãn**.
Gần như không có ví dụ nào của cả hai lớp ở cùng một acquisition style.

Đây là vi phạm điều kiện positivity ở mức gần như toàn phần.

## Kết luận: điều kiện nhận dạng không thỏa

Hai kết quả tưởng như mâu thuẫn, thực ra bổ sung nhau:

**Liên kết yếu giữa hai hướng** nói rằng mạng *tình cờ* mã hóa hai thứ theo hai
hướng phần lớn khác nhau. Nó không nói dữ liệu cho phép *kiểm chứng* hướng nào
là bệnh lý.

**Chồng lấn 0,4%** nói rằng không thể ước lượng "viêm phổi trông như thế nào khi
giữ cố định acquisition style", vì gần như không tồn tại cặp quan sát như vậy.

Một mô hình adversarial huấn luyện trên dữ liệu này sẽ tối ưu một mục tiêu mà
dữ liệu không phân biệt được nghiệm. Việc nó "hoạt động" trên chính dataset này
sẽ không chứng minh được điều gì.

Theo khung phân loại đã thống nhất, đây là **trường hợp D**: dataset không đủ để
học deconfounding một cách an toàn.

## Việc nên làm và không nên làm

**Không nên.** Triển khai GRL, DANN, GroupDRO hay consistency learning trên
riêng dataset này. Không phải vì kỹ thuật sai, mà vì không có cách xác nhận nó
học đúng thứ.

**Nên.** Đóng gói phần này thành đóng góp forensic. Chuỗi bằng chứng đã đầy đủ
và mỗi mắt xích đều tái lập được:

```
OOF group AUC 0,9994 gần bão hòa
→ benchmark specificity 0,671
→ dịch chuyển tập trung ở lớp NORMAL (KS 0,60–0,70 so với 0,07–0,12)
→ tỉ lệ khung chỉ giải thích 0,03 AUC
→ calibration học từ nguồn không chuyển được, còn làm xấu thêm
→ bảng lượng tử hóa JPEG đánh dấu lớp ở 96,6% ảnh development
→ hai can thiệp preprocessing đều trượt manipulation check
→ biểu diễn mã hóa miền ở AUC 0,94–0,98 ngay từ layer1
→ chồng lấn giữa hai lớp: 0,4%
```

**Cần thêm để đi tiếp.** Dữ liệu từ nhiều nguồn, có cả hai lớp trải trên cùng
dải acquisition style. Không có nó thì mọi phương pháp invariance đều không
kiểm chứng được.

## Việc còn lại chưa làm: review mù nhóm NORMAL khó

Probe FP/TN đạt AUC 0,98, nhưng chưa biết các ca đó *là gì*. Cần review mù 52
group FP của B1 cùng một mẫu TN đối chứng, phân loại: bình thường rõ, lỗi chất
lượng ảnh, crop/tư thế, bất thường khác không phải viêm phổi, nghi có opacity,
nhãn không chắc.

Nếu FP chứa nhiều bất thường không phải viêm phổi, bài toán không chỉ là
acquisition shift mà còn là **nhãn NORMAL không đồng nhất** — và hướng đi sẽ là
hard-negative learning hoặc bài toán abnormal-vs-normal, không phải domain
invariance.

Đây là bước cần người có nền y khoa, không phải bước tính toán.

## File sinh ra

```
results_representation_probes.csv
results_direction_alignment.csv
results_error_probe_checks.csv
results_direction_alignment_null.csv
```

Kế hoạch phân tích phần đọc phim đã khóa tại
[`clinical_review_preregistration.md`](clinical_review_preregistration.md).
