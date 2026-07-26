# Phase 2C — Biểu diễn ẩn mã hóa gì, và dữ liệu có cho phép gỡ không

Không huấn luyện lại mạng nào. Probe tuyến tính, cross-validation theo group,
encoder đóng băng.

Tái tạo: `python3 scripts/representation_audit.py`

## Kết quả 1 — Miền được mã hóa mạnh, và chỉ trong lớp NORMAL

Probe tuyến tính đọc từ activation đã pooling:

| Tầng | B0: domain trong NORMAL | B0: trong PNEUMONIA | B1: NORMAL | B1: PNEUMONIA |
|---|---:|---:|---:|---:|
| layer1 | **0,941** | 0,661 | **0,937** | 0,729 |
| layer2 | **0,954** | 0,694 | **0,950** | 0,705 |
| layer3 | **0,978** | 0,727 | **0,956** | 0,693 |
| layer4 | **0,974** | 0,701 | **0,956** | 0,695 |
| penultimate | **0,974** | 0,701 | **0,956** | 0,695 |

Từ `layer1` trở đi, biểu diễn đã phân biệt được ảnh NORMAL đến từ development
hay benchmark ở AUC 0,94. Với PNEUMONIA chỉ 0,66–0,73.

Bất đối xứng quan sát được ở mức score giờ tái hiện ở **mức biểu diễn**.

## Kết quả 2 — Embedding mã hóa đặc trưng thu nhận

R² khi hồi quy từ penultimate embedding:

| Mục tiêu | B0 | B1 |
|---|---:|---:|
| file_size_per_pixel | **0,868** | **0,814** |
| megapixels gốc | **0,844** | **0,838** |
| noise_estimate | 0,661 | 0,613 |
| laplacian_variance | 0,363 | 0,285 |

Mạng giữ lại thông tin về cách ảnh được thu nhận, không chỉ nội dung.

## Kết quả 3 — Mạng "biết trước" nó sẽ sai ở đâu

Probe FP so với TN trong riêng benchmark NORMAL:

| Cấu hình | AUC |
|---|---:|
| B0 | **0,984** |
| B1 | **0,978** |

Cao hơn hẳn mức 0,84–0,89 khi dự đoán từ đặc trưng thu nhận thủ công. Biểu diễn
tổ chức ảnh NORMAL thành nhóm dễ và nhóm khó gần như tách hoàn toàn.

## Kết quả 4 — Hướng miền gần vuông góc với hướng bệnh

Cosine giữa vector probe miền và hiệu hai hàng của lớp phân loại:

| Cấu hình | cosine |
|---|---:|
| B0 | +0,110 |
| B1 | +0,063 |

Hai hướng ngẫu nhiên trong không gian 512 chiều cho cosine ~0 với độ lệch chuẩn
1/√512 ≈ 0,044. Nên +0,11 và +0,06 chỉ nhỉnh hơn ngẫu nhiên chút ít.

Đọc theo khung quyết định đã đặt: đây **không** phải trường hợp hai hướng đồng
tuyến. Về mặt hình học, có thể triệt hướng miền mà không phá hướng bệnh.

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

**Hướng gần vuông góc** nói rằng mạng *tình cờ* mã hóa hai thứ ở hai hướng khác
nhau. Nó không nói dữ liệu cho phép *kiểm chứng* hướng nào là bệnh lý.

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
```
