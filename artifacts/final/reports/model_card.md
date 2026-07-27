# Model card — phát hiện viêm phổi trên X-quang ngực trẻ em

## Mô hình

Trung bình xác suất có trọng số bằng nhau của hai CNN:

| Thành viên | Kiến trúc | Tiền xử lý | Loss |
|---|---|---|---|
| ResNet18 B1 | ImageNet pretrained | stretch 224×224, augment mạnh | weighted CE |
| DenseNet121 v5 | ImageNet pretrained | stretch 224×224, augment mạnh | weighted CE |

Mỗi thành viên là ensemble 5 fold. Điểm cuối là trung bình cộng của hai xác
suất, không có trọng số học được.

**Ngưỡng: 0,587268.** Chọn trên out-of-fold gộp bằng cách duyệt mọi điểm số
quan sát được, lấy ngưỡng cao nhất còn giữ độ nhạy ≥97%. Benchmark không tham
gia vào việc chọn ngưỡng.

**Đơn vị dự đoán: filename-derived group**, không phải từng ảnh. Nhiều ảnh của
cùng một nhóm được gộp bằng trung bình xác suất.

## Dữ liệu

Kermany chest X-ray trẻ em, 5.856 ảnh, 4.097 filename-derived group. Chia
5 fold theo group; không group hay hash nào nằm ở nhiều split.

Nhóm suy từ tên file là **proxy cho bệnh nhân**, không phải patient ID chính
thức. Khóa nhóm gồm cả phân nhóm viêm phổi, vì `person1_bacteria` và
`person1_virus` là hai chuỗi đếm độc lập và gộp chúng tạo ra 170 chồng lấn giả.

## Hiệu năng

Trên known engineering benchmark, mức group:

| | |
|---|---:|
| Độ nhạy | 0,9951 (202/203) |
| **Độ đặc hiệu** | **0,8222** (185/225) |
| ROC-AUC | 0,9792 |
| HSAS@97 | 0,8369 |
| TN / FP / FN / TP | 185 / 40 / 1 / 202 |

So với ResNet18 đơn: giảm từ 52 xuống 40 ca báo nhầm, không tăng ca bỏ sót.
McNemar p = 0,0005, KTC 95% của Δđộc đặc hiệu [+0,0267, +0,0844].

## Cách đọc con số này

Tập test **đã được xem nhiều lần** qua các giai đoạn trước và đã định hướng
thiết kế. Đây là **known engineering benchmark**, không phải holdout nguyên
vẹn, và con số trên là ước lượng **lạc quan** của khả năng khái quát hóa.

Ngoài ra, đặc trưng thu nhận ảnh tương quan mạnh với nhãn trong tập này: bảng
lượng tử hóa JPEG một mình xác định được lớp ở 96,6% ảnh development, và chồng
lấn giữa hai lớp theo vector nhiễu chỉ 0,4%. Chi tiết ở
[`phase2a_revised_report.md`](../../../reports/phase2a_revised_report.md).

## Dùng được và không dùng được

**Có thể dùng** làm bộ sàng lọc độ nhạy cao trong nghiên cứu, hoặc làm mốc so
sánh cho công việc tiếp theo trên cùng dataset.

**Không dùng được** cho quyết định lâm sàng. Không có external validation,
không có đánh giá của bác sĩ, và lớp NORMAL trong dataset này không được xác
minh là không có bất thường nào khác.

Ở tỉ lệ mắc thực tế thấp hơn tập test (62,5% ảnh là viêm phổi), precision sẽ
thấp hơn nhiều so với con số 0,82 quan sát được ở đây.

## Tái lập

```bash
python3 scripts/build_final_results.py
```

Mọi con số trong tài liệu này được tính lại từ prediction đã lưu bằng một
implementation duy nhất. Băm SHA-256 trong
[`artifact_manifest.csv`](../provenance/artifact_manifest.csv).
