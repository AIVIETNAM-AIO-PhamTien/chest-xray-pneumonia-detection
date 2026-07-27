# Báo cáo cuối — cải thiện độ đặc hiệu ở độ nhạy cao

Mọi con số dưới đây được tính lại từ prediction đã lưu bằng
`scripts/build_final_results.py`, một implementation duy nhất cho ngưỡng, HSAS
và cách gộp group. Không con số nào được chép từ bảng cũ.

## Câu hỏi

Backbone mạnh hơn, tinh chỉnh theo hard-negative, đặc trưng chuyên biệt cho
X-quang ngực, và ensemble — có giảm được số ca bình thường bị báo nhầm, trong
khi giữ độ nhạy ≥97%?

## Trả lời

Có, một phần, và **không phải bằng những thứ tưởng sẽ hiệu quả**.

| Mô hình | AUC | HSAS@97 | Độ nhạy | Độ đặc hiệu | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 B1 | 0,9779 | 0,8298 | 0,9951 | 0,7689 | 52 | 1 |
| DenseNet121 v5 | 0,9801 | 0,8386 | 0,9951 | 0,8044 | 44 | 1 |
| DeiT-Small | 0,9789 | 0,8009 | 0,9951 | 0,6578 | 77 | 1 |
| **R+D ensemble** | 0,9792 | 0,8369 | **0,9951** | **0,8222** | **40** | 1 |

52 → 40 ca báo nhầm, giảm 23,1%, độ nhạy không đổi.

## So sánh ghép cặp

| So với | Sửa | Phá | Ròng | McNemar p | Δđộ đặc hiệu KTC 95% |
|---|---:|---:|---:|---:|---|
| ResNet18 B1 *(chính)* | 12 | 0 | **+12** | **0,0005** | [+0,0267; +0,0844] |
| DenseNet121 v5 *(phụ)* | 6 | 2 | +4 | 0,2891 | [−0,0044; +0,0444] |
| DeiT-Small *(phụ)* | 37 | 0 | +37 | 0,0000 | [+0,1200; +0,2133] |

Cần nói thẳng: **ensemble không chứng minh được là hơn DenseNet121 về mặt thống
kê.** Bốn ca ròng, p = 0,29, khoảng tin cậy chứa 0. Nó được giữ vì đạt mục tiêu
kỹ thuật đã khóa, không giảm độ nhạy, và lợi ích tích lũy so với baseline
ResNet là rõ ràng.

## Bốn thí nghiệm không cải thiện được gì

| Thí nghiệm | Kết quả | Mở benchmark? |
|---|---|---|
| Hard-negative fine-tuning | 5/5 fold giữ epoch 0; prediction **trùng bit** DenseNet | có, và trùng khớp |
| DeiT-Small | AUC tương đương, độ đặc hiệu 0,6578 | có |
| R+D+DeiT | sửa 0 ca, phá 37 ca | có |
| XRV verifier đóng băng | không qua cổng phía nguồn | **không** |
| ResNet18 v6 (C2) | E0 thắng trên OOF | **không** |
| Rank ensemble | ngưỡng không chuyển được qua đổi tỉ lệ lớp | có, trong audit |

Hai thí nghiệm cuối **không được phép chạm vào benchmark** vì cổng phía nguồn
không đạt — quyết định do code cưỡng chế, không phải kỷ luật.

## Phát hiện trung tâm

> **AUC toàn cục ngang nhau không có nghĩa hiệu năng ở điểm làm việc ngang
> nhau, cũng không có nghĩa ngưỡng chuyển được.**

Ba bằng chứng độc lập:

**DeiT** có AUC 0,9789, ngang hai CNN, nhưng độ đặc hiệu 0,6578 và khoảng cách
OOF→benchmark 0,328 so với 0,186 của DenseNet.

**C2** cho ra một ứng viên ensemble có độ đặc hiệu OOF **cao hơn** đúng hai ca,
nhưng HSAS@97 thấp hơn 0,0144 — gấp bảy lần biên hòa. Xếp hạng ở vùng ngưỡng
thực sự nằm thì kém rõ rệt.

**Chọn checkpoint theo AUC** ở fold 0 giữ epoch 10 vì hơn 0,0001 AUC, trong khi
epoch 5 hơn 0,0083 độ đặc hiệu.

## Trần của mô hình hiện tại

Đọc nhãn benchmark để đo trần (chỉ chẩn đoán, ngưỡng này không dùng được):

| FN tối đa | FP oracle | Độ đặc hiệu |
|---:|---:|---:|
| 1 | 31 | 86,2% |
| 3 | 25 | 88,9% |
| 6 | **22** | 90,2% |

Khoảng cách 40 → 20 tách làm ba: 9 ca do đặt ngưỡng, 9 ca do ngân sách FN,
**2 ca cuối không đạt được** với xếp hạng hiện tại. Mục tiêu FP=20 nằm dưới
trần này.

## Điều gì đã không xảy ra

Bốn can thiệp liên tiếp đều null, và chúng nhất quán với nhau. Nhóm NORMAL khó
này không tách được bằng: trọng số hard-negative, kiến trúc Transformer, verifier
tuyến tính trên hai không gian đặc trưng X-quang đóng băng, hay chọn lại
checkpoint.

Cần viết chính xác: điều này **không** chứng minh Transformer kém CNN nói chung,
hay pretraining chuyên biệt X-quang vô dụng. Nó chứng minh **dưới protocol này,
với các cổng đã đăng ký trước, không cái nào cung cấp bằng chứng phía nguồn đủ
để biện minh cho việc áp dụng.**

## Vì sao dừng

Stop rule đã đăng ký: dừng sau DenseNet, hard-negative, DeiT và ensemble cuối
nếu không phương án nào tăng ≥2 điểm độ đặc hiệu. Cả bốn đã xong. Hai thí
nghiệm mở rộng (XRV verifier, C2) cũng không qua cổng.

Muốn đi xa hơn cần thứ dự án này không có: dữ liệu từ nguồn khác, hard-negative
có nhãn giống phân phối đích, hoặc một tập hiệu chuẩn từ cùng phân phối đích.

## Tái lập

```bash
python3 scripts/build_final_results.py
```

Băm SHA-256 của mọi artifact trong
[`artifact_manifest.csv`](../provenance/artifact_manifest.csv). Commit sinh ra
gói này ghi trong [`commit_manifest.json`](../provenance/commit_manifest.json).
