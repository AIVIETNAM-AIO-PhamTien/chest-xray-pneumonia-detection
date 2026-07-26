# Calibration audit — Baseline v4

Không huấn luyện lại mạng nào. Toàn bộ phân tích chạy trên prediction đã đóng
băng: OOF theo từng fold, benchmark theo từng fold, và benchmark ensemble.

Đơn vị suy luận là **filename-derived group**. Kết quả mức ảnh chỉ để mô tả.

Tái tạo: `python3 scripts/calibration_audit.py`

## Câu hỏi

Phân rã phần độ đặc hiệu bị mất khi chuyển từ OOF sang benchmark thành ba
thành phần cần ba cách chữa khác nhau:

1. mất khả năng phân biệt dưới distribution shift;
2. điểm làm việc không chuyển được từ nguồn sang đích;
3. xác suất không đúng nghĩa xác suất.

## Phép kiểm tra tính đúng của code

Temperature, intercept-only, Platt và beta đều đơn điệu tăng theo score. Nếu áp
sau khi đã gộp fold, rồi chọn lại ngưỡng trên cùng thứ hạng OOF với cùng ràng
buộc độ nhạy, thì `f(p) >= f(t)` đúng khi và chỉ khi `p >= t` — **không quyết
định nào được phép đổi**. Mọi mức tăng độ đặc hiệu báo trong thiết lập đó là lỗi
lập trình, không phải phát hiện.

`monotonic_invariance_check` xác nhận **ĐẠT trên cả 4 cấu hình**. Kết quả
post-ensemble phía dưới đọc được là nhờ vậy.

## Kết quả 1 — Ở nguồn, calibration hoạt động đúng như quảng cáo

OOF cross-fitted (fit trên 4 fold, áp lên fold còn lại):

| Cấu hình | ECE raw | ECE platt | Brier raw | Brier platt |
|---|---:|---:|---:|---:|
| letterbox+nhẹ | 0,0067 | **0,0013** | 0,0095 | 0,0087 |
| stretch+nhẹ | 0,0086 | **0,0020** | 0,0106 | 0,0101 |
| letterbox+mạnh | 0,0064 | **0,0020** | 0,0112 | 0,0114 |
| stretch+mạnh | 0,0232 | **0,0028** | 0,0206 | 0,0165 |

ECE giảm 3–8 lần. Mô hình vốn đã gần hiệu chuẩn tốt ở nguồn (ECE 0,006–0,023),
và calibration đưa nó về gần như hoàn hảo.

## Kết quả 2 — Ở đích, calibration làm mọi thứ TỆ HƠN

Benchmark, pre-ensemble:

| Cấu hình | Brier raw | Brier intercept | NLL raw | NLL intercept | ECE raw | ECE intercept |
|---|---:|---:|---:|---:|---:|---:|
| letterbox+nhẹ | 0,203 | 0,232 | 1,108 | 1,391 | 0,235 | 0,265 |
| stretch+nhẹ | 0,149 | 0,185 | 0,657 | 0,883 | 0,185 | 0,223 |
| letterbox+mạnh | 0,194 | 0,217 | 0,895 | 1,023 | 0,236 | 0,261 |
| stretch+mạnh | **0,086** | 0,139 | **0,338** | 0,556 | **0,121** | 0,187 |

Mọi chỉ số đều xấu đi. Ngoại lệ duy nhất là temperature scaling cải thiện NLL
một chút (stretch+mạnh: 0,338 → 0,321).

Lý do đọc được từ calibration intercept trên benchmark raw: **−2,07 đến −3,03**.
Mô hình lệch hệ thống về phía dự đoán viêm phổi trên benchmark, trong khi ở OOF
nó gần như không lệch. Calibrator học từ OOF không thể biết về độ lệch chỉ tồn
tại ở đích, và đẩy intercept đi xa hơn nữa theo hướng sai (−3,1 đến −3,9).

`figures/reliability_oof_vs_benchmark.png` cho thấy trực tiếp: hàng trên nằm
trên đường chéo, hàng dưới bám đáy rồi vọt đứng ở p≈1, và đường đã hiệu chuẩn
nằm dưới đường raw.

Đây là chứng minh rằng miscalibration trên benchmark do **distribution shift**
gây ra, không phải do mô hình vốn hiệu chuẩn kém — nên một ánh xạ học từ nguồn
không sửa được.

## Kết quả 3 — Phục hồi độ đặc hiệu

Operating point B (ngưỡng chọn từ OOF đã hiệu chuẩn, cross-fitted):

### Post-ensemble: 0% hoặc âm

| Cấu hình | Phục hồi tốt nhất |
|---|---:|
| letterbox+nhẹ | 0,000 |
| stretch+nhẹ | 0,000 |
| letterbox+mạnh | −0,013 |
| stretch+mạnh | −0,022 |

Đúng như bất biến toán học dự đoán. Số âm nhỏ đến từ việc gộp ảnh thành group
sau khi hiệu chuẩn, làm phép biến đổi không còn đơn điệu ở mức group.

### Pre-ensemble: có phục hồi, chỉ ở nhánh stretch

| Cấu hình | Calibrator | Raw | Sau cal | Oracle | Δ | Tỉ lệ | KTC 95% | Độ nhạy |
|---|---|---:|---:|---:|---:|---:|---|---:|
| stretch+mạnh | intercept | 0,769 | **0,809** | 0,893 | +0,040 | 32,1% | [+0,018; +0,067] | 0,995 |
| stretch+mạnh | beta | 0,769 | **0,809** | 0,893 | +0,040 | 32,1% | [+0,018; +0,067] | 0,995 |
| stretch+mạnh | platt | 0,769 | 0,804 | 0,893 | +0,036 | 28,6% | [+0,013; +0,062] | 0,995 |
| stretch+nhẹ | intercept | 0,764 | 0,773 | 0,853 | +0,009 | 10,0% | [+0,000; +0,022] | 0,985 |
| letterbox (cả hai) | — | | | | ≤ 0 | | | |

Độ nhạy giữ nguyên 0,995 với KTC [0; 0] — mức tăng này không đánh đổi độ nhạy.

## Kết quả 4 — Nhưng lựa chọn đó không hợp lệ

Pre-ensemble gần như không đổi khả năng phân biệt:

| Cấu hình | AUC post | AUC pre | ΔAUC | Tương quan hạng |
|---|---:|---:|---:|---:|
| stretch+mạnh, intercept | 0,9779 | 0,9777 | −0,0002 | 0,9985 |

AUC không tăng. Mức +0,040 đến từ việc xáo nhẹ thứ hạng quanh ngưỡng, tình cờ
đẩy 9 group NORMAL sang đúng phía. Không có cơ chế nào đảm bảo điều đó lặp lại
trên dữ liệu khác.

Nghiêm trọng hơn: **phân biệt pre-ensemble với post-ensemble không tồn tại trên
OOF.** Dự đoán OOF vốn là single-model — không có gì để gộp. Nên không có tín
hiệu nguồn nào chỉ ra nên chọn pre thay vì post; lựa chọn đó chỉ đọc được từ
bảng benchmark.

Theo đúng quy tắc đã đặt — benchmark không được dùng để chọn bất cứ thứ gì —
kết quả này là **thăm dò, không được áp dụng**.

## Kết luận

Trả lời trực tiếp ba câu hỏi ban đầu:

**Mất phân biệt.** Là thành phần lớn nhất ở baseline. Trần do phân biệt trên
benchmark là 0,773 so với 0,997 ở OOF. Không calibration nào chạm được vào phần
này.

**Điểm làm việc không chuyển được.** Có thật, chiếm 31–58% tùy cấu hình.

**Xác suất sai nghĩa.** Có thật và lớn ở đích (ECE 0,12–0,24 so với 0,006–0,023
ở nguồn), nhưng **source-only calibration không sửa được** — nó làm xấu thêm.

Đây là trường hợp 4 trong khung phân loại: khoảng cách operating point tồn tại
về mặt chẩn đoán nhưng không sửa được bằng calibration học từ OOF. Muốn sửa cần
dữ liệu hiệu chuẩn giống đích hoặc domain adaptation — cả hai đều đòi nhãn từ
phân phối đích, thứ dự án hiện chưa có.

## Việc không nên làm tiếp

Không nên áp dụng pre-ensemble + intercept cho stretch+mạnh dù nó cho +0,040.
Lựa chọn đó sinh ra từ việc đọc benchmark, AUC không cải thiện, và không có
cách nào xác nhận từ nguồn.

## Việc đáng làm tiếp

Phần lớn nhất — mất khả năng phân biệt, 42–69% khoảng cách — vẫn chưa có lời
giải. Tỉ lệ khung đã bị loại trừ (chỉ 0,03 AUC). Nghi phạm còn lại chưa kiểm:
độ sáng, độ tương phản, độ phân giải gốc, độ nét/nhiễu, tỉ lệ cơ thể trong
khung, thống kê viền, họ tên file.

## File sinh ra

```
results_calibration_metrics.csv
results_calibration_operating_points.csv
results_calibration_gap_decomposition.csv
results_pre_vs_post_ensemble_calibration.csv
figures/reliability_oof_vs_benchmark.png
figures/score_distribution_oof_vs_benchmark.png
```
