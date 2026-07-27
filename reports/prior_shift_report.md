# Phase 2A.0 — Prior shift và giả định label shift

Chạy trên prediction đã đóng băng, không huấn luyện lại. Đơn vị suy luận là
filename-derived group.

Tái tạo: `python3 scripts/prior_shift_audit.py`

## Prevalence

| Mức | Nguồn | Đích | Δ log-odds |
|---|---:|---:|---:|
| ảnh | 0,7422 (n=5.232) | 0,6250 (n=624) | −0,546 |
| **filename-group** | **0,6678** (n=3.669) | **0,4743** (n=428) | **−0,801** |

Ở mức group — đơn vị suy luận chính — dịch chuyển mạnh hơn mức ảnh đáng kể,
vì group PNEUMONIA trong development chứa nhiều ảnh hơn.

Calibration intercept quan sát được trên benchmark nằm trong khoảng −2,07 đến
−3,30. Prior shift (−0,801) giải thích khoảng **24–39%** độ lệch đó. Phần lớn
còn lại đến từ nguồn khác.

## Kết quả chính — giả định label shift bị vi phạm, và vi phạm bất đối xứng

Khoảng cách giữa phân phối score ở nguồn và ở đích, tính riêng theo từng lớp,
trên thang log-odds:

| Cấu hình | KS NORMAL | KS PNEUMONIA | W₁ NORMAL | W₁ PNEUMONIA |
|---|---:|---:|---:|---:|
| letterbox+nhẹ | **0,702** | 0,108 | **10,46** | 0,59 |
| stretch+nhẹ | **0,668** | 0,184 | **8,60** | 1,18 |
| letterbox+mạnh | **0,703** | 0,140 | **8,26** | 0,84 |
| stretch+mạnh | **0,604** | 0,200 | **5,73** | 1,44 |

Đây là phát hiện quan trọng nhất của giai đoạn này.

**Lớp PNEUMONIA gần như không đổi** giữa hai split (KS 0,11–0,20, W₁ 0,6–1,4).
**Lớp NORMAL đổi hoàn toàn** (KS 0,60–0,70, W₁ 5,7–10,5 đơn vị log-odds).

Chênh lệch 7–10 lần. Đây không phải label shift — đây là **covariate shift có
điều kiện theo lớp, tập trung hầu như toàn bộ vào lớp NORMAL**.

Điều đó giải thích trực tiếp mọi thứ quan sát được từ đầu dự án:

- độ nhạy giữ nguyên 0,995 trên mọi cấu hình — ca viêm phổi ở benchmark trông
  giống hệt ca viêm phổi ở development;
- độ đặc hiệu sụp — ca bình thường ở benchmark không giống ca bình thường ở
  development;
- calibration học từ nguồn làm xấu thêm ở đích — nó học quan hệ từ một phân
  phối NORMAL không còn tồn tại;
- prior correction không cứu được — không phải bài toán tỉ lệ trộn.

## Ước lượng prevalence đích không dùng nhãn đích

Cả hai phương pháp đều giả định label shift, nên khi giả định sai chúng hỏng:

| Cấu hình | EM | BBSE | Sai số EM | Sai số BBSE |
|---|---:|---:|---:|---:|
| letterbox+nhẹ | 0,7182 | 0,6466 | **+0,244** | +0,172 |
| stretch+nhẹ | 0,6554 | 0,6107 | +0,181 | +0,136 |
| letterbox+mạnh | 0,7229 | 0,6611 | **+0,249** | +0,187 |
| stretch+mạnh | 0,5710 | 0,6053 | +0,097 | +0,131 |

Thật là 0,4743. Cả hai đều **ước lượng quá cao** ở mọi cấu hình, sai số +0,10
đến +0,25. Không có ước lượng prevalence nào dùng được.

## Hiệu chỉnh prior thay đổi được gì

Ngưỡng giữ nguyên từ OOF; chỉ score được dịch.

| Cấu hình | Hiệu chỉnh | Đặc hiệu | Độ nhạy | Brier | Intercept |
|---|---|---:|---:|---:|---:|
| letterbox+nhẹ | không | 0,702 | 0,995 | 0,203 | −3,03 |
| | oracle | 0,711 | 0,985 | 0,177 | −2,63 |
| | EM | 0,693 | 0,995 | 0,211 | −3,15 |
| stretch+nhẹ | không | 0,764 | 0,990 | 0,149 | −2,75 |
| | oracle | 0,836 | 0,985 | 0,120 | −2,24 |
| | EM | 0,764 | 0,990 | 0,147 | −2,71 |
| letterbox+mạnh | không | 0,676 | 0,995 | 0,194 | −3,30 |
| | oracle | 0,724 | 0,985 | 0,164 | −2,77 |
| | EM | 0,658 | 0,995 | 0,205 | −3,48 |
| **stretch+mạnh** | không | 0,769 | 0,995 | 0,086 | −2,07 |
| | **oracle** | **0,840** | **0,995** | **0,065** | −1,43 |
| | **EM** | **0,813** | **0,995** | **0,074** | −1,74 |

Oracle dùng prevalence thật của benchmark, **không triển khai được**, chỉ để đo
trần.

Hai quan sát:

**Ở nhánh letterbox, hiệu chỉnh gần như vô dụng** và EM còn làm tệ đi.

**Ở `stretch+mạnh`, EM cho +0,044 độ đặc hiệu mà không mất độ nhạy** — và EM
không dùng nhãn đích, nên về nguyên tắc triển khai được. Nhưng nó ước lượng
0,571 trong khi thật là 0,474: nó giúp **vì lý do sai**, chỉ tình cờ đẩy ngưỡng
đúng hướng. Và không có tín hiệu nào ở nguồn cho biết nên dùng EM cho cấu hình
này mà không dùng cho cấu hình kia. Cùng vấn đề đã gặp ở calibration audit.

## Kết luận

Prior shift có thật (−0,801 log-odds ở mức group) và giải thích 24–39% độ lệch
calibration intercept. Nhưng nó **không phải cơ chế chính**, và không sửa được
bằng công cụ nào không dùng nhãn đích.

Cơ chế chính là **phân phối ảnh NORMAL ở benchmark khác hẳn phân phối ảnh
NORMAL ở development**, trong khi ảnh PNEUMONIA thì không.

## Điều này đổi hướng Phase 2A.1

Câu hỏi ban đầu dự định là: *đặc trưng thu nhận nào phân biệt NORMAL với
PNEUMONIA?* Đó chính là câu hỏi mà tỉ lệ khung đã trả lời, và câu trả lời chỉ
đáng 0,03 AUC.

Câu hỏi đúng sau kết quả này là:

> **Đặc trưng thu nhận nào phân biệt ảnh NORMAL ở development với ảnh NORMAL ở
> benchmark?**

Đây là câu hỏi khác hẳn, sắc hơn, và nhắm thẳng vào hard-NORMAL subgroup. Nó
cũng kiểm được bằng đúng bộ feature đã đăng ký trước, chỉ đổi phép so sánh:
thay vì `NORMAL vs PNEUMONIA`, so `NORMAL nguồn vs NORMAL đích`.

Phép so cũ vẫn nên giữ làm control, nhưng không còn là câu hỏi chính.

## File sinh ra

```
results_prior_shift.csv
results_label_shift_assumptions.csv
```
