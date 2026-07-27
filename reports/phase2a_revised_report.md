# Phase 2A — Nguồn gốc của dịch chuyển trong lớp NORMAL

Không huấn luyện lại mạng nào. Đơn vị suy luận là filename-derived group.

Tái tạo:

```bash
python3 scripts/fold_matched_score_shift.py
python3 scripts/extract_nuisance_features.py
python3 scripts/nuisance_domain_screen.py
python3 scripts/nuisance_matching.py
python3 scripts/nuisance_domain_models.py
```

## Tóm tắt

Ảnh NORMAL trong tập train được lưu ở **JPEG quality 95,5**; mọi nhóm còn lại —
kể cả NORMAL trong tập test — ở **quality 75,0**. Xác nhận trực tiếp từ bảng
lượng tử hóa trong header, không phải suy ra từ dung lượng file.

Chênh lệch này nhìn thấy được trong pixel, tách hai lớp trong development gần
như hoàn hảo, và biến mất hoàn toàn ở benchmark.

Một cơ chế duy nhất giải thích mọi quan sát của dự án.

## Bước 0 — Đo lại shift bằng cùng một scoring function

Phép đo trước đặt OOF single-model cạnh benchmark ensemble. Đo lại bằng chính
mô hình fold *k* trên cả hai phía:

| Cấu hình | KS NORMAL (trung vị, 5 fold) | KS PNEUMONIA | Tỉ số KS | Tỉ số W₁ |
|---|---|---|---:|---:|
| letterbox+nhẹ | 0,673 [0,622–0,692] | 0,118 | **5,7×** | 13,9× |
| stretch+nhẹ | 0,604 [0,586–0,653] | 0,073 | **8,3×** | 19,3× |
| letterbox+mạnh | 0,661 [0,594–0,689] | 0,096 | **6,9×** | 10,9× |
| stretch+mạnh | 0,571 [0,473–0,597] | 0,083 | **6,9×** | 9,0× |

Bất đối xứng giữ nguyên, ổn định qua clip 1e-4 đến 1e-8. Không phải hiệu ứng đo
lường.

## Tầng 1 — Đặc trưng nào phân biệt NORMAL nguồn với NORMAL đích

41 feature, 4 họ đăng ký trước, trích trên ảnh gốc trước resize.

| Feature | NORMAL dev→ben | PNEUMONIA dev→ben | Lớp trong dev | Lớp trong ben |
|---|---:|---:|---:|---:|
| **file_size_per_pixel** | **0,998** | 0,528 | **0,999** | **0,505** |
| **noise_estimate** | **0,990** | 0,541 | **0,985** | 0,598 |
| file_size_bytes | 0,946 | 0,502 | 0,998 | 0,892 |
| laplacian_variance | 0,808 | 0,596 | 0,504 | 0,850 |
| aspect | 0,704 | 0,506 | 0,882 | 0,700 |

Đọc hai cột cuối của dòng đầu: trong development, dung lượng trên mỗi pixel tách
NORMAL khỏi PNEUMONIA ở **AUC 0,999**. Ở benchmark, **0,505** — đúng bằng ngẫu
nhiên.

Chỉ 5/41 feature qua cổng sàng lọc. Họ photometry không có feature nào.

## Kiểm chứng ở cấp header JPEG

Dung lượng trên mỗi pixel **không** tự chứng minh chất lượng nén khác nhau — nó
còn phụ thuộc entropy nội dung, nhiễu cảm biến và độ sắc nét. Bảng lượng tử hóa
thì khác: nó là thiết lập của bộ mã hóa, nằm trong file, không phụ thuộc ảnh.

| Thư mục | n | Quality ước lượng | p25–p75 | qtable phổ biến nhất |
|---|---:|---:|---|---|
| **train/NORMAL** | 1.341 | **95,5** | 95,5–95,5 | `bc79aa8c5699` (96,9%) |
| train/PNEUMONIA | 3.875 | 75,0 | 75,0–75,0 | `5fe3e571bb5a` (89,3%) |
| test/NORMAL | 234 | 75,0 | 75,0–75,0 | `5fe3e571bb5a` (100%) |
| test/PNEUMONIA | 390 | 75,0 | 75,0–75,0 | `5fe3e571bb5a` (90,3%) |

Bảng thực tế, góc 4×4 trên-trái (hệ số nhỏ = nén ít):

```
train/NORMAL      [[1 1 1 2]     tổng 64 hệ số =  330
                   [1 1 1 2]
                   [1 1 2 3]
                   [2 2 3 4]]

mọi nhóm khác     [[8 6 5 8]     tổng 64 hệ số = 1858
                   [6 6 7 10]
                   [7 7 8 12]
                   [7 9 11 15]]
```

`train/PNEUMONIA` và `test/NORMAL` dùng **bảng giống hệt nhau**. `train/NORMAL`
dùng bảng riêng, lượng tử hóa thô hơn 5,6 lần ở các nhóm còn lại.

**Bảng lượng tử hóa gần như xác định được lớp trong development:**

| Miền | Tỉ lệ ảnh dùng qtable dùng chung giữa hai lớp |
|---|---:|
| development | **3,4%** |
| benchmark | **93,9%** |

Trong tập train, 96,6% ảnh dùng một bảng lượng tử hóa **chỉ thuộc về một lớp**.
Nhãn đọc được từ header JPEG.

## Cùng kết luận, đọc thẳng từ đĩa

| Thư mục | n | KB trung vị | MP trung vị | **KB/MP** |
|---|---:|---:|---:|---:|
| train/NORMAL | 1.341 | 544,0 | 2,27 | **245,1** |
| train/PNEUMONIA | 3.875 | 71,5 | 0,89 | 79,3 |
| test/NORMAL | 234 | 172,1 | 2,47 | **79,5** |
| test/PNEUMONIA | 390 | 62,4 | 0,79 | 76,1 |

`train/NORMAL` là ngoại lệ duy nhất. Ba nhóm còn lại nằm trong khoảng 76–80
KB/MP; nhóm này ở 245.

Tín hiệu này **nhìn thấy được trong pixel**: Spearman giữa
`file_size_per_pixel` và `noise_estimate` là **+0,941**. Mạng không đọc được
dung lượng file, nhưng đọc được độ nhiễu/hạt mà nén ít để lại.

## Tầng 2 — Liên hệ với score và lỗi trong riêng benchmark NORMAL

|Feature | \|rho\| TB | Cliff's delta TB | Số cấu hình p<0,05 |
|---|---:|---:|---:|
| file_size_bytes | 0,493 | −0,566 | **4/4** |
| aspect | 0,485 | +0,558 | 4/4 |
| laplacian_variance | 0,333 | +0,459 | 4/4 |
| noise_estimate | 0,247 | +0,357 | 4/4 |
| file_size_per_pixel | 0,177 | +0,257 | 3/4 |

## Tầng 3 — Cân bằng có làm khoảng cách score giảm không

Ghép cặp NORMAL nguồn ↔ NORMAL đích, chỉ dùng feature và nhãn miền, không dùng
score mô hình.

### Hai feature mạnh nhất KHÔNG cân bằng được

| Feature | SMD trước | Cặp ghép được | Tỉ lệ |
|---|---:|---:|---:|
| file_size_per_pixel | −6,01 | 3 | **1,3%** |
| noise_estimate | −2,29 | 3 | **1,3%** |
| propensity đa biến | 5,50 | **0** | **0,0%** |

Chưa tới 2% ảnh NORMAL ở benchmark tìm được ảnh NORMAL ở development trong
phạm vi 0,2 SD. Propensity đa biến **không có một cặp nào**. Hai miền không có
vùng chồng lấn chung — bản thân điều đó là bằng chứng mạnh nhất về mức tách
biệt.

### Nơi cân bằng được, khoảng cách giảm một nửa

W₁ giữa score NORMAL nguồn và NORMAL đích, so với subset ngẫu nhiên cùng cỡ:

| Cấu hình | Feature | W₁ trước | W₁ sau | Suy giảm | Ngẫu nhiên | p |
|---|---|---:|---:|---:|---:|---:|
| letterbox+nhẹ | file_size_bytes | 9,94 | 5,04 | **49,3%** | −0,0% | 0,0000 |
| stretch+nhẹ | file_size_bytes | 8,23 | 3,96 | **51,8%** | 0,1% | 0,0000 |
| letterbox+mạnh | file_size_bytes | 8,14 | 4,34 | **46,7%** | 0,1% | 0,0000 |
| stretch+mạnh | file_size_bytes | 5,66 | 2,51 | **55,7%** | 0,1% | 0,0000 |
| letterbox+nhẹ | aspect | 9,94 | 8,50 | 14,6% | −0,1% | 0,0000 |
| letterbox+nhẹ | laplacian_variance | 9,94 | 11,87 | **−19,4%** | −0,0% | 1,0000 |

Cân bằng dung lượng file xóa được khoảng **một nửa** khoảng cách score, lặp lại
trên cả bốn cấu hình. Tỉ lệ khung chỉ xóa được 13–17%. Cân bằng
`laplacian_variance` làm khoảng cách **tệ hơn**.

## Phase 2A.2 — Mô hình đa biến

Chỉ dùng đặc trưng thu nhận, không thấy một pixel phổi nào:

| Lớp | Mô hình | AUC OOF | Theo từng fold |
|---|---|---:|---|
| **NORMAL** | logistic | **1,0000** | 1,000 ×5 |
| **NORMAL** | boosting | **0,9999** | 1,000 ×5 |
| PNEUMONIA | logistic | 0,6853 | 0,632–0,729 |
| PNEUMONIA | boosting | 0,7743 | 0,740–0,821 |

Tách **hoàn hảo** development NORMAL khỏi benchmark NORMAL. Cùng bộ feature chỉ
đạt 0,69–0,77 với PNEUMONIA.

Dự đoán ảnh NORMAL nào ở benchmark sẽ thành false positive, chỉ từ đặc trưng
thu nhận:

| Cấu hình | AUC OOF |
|---|---:|
| resnet18 | 0,869 |
| stretch_nhe | 0,888 |
| augment_manh | 0,836 |
| stretch_manh | 0,858 |

## Cơ chế

Năm quan sát, một nguyên nhân:

1. Ảnh NORMAL trong train được lưu ở JPEG quality 95,5; mọi nhóm khác ở 75,0.
2. Trong development, điều đó cho một bộ phân loại gần hoàn hảo không liên quan
   gì tới phổi (AUC 0,999). Mô hình học nó → **OOF group AUC 0,9994**.
3. Ở benchmark, đặc trưng này không mang thông tin lớp nào (AUC 0,505).
4. Shortcut vẫn kích hoạt: ảnh NORMAL ở benchmark "nén giống PNEUMONIA" → bị
   chấm điểm cao → **độ đặc hiệu sụp còn 0,67**.
5. Ảnh PNEUMONIA không đổi giữa hai miền → **độ nhạy giữ 0,995**.

## Đánh giá theo năm tiêu chí đã đăng ký

| Tiêu chí | Kết quả |
|---|---|
| 1. Dịch chuyển nguồn→đích trong NORMAL | AUC 0,998 (đơn biến), 1,000 (đa biến) |
| 2. Yếu hơn rõ trong PNEUMONIA | 0,528 và 0,685–0,774 |
| 3. Liên hệ với score/lỗi ở benchmark NORMAL | 4/4 cấu hình; dự đoán FP AUC 0,84–0,89 |
| 4. Cân bằng làm giảm vượt đối chứng ngẫu nhiên | 46,7–55,7%, p = 0,0000 |
| 5. Lặp lại trên ≥2 cấu hình | Cả 4 |

Đạt cả năm. Theo định nghĩa đã thống nhất, đây là **strong mitigation target**.

## Diễn giải được phép và chưa được phép

**Được phép.** Mức nén ảnh khác nhau giữa hai lớp trong tập train tạo ra một
đặc trưng tắt nhìn thấy được trong pixel; nó giải thích phần lớn khoảng cách
OOF→benchmark và tập trung vào lớp NORMAL.

**Chưa được phép.** Chưa chứng minh trực tiếp mạng dùng dấu vết nén — mới có ba
tầng bằng chứng quan sát cộng một phép cân bằng. Bằng chứng nhân quả cần can
thiệp trên chính ảnh: nén lại toàn bộ dataset về một mức chất lượng rồi huấn
luyện lại. Đó là thí nghiệm tiếp theo, và nó rẻ.

## Hệ quả cho phần còn lại của dự án

Hướng geometry-invariant learning nên dừng. Tỉ lệ khung xóa được 13–17% khoảng
cách; mức nén xóa được 47–56%, và không cân bằng nổi bằng ghép cặp.

Thí nghiệm tiếp theo đáng làm nhất:

1. **Chuẩn hóa nén.** Giải mã và lưu lại toàn bộ 5.856 ảnh ở cùng một mức chất
   lượng JPEG, huấn luyện lại. Nếu OOF AUC tụt từ 0,9994 và khoảng cách
   OOF→benchmark thu hẹp, cơ chế được xác nhận bằng can thiệp.
2. **Augmentation theo mức nén.** Nén lại ngẫu nhiên trong lúc huấn luyện.
3. Cả hai đều rẻ hơn nhiều so với adversarial training, và nhắm đúng thứ đo
   được.

## Cảnh báo cho các kết quả công bố trên dataset này

Tập chia gốc cho phép phân biệt hai lớp gần như hoàn hảo chỉ bằng dấu vết mã
hóa: 96,6% ảnh trong development mang bảng lượng tử hóa chỉ thuộc một lớp. Do
đó **các kết quả nội bộ trên 99% công bố trên tập chia này tương thích với việc
khai thác shortcut, trừ khi đặc tính mã hóa đã được kiểm soát tường minh.**

Điều này không chứng minh rằng mọi mô hình đã công bố đều dựa vào shortcut —
mỗi nghiên cứu có thể chia lại, resize hoặc mã hóa lại. Nhưng nó là cảnh báo
đối với mọi kết quả dùng tập chia development gốc mà không audit mã hóa.

## File sinh ra

```
results_fold_matched_score_shift.csv
nuisance_feature_manifest.csv
results_normal_domain_shift.csv
results_nuisance_error_association.csv
results_nuisance_matching.csv
results_multivariable_domain_models.csv
```
