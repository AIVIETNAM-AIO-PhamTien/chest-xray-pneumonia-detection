# Handbook: Deep Learning cho phân loại viêm phổi trên ảnh X-quang ngực

> Tài liệu tổng hợp nội bộ dành cho nhóm thực hiện project **Chest X-ray Pneumonia Classification**.
>
> Phiên bản tổng hợp: 22/07/2026. Nội dung được rút ra từ toàn bộ PDF, [`tonghop.md`](./tonghop.md) và [`tonghop2.md`](./tonghop2.md) trong thư mục `paper/`. Các case study chỉ có trong ghi chú thứ cấp đã được đối chiếu với trang/toàn văn chính thức khi nguồn cho phép. Các con số được ghi rõ là **paper-reported** khi chúng là tuyên bố của tác giả và chưa được nhóm tái lập.

## Mục lục

1. [Mục tiêu và cách sử dụng tài liệu](#1-mục-tiêu-và-cách-sử-dụng-tài-liệu)
2. [Danh mục tài liệu nguồn](#2-danh-mục-tài-liệu-nguồn)
3. [Bức tranh tổng quát của lĩnh vực](#3-bức-tranh-tổng-quát-của-lĩnh-vực)
4. [Kiến thức nền về X-quang ngực và viêm phổi](#4-kiến-thức-nền-về-x-quang-ngực-và-viêm-phổi)
5. [Các bài toán học máy trên CXR](#5-các-bài-toán-học-máy-trên-cxr)
6. [Dữ liệu và các dataset quan trọng](#6-dữ-liệu-và-các-dataset-quan-trọng)
7. [Một mô hình đọc ảnh như thế nào](#7-một-mô-hình-đọc-ảnh-như-thế-nào)
8. [Các họ kiến trúc quan trọng](#8-các-họ-kiến-trúc-quan-trọng)
9. [Tiền xử lý, augmentation và mất cân bằng lớp](#9-tiền-xử-lý-augmentation-và-mất-cân-bằng-lớp)
10. [Training và evaluation đúng cách](#10-training-và-evaluation-đúng-cách)
11. [Explainability, localization và robustness](#11-explainability-localization-và-robustness)
12. [Tổng hợp từng paper](#12-tổng-hợp-từng-paper)
13. [Giải phẫu paper CNN–BiGRU–SNN 2025](#13-giải-phẫu-paper-cnnbigrusnn-2025)
14. [Các mâu thuẫn và rủi ro trong paper chính](#14-các-mâu-thuẫn-và-rủi-ro-trong-paper-chính)
15. [Protocol reproduction đề xuất cho nhóm](#15-protocol-reproduction-đề-xuất-cho-nhóm)
16. [Roadmap thí nghiệm](#16-roadmap-thí-nghiệm)
17. [Checklist đọc và review một paper mới](#17-checklist-đọc-và-review-một-paper-mới)
18. [Glossary](#18-glossary)

---

## 1. Mục tiêu và cách sử dụng tài liệu

Tài liệu này có bốn mục tiêu:

1. Cung cấp nền tảng chung để mọi thành viên hiểu bài toán y khoa và bài toán machine learning đang làm.
2. Giải thích các mô hình “đọc” ảnh như thế nào, thay vì chỉ liệt kê tên kiến trúc.
3. Tổng hợp chi tiết đóng góp, kết quả và hạn chế của từng paper trong thư mục.
4. Chuyển literature review thành các quyết định kỹ thuật có thể triển khai và kiểm chứng trong project.

Khi đọc, luôn phân biệt ba loại thông tin:

- **Established concept**: kiến thức kỹ thuật được nhiều nguồn thống nhất, ví dụ convolution, patient-level split hoặc transfer learning.
- **Paper-reported result**: kết quả do một paper công bố nhưng nhóm chưa tái lập.
- **Project decision**: quyết định do nhóm chọn để xử lý một điểm mơ hồ hoặc xây dựng protocol đáng tin cậy.

Không nên xem accuracy cao nhất trong một bảng survey là mô hình tốt nhất. Kết quả chỉ so sánh được khi cùng dataset snapshot, split, preprocessing, test set và định nghĩa metric.

### Lộ trình đọc theo vai trò

- **Thành viên mới:** đọc mục 3 → 8 để hiểu bài toán và các họ mô hình.
- **Người làm data:** tập trung mục 6, 9, 10 và 15.
- **Người code model:** đọc kỹ mục 7, 8, 13, 14 và 16.
- **Người chạy experiment:** đọc mục 9, 10, 15 và 16 trước khi train.
- **Người viết report/presentation:** dùng mục 12–14 và luôn giữ nhãn `paper-reported` cho số chưa reproduce.
- **Reviewer:** sử dụng checklist ở mục 17.

---

## 2. Danh mục tài liệu nguồn

### 2.1. Các PDF trong folder

| File | Tài liệu | Phạm vi | Vai trò trong project |
|---|---|---|---|
| [`1702.05747v2.pdf`](./1702.05747v2.pdf) | Litjens et al., *A Survey on Deep Learning in Medical Image Analysis* | Survey hơn 300 nghiên cứu medical imaging đến đầu năm 2017 | Nền tảng về task, CNN/RNN/AE và các thách thức riêng của dữ liệu y tế |
| [`2103.08700v1.pdf`](./2103.08700v1.pdf) | Sogancioglu/Çallı et al., *Deep Learning for Chest X-ray Analysis: A Survey* | 295 paper CXR giai đoạn 2015–2021 | Nguồn phương pháp luận CXR toàn diện nhất trong folder |
| [`2408.13315v1.pdf`](./2408.13315v1.pdf) | Xu, *A systematic review: Deep learning-based methods for pneumonia region detection* | Review ngắn về detection/localization pneumonia | Nguồn định hướng one-stage/two-stage; phải đọc có phê bình |
| [`jimaging-10-00176.pdf`](./jimaging-10-00176.pdf) | Siddiqi & Javaid, *Deep Learning for Pneumonia Detection in Chest X-ray Images: A Comprehensive Survey* | Survey pneumonia/COVID-19, xuất bản 2024 | Tổng hợp dataset, CNN, transfer learning, ensemble, ViT, XAI và research gaps |
| [`s41598-025-23664-x.pdf`](./s41598-025-23664-x.pdf) | Slimi et al., *Trustworthy pneumonia detection in chest X-ray imaging through attention-guided deep learning* | Mô hình hybrid CNN–BiGRU–SNN, 2025 | Paper chính project đang cố gắng reproduce và audit |

### 2.2. Ghi chú literature review có sẵn

[`tonghop.md`](./tonghop.md) bổ sung các case study không có PDF riêng trong folder:

- CheXNet/DenseNet121 trên ChestX-ray14.
- Kermany et al. và pediatric pneumonia dataset.
- Ensemble transfer learning.
- So sánh ImageNet pretrained feature với train from scratch.
- Attention-enhanced ResNet và focal loss.

[`tonghop2.md`](./tonghop2.md) bổ sung bốn lượt đọc:

- Slimi et al. 2025 về CNN–BiGRU–SNN; paper này đã có PDF đầy đủ trong folder.
- Singh et al. 2024 về `DEIT_Base_Patch16_224` cho phân loại pneumonia.
- Sana et al. 2025 về transfer learning VGG-16, custom DNN head và Grad-CAM.
- Srivastava et al. 2025 về phân loại bốn lớp COVID-19/TB/pneumonia/normal bằng các CNN và feature-level SMOTE.

Hai file Markdown là **ghi chú thứ cấp**, không ngang hàng với toàn văn paper. Khi tổng hợp, tài liệu này dùng thứ tự ưu tiên:

```text
PDF/toàn văn chính thức
    > bảng, hình và công thức trong paper
    > abstract/trang metadata chính thức
    > tonghop.md hoặc tonghop2.md
    > suy luận của người đọc
```

Nếu ghi chú và paper khác nhau, handbook giữ nội dung của paper và nêu mâu thuẫn. Những câu kiểu “thường dùng”, “có thể”, “nhiều nghiên cứu làm” trong ghi chú không được biến thành hyperparameter hoặc đóng góp của paper nếu paper không nói như vậy.

---

## 3. Bức tranh tổng quát của lĩnh vực

### 3.1. Quá trình phát triển

Lĩnh vực đã dịch chuyển theo chuỗi sau:

```text
Rule-based image processing
    ↓
Handcrafted features + classical ML
    ↓
CNN học feature end-to-end
    ↓
Transfer learning từ ImageNet
    ↓
Residual/dense/efficient architectures
    ↓
Attention, ensemble và multi-task learning
    ↓
Transformer, self-supervised learning, domain adaptation
    ↓
Hybrid models, XAI, robustness và clinical deployment
```

Litjens et al. khảo sát 308 paper và nhận thấy deep learning đã đi vào hầu hết nhánh medical image analysis chỉ trong vài năm. Survey CXR 2021 tiếp tục ghi nhận 295 paper, trong đó 209 paper dùng ít nhất một public dataset.

### 3.2. Bài học xuyên suốt các survey

- Chất lượng dữ liệu và nhãn thường quan trọng hơn việc đổi backbone.
- Một architecture mới không tự giải quyết domain shift, label noise hoặc leakage.
- Preprocessing, augmentation, sampling và receptive field có thể tạo khác biệt lớn hơn một thay đổi nhỏ trong kiến trúc.
- Public dataset thúc đẩy nghiên cứu nhưng đồng thời khiến nhiều công trình tối ưu cho benchmark thay vì nhu cầu lâm sàng.
- Model được test trên cùng nguồn dữ liệu với training thường suy giảm khi chuyển sang bệnh viện khác.
- Hệ thống hỗ trợ bác sĩ cần localization, uncertainty và integration vào workflow, không chỉ một danh sách probability.
- “Radiologist-level” trong một test protocol cụ thể không có nghĩa model có thể thay thế bác sĩ ngoài thực tế.

### 3.3. Vì sao accuracy rất cao vẫn có thể không đáng tin

Một mô hình có thể đạt 99% vì:

- Ảnh của cùng bệnh nhân xuất hiện ở train và test.
- File trùng được chia vào nhiều split.
- Class khác nhau đến từ máy chụp hoặc bệnh viện khác nhau.
- Model đọc chữ, marker, border, portable-device pattern hoặc artifact thay vì phổi.
- Test set nhỏ hoặc được dùng lặp lại để chọn hyperparameter.
- Bài toán quá đơn giản: trẻ em normal đối đầu với pneumonia rõ ràng đã qua lọc chất lượng.
- Metric được tính trên split không được mô tả đầy đủ.

Vì vậy, mục tiêu của project là **reproduction có kiểm chứng**, không phải bằng mọi cách đạt con số paper công bố.

---

## 4. Kiến thức nền về X-quang ngực và viêm phổi

### 4.1. CXR là gì?

Chest radiograph (CXR) là ảnh chiếu 2D của cấu trúc ba chiều trong lồng ngực. Nó phổ biến vì:

- Chi phí thấp.
- Chụp nhanh.
- Liều bức xạ thấp hơn CT đáng kể.
- Có máy portable cho bệnh nhân tại giường.
- Phù hợp với cơ sở y tế hạn chế nguồn lực.

Nhược điểm chính là các cấu trúc bị chồng lên nhau theo hướng chiếu. Một opacity sau tim, sau xương sườn hoặc gần cơ hoành có thể rất khó nhìn.

CXR lâm sàng thường được lưu dưới dạng DICOM với dynamic range lớn và kích thước có thể khoảng 2.000–4.000 pixel mỗi chiều. Nhiều public dataset đã chuyển sang 8-bit JPEG/PNG và resize nhỏ hơn. Việc này giúp chia sẻ/train dễ hơn nhưng có thể làm mất dynamic range, chi tiết nhỏ và metadata acquisition.

### 4.2. Các tư thế ảnh

| View | Cách chụp | Đặc điểm |
|---|---|---|
| PA | Tia đi từ sau ra trước; bệnh nhân thường đứng | View chuẩn phổ biến, ít phóng đại tim hơn AP |
| AP | Tia đi từ trước ra sau; thường chụp nằm/portable | Hình tim có thể lớn hơn; thường liên quan bệnh nhân nặng và thiết bị hỗ trợ |
| Lateral | Chụp nghiêng | Bổ sung thông tin vùng bị che trên ảnh frontal |

View có thể trở thành confounder. Nếu hầu hết pneumonia là AP portable còn normal là PA đứng, model có thể học view thay vì pathology.

### 4.3. Dấu hiệu liên quan viêm phổi

Các paper nhắc đến:

- Lung opacity hoặc vùng trắng tăng đậm độ.
- Lobar hoặc alveolar consolidation.
- Interstitial infiltrates.
- Mờ các landmark giải phẫu.
- Mất rõ bờ tim hoặc cơ hoành.
- Pleural effusion trong một số trường hợp.

Đây không phải các dấu hiệu chỉ có ở pneumonia. Edema, atelectasis, xuất huyết, khối u hoặc kỹ thuật chụp kém cũng có thể tạo biểu hiện tương tự.

### 4.4. CXR không đủ để khẳng định căn nguyên

Bác sĩ thường kết hợp:

- Triệu chứng và khám lâm sàng.
- Tuổi, tiền sử và yếu tố nguy cơ.
- Xét nghiệm máu, vi sinh hoặc PCR.
- Ảnh trước đó của cùng bệnh nhân.
- View lateral hoặc CT khi cần.

Một model chỉ nhìn một JPEG nên được mô tả là **image classifier/screening aid**, không phải hệ thống tự động xác định chẩn đoán đầy đủ.

---

## 5. Các bài toán học máy trên CXR

### 5.1. Image-level classification

Đầu vào là toàn ảnh; đầu ra là một hoặc nhiều nhãn.

- Binary: `NORMAL` và `PNEUMONIA`.
- Multiclass: normal, bacterial pneumonia, viral pneumonia, COVID-19, v.v.
- Multi-label: mỗi ảnh có thể đồng thời có pneumonia, effusion, edema, cardiomegaly...

Project hiện tại chủ yếu là binary classification.

### 5.2. Object detection/localization

Mô hình vừa dự đoán class vừa dự đoán bounding box vùng nghi ngờ. Metric phù hợp gồm IoU và mAP, không chỉ accuracy.

- One-stage: YOLO, SSD, RetinaNet, EfficientDet.
- Two-stage: Faster R-CNN, Mask R-CNN.

Nhận định “two-stage luôn chính xác hơn one-stage” chỉ là xu hướng lịch sử, không phải quy luật. Kết quả phụ thuộc dataset, backbone, resolution và mục tiêu latency.

### 5.3. Segmentation

Mỗi pixel được gán nhãn, ví dụ lung field hoặc opacity. U-Net là kiến trúc nền tảng:

```text
Encoder giảm kích thước và học context
        ↘ skip connections ↙
Decoder tăng kích thước và phục hồi chi tiết
```

Lung segmentation có thể giúp giới hạn model vào vùng phổi, nhưng cũng có nguy cơ loại mất dấu hiệu ngoại vi hoặc artifact lâm sàng quan trọng nếu mask kém.

### 5.4. Regression và severity scoring

Output là giá trị liên tục hoặc ordinal, ví dụ mức độ nặng, tỷ lệ tim-ngực hoặc nguy cơ dài hạn.

### 5.5. Image generation/enhancement

- Bone suppression.
- Denoising và super-resolution.
- Tạo synthetic images bằng GAN.
- Chuẩn hóa style giữa các bệnh viện.

Synthetic data phải được kiểm tra kỹ để tránh memorization, artifact hoặc thay đổi nhãn bệnh lý.

### 5.6. Domain adaptation và federated learning

- Domain adaptation học representation ít phụ thuộc bệnh viện/máy chụp.
- Federated learning cho phép nhiều bệnh viện cùng train mà không chuyển raw data về một nơi.

Hai hướng này giải quyết các vấn đề khác nhau và không tự đảm bảo privacy, fairness hoặc generalization.

### 5.7. Report generation và retrieval

Các survey CXR chỉ ra rằng workflow thực tế còn bao gồm so sánh ảnh cũ và tạo báo cáo. Đây là khoảng trống lớn vì phần lớn literature chỉ làm image-level classification.

---

## 6. Dữ liệu và các dataset quan trọng

### 6.1. Pediatric Pneumonia/Kermany dataset

Đây là dataset project đang sử dụng.

| Thuộc tính | Mô tả |
|---|---|
| Nguồn | Guangzhou Women and Children’s Medical Center |
| Dân số | Trẻ 1–5 tuổi |
| View paper mô tả | AP |
| Định dạng | JPEG, chủ yếu grayscale, nhiều resolution |
| Nhãn phổ biến | Normal/Pneumonia; tên file pneumonia còn chứa bacterial/viral |
| Snapshot thực tế của nhóm | 5.856 ảnh |
| Split gốc | Train 5.216, validation 16, test 624 |

Phân bố snapshot của nhóm:

| Split | Normal | Pneumonia | Tổng |
|---|---:|---:|---:|
| Train | 1.341 | 3.875 | 5.216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |
| **Tổng** | **1.583** | **4.273** | **5.856** |

Lưu ý:

- Literature ghi không nhất quán 5.856, 5.858 hoặc 5.863 ảnh.
- Dataset chỉ đại diện cho bệnh nhi tại một trung tâm.
- Nhiều ảnh có cùng `person ID`, vì vậy split theo ảnh có nguy cơ patient leakage.
- Dataset trong workspace hiện có một cây thư mục lồng trùng; loader phải chặn việc đếm hai lần.
- Binary label không mô tả các abnormality khác trong class normal/negative.

### 6.2. ChestX-ray14

| Thuộc tính | Giá trị survey 2021 |
|---|---:|
| Ảnh | 112.120 |
| Bệnh nhân | 30.805 |
| Resolution phát hành | 1024×1024, 8-bit grayscale |
| Nhãn | 14 abnormality, trích tự động từ report |
| Localization subset | Bounding box cho một tập nhỏ |

Ưu điểm là quy mô lớn; hạn chế là label noise do report parsing và số positive pneumonia tương đối ít.

### 6.3. CheXpert

- 224.316 CXR từ 65.240 bệnh nhân.
- Có PA, AP và lateral.
- Nhãn gồm present, absent, uncertain và no-mention.
- Phần lớn nhãn được tạo bằng rule-based report labeler; một tập nhỏ có expert consensus.

### 6.4. MIMIC-CXR

- Khoảng 371.920 ảnh từ 64.588 bệnh nhân.
- Có report gốc và DICOM ở phiên bản mới.
- Phù hợp cho pretraining, multi-label classification và image-report learning.

### 6.5. PadChest

- Khoảng 160 nghìn ảnh, 67 nghìn bệnh nhân và 110 nghìn study.
- Có nhiều view và taxonomy nhãn rộng.
- Một phần nhãn do bác sĩ đọc report, phần còn lại được tự động hóa.

### 6.6. PLCO và Open-i

- PLCO: screening cohort lớn, nhãn và location được radiologist cung cấp.
- Open-i: 7.910 ảnh từ 3.955 study/bệnh nhân, có report và MeSH findings.

### 6.7. RSNA Pneumonia Detection Challenge

- Khoảng 30.000 CXR theo survey 2021; một số nguồn ghi 26.684 ảnh sau lọc/chia challenge.
- Có bounding box do radiologist đánh dấu opacity.
- Ba nhóm: normal, lung opacity và abnormal nhưng không có opacity.
- Phù hợp hơn Kermany cho detection/localization.

### 6.8. Dataset nhỏ và COVID-19 datasets

Các survey liệt kê JSRT/SCR, Shenzhen, Montgomery, BIMCV, COVIDGR, COVID-CXR, SIIM-ACR và nhiều dataset tổng hợp khác.

Nguy cơ đặc biệt của COVID datasets:

- Ảnh được lấy từ bài báo, screenshot hoặc nhiều nguồn không đồng nhất.
- Adult COVID bị ghép với pediatric normal/non-COVID pneumonia.
- Class và source gần như trùng nhau, tạo source shortcut.
- Duplicate lan truyền qua nhiều dataset tổng hợp.

### 6.9. Quy tắc vàng khi dùng dataset công khai

1. Xác định patient ID trước khi split.
2. Hash file để phát hiện duplicate.
3. Lưu manifest gồm path, class, patient, source, view và hash.
4. Không infer class ID từ thứ tự alphabet của folder.
5. Chỉ fit transform có state, sampler và balancing trên train.
6. Không dùng test để chọn threshold, epoch hoặc hyperparameter.
7. Giữ một external dataset nếu muốn claim generalization.
8. Kiểm tra class có bị đồng nhất với bệnh viện, tuổi hoặc view hay không.

### 6.10. Audit snapshot đang có trong workspace

Kết quả kiểm tra ngày 22/07/2026 trên **một bản chính** của cây `chest_xray/`:

| Hạng mục | Kết quả |
|---|---:|
| Tổng file ảnh | 5.856 |
| File không mở được | 0 |
| Hash nội dung duy nhất | 5.824 |
| Nhóm duplicate chính xác trong cùng split | 30 |
| Nhóm duplicate chính xác xuyên split | 0 |
| Ảnh mode grayscale `L` | 5.573 |
| Ảnh mode `RGB` | 283 |
| Số kích thước `(width, height)` khác nhau | 4.803 |

Filesystem còn có cây `chest_xray/chest_xray/` lặp lại cùng dataset. Nếu loader quét đệ quy từ folder ngoài cùng, nó sẽ nhìn thấy 11.712 file thay vì 5.856.

Tên file pneumonia cho thấy:

- Train có 3.875 ảnh nhưng chỉ khoảng 1.635 `person ID` khác nhau.
- 1.005 ID trong train có nhiều hơn một ảnh.
- Một ID có thể xuất hiện tới 30 ảnh.
- Khoảng 170 ID dạng `person<number>` xuất hiện ở cả train và test gốc.

Con số cuối cần được xem là **tín hiệu audit**, chưa phải kết luận chắc chắn rằng 170 người bị overlap: nhóm phải xác minh quy ước đặt ID từ nguồn Kermany. Tuy nhiên, chỉ riêng việc một ID có nhiều ảnh đã đủ để yêu cầu group-aware splitting cho validation mới.

---

## 7. Một mô hình đọc ảnh như thế nào

### 7.1. Từ file JPEG đến tensor

Một pipeline điển hình:

```text
JPEG/DICOM
→ decode pixel
→ chọn/chuẩn hóa channel
→ resize/crop
→ scale intensity
→ tensor [B, C, H, W]
→ model
→ logits
→ probability
→ threshold/argmax
→ prediction
```

Với batch 32 ảnh RGB giả lập ở 128×128:

```text
images.shape = [32, 3, 128, 128]
```

Ba channel có thể chỉ là cùng một ảnh grayscale được lặp ba lần. Việc lặp channel không tạo thêm thông tin; mục tiêu là tương thích với backbone pretrained trên RGB.

### 7.2. Convolution đọc pattern cục bộ

Một kernel 3×3 trượt qua ảnh và tính weighted sum. Ở layer đầu, filter thường phản ứng với:

- Edge.
- Hướng sáng-tối.
- Texture đơn giản.

Ở layer sâu hơn, receptive field lớn dần và feature có thể biểu diễn:

- Bờ xương sườn.
- Lung texture.
- Vùng opacity.
- Quan hệ giữa vùng phổi và anatomy xung quanh.

Output convolution là feature map, không phải probability.

### 7.3. Activation, normalization và pooling

- ReLU/LeakyReLU đưa non-linearity vào network.
- BatchNorm chuẩn hóa activation trong batch và có running statistics dùng ở inference.
- MaxPool hoặc stride giảm spatial resolution, tăng receptive field và giảm compute.

Ví dụ ba block CNN với pooling sau mỗi block:

| Bước | Shape minh họa |
|---|---|
| Input | `[B, 3, 128, 128]` |
| Conv 3→32 | `[B, 32, 128, 128]` |
| Pool | `[B, 32, 64, 64]` |
| Conv 32→64 | `[B, 64, 64, 64]` |
| Pool | `[B, 64, 32, 32]` |
| Conv 64→128 | `[B, 128, 32, 32]` |
| Pool | `[B, 128, 16, 16]` |
| Flatten | `[B, 32768]` |

Đây là phép suy shape chuẩn theo mô tả, không phải shape chắc chắn của paper SNN; paper ghi một con số khác và được phân tích ở phần sau.

### 7.4. Từ feature map đến quyết định

Có hai cách phổ biến:

- Flatten toàn bộ feature map rồi dùng fully connected layer.
- Global average pooling mỗi channel thành một scalar rồi dùng classifier.

Global average pooling thường ít parameter và ít overfit hơn flatten lớn.

Classifier sinh **logits**. Logit chưa phải probability:

- Một output: dùng `BCEWithLogitsLoss`; inference dùng sigmoid.
- Hai output: dùng `CrossEntropyLoss`; inference dùng softmax.

Không nên sigmoid trước `BCEWithLogitsLoss` hoặc softmax trước `CrossEntropyLoss`, vì loss đã thực hiện phép biến đổi ổn định số học bên trong.

### 7.5. Training khác inference như thế nào?

Trong training:

```text
image → logits → loss(label, logits)
      → backpropagation
      → optimizer cập nhật weight
```

Trong inference:

```text
image → logits → probability → threshold → prediction
```

Inference phải dùng `model.eval()` để dropout tắt và BatchNorm dùng running statistics.

### 7.6. Model thật sự có thể đang đọc gì?

Ngoài bệnh lý, model có thể đọc:

- Marker trái/phải.
- Chữ, viền, crop và compression.
- Portable AP pattern.
- Tuổi và kích thước cơ thể.
- Thiết bị hỗ trợ.
- Style của bệnh viện hoặc máy chụp.

Do đó, accuracy và heatmap phải được kết hợp với audit dữ liệu, external validation và error analysis.

### 7.7. Cách tự kiểm tra tensor shape trong một architecture table

Khi đọc paper, không nên sao chép layer theo tên rồi hy vọng chúng nối được với nhau. Với convolution hai chiều, kích thước output theo mỗi chiều là:

\[
H_{out}=\left\lfloor\frac{H_{in}+2P-D(K-1)-1}{S}+1\right\rfloor
\]

Trong đó (K) là kernel, (S) là stride, (P) là padding và (D) là dilation. Công thức tương tự cho chiều rộng.

Ví dụ `kernel=3`, `stride=1`, `padding=1`, `dilation=1` giữ nguyên height/width. MaxPool 2×2 stride 2 thường giảm mỗi chiều một nửa.

Số parameter của `Conv2d`, nếu có bias:

\[
(C_{in}\cdot K_h\cdot K_w+1)\cdot C_{out}
\]

Số parameter của linear layer:

\[
(F_{in}+1)\cdot F_{out}
\]

Các sanity check bắt buộc:

1. Output channel của layer trước phải bằng input channel của layer sau.
2. Tích `C×H×W` phải đúng với `in_features` của linear layer sau flatten.
3. Tensor đưa vào GRU với `batch_first=True` phải có shape `[B, T, F]`.
4. BiGRU hidden size `H` trả output cuối chiều `2H`, không phải `H`.
5. Với `L` layer BiGRU, `h_n` thường có shape `[2L, B, H]`; phải chọn/ghép đúng direction và layer.
6. SNN state phải reset giữa các sample/batch theo đúng thiết kế.
7. Tổng parameter tính từ layer phải xấp xỉ con số paper báo cáo.

Nếu shape chỉ khớp sau khi tự đổi thứ tự block hoặc thêm projection không được paper nói tới, đó là một **implementation assumption** và phải được ghi vào reproduction report.

---

## 8. Các họ kiến trúc quan trọng

### 8.1. Classical ML với handcrafted features

Pipeline:

```text
CXR → LBP/HOG/SIFT/radiomics → feature vector → SVM/RF → label
```

Ưu điểm là dễ phân tích feature; hạn chế là feature không được tối ưu end-to-end và phụ thuộc nhiều vào thiết kế thủ công.

### 8.2. CNN train from scratch

CNN tự học feature từ dữ liệu đích. Phù hợp khi dataset đủ lớn hoặc model nhỏ. Với Kermany dataset, model lớn train từ scratch dễ overfit.

### 8.3. VGG

VGG xếp nhiều convolution 3×3 liên tiếp:

```text
[Conv 3×3] × N → Pool → tăng channel → ... → classifier
```

Ưu điểm là đơn giản; nhược điểm là nhiều parameter và compute, đặc biệt ở fully connected layers.

### 8.4. ResNet

Residual block học phần dư:

\[
y = F(x) + x
\]

Skip connection giúp gradient đi qua network sâu dễ hơn. ResNet18 là baseline nhẹ, ResNet50 sâu hơn và thường dùng transfer learning.

### 8.5. DenseNet

Trong dense block, mỗi layer nhận concatenation output của tất cả layer trước:

\[
x_l = H_l([x_0, x_1, ..., x_{l-1}])
\]

Ưu điểm:

- Feature reuse.
- Gradient flow tốt.
- DenseNet121 đã trở thành backbone phổ biến trong CXR sau CheXNet.

Nhược điểm là activation memory có thể lớn vì concatenation.

### 8.6. Inception và Xception

- Inception xử lý song song nhiều kernel/receptive field rồi ghép feature.
- Xception dùng depthwise separable convolution để tách spatial convolution và channel mixing.

Hai họ này thường xuất hiện trong transfer-learning baseline của pneumonia classification.

### 8.7. MobileNet và EfficientNet

- MobileNet tối ưu cho thiết bị hạn chế compute bằng depthwise separable convolution.
- EfficientNet compound-scales depth, width và resolution thay vì tăng một chiều riêng lẻ.

Model nhỏ hơn không mặc nhiên tốt hơn nếu preprocessing, calibration hoặc external performance kém.

### 8.8. Transfer learning

Có ba chiến lược:

1. **Frozen feature extractor**: đóng băng backbone, chỉ train head.
2. **Partial fine-tuning**: train head và một số block cuối.
3. **Full fine-tuning**: train toàn bộ với learning rate nhỏ.

Quy trình an toàn:

```text
Train head với backbone frozen
→ kiểm tra validation
→ unfreeze dần block cuối
→ giảm learning rate
→ fine-tune
```

Khi dùng ImageNet weights, phải dùng normalization tương ứng hoặc chứng minh normalization khác tốt hơn qua thí nghiệm có kiểm soát.

### 8.9. Ensemble

Các cách kết hợp:

- Trung bình probability.
- Weighted average.
- Majority vote.
- Stacking bằng meta-classifier.
- Concatenate embedding rồi học fusion head.

Chỉ fit trọng số ensemble/meta-classifier trên validation, không dùng test.

### 8.10. Channel và spatial attention

Channel attention trả lời: **feature channel nào quan trọng?**

Spatial attention trả lời: **vị trí nào trên feature map quan trọng?**

CBAM thường áp dụng channel attention rồi spatial attention. SE block chủ yếu recalibrate channel.

Trainable attention thay đổi feature và có thể ảnh hưởng prediction. Đây là khái niệm khác với Grad-CAM post-hoc.

### 8.11. Vision Transformer

Pipeline ViT đơn giản:

```text
Image
→ chia thành patch
→ linear patch embedding
→ thêm positional embedding và class token
→ multi-head self-attention
→ MLP blocks
→ classification head
```

Self-attention tính quan hệ giữa các token/patch, giúp model nắm context toàn cục. Tuy nhiên:

- Cần pretraining hoặc dữ liệu đủ lớn.
- Complexity attention tăng theo bình phương số token trong ViT chuẩn.
- Attention weights không tự động là explanation đáng tin.
- Hybrid CNN–Transformer có thể dùng CNN học local texture rồi Transformer học global relation.

**Ví dụ cụ thể từ Singh et al. 2024:** model thực nghiệm là `DEIT_Base_Patch16_224` dùng pretrained weights. Tên model cho biết ảnh 224×224 được chia thành patch 16×16:

```text
224 / 16 = 14 patch trên mỗi chiều
14 × 14 = 196 image tokens
196 tokens + 1 class token = 197 tokens
mỗi patch RGB thô có 16 × 16 × 3 = 768 giá trị
```

Luồng đọc ảnh có thể hiểu là:

```text
[B, 3, 224, 224]
→ 196 patch không chồng lấp
→ patch embedding + positional embedding
→ chuỗi 197 token qua các Transformer encoder
→ lấy representation của class token
→ classification head cho hai lớp
```

Khác CNN, ViT không trượt kernel qua từng vùng lân cận. Mỗi attention head tạo `query`, `key`, `value` cho token và trộn thông tin dựa trên độ tương đồng. Vì vậy patch ở hai vị trí xa nhau có thể tương tác ngay trong một block. Tuy nhiên, CNN sâu cũng có thể đạt receptive field toàn cục; khác biệt nằm ở inductive bias và cơ chế tương tác, không phải “CNN chỉ nhìn local”. Token hóa theo patch 16×16 làm yếu inductive bias cục bộ và có thể khiến việc học dấu hiệu rất nhỏ khó hơn nếu dữ liệu/pretraining không đủ; nó không mặc nhiên xóa thông tin vì patch thô 768 giá trị được chiếu sang embedding. Model này có khoảng 85,8 triệu trainable parameter theo paper, nên không phải lựa chọn nhẹ chỉ vì dùng attention. Case study và các lỗi đếm dữ liệu của paper được phân tích ở mục 12.8.

### 8.12. GRU/BiGRU

GRU giữ hidden state và dùng gate để điều khiển thông tin.

Update gate:

\[
z_t = \sigma(W_zx_t + U_zh_{t-1} + b_z)
\]

Reset gate:

\[
r_t = \sigma(W_rx_t + U_rh_{t-1} + b_r)
\]

Candidate state:

\[
\tilde{h}_t = \tanh(W_hx_t + U_h(r_t \odot h_{t-1}) + b_h)
\]

Final state thường là tổ hợp của state cũ và candidate state. BiGRU chạy hai hướng và concatenate output forward/backward.

GRU có ý nghĩa tự nhiên với chuỗi, video hoặc nhiều slice. Với một ảnh tĩnh bị lặp y hệt qua 25 timestep, “temporal dependency” là nhân tạo và phải được chứng minh bằng ablation.

### 8.13. Spiking Neural Network và LIF neuron

SNN truyền thông tin bằng spike nhị phân. LIF neuron tích lũy membrane potential:

\[
V_{t+1} = \beta V_t + (1-\beta)I_t
\]

Khi vượt threshold:

\[
S_t = \mathbb{1}[V_t \geq V_{th}]
\]

Sau spike, membrane được reset. Vì hàm step không khả vi, training end-to-end thường thay gradient thật bằng surrogate gradient.

Các kiểu encoding ảnh tĩnh thành spike:

| Cách mã hóa | Cách hoạt động | Ưu điểm | Rủi ro |
|---|---|---|---|
| Rate coding | Cường độ quyết định số spike/tần suất firing | Đơn giản, chịu jitter thời gian | Cần nhiều timestep và nhiều spike |
| Poisson rate coding | Lấy mẫu spike ngẫu nhiên với xác suất phụ thuộc cường độ | Tạo variability | Có stochastic noise; cần seed và nhiều timestep |
| Latency/intensity coding | Cường độ quyết định thời điểm spike đầu tiên | Sparse, có thể truyền thông tin nhanh | Rất nhạy với normalization, polarity và quy tắc ánh xạ |
| Population coding | Một giá trị được mã bởi nhiều neuron | Representation giàu hơn | Tăng neuron, memory và compute |

`Repeated-feature input` trong paper không phải một spike-encoding scheme chuẩn. Nó là cách tạo chuỗi bằng việc lặp cùng CNN feature qua các timestep rồi để LIF/GRU tích lũy: dễ code end-to-end nhưng không tạo temporal information mới.

Paper Slimi et al. tuyên bố ở Introduction rằng dùng latency-intensity encoding: vùng sáng spike sớm, vùng tối spike muộn. Tuy nhiên phần architecture/implementation lại mô tả CNN feature được lặp qua 25 timestep và không công bố hàm ánh xạ intensity→latency, normalization, polarity, threshold hay tensor spike đầu vào. Vì vậy latency-intensity hiện là **paper claim chưa operationalize đủ để reproduce**, không được tự suy diễn thành code.

Paper dẫn prior benchmarks nói SNN trên Intel Loihi hoặc IBM TrueNorth có thể dùng điện thấp hơn tới 100× cho **một số workload**. Đây không phải phép đo của CNN–BiGRU–SNN trong paper. Lợi thế năng lượng chỉ đáng tin khi đo trên cùng workload và hardware cụ thể; mô phỏng 25 timestep trên GPU có thể tốn compute hơn CNN thông thường.

### 8.14. One-stage detector

YOLO/SSD/RetinaNet dự đoán class và box trong một lượt. Ưu điểm là nhanh; RetinaNet dùng focal loss để giảm ảnh hưởng của easy background examples.

### 8.15. Two-stage detector

Faster R-CNN:

```text
Backbone feature map
→ Region Proposal Network
→ ROI features
→ box refinement + classification
```

Mask R-CNN thêm nhánh segmentation mask. Two-stage thường mạnh cho object nhỏ nhưng chậm hơn.

### 8.16. U-Net

Encoder học semantic context; decoder phục hồi resolution. Skip connections truyền chi tiết từ encoder sang decoder. Metric phổ biến là Dice và IoU.

---

## 9. Tiền xử lý, augmentation và mất cân bằng lớp

### 9.1. Decode và channel handling

Dataset của nhóm có cả ảnh mode `L` và `RGB`. Pipeline phải thống nhất:

```text
decode → convert grayscale → repeat 3 channels nếu backbone cần RGB
```

Không được để một số ảnh có một channel và số khác có ba channel mà không kiểm soát.

### 9.2. Resize

Resize 128×128 giảm compute nhưng có thể làm mất opacity nhỏ. Resize trực tiếp cũng làm biến dạng aspect ratio.

Các lựa chọn cần so sánh:

- Direct resize.
- Resize giữ tỷ lệ + padding.
- Center/random crop sau resize.

Resolution phải là hyperparameter được ghi trong config, không hardcode.

### 9.3. Intensity normalization

Hai hướng phổ biến:

- Scale pixel về `[0,1]`, sau đó normalize theo mean/std của dataset.
- Dùng ImageNet mean/std khi fine-tune pretrained RGB backbone.

Phải dùng cùng transform ở validation/test, trừ augmentation ngẫu nhiên.

### 9.4. Augmentation hợp lý

Các paper đề cập:

- Horizontal flip.
- Rotation nhẹ.
- Zoom/crop.
- Brightness/contrast shift.
- Translation.

Nguyên tắc:

- Chỉ augment train.
- Không dùng transform làm mất pathology.
- Rotation/zoom phải ở mức có ý nghĩa với kỹ thuật chụp.
- Horizontal flip có thể hữu ích cho binary classification nhưng làm đảo laterality marker và giải phẫu; phải audit nếu task cần vị trí trái/phải.
- Không dùng vertical flip cho CXR thông thường.

### 9.5. Mất cân bằng lớp

Kermany dataset có nhiều pneumonia hơn normal. Các lựa chọn:

| Phương pháp | Ưu điểm | Rủi ro |
|---|---|---|
| Class-weighted loss | Đơn giản, không tạo ảnh giả | Cần chọn weight và theo dõi calibration |
| WeightedRandomSampler | Batch cân bằng hơn | Có thể thấy lặp lại minority sample nhiều lần |
| Focal loss | Tập trung hard examples | Hyperparameter alpha/gamma cần tune |
| Minority augmentation | Tăng biến thiên | Transform kém có thể làm sai anatomy |
| Undersampling | Giảm majority dominance | Vứt bỏ dữ liệu |
| SMOTE | Bám theo paper chính | Không tự nhiên khi áp dụng trực tiếp trên pixel/feature map |
| GAN synthetic data | Có thể tăng đa dạng | Artifact, memorization và label fidelity khó kiểm chứng |

### 9.6. Vì sao SMOTE trên ảnh cần thận trọng?

SMOTE nội suy vector:

\[
x_{new}=x_i+\lambda(x_j-x_i)
\]

Nếu flatten ảnh rồi nội suy, pixel tương ứng giữa hai bệnh nhân không đại diện cùng cấu trúc giải phẫu. Ảnh tổng hợp có thể không có ý nghĩa y khoa. Nếu SMOTE trên embedding, embedding phải được học chỉ từ train và cần mô tả rõ layer/domain áp dụng.

Feature-level SMOTE tạo **vector đặc trưng tổng hợp**, không tạo thêm ảnh CXR thật. Vì vậy câu “SMOTE tạo dataset 6.000 ảnh” phải được hiểu cẩn thận: model có thể nhìn thấy 6.000 feature vectors, nhưng không có 6.000 radiograph độc lập để bác sĩ kiểm tra. Pipeline hợp lệ phải là:

```text
patient/source-level split
→ fit feature extractor theo protocol chỉ dùng train
→ trích feature train
→ fit và apply SMOTE chỉ trên feature train
→ train classifier
→ validation/test giữ nguyên, không SMOTE
```

Nếu SMOTE được fit trước split, hoặc synthetic sample xuất hiện trong validation/test, metric không còn đo generalization trên bệnh nhân thật. Với dữ liệu ghép từ nhiều nguồn, SMOTE còn có thể nội suy shortcut của dataset/máy chụp thay vì pathology.

Baseline chính nên ưu tiên weighted cross-entropy, sampler hoặc focal loss. SMOTE nên là experiment riêng để reproduce paper.

---

## 10. Training và evaluation đúng cách

### 10.1. Split theo bệnh nhân

Không dùng `train_test_split` theo từng ảnh nếu một bệnh nhân có nhiều ảnh.

```text
patient IDs
→ group-aware train/validation/test split
→ mọi ảnh của một bệnh nhân chỉ thuộc một split
```

Với cross-validation, ưu tiên `StratifiedGroupKFold` khi có thể vừa giữ class balance vừa giữ group.

### 10.2. Ba vai trò khác nhau của split

- Train: cập nhật weight.
- Validation: chọn epoch, threshold và hyperparameter.
- Test: chỉ dùng khi model/protocol đã khóa.

Nếu test được xem nhiều lần trong quá trình phát triển, nó đã trở thành validation set.

### 10.3. Loss và optimizer

Paper chính dùng:

- Cross-entropy.
- Adam.
- Learning rate `1e-5`.
- Weight decay `1e-5`.
- OneCycleLR.

Learning rate phù hợp phụ thuộc train from scratch hay fine-tuning. `1e-5` có thể hợp lý cho pretrained backbone nhưng rất nhỏ với một số head train mới.

### 10.4. Early stopping và checkpoint

Checkpoint tốt nhất phải dựa trên validation metric đã xác định trước, ví dụ validation loss hoặc PR-AUC. Không chọn checkpoint theo test accuracy.

Artifact tối thiểu:

- Best model state.
- Resolved config.
- Random seed.
- Dataset manifest/hash.
- Environment và git commit.
- Training curves.
- Predictions trên validation/test.

### 10.5. Confusion matrix

Với pneumonia là positive class:

| | Predicted normal | Predicted pneumonia |
|---|---:|---:|
| Actual normal | TN | FP |
| Actual pneumonia | FN | TP |

### 10.6. Các metric

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

\[
Precision=\frac{TP}{TP+FP}
\]

\[
Recall/Sensitivity=\frac{TP}{TP+FN}
\]

\[
Specificity=\frac{TN}{TN+FP}
\]

\[
F1=2\cdot\frac{Precision\cdot Recall}{Precision+Recall}
\]

Ngoài ra cần:

- ROC-AUC.
- PR-AUC, hữu ích khi class imbalance.
- Balanced accuracy.
- Calibration/Brier score.
- 95% confidence interval bằng bootstrap theo bệnh nhân.

### 10.7. Threshold

Threshold 0,5 không phải lúc nào tối ưu. Có thể chọn threshold trên validation theo:

- Youden’s J.
- F1 tối đa.
- Sensitivity tối thiểu theo yêu cầu screening.
- Cost FP/FN do nhóm định nghĩa.

Sau khi chọn, threshold phải được khóa trước khi test.

### 10.8. Cross-validation

Mỗi fold cần:

- Group split độc lập.
- Fit preprocessing/balancing chỉ trên training fold.
- Validation fold không được dùng để train sampler/SMOTE.
- Báo mean, standard deviation và từng fold.

Không được chọn fold tốt nhất làm kết quả cuối.

### 10.9. External validation

Đây là phép thử generalization quan trọng nhất. Performance suy giảm trên bệnh viện mới là hiện tượng thường gặp, không phải ngoại lệ.

### 10.10. Fair comparison

Hai model chỉ nên đặt cạnh nhau nếu cùng:

- Dataset snapshot và exclusion criteria.
- Patient/group split.
- Input resolution.
- Preprocessing.
- Training budget.
- Threshold policy.
- Metric implementation.

So sánh con số lấy từ các paper khác dataset chỉ mang tính tham khảo lịch sử.

---

## 11. Explainability, localization và robustness

### 11.1. CAM và Grad-CAM

CAM dùng weighted combination của feature map trong kiến trúc phù hợp. Grad-CAM dùng gradient của class score đối với feature map:

1. Forward để lấy class score.
2. Backprop gradient về convolution layer được chọn.
3. Global-average gradient để tạo channel weights.
4. Weighted sum feature maps.
5. ReLU và resize heatmap lên ảnh gốc.

Grad-CAM là post-hoc explanation. Nó không làm model chính xác hơn nếu chỉ chạy sau classification.

### 11.2. Heatmap không phải bằng chứng lâm sàng

Heatmap đẹp có thể gây hiểu nhầm vì:

- Resolution thấp.
- Nhạy với layer được chọn.
- Có thể không trung thành với prediction.
- Chỉ cho vùng liên quan, không nói feature nào hoặc quan hệ nhân quả.
- Có thể tập trung ngoài phổi do dataset bias.

### 11.3. Cách đánh giá XAI tốt hơn

- Overlay với lung mask hoặc lesion annotation.
- Đo localization/pointing-game metric nếu có ground truth.
- Review cả true positive, true negative, false positive và false negative.
- Randomization sanity check.
- Occlusion test: che vùng heatmap cao và quan sát score thay đổi.
- Đánh giá bởi bác sĩ/radiologist nếu đưa ra claim clinical relevance.

### 11.4. Robustness corruption

Cần đo metric trên toàn test set ở nhiều severity:

- Gaussian blur.
- Gaussian noise.
- Salt-and-pepper noise.
- Speckle noise.
- Contrast/brightness shift.
- JPEG compression.
- Crop, rotation và motion blur.

Báo performance curve theo severity, không chỉ hiển thị vài ảnh còn được dự đoán đúng.

### 11.5. Adversarial robustness

Adversarial perturbation được tối ưu để thay đổi prediction và khác random noise. Survey 2024 nhấn mạnh rằng CXR classifier có thể dễ bị tấn công; robustness evaluation nên gồm FGSM/PGD trong phạm vi threat model rõ ràng nếu đây là mục tiêu project.

Một nghiên cứu được survey trích dẫn báo universal adversarial perturbation có attack success rate trên 80% ở cả targeted và untargeted setting, và adversarial retraining trong setup đó không loại bỏ được phần lớn tác động. Đây là kết quả của một nghiên cứu cụ thể, không phải tỷ lệ chung cho mọi CXR model, nhưng đủ cho thấy random Gaussian noise không đại diện đầy đủ cho robustness.

### 11.6. Domain robustness

Corruption robustness không thay thế external validation. Một model chịu được Gaussian noise vẫn có thể thất bại khi đổi bệnh viện, dân số, view hoặc protocol chụp.

---

## 12. Tổng hợp từng paper

### 12.1. Litjens et al. 2017 — Survey medical image analysis

**Phạm vi:** 308 công trình về classification, detection, segmentation, registration, retrieval, image generation và nhiều cơ quan/modalities.

**Đóng góp quan trọng:**

- Hệ thống hóa CNN, multi-stream/multi-scale architecture, RNN, autoencoder, RBM và generative approaches.
- Cho thấy CNN đã thay handcrafted systems trong nhiều challenge.
- Nhấn mạnh exact architecture không phải yếu tố duy nhất; data preparation và task-specific design rất quan trọng.
- Input size và receptive field phải phù hợp với resolution/context cần để giải bài toán.

**Các thách thức y tế được nêu:**

- Thiếu nhãn chất lượng hơn là thiếu raw images.
- Expert annotation đắt và chậm.
- Inter-reader disagreement và label uncertainty.
- Class imbalance và within-class heterogeneity.
- Binary normal/abnormal thường đơn giản hóa quá mức.
- Cần kết hợp image với metadata và clinical context.
- Black-box behavior và uncertainty cản trở clinical acceptance.

**Ý nghĩa cho project:** data audit, expert-label uncertainty, receptive field và interpretability phải được xem là phần lõi, không phải việc bổ sung sau model.

### 12.2. Sogancioglu/Çallı et al. 2021 — Survey CXR

**Phạm vi:** 295 paper, phân loại theo image-level prediction, segmentation, localization, generation, domain adaptation và các task khác; đồng thời khảo sát dataset và sản phẩm thương mại.

Phân bố paper theo task mà survey báo cáo:

| Task | Số paper |
|---|---:|
| Image-level prediction | 187 |
| Segmentation | 58 |
| Localization | 30 |
| Image generation | 35 |
| Domain adaptation | 11 |
| Other applications | 14 |

Một paper có thể thuộc tối đa hai nhóm nên tổng các dòng lớn hơn 295. Trong 58 nghiên cứu segmentation, 29 bài dùng U-Net hoặc một biến thể gần U-Net.

**Kết luận chính:**

- 209/295 paper dùng public dataset.
- NLP-extracted labels hữu ích nhưng không nên mặc nhiên dùng làm gold-standard test labels.
- Nhiều paper dùng off-the-shelf architectures mà không có contribution hoặc protocol đủ mạnh.
- 142 công trình bị loại vì chất lượng khoa học; 112 trong đó có vấn đề dataset construction.
- 61 nghiên cứu COVID ghép adult COVID với pediatric control, tạo confounding nghiêm trọng.
- Ensemble thường có performance cao trong challenge nhưng chưa giải quyết clinical translation.
- Không có bằng chứng rõ rằng một backbone luôn tốt nhất cho mọi CXR task.

Survey còn ghi nhận 21 sản phẩm thương mại có CE mark và/hoặc FDA clearance tại thời điểm khảo sát. Trong đó 17/21 cung cấp localization cho ít nhất một abnormality, năm sản phẩm hỗ trợ ưu tiên worklist và năm sản phẩm tạo một dạng draft report. Đây chỉ là snapshot sản phẩm, không phải đánh giá rằng tất cả đã được external validate ngang nhau.

**Clinical translation:**

- AI ban đầu nên hỗ trợ, không thay radiologist.
- Localization/bounding box có ích hơn một danh sách dài probability.
- Cần dùng lateral view, history, symptoms, test results và prior images.
- Worklist prioritization, normal-case triage, interval change và draft report là các use case thực tế.

**Ý nghĩa cho project:** đây là nguồn mạnh nhất để thiết kế evaluation và phê bình các claim accuracy cao.

### 12.3. Xu 2023 — Review pneumonia region detection

**Nội dung chính:**

- Liệt kê ChestX-ray14, RSNA, pediatric pneumonia và một số CT dataset.
- Trình bày preprocessing: labeling, resize, augmentation, normalization và split.
- So sánh one-stage với two-stage detector.
- Tổng hợp transfer learning, ensemble, Mask R-CNN, RetinaNet và YOLO.
- Đề xuất multimodal imaging, clinical data, interpretability và real-world deployment.

**Điểm hữu ích:** cung cấp bản đồ nhanh về classification/localization và trade-off speed–accuracy.

**Điểm phải đọc thận trọng:**

- Review rất ngắn và methodology tìm kiếm chưa mạnh.
- Trộn image classification, ensemble và object-detection stage terminology.
- Gọi CheXNeXt ensemble là “two-stage” không cùng nghĩa với two-stage detector kiểu Faster R-CNN.
- Một số số lượng dataset không nhất quán.
- Kết luận two-stage chính xác hơn one-stage là khái quát hóa quá mức.

### 12.4. Siddiqi & Javaid 2024 — Comprehensive pneumonia survey

**Research questions:** nền tảng, public datasets, thống kê, kỹ thuật gần đây và challenges.

**Phương pháp:** tìm trên IEEE Xplore, ScienceDirect, SpringerLink và ACM DL; truy vấn ban đầu trả về 789 mục, sau đó áp dụng inclusion/exclusion và quality criteria.

**Taxonomy:**

- Preprocessing: augmentation và segmentation.
- Classification: CNN, transfer learning, hybrid và ensemble.
- XAI.
- Dataset và trends.
- ViT và adversarial robustness.

**Thống kê paper báo:**

- 140 paper liên quan trực tiếp được chia thành 22 non-COVID, 48 COVID và 70 cả hai.
- Classification task: 19 binary non-COVID, 41 binary COVID, 34 multiclass và 56 nghiên cứu cả binary/multiclass.
- Một đoạn khác nói đã “explored” 262 studies; phạm vi đếm này không hoàn toàn nhất quán với 140 paper chính.

**Kết luận chính:**

- Transfer learning và custom CNN vẫn là phương pháp chủ đạo.
- Ensemble/hybrid thường báo accuracy cao.
- ViT là hướng tiềm năng nhờ global context và transfer learning.
- Các vấn đề mở: biased datasets, code/data availability, explainability, fair comparison, imbalance và adversarial attacks.

**Điểm phê bình:**

- Các accuracy 94–99% trong Table 5 đến từ dataset/protocol khác nhau và không thể xếp hạng trực tiếp.
- Claim ViT ít parameter hoặc ít compute hơn CNN không đúng tổng quát.
- Paper vừa cảnh báo fair comparison vừa trình bày nhiều con số cạnh nhau dễ gây cảm giác có thể so sánh.
- Số lượng studies và thời gian bao phủ được mô tả chưa hoàn toàn thống nhất.

### 12.5. CheXNet trong ghi chú nhóm

**Kiến trúc:** DenseNet121 pretrained trên ImageNet, thay classifier để phát hiện pneumonia và mở rộng multi-label 14 pathology.

**Pipeline được ghi chú:**

- Resize 224×224.
- ImageNet normalization.
- Random horizontal flip.
- Patient-level split.
- Weighted binary cross-entropy.
- CAM để localization yếu.

**Kết quả được ghi chú:** F1 khoảng 0,435 trên test set 420 ảnh, so với trung bình bốn radiologist khoảng 0,387 trong protocol cụ thể.

**Cách diễn giải đúng:** model vượt mức trung bình reader trong test setup đó; không chứng minh thay thế radiologist nói chung.

### 12.6. Kermany et al. 2018 trong ghi chú nhóm

**Đóng góp:** chứng minh transfer learning có hiệu quả trên OCT và pediatric CXR; dataset pediatric pneumonia sau đó trở thành benchmark phổ biến.

**Kết quả CXR được ghi chú:** accuracy khoảng 92,8%, sensitivity 93,2%, specificity 90,1%; bài toán bacterial-vs-viral khó hơn normal-vs-pneumonia.

**Ý nghĩa:** đây là paper gắn trực tiếp với nguồn dataset mà project dùng, nhưng dataset nhỏ và có domain rất hẹp.

### 12.7. Các ghi chú transfer learning/ensemble/attention khác

- Kundu et al.: ensemble GoogLeNet + ResNet18 + DenseNet121 để tận dụng representation bổ sung.
- Gu & Lee 2024: frozen ImageNet backbones thường hội tụ nhanh và tốt hơn cùng architecture train from scratch; ResNet18 được ghi chú tăng test accuracy từ khoảng 89,6% lên 93,1%.
- Li 2024: attention-enhanced ResNet kết hợp channel/spatial attention và enhanced focal loss; accuracy được ghi chú khoảng 98% trong experimental setup của paper.

Các con số này là secondary notes và phải kiểm tra paper gốc trước khi dùng trong bảng kết quả chính thức.

### 12.8. Singh et al. 2024 — DeiT/ViT trên pediatric CXR

**Nguồn chính thức:** Singh et al., *Efficient pneumonia detection using Vision Transformers on chest X-rays*, Scientific Reports 14, 2487 (2024), [DOI 10.1038/s41598-024-52703-2](https://doi.org/10.1038/s41598-024-52703-2). Paper xuất bản ngày 30/01/2024. Đây là case study mới được dẫn từ [`tonghop2.md`](./tonghop2.md) và đã được đối chiếu với toàn văn chính thức.

#### Câu hỏi nghiên cứu

Paper muốn kiểm tra liệu Transformer học quan hệ toàn cục giữa các patch có phân loại normal/pneumonia tốt hơn các CNN pretrained trên pediatric Kermany CXR hay không.

#### Model thực tế là gì?

Tên bài dùng “Vision Transformer”, nhưng implementation ghi rõ sử dụng pretrained weights `DEIT_Base_Patch16_224`. Đây là thông tin quan trọng hơn mô tả ViT chung:

```text
Input [B, 3, 224, 224]
→ patch 16×16, tổng 14×14 = 196 patch
→ linear patch embeddings
→ thêm positional embedding và class token
→ Transformer encoder blocks
→ class-token representation
→ two-class classification head
```

Paper báo `85.800.194` trainable parameters và `0` non-trainable parameters cho model đề xuất. Điều này lớn hơn nhiều CNN frozen-feature baselines trong Table 6; do đó kết quả không chứng minh Transformer nhẹ hơn CNN.

#### Dữ liệu và huấn luyện paper báo cáo

| Thành phần | Giá trị paper-reported |
|---|---:|
| Resize | 224×224 |
| Split theo số ảnh | 4.684 train / 586 validation / 586 test |
| Batch size | 16 |
| Loss | `CrossEntropyLoss` |
| Optimizer | Adam |
| Learning rate | `1e-5` |
| LR multiplicative factor | `0,995` |
| Epoch | 30 |
| Hardware mô tả | i5, GPU 2 GB, RAM 8 GB, CUDA 10 |

Giá trị 30 epoch nằm trong Table 5, không nằm trong bảng hyperparameter Table 4. Cùng phần kết quả lại viết precision hội tụ sau 35 epoch, nên mô tả epoch/curve của paper tự mâu thuẫn.

`tonghop2.md` ghi binary cross-entropy, nhưng toàn văn ghi `CrossEntropyLoss`. Hai loss có thể tương đương về mục tiêu khi implementation đúng, song số output logit và label format khác nhau; khi reproduce phải theo code hoặc mô tả chính thức, không trộn hai API.

Paper không mô tả patient-grouped split. Vì dataset có nhiều ảnh cho một `person ID`, con số dưới đây có nguy cơ optimistic nếu split lại ngẫu nhiên ở mức ảnh.

#### Kết quả paper báo cáo

| Metric | Giá trị |
|---|---:|
| Accuracy | 97,61% |
| Sensitivity | 0,949 |
| Specificity | 0,981 |
| F-score | 0,952 |
| AUC | 0,966 |

Confusion matrix được ghi:

```text
TP = 152, TN = 420, FP = 6, FN = 8
Tổng = 586
```

Tính lại trực tiếp:

- Accuracy = `(152 + 420) / 586` = 97,61%.
- Precision của class được gọi positive = `152 / (152 + 6)` ≈ 96,20%.
- Recall của class positive = `152 / (152 + 8)` = 95,00%.
- F1 của class positive ≈ 95,60%.
- True-negative rate = `420 / (420 + 6)` ≈ 98,59%.

Accuracy khớp hoàn toàn. F1 ≈ 0,956 và true-negative rate ≈ 0,986 không tái tạo được các giá trị 0,952 và 0,981 bằng định nghĩa nhị phân chuẩn; class order hoặc cách aggregation không được mô tả đủ, và rounding đơn thuần không giải thích được toàn bộ chênh lệch.

#### Audit số lượng và class mapping

Paper có ba lớp mâu thuẫn số học:

1. Prose ghi tổng `5.863` ảnh, trong khi Table 2 ghi:

   ```text
   4.273 pneumonia + 1.583 normal = 5.856
   ```

2. Table 3 có tổng split đúng `4.684 + 586 + 586 = 5.856`, nhưng cộng theo lớp lại sai:

   ```text
   Pneumonia: 3.205 + 360 + 330 = 3.895, không phải 4.273
   Normal:    1.479 + 226 + 256 = 1.961, không phải 1.583
   ```

3. Test split trong Table 3 được mô tả có 330 pneumonia và 256 normal. Confusion matrix lại hàm ý 160 positive và 426 negative. Đáng chú ý, `160 ≈ 10% × 1.583 Normal` và `426 ≈ 10% × 4.273 Pneumonia`. Vì vậy confusion matrix có vẻ đến từ một split 10% khác và **Normal có thể được đặt là positive class**; sensitivity 95% khi đó là recall của Normal, không chắc là khả năng tránh bỏ sót pneumonia như cách diễn giải trong `tonghop2.md`.

Không thể biết bảng split sai, confusion matrix dùng split khác, hay class label bị đảo nếu không có code/split manifest. Nhóm không nên dùng 97,61% làm reproduction target cứng.

#### So sánh với CNN có thực sự fair?

Paper nói giữ cùng hyperparameter để so sánh pretrained CNN. Tuy nhiên:

- Các CNN trong Table 6 có phần lớn parameter non-trainable, còn DeiT có toàn bộ 85,8 triệu parameter trainable.
- Một learning rate và schedule giống nhau không đồng nghĩa mỗi architecture được tune công bằng.
- Chưa thấy nhiều seed hoặc confidence interval giữa các lần train.
- Cùng image source chưa đảm bảo cùng patient split hoặc tránh duplicate.

Do đó kết luận hợp lý là **DeiT đạt kết quả cao trong protocol paper**, không phải “ViT luôn vượt mọi CNN”.

#### Giá trị cho project

- Là comparator Transformer rõ ràng nếu nhóm đủ compute.
- Cho thấy cách biến ảnh thành token cụ thể, thay vì gọi chung là self-attention.
- Là ví dụ tốt rằng metric có thể đúng về phép tính nhưng dataset table vẫn mâu thuẫn.
- Reproduction nên dùng manifest riêng, patient-aware split, confusion matrix có class order rõ, và báo cả parameter/FLOP/latency.

### 12.9. Sana et al. 2025 — VGG-16 + custom DNN + Grad-CAM

**Nguồn chính thức:** Sana, Biswas & Islam, *Enhancing Pneumonia Detection from Chest Radiographs Through a VGG-16-based Deep Learning Approach*, European Journal of Clinical and Biomedical Sciences 11(5), 60–72 (2025), [DOI 10.11648/j.ejcbs.20251105.11](https://doi.org/10.11648/j.ejcbs.20251105.11). Trang chính thức ghi ngày xuất bản 11/12/2025. Mục này đã được đối chiếu với PDF toàn văn, không chỉ abstract được tóm tắt trong `tonghop2.md`.

#### Dữ liệu và preprocessing

Paper dùng đúng snapshot 5.856 ảnh với split gốc:

| Split | Số ảnh |
|---|---:|
| Train | 5.216 |
| Validation | 16 |
| Test | 624 |

Pipeline mô tả:

```text
decode image
→ resize 224×224
→ đưa về 3 channel cho VGG-16
→ train-only augmentation: rotation, width/height shift, zoom, shear, intensity variation
→ scale pixel về [0, 1]
```

Paper không công bố biên độ cụ thể của từng augmentation trong đoạn phương pháp, nên không được tự gán degree/range từ một implementation khác.

#### Kiến trúc đọc ảnh

```text
Image [B, 3, 224, 224]
→ VGG-16 convolutional base pretrained trên ImageNet
→ Global Average Pooling
→ Dense 512 → ReLU → BatchNorm → Dropout 0,40
→ Dense 256 → ReLU → BatchNorm → Dropout 0,40
→ Dense 128 → ReLU → BatchNorm → Dropout 0,40
→ Dense 1 → Sigmoid
→ P(pneumonia)
```

Global Average Pooling trung bình từng feature map theo hai chiều không gian, tạo một số cho mỗi channel. Nó tránh flatten một feature map lớn vào fully connected layer và làm head nhỏ hơn VGG cổ điển.

Paper viết “top four convolutional layers” được sửa/fine-tune để học feature liên quan pneumonia, nhưng không nêu rõ tên layer, freeze mask hoặc learning-rate phân biệt. Khi reproduce cần log danh sách `trainable` layer; không nên chỉ ghi chung “fine-tune VGG-16”.

#### Training setup

| Thành phần | Giá trị paper-reported |
|---|---:|
| Loss | Binary cross-entropy |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Epoch | 20 |
| Batch size | 32 |
| Data loader | Keras `ImageDataGenerator` |
| Output | 1 sigmoid unit |

#### Kết quả và kiểm tra lại

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| ResNet50 | 90,06% | 86,77% | 99,23% | 92,58% |
| DenseNet169 | 92,15% | 97,49% | 89,74% | 93,46% |
| MobileNet | 91,99% | 92,71% | 94,62% | 93,65% |
| InceptionV3 | 90,71% | 87,56% | 99,23% | 93,03% |
| Xception | 91,99% | 89,35% | 98,97% | 93,92% |
| **VGG-16 đề xuất** | **92,79%** | **94,12%** | **94,36%** | **94,24%** |

Paper báo AUC khoảng 0,98. Confusion matrix của VGG-16 là:

```text
TN = 211, FP = 23
FN = 22,  TP = 368
```

Các số này cho đúng test gốc 234 normal + 390 pneumonia và tái tạo các metric:

- Accuracy = `579/624` = 92,79%.
- Precision = `368/(368+23)` = 94,12%.
- Recall = `368/(368+22)` = 94,36%.
- F1 ≈ 94,24%.

Đây là một điểm mạnh về reporting nội bộ: confusion matrix, class distribution và scalar metrics khớp nhau.

#### Grad-CAM đóng vai trò gì?

Grad-CAM lấy gradient của class score đối với feature maps ở convolutional layer cuối, tạo heatmap vùng ảnh ảnh hưởng đến prediction. Nó được chạy **sau** forward pass để diễn giải; không phải attention layer được train và không làm accuracy tăng.

Muốn dùng Grad-CAM có trách nhiệm, nhóm phải kiểm tra heatmap có nằm trong phổi, có bám chữ/marker/border không, và nếu có annotation thì đo localization. Một số hình đẹp không chứng minh clinical transparency.

#### Hạn chế cần nhớ

- Validation chỉ có 16 ảnh, quá nhỏ để chọn hyperparameter hoặc early stopping ổn định.
- Test 624 ảnh vẫn từ cùng một trung tâm pediatric; không phải external validation.
- Không thấy patient-level grouping, nhiều seed hoặc confidence interval.
- VGG-16 không mặc nhiên computationally efficient; paper không cung cấp benchmark latency/FLOP/energy đủ để kết luận real-time.
- “Outperform SOTA” chỉ phản ánh các baseline/protocol paper chọn; không được so trực tiếp với 97,61% của paper ViT vì split và training khác nhau.
- Khả năng tích hợp telemedicine/clinical decision support là hướng ứng dụng đề xuất, chưa phải clinical deployment đã được đánh giá.

#### Giá trị cho project

Đây là baseline thực dụng hơn hybrid SNN để nhóm xây trước: architecture rõ, metric tự khớp, test set nguyên bản và Grad-CAM dễ triển khai. Tuy vậy nhóm nên thay validation 16 ảnh bằng patient-aware validation lấy từ train, khóa test gốc cho lần đánh giá cuối, và thêm external/source-shift test nếu có.

### 12.10. Srivastava et al. 2025 — Multi-class VGG-19 và feature-level SMOTE

**Nguồn chính thức:** Srivastava et al., *Multi-class deep learning architecture for COVID-19, tuberculosis, and pneumonia classification using chest X-ray images*, Journal of Medical Imaging and Radiation Sciences 56(6), 102115 (2025), [DOI 10.1016/j.jmir.2025.102115](https://doi.org/10.1016/j.jmir.2025.102115). Lưu ý DOI dùng chuỗi `jmir`, không phải `jmirs` như ghi trong `tonghop2.md`.

#### Phạm vi bài toán

Đây không còn là binary pediatric pneumonia classification. Output gồm bốn class:

```text
Normal | Pneumonia | Tuberculosis | COVID-19
```

Paper so sánh ResNet-50, EfficientNet, DenseNet và VGG-19 trong một pipeline transfer-learning/multi-class. Mục tiêu là giảm thiên lệch do COVID-19 có ít mẫu hơn các class khác.

#### Pipeline paper/abstract báo cáo

```text
ảnh CXR từ nhiều public sources
→ resize + normalization + augmentation
→ CNN feature extraction
→ feature-level SMOTE cho minority classes
→ balanced representation: 6.000 mẫu, khoảng 1.500/class
→ multi-class classifier
```

VGG-19 đạt paper-reported test accuracy 97,5%; precision, recall và F1 đều trên 96% cho mỗi class. Paper nói VGG-19 tốt hơn ResNet-50, EfficientNet và DenseNet trong protocol của họ.

#### Ranh giới xác minh

Folder không có PDF paper này. `tonghop2.md` và abstract/trang chính thức xác nhận bốn class, feature-level SMOTE, 6.000 balanced samples, các backbone và result tổng quát. Chúng **không đủ** để xác nhận chính xác:

- nguồn và số bệnh nhân của từng class;
- patient/source-level split;
- layer dùng làm embedding;
- SMOTE chạy trước hay sau split;
- pretrained weights, freeze policy và hyperparameter;
- model selection dùng validation hay test;
- confusion matrix và confidence interval từng class.

Do đó các chi tiết kiểu “softmax/cross-entropy”, input 224×224 hoặc tỷ lệ split không được đưa vào như fact nếu chưa xem full text/code.

#### “6.000 ảnh sau SMOTE” cần hiểu thế nào?

SMOTE ở feature level nội suy vector:

\[
z_{new}=z_i+\lambda(z_j-z_i),\qquad 0\leq\lambda\leq1
\]

`z` là embedding, không phải radiograph. Cách nói chính xác là **6.000 feature samples dùng cho training**, không phải thu thập hoặc tạo được 6.000 ảnh CXR có thể đọc bằng mắt.

Protocol leakage-resistant phải là:

```text
deduplicate + group theo patient/source
→ split train/validation/test
→ fit/fine-tune extractor chỉ trong training protocol
→ trích train embeddings
→ fit SMOTE chỉ trên train embeddings
→ train classifier
→ validation/test chỉ gồm ảnh thật, không synthetic balance
```

Nếu ghi chú nói “testing trên balanced set sau SMOTE”, nhóm tuyệt đối không copy protocol đó khi chưa kiểm tra toàn văn. SMOTE trước split làm synthetic point chia sẻ neighbor gốc với train/test và có thể đẩy metric lên rất cao.

#### Domain/source shortcut còn nguy hiểm hơn imbalance

Khi bốn class được ghép từ nhiều dataset:

- COVID-19 có thể đến từ một bệnh viện/máy chụp khác TB.
- Pneumonia/normal có thể là trẻ em trong khi TB/COVID là người lớn.
- Marker, crop, độ phân giải hoặc processing pipeline có thể tiết lộ nguồn.
- Cùng file có thể xuất hiện ở nhiều public collection dưới tên khác.

Model có thể đạt 97,5% bằng cách nhận diện source/age/acquisition thay vì phân biệt pathology. Chính paper cũng thừa nhận dữ liệu heterogeneous có thể tạo dataset-specific bias và gọi model là research framework, hiện **non-clinical**.

#### Giá trị cho project

- Cho thấy backbone cũ vẫn cạnh tranh nếu data pipeline phù hợp.
- Là case audit tốt cho feature-level SMOTE và source leakage.
- Phù hợp làm future extension sau khi binary baseline đáng tin, không nên đưa ngay vào core scope.
- Nếu mở rộng multi-class, nhóm cần source-balanced external test và báo metric từng class; overall accuracy là chưa đủ.

### 12.11. Những điểm trong `tonghop2.md` không được sao chép nguyên văn

| Ghi chú trong `tonghop2.md` | Đối chiếu/cách dùng đúng |
|---|---|
| “Test set gốc chỉ có 16 sample” | Sai: validation gốc có 16; test gốc có 624 ảnh. |
| Gọi CNN–BiGRU–SNN là “ensemble” | Đây là hybrid/sequential multi-block model; không có nhiều predictor độc lập được ensemble. |
| “Attention-guided module xác định độ quan trọng pixel khi dự đoán” | Paper nói attention map post-classification; nó không tham gia forward prediction. |
| Công thức gần dạng `origin + (origin-neighbor)×random` | Dấu sai hướng; SMOTE chuẩn là `origin + λ(neighbor-origin)`. |
| Augmentation được gọi chung là noise robustness | Flip/zoom/brightness khi train khác corruption test như Gaussian noise/motion artifact. |
| “Pixel sáng là consolidation quan trọng, pixel tối ít quan trọng” | Đây là narrative quá mạnh: xương, marker và border cũng sáng; polarity/normalization phải được định nghĩa. |
| Loihi/TrueNorth tiết kiệm gần 100× | Là prior hardware-specific claim cho một số workload, không phải measurement của proposed model trên GPU. |
| Các marker `[nature]`, `[sciencedirect]`, `[mdpi]` | Không phải citation đầy đủ; dùng DOI hoặc link paper chính thức. |
| “Thông thường paper dùng...” hoặc “có thể extrapolate...” | Là suy luận của người tóm tắt, không được ghi thành method/result của tác giả. |

Nhờ vậy `tonghop2.md` vẫn hữu ích như danh sách paper và câu hỏi đọc, còn handbook giữ vai trò bản đã chuẩn hóa bằng chứng.

---

## 13. Giải phẫu paper CNN–BiGRU–SNN 2025

### 13.1. Mục tiêu paper

Paper tuyên bố kết hợp:

- CNN để học spatial features.
- BiGRU để học temporal dependencies.
- SNN/LIF để xử lý spike và tăng robustness/energy efficiency.
- Attention map để giải thích prediction.

Mục tiêu cuối là accuracy cao, chịu noise, dễ giải thích và phù hợp môi trường hạn chế tài nguyên.

### 13.2. Dataset và preprocessing paper mô tả

- Pediatric Kermany dataset.
- Resize 128×128.
- Convert grayscale và lặp thành ba channel.
- Horizontal flip.
- Zoom 0,9–1,1.
- Brightness ±20%.
- SMOTE để cân bằng class.
- Merge train và validation gốc, sau đó split lại 80/20.

Paper ghi 3.883 pneumonia và 1.349 normal sau merge, tổng đúng là 5.232 ảnh.

Paper nói lặp grayscale thành ba channel để phù hợp “pretrained CNN input”, nhưng proposed model là custom CNN `3→32→64→128`; không có pretrained backbone nào được mô tả trong proposed path. Paper cũng không công bố interpolation, pixel range, mean/std normalization, xác suất augmentation, fill mode hoặc thứ tự transform. Vì vậy các thông số này phải được đánh dấu là quyết định reconstruction.

### 13.3. Spatial Feature Extraction block

Paper Table 1 mô tả:

```text
Input [B, 3, 128, 128]
→ Conv2d 3→32, kernel 3, stride 1, padding 1
→ BatchNorm → LeakyReLU → MaxPool 2×2
→ Conv2d 32→64 → BatchNorm → LeakyReLU → MaxPool
→ Conv2d 64→128 → BatchNorm → LeakyReLU → MaxPool
```

Nếu có pooling sau cả ba convolution, shape hợp lý là:

```text
[B, 3, 128, 128]
→ [B, 32, 64, 64]
→ [B, 64, 32, 32]
→ [B, 128, 16, 16]
→ flatten [B, 32768]
```

Paper lại ghi projection input 65.536, nên computational graph chưa được mô tả nhất quán.

### 13.4. Temporal Dynamics Modeling block

Thông số paper:

- 25 timestep.
- Hai layer bidirectional GRU.
- Hidden size 256.
- Dropout 0,3.
- Forward/backward output được concatenate, tạo 512 feature mỗi timestep nếu dùng output chuẩn.

Paper nói CNN feature của ảnh tĩnh được lặp qua 25 timestep. Vì input mỗi timestep giống nhau, GRU không quan sát temporal data thật; nó học dynamics phát sinh từ hidden-state recurrence trên một vector lặp.

Với `batch_first=True`, tensor hợp lý phải có dạng `[B, 25, d]`, và output BiGRU là `[B, 25, 512]`. Paper không công bố `d` hay thao tác reshape tạo chuỗi. Nếu BiGRU nhận trực tiếp flatten 32.768 chiều, riêng GRU hai layer hai hướng đã có khoảng 51,91 triệu parameter, trái với tổng 8,63 triệu paper báo. Do đó gần như chắc chắn phải có projection trước input GRU dù Figure 6 không vẽ rõ điều này.

### 13.5. Feature projection

Table 1 ghi:

```text
Linear 65,536 → 128
Linear 128 → 64
```

Table 1 đặt temporal block trước projection, nhưng Figure 6 cho thấy flatten được tách thành hai nhánh song song:

```text
flatten → feature projection ─┐
                             ├→ spiking/fusion block
flatten → temporal BiGRU ─────┘
```

Paper vẫn không mô tả input dimension thật của GRU. Ngoài ra, hai layer đều bị ghi là `fc1`; layer thứ hai nhiều khả năng phải là `fc2`. Không có activation, normalization hoặc dropout giữa `65.536→128→64` được nêu.

### 13.6. Spiking block

- LIF neuron với `beta = 0.95`.
- 25 discrete timestep.
- Surrogate-gradient backpropagation.
- Mean spike aggregation qua thời gian.
- Reset/detach hidden state giữa mini-batch để tránh gradient leakage.
- Paper nói dùng `snnTorch`.

Figure 4 bổ sung phần fusion:

```text
spatial/projected branch → LIF1, beta 0.95 → feature 256 ─┐
                                                         ├→ concat 768
BiGRU temporal feature 512 ───────────────────────────────┘
                                                         ↓
                                             LIF2/spike encoding
                                                         ↓
                                                decision head
```

Con số 512 khớp hai hướng GRU hidden 256. Nhưng projection kết thúc ở 64 chiều còn Figure 4 yêu cầu LIF1/spatial feature 256 chiều. Paper thiếu ít nhất một phép biến đổi tương tự `Linear 64→256`, hoặc thiếu mô tả về kích thước LIF1.

Paper không công bố threshold, reset rule, initial membrane hay surrogate-gradient function. `detach_hidden()` trong snnTorch chỉ cắt computational graph, không đồng nghĩa đặt membrane về zero; reproduction phải reset state một cách tường minh ở đầu mỗi sequence.

Paper có lúc mô tả latency-intensity encoding ở phần giới thiệu, nhưng phần implementation lại nói lặp extracted spatial feature qua 25 timestep. Đây là hai mô tả encoding không hoàn toàn đồng nhất.

Introduction còn nhắc “spiking convolutional layers” và adaptive pooling, nhưng architecture chính chỉ thể hiện Conv2d thông thường, MaxPool và LIF sau CNN/GRU. Không có thuật toán chuyển intensity thành spike time, nên không thể khẳng định một latency encoder độc lập đã được triển khai.

### 13.7. Decision head

Paper ghi:

```text
FC 768 → 512
→ Dropout 0.4
→ FC 512 → 128
→ Dropout 0.4
→ FC 128 → 2
```

Figure 4 giải thích 768 là `256 spatial/spiking + 512 temporal`. Điểm chưa rõ là phép biến đổi `64→256`, fusion dùng spike hay membrane, LIF2 output gì và mean aggregation được đặt trước hay sau fusion.

Paper cũng không ghi activation giữa các linear layer của head. Nếu thật sự không có activation, chuỗi linear layer ở inference gần tương đương một linear transformation duy nhất; nhiều khả năng ReLU/LeakyReLU đã bị bỏ sót trong mô tả.

### 13.8. Data flow có thể hiểu từ paper

```text
CXR → CNN → flatten
             ├→ projection 65.536?→128→64 → missing 64→256? → LIF1 ─┐
             │                                                       ├→ concat 768
             └→ sequence 25 bước → 2-layer BiGRU → feature 512 ─────┘
                                                                     ↓
                                                     LIF2/mean spikes?
                                                                     ↓
                                                   FC 768→512→128→2
```

Dấu `?` là ambiguity thật của paper; implementation phải chọn một interpretation và lưu model summary/shape trace.

### 13.9. Training setup paper báo cáo

| Thành phần | Paper-reported value |
|---|---|
| Hardware | Kaggle Tesla P100 |
| Optimizer | Adam |
| Learning rate | `1e-5` |
| Weight decay | `1e-5` |
| Scheduler | OneCycleLR |
| Loss | Categorical cross-entropy |
| Batch size | 32 |
| Early stopping | Patience 5 |
| Training duration | Vừa ghi scheduler 30 epoch, vừa ghi train 7 epoch |

Figure 9 biểu diễn `PNEUMONIA=[1,0]` và `NORMAL=[0,1]`, tức pneumonia có thể là class index 0 trong code tác giả. Đây là chiều ngược với mapping `NORMAL=0`, `PNEUMONIA=1` mà project dự kiến dùng. Class mapping phải được lưu trong artifact; khi so metric/AUC cần map positive class rõ ràng, không dựa vào thứ tự folder.

### 13.10. Ablation paper báo cáo

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Không GRU | 96,32% | 96,22% | 96,20% | 96,54% | 96,00% |
| Có GRU | 99,35% | 99,10% | 99,50% | 99,10% | 99,10% |

Paper diễn giải GRU giúp convergence mượt, nhanh và generalization tốt hơn. Tuy nhiên chỉ một ablation `with/without GRU` chưa tách được contribution của CNN, SNN, encoding, projection và parameter budget.

### 13.11. Baseline paper báo cáo

| Model | Accuracy | F1 | AUC | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Xception | 95,45 | 95,36 | 96,18 | 95,33 | 95,33 |
| InceptionV3 | 94,21 | 94,40 | 95,12 | 93,70 | 93,72 |
| DenseNet121 | 93,75 | 93,50 | 94,30 | 94,23 | 94,23 |
| ResNet50 | 91,55 | 91,11 | 93,88 | 92,08 | 92,10 |
| VGG16 | 80,33 | 79,39 | 85,40 | 74,16 | 74,17 |
| Proposed | 99,35 | 99,10 | 99,10 | 99,10 | 99,50 |

Không đủ thông tin để xác nhận các baseline dùng cùng split, tuning budget, pretrained weights và threshold.

### 13.12. Cross-validation paper báo cáo

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---:|---:|---:|---:|---:|---:|
| 98,39 | 99,19 | 98,87 | 99,03 | 99,11 | 98,92 |

Paper không mô tả đủ group/patient handling và fold construction để tái lập duy nhất.

### 13.13. Complexity và energy paper báo cáo

| Metric | Value |
|---|---:|
| Parameters | 8.631.042 |
| MACs | 4.433,96 M |
| FLOPs | 4,48 G |
| Approx GPU memory | 3.760 MB |
| Training time cho 1 sample/epoch | 0,0902 s |
| Energy cho 1 sample/epoch | 0,0038 Wh |

Paper thừa nhận energy benefit của SNN phụ thuộc neuromorphic hardware. Không có benchmark trực tiếp trên Loihi/TrueNorth hoặc phép đo công suất end-to-end, vì vậy con số năng lượng chỉ nên ghi là paper estimate.

Hai sanity check bổ sung:

- `4,43 G MACs` gần bằng `4,48 G FLOPs`, trong khi nhiều convention tính một MAC xấp xỉ hai FLOPs. Profiler cũng có thể bỏ qua recurrent loop hoặc custom SNN operation; không có profiler code để kiểm tra.
- Từ `0,0038 Wh` trong `0,0902 s`, công suất suy ra xấp xỉ 151,7 W. Điều này gợi ý energy có thể chỉ được ước lượng bằng thời gian nhân một power constant khoảng 150 W, không phải đo toàn hệ thống.

Paper còn claim giảm computational cost 40% nhưng không cung cấp bảng đối chứng đủ để xác nhận con số này.

### 13.14. Robustness paper trình bày

Ba corruption:

- Gaussian blur: sigma 0,2; 0,4; 0,6.
- Salt-and-pepper noise: probability 0,05; 0,10; 0,15.
- Speckle noise: sigma 0,1; 0,2; 0,3.

Figure cho thấy một ảnh normal và một ảnh pneumonia vẫn được phân loại đúng ở các mức trên. Đây là minh họa định tính, không phải robustness benchmark toàn dataset.

Các noise trong Figure 9 có màu dù input được mô tả là một grayscale channel lặp ba lần. Có thể noise đã được sample độc lập theo channel, tạo artifact không đại diện cho sensor noise CXR. Một reproduction hợp lý nên corrupt ảnh grayscale trước, dùng cùng corruption cho ba channel, chạy toàn test set với nhiều noise seed và báo delta metric so với clean baseline.

### 13.15. Attention/XAI trong paper

Paper xác nhận attention map được tạo **sau classification**, tương tự Grad-CAM, và không ảnh hưởng prediction. Vì vậy:

- Nó không phải trainable attention-guided classifier theo nghĩa chặt.
- Không thể dùng attention map để giải thích phần accuracy tăng.
- Cần gọi là post-hoc activation/attention visualization trong reproduction report.

Paper không công bố Grad-CAM/Grad-CAM++ hay thuật toán cụ thể, target layer, class score, cách aggregate qua 25 timestep hoặc normalization. Figure chỉ đưa ba prediction đúng; một số activation nằm ở shoulder, border, upper chest hoặc vùng ngoài lung field. Dataset cũng không có lesion mask/bounding box để hỗ trợ claim “align với radiologist annotations”, và paper không báo một reader study hay localization metric nào.

---

## 14. Các mâu thuẫn và rủi ro trong paper chính

### 14.1. Dataset count

Paper viết dataset có 5.863 ảnh nhưng cũng ghi 3.883 pneumonia + 1.349 normal:

```text
3.883 + 1.349 = 5.232
```

5.232 chính là train+validation của snapshot phổ biến, không gồm test gốc 624 ảnh.

### 14.2. Split và confusion matrix

80/20 của 5.232 cho khoảng:

- Train: 4.185 ảnh.
- Holdout: 1.047 ảnh.

Nhưng confusion matrix có:

```text
768 + 772 + 3 + 7 = 1.550 mẫu
```

Không có split được mô tả nào tạo đúng 1.550 mẫu.

### 14.3. Dấu hiệu forensic về SMOTE-before-split

Có một phép tính khớp chính xác kích thước confusion matrix:

```text
Original train: Pneumonia 3.875, Normal 1.341
→ SMOTE tăng Normal thành 3.875
→ balanced total = 7.750
→ 20% × 7.750 = 1.550
```

Confusion matrix gồm 771 normal và 779 pneumonia, gần cân bằng nhưng không đúng 775/775; điều này phù hợp với random split không stratify từ một tập đã được cân bằng.

Một giả thuyết có sức giải thích là:

```text
original train
→ SMOTE trên toàn bộ tập
→ sau đó mới split 80/20
→ evaluation có 1.550 mẫu
```

Đây là **suy luận forensic**, không phải điều paper xác nhận. Nếu đúng, synthetic samples được sinh từ toàn bộ dữ liệu trước split có thể làm train và evaluation chia sẻ thông tin, khiến accuracy lạc quan nghiêm trọng. Nhóm nên chạy một experiment riêng để kiểm tra liệu protocol sai này có tái tạo confusion-matrix size hay không, nhưng không được dùng nó làm kết quả khoa học hợp lệ.

### 14.4. Metric không khớp confusion matrix

Nếu `TN=768`, `TP=772`, `FP=3`, `FN=7` thì:

- Accuracy ≈ 99,35%.
- Precision pneumonia ≈ 99,61%.
- Recall pneumonia ≈ 99,10%.
- F1 ≈ 99,36%.

Không khớp precision 99,1%, recall 99,5% và F1 99,1% paper công bố.

Ngay cả các scalar paper báo cũng không tự khớp công thức F1:

- Với precision 99,1 và recall 99,5, F1 phải xấp xỉ 99,30%, không phải 99,1%.
- Với no-GRU precision 96,22 và recall 96,20, F1 phải xấp xỉ 96,21%, không phải 96,54%.
- Harmonic mean không thể lớn hơn cả precision và recall cùng được dùng trong công thức, nên F1 96,54% là bất khả thi nếu các giá trị đến từ cùng aggregation.

Chỉ có thể có chênh lệch nếu paper dùng các averaging scheme khác nhau, nhưng paper không công bố điều đó.

### 14.5. Epoch inconsistency

- OneCycleLR được mô tả chạy đủ 30 epoch.
- Experimental setup lại ghi “seven total epochs”.
- Curves/ablation được diễn giải tới epoch 30.

Project decision hợp lý: `max_epochs=30`, early stopping patience 5, đồng thời báo epoch thực dừng.

### 14.6. Shape inconsistency

- Ba pool từ input 128×128 với output channel 128 tạo flatten 32.768.
- Paper ghi projection từ 65.536.
- Figure 4 cho biết 768 = 256 spatial/spiking + 512 BiGRU, nhưng projection chỉ kết thúc ở 64 và không có bước 64→256.
- GRU input size và cách reshape flatten thành chuỗi không được công bố.

Vì vậy không thể tuyên bố code là exact reproduction nếu chưa có code gốc hoặc clarification từ tác giả.

### 14.7. Parameter inconsistency tiềm tàng

Chỉ riêng Linear `65.536 → 128` đã có khoảng 8,39 triệu weights. Tính gần đúng:

| Khối | Parameter theo mô tả paper |
|---|---:|
| Ba CNN block | 93.696 |
| Projection 65.536→128→64 | 8.396.992 |
| Head 768→512→128→2 | 459.650 |
| **Tổng chưa có BiGRU/LIF** | **8.950.338** |

Tổng này đã lớn hơn con số paper báo `8.631.042` trước khi cộng BiGRU. Vì vậy reported parameter count không thể thuộc nguyên văn kiến trúc Table 1.

### 14.8. Temporal modeling ambiguity

Lặp cùng một feature 25 lần không tạo dữ liệu thời gian mới. GRU có thể đóng vai trò một nonlinear recurrent transformation, nhưng paper chưa chứng minh lợi ích đến từ temporal information thay vì chỉ tăng capacity.

Control cần có:

- Feed-forward MLP với parameter count tương đương.
- Unidirectional GRU.
- BiGRU với 1, 5, 10, 25 timestep.
- Shuffle/repeat control.

### 14.9. Attention terminology

Title gọi “attention-guided” nhưng method nói attention chỉ post-classification và không ảnh hưởng prediction. Kết luận lại mô tả embedded attention làm tăng accuracy. Đây là mâu thuẫn khái niệm.

### 14.10. Robustness evidence yếu

Hai ảnh minh họa không đủ để claim robust. Cần metric toàn test set, confidence interval và severity curve.

### 14.11. State-of-the-art comparison sai phạm vi

Một số reference trong Table 4 là nghiên cứu Alzheimer/MRI chứ không phải pneumonia CXR. Vì vậy bảng này không nên được dùng làm bằng chứng state of the art.

### 14.12. Energy claim chưa được xác nhận

SNN chạy qua 25 timestep trên GPU không tự nhiên tiết kiệm. Muốn claim energy efficiency cần:

- Xác định hardware.
- Đo wall power/energy end-to-end.
- So sánh cùng batch, precision và throughput.
- Báo spike rate/sparsity.
- Nếu dùng neuromorphic hardware, báo conversion và accuracy sau deployment.

### 14.13. Patient leakage

Paper nói stratified 80/20 ở mức ảnh, không đề cập patient grouping. Dataset có nhiều ảnh cho cùng `person ID`, nên reported performance có nguy cơ optimistic.

### 14.14. Latency-intensity narrative không được nối với computational graph

Introduction của PDF paper chính tuyên bố:

```text
pixel intensity
→ latency-intensity encoding
→ temporally precise spike trains
→ spiking convolutional layers
→ adaptive pooling
→ classification
```

Nhưng phần method/table/figure lại mô tả:

```text
static CXR
→ conventional Conv + BatchNorm + LeakyReLU + MaxPool
→ flatten/projection
→ lặp feature qua 25 timestep + BiGRU/LIF
→ dense head
```

Paper không công bố công thức chuyển intensity thành spike time, số spike/pixel, xử lý pixel 0, polarity sau normalization, temporal binning, target tensor hoặc spiking-convolution definition. Không có cầu nối đủ rõ giữa hai mô tả.

Vì thế project phải coi đây là hai giả thuyết riêng:

1. **Reconstruct architecture figures:** CNN feature + repeated temporal sequence + BiGRU/LIF.
2. **True latency encoder ablation:** triển khai một encoder được định nghĩa đầy đủ và ghi rõ đây là extension của nhóm, không phải exact reproduction.

Không được chọn một công thức latency encoder tùy ý rồi tuyên bố đã tái tạo paper. Cũng không được giả định vùng sáng luôn là pathology: xương, tim, marker, text và border đều có thể sáng hơn nhu mô phổi.

---

## 15. Protocol reproduction đề xuất cho nhóm

### 15.1. Hai protocol song song

#### Protocol A — Paper-compatible

- Merge train+val gốc.
- Stratified 80/20 image-level split với seed cố định.
- Giữ test gốc để đánh giá bổ sung nếu phù hợp.
- Dùng để xem nhóm có tái hiện xu hướng paper hay không.
- Gắn nhãn rõ `paper_compatible`, không gọi đây là protocol lâm sàng tối ưu.

#### Protocol B — Leakage-resistant

- Xây patient/group ID từ metadata/tên file sau khi kiểm chứng.
- Group-aware split.
- Deduplicate trước split.
- Dùng test gốc chỉ sau khi audit patient overlap; nếu overlap thật, tạo grouped holdout mới hoặc dùng external dataset.
- Đây là kết quả chính để đánh giá generalization nội bộ.

### 15.2. Data manifest

Mỗi sample nên có:

```text
relative_path
split_original
split_experiment
class_name
class_id
patient_or_group_id
image_hash
width
height
channel_mode
source
view_if_known
```

### 15.3. Baseline trước hybrid model

Tối thiểu:

1. Majority-class/dummy baseline.
2. Small CNN train from scratch.
3. ResNet18 pretrained.
4. DenseNet121 pretrained.
5. Full hybrid CNN–BiGRU–SNN.

Nếu hybrid không vượt baseline đáng kể qua nhiều seed và confidence interval, complexity tăng không được biện minh.

### 15.4. Cách xử lý ambiguity kiến trúc

Tạo ít nhất hai config có tên rõ:

- `paper_literal`: bám table tối đa có thể.
- `paper_resolved`: sửa shape/order theo computational graph hợp lý.

Mỗi run lưu:

- Shape trace từng block.
- Model summary.
- Parameter count.
- Giải thích mọi deviation so với paper.

### 15.5. Balancing experiments

So sánh riêng:

- Unweighted cross-entropy.
- Class-weighted cross-entropy.
- Weighted sampler.
- Focal loss.
- SMOTE theo interpretation đã ghi rõ.

Không thay nhiều yếu tố cùng lúc.

Nếu thử embedding-SMOTE:

- Feature extractor chỉ được fit/fine-tune trong training fold.
- SMOTE chỉ được `fit_resample` trên training embeddings.
- Validation/test giữ nguyên class distribution và chỉ chứa sample thật.
- Lưu số lượng ảnh thật tách biệt với số synthetic feature vectors.
- Với cross-validation, extractor và SMOTE phải được fit lại độc lập trong từng fold.

### 15.6. Reproducibility controls

- Seed Python/NumPy/PyTorch/DataLoader.
- Ghi deterministic setting và trade-off performance.
- Lưu exact dependency lock.
- Không report chỉ seed tốt nhất.
- Ít nhất 3–5 seed cho experiment chính nếu compute cho phép.

### 15.7. Reporting language

Dùng các cột:

| Result type | Ý nghĩa |
|---|---|
| `paper_reported` | Con số paper công bố |
| `reproduced` | Nhóm chạy theo interpretation đã ghi |
| `patient_grouped` | Kết quả protocol chống leakage |
| `external` | Kết quả nguồn dữ liệu khác |

Không dùng từ “trustworthy”, “clinical-ready” hoặc “radiologist-level” nếu chưa có external clinical validation phù hợp.

---

## 16. Roadmap thí nghiệm

### Giai đoạn 0 — Data audit

- Đếm class/split.
- Detect nested duplicate tree.
- Verify ảnh đọc được.
- Hash duplicate.
- Parse và audit patient/group IDs.
- Kiểm tra overlap giữa splits.
- Vẽ resolution, aspect ratio và channel-mode distribution.

### Giai đoạn 1 — Baseline đáng tin

- Small CNN.
- ResNet18.
- DenseNet121.
- Optional Transformer comparator: pretrained DeiT-Tiny nếu ưu tiên chi phí, hoặc đúng `DEIT_Base_Patch16_224` nếu reproduce Singh et al.; không trộn kết quả hai kích thước model.
- Weighted loss.
- Patient-level metrics và confidence interval.

### Giai đoạn 2 — Reconstruct paper blocks

- Spatial CNN với shape tests.
- Projection layer.
- GRU/BiGRU với sequence-shape tests.
- LIF block với state-reset tests.
- Decision head.
- Full forward/backward smoke test.

### Giai đoạn 3 — Ablation

| ID | Experiment | Câu hỏi |
|---|---|---|
| A0 | CNN only | Spatial baseline đạt bao nhiêu? |
| A1 | CNN + MLP matched parameters | Tăng capacity có giải thích gain không? |
| A2 | CNN + GRU | GRU đóng góp gì? |
| A3 | CNN + SNN | SNN đóng góp gì? |
| A4 | CNN + GRU + SNN | Full hybrid có tốt hơn từng phần? |
| A5 | 1/5/10/25 timestep | Temporal length ảnh hưởng ra sao? |
| A6 | weighted loss/focal/SMOTE | Balancing nào hiệu quả? |

### Giai đoạn 4 — Robustness và XAI

- Corruption benchmark toàn test set.
- Grad-CAM cho TP/TN/FP/FN.
- Lung-overlap/occlusion sanity checks.
- Calibration trước và sau corruption.

### Giai đoạn 5 — Generalization

- Patient-grouped protocol.
- External dataset.
- Subgroup analysis nếu có metadata.
- Error taxonomy với review thủ công.
- Chỉ sau binary baseline: cân nhắc extension bốn lớp COVID-19/TB/pneumonia/normal với source-aware split; không ghép public datasets rồi random-split ở mức ảnh.

### Giai đoạn 6 — Efficiency

- Parameter, FLOPs/MACs.
- Peak memory.
- Latency và throughput.
- Energy measurement nếu có thiết bị phù hợp.
- So sánh cùng hardware, batch size và precision.

---

## 17. Checklist đọc và review một paper mới

### Dataset

- Dataset nào, snapshot/version nào?
- Bao nhiêu ảnh và bao nhiêu bệnh nhân?
- Class count có cộng đúng không?
- Có duplicate hoặc source overlap không?
- Split theo ảnh hay bệnh nhân?
- Nhãn đến từ radiologist, report parsing hay filename?

### Model

- Input shape là gì?
- Từng block biến đổi tensor ra sao?
- Output là một logit hay nhiều logits?
- Pretrained hay scratch?
- Parameter/FLOPs có khớp architecture không?
- Attention tham gia prediction hay chỉ post-hoc?

### Training

- Loss, optimizer, scheduler?
- Learning rate, weight decay, batch size, epoch?
- Augmentation và balancing áp dụng ở split nào?
- Early stopping dùng metric nào?
- Có nhiều seed không?

### Evaluation

- Test set có thực sự held-out?
- Threshold chọn ở đâu?
- Positive class là gì?
- Precision/recall/F1 có khớp confusion matrix không?
- Có confidence interval hoặc statistical test không?
- Có external validation không?

### Claims

- Baseline có cùng protocol không?
- “State of the art” có so cùng dataset không?
- “Robust” có đo toàn dataset và nhiều severity không?
- “Explainable” có quantitative evaluation không?
- “Energy efficient” có đo trên hardware thật không?
- Claim lâm sàng có vượt quá evidence không?

---

## 18. Glossary

| Thuật ngữ | Nghĩa |
|---|---|
| AP/PA | Hướng chiếu X-quang trước-sau/sau-trước |
| CXR | Chest X-ray/chest radiograph |
| CNN | Convolutional Neural Network |
| ViT | Vision Transformer; xử lý ảnh như chuỗi patch token |
| DeiT | Data-efficient Image Transformer; họ ViT với recipe/pretraining hướng tới hiệu quả dữ liệu |
| Patch token | Vector biểu diễn một patch ảnh đưa vào Transformer |
| Feature map | Tensor đặc trưng không gian do convolution tạo ra |
| Embedding | Vector đặc trưng nén biểu diễn một ảnh/patch/sample |
| Receptive field | Vùng input ảnh hưởng đến một activation |
| Logit | Score chưa chuyển thành probability |
| Transfer learning | Tái sử dụng weight/feature từ task nguồn |
| Fine-tuning | Tiếp tục train pretrained model trên task đích |
| GRU/BiGRU | Gated Recurrent Unit/chạy hai hướng |
| SNN | Spiking Neural Network |
| LIF | Leaky Integrate-and-Fire neuron |
| Surrogate gradient | Gradient thay thế cho spike function không khả vi |
| Timestep | Bước mô phỏng thời gian trong recurrent/SNN processing |
| CAM/Grad-CAM | Post-hoc class activation visualization |
| XAI | Explainable Artificial Intelligence |
| IoU | Intersection over Union |
| mAP | Mean Average Precision cho detection |
| Dice | Metric overlap thường dùng segmentation |
| ROC-AUC | Area under ROC curve |
| PR-AUC | Area under precision–recall curve |
| Calibration | Mức probability dự đoán phản ánh tần suất đúng thực tế |
| Domain shift | Phân phối dữ liệu triển khai khác training |
| Data leakage | Thông tin ngoài train hợp lệ lọt vào quá trình học/chọn model |
| Source leakage | Model dựa vào dấu hiệu nguồn dữ liệu/máy chụp thay vì pathology |
| SMOTE | Nội suy giữa các vector minority-class để tạo synthetic samples |
| Patient-level split | Mọi ảnh cùng bệnh nhân nằm trong một split |
| Ablation | Loại/thay một thành phần để đo contribution |
| External validation | Đánh giá trên nguồn dữ liệu độc lập |

---

## Kết luận dành cho nhóm

Thông điệp cốt lõi từ toàn bộ literature trong folder là:

```text
Data quality và split protocol
        ↓
Baseline đơn giản nhưng đáng tin
        ↓
Ablation công bằng
        ↓
External validation
        ↓
XAI, robustness và efficiency có đo lường
        ↓
Chỉ sau đó mới đưa ra claim mạnh
```

CNN–BiGRU–SNN là một hướng nghiên cứu thú vị, nhưng paper 2025 không cung cấp computational graph và evaluation protocol đủ nhất quán để tái tạo duy nhất. Giá trị khoa học của project sẽ đến từ việc nhóm:

1. Ghi rõ mọi interpretation.
2. Phát hiện và định lượng leakage.
3. So sánh với transfer-learning baseline mạnh.
4. Tách contribution thật của GRU và SNN.
5. Phân biệt kết quả paper-reported với kết quả reproduced.
6. Không đánh đồng accuracy cao trên một pediatric benchmark với khả năng chẩn đoán lâm sàng.
