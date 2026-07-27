# Hạn chế

## Tập test không còn nguyên vẹn

624 ảnh này đã được đọc ở nhiều giai đoạn: nó sinh ra giả thuyết về tỉ lệ khung,
định hướng thiết kế 2×2, và xuất hiện trong mọi bảng so sánh từ v2 trở đi. Đây
là **known engineering benchmark**, và mọi con số trên nó là ước lượng **lạc
quan**.

Một ước lượng không thiên lệch cần bệnh nhân mới, từ nguồn khác.

## Nhãn tương quan mạnh với cách chụp ảnh

Trong tập development, đặc trưng thu nhận **một mình** gần như xác định được
nhãn:

- bảng lượng tử hóa JPEG chỉ thuộc một lớp ở **96,6%** ảnh;
- `train/NORMAL` lưu ở JPEG quality 95,5, mọi nhóm khác ở 75,0;
- chồng lấn giữa hai lớp theo vector nhiễu: **0,4%**;
- probe tuyến tính đọc miền từ biểu diễn ẩn ở AUC 0,94–0,98 ngay từ `layer1`.

Nghĩa là mô hình có sẵn một đường tắt rất dễ học ngoài bệnh lý. Hai can thiệp
tiền xử lý (chuẩn hóa JPEG, hạ độ phân giải) đều **trượt kiểm tra manipulation**
— dấu vết không gỡ được bằng preprocessing.

Chi tiết: [`phase2a_revised_report.md`](../../../reports/phase2a_revised_report.md),
[`phase2c_representation_audit.md`](../../../reports/phase2c_representation_audit.md).

## Một trung tâm, một nhóm tuổi

Toàn bộ dữ liệu là bệnh nhi tại một cơ sở. Không có thông tin về máy chụp, quy
trình, hay phân bố bệnh nhân. Không suy rộng được sang người lớn, sang trung tâm
khác, hay sang thiết bị khác.

## Nhóm bệnh nhân là suy đoán

Nhóm suy từ tên file, không phải patient ID chính thức. Tôi đã chứng minh khóa
nhóm hiện tại cho 0/4.097 chồng lấn giữa các split, nhưng đó là bằng chứng
không có rò rỉ **theo khóa này**, không phải bằng chứng khóa này đúng.

## Validation nội bộ gần bão hòa

Đây là hạn chế phương pháp lặp lại xuyên suốt dự án:

- OOF group AUC: biên độ 0,0014 trên 7 cấu hình;
- OOF spec@sens97: chỉ còn 1–6 ca báo nhầm mỗi fold;
- tương quan hạng OOF↔test trong thiết kế 2×2: **−0,80**.

Ba lần OOF nghiêng về phương án chuyển kém hơn (resize ở v4, rank ensemble, và
suýt nữa là R+D+T). Cả ba đều bị chặn bởi ràng buộc đặt trước, không phải bởi
nhìn benchmark. Nhưng điều đó nghĩa là **chọn mô hình ở đây phụ thuộc nhiều vào
biên hòa hơn là vào tín hiệu**.

## Chưa đạt mục tiêu tốt

Đạt mục tiêu tối thiểu 82%, **không** đạt 85%. Trần oracle của xếp hạng hiện tại
là 22 ca báo nhầm, nên mục tiêu 20 nằm ngoài tầm mà không đổi biểu diễn.

## Lớp NORMAL chưa được xác minh

Không ai có chuyên môn y khoa đọc lại các ca báo nhầm. Nhãn nhị phân không mô tả
các bất thường không phải viêm phổi, nên một phần trong 40 ca FP có thể là ảnh
**thật sự bất thường** bị gán NORMAL. Gói đọc phim mù đã chuẩn bị nhưng **chưa
ai đọc**.

## Không tuyên bố quá phạm vi

Kết quả DeiT kém **không** chứng minh Transformer kém CNN — nó chỉ nói dưới
protocol này, với cấu hình này, nó chuyển kém.

Verifier XRV null **không** chứng minh pretraining chuyên biệt X-quang vô dụng —
nó nói verifier **tuyến tính** trên **hai** không gian đặc trưng **đóng băng**,
dưới cascade này, không qua cổng phía nguồn.

Hard-negative null **không** chứng minh hard-negative learning vô dụng — nó nói
validation theo từng fold ở điểm làm việc này không đủ độ phân giải để chọn được
cải thiện.

Không tuyên bố SOTA. Không tuyên bố sẵn sàng triển khai lâm sàng.

## So sánh với literature không hợp lệ

Các báo cáo 99%+ trên tập chia gốc dùng protocol khác — thường chia theo ảnh
chứ không theo bệnh nhân, và không kiểm soát đặc tính mã hóa. Vì đặc trưng mã
hóa một mình đã tách được hai lớp ở AUC 0,999 trong development, những kết quả
đó **tương thích với việc khai thác shortcut**, trừ khi có audit mã hóa tường
minh.

Điều này không chứng minh mô hình nào đã công bố dựa vào shortcut.
