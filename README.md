# Dự án: Phân loại Ảnh Động vật (4 Lớp)

Dự án này xây dựng một mô hình học máy để phân loại hình ảnh của 4 loài động vật: chó, mèo, sư tử, gà. Quy trình được thực hiện đầy đủ từ việc chuẩn bị dữ liệu, huấn luyện mô hình đến triển khai thành một ứng dụng web đơn giản.

## Kế hoạch Thực hiện (To-Do List)

### Giai đoạn 1: Khởi tạo và Chuẩn bị Dữ liệu

#### [ ] Khởi tạo môi trường:
- [x] Clone repository.
- [x] Tạo virtual environment và kích hoạt.
- [x] Cài đặt các thư viện cần thiết từ requirements.txt.

#### [ ] Thu thập Dữ liệu (data/raw):
- [x] Tìm và tải xuống bộ dữ liệu hình ảnh cho 4 loài động vật.
- [x] Tổ chức dữ liệu gốc vào các thư mục con tương ứng: data/raw/dogs, data/raw/cats, data/raw/lions, data/raw/chickens.

#### [ ] Tiền xử lý và Phân chia Dữ liệu (src/data_utils.py):
- [ ] Viết script để chia dữ liệu thành các tập train, validation, và test.
- [ ] Thực hiện xử lý (ví dụ: thay đổi kích thước ảnh) và lưu vào data/processed/.

### Giai đoạn 2: Phân tích và Xây dựng Mô hình

#### [ ] Phân tích Khám phá Dữ liệu - EDA (notebooks/EDA.ipynb):
- [ ] Phân tích sự phân bổ số lượng ảnh của mỗi lớp.
- [ ] Trực quan hóa một vài ảnh mẫu để kiểm tra chất lượng.
- [ ] Kiểm tra sự đa dạng về kích thước, góc chụp của ảnh.

#### [ ] Xây dựng Mô hình (src/models.py):
- [ ] Mô hình cơ bản: Định nghĩa một kiến trúc CNN đơn giản để làm baseline.
- [ ] Mô hình nâng cao: Định nghĩa một mô hình sử dụng Transfer Learning (ví dụ: ResNet50, EfficientNet) để đạt hiệu suất cao hơn.

### Giai đoạn 3: Huấn luyện và Tinh chỉnh

#### [ ] Viết mã Huấn luyện (src/train.py):
- [ ] Xây dựng hàm nạp dữ liệu (data loader) có áp dụng Data Augmentation (tăng cường dữ liệu).
- [ ] Viết vòng lặp huấn luyện (training loop) để huấn luyện mô hình.
- [ ] Tích hợp việc lưu lại model tốt nhất và ghi log quá trình huấn luyện vào outputs/.

#### [ ] Tiến hành Huấn luyện:
- [ ] Huấn luyện mô hình CNN cơ bản.
- [ ] Huấn luyện mô hình Transfer Learning.
- [ ] Theo dõi các chỉ số (loss, accuracy) trên tập validation.

### Giai đoạn 4: Đánh giá Mô hình

#### [ ] Viết mã Đánh giá (src/evaluate.py):
- [ ] Viết script để tải mô hình đã huấn luyện từ outputs/models/.
- [ ] Đánh giá hiệu suất của mô hình trên tập test.

#### [ ] Phân tích Kết quả:
- [ ] Tính toán độ chính xác (accuracy) tổng thể và trên từng lớp.
- [ ] Tạo và phân tích Ma trận nhầm lẫn (Confusion Matrix) để xem mô hình hay nhầm lẫn giữa các lớp nào. Lưu biểu đồ vào outputs/figures/.

#### [ ] Lặp lại:
- [ ] Dựa trên kết quả, quay lại tinh chỉnh (fine-tune) mô hình hoặc các siêu tham số (hyperparameters) nếu cần.

### Giai đoạn 5: Triển khai (Deployment)

#### [ ] Xây dựng Ứng dụng Web (app/app.py):
- [ ] Sử dụng Flask hoặc FastAPI để tạo một API đơn giản.
- [ ] API nhận đầu vào là một hình ảnh và trả về dự đoán của mô hình.

#### [ ] Đóng gói Ứng dụng (app/Dockerfile):
- [ ] Viết Dockerfile để đóng gói ứng dụng và các dependencies.

#### [ ] Kiểm thử:
- [ ] Build và chạy Docker container để kiểm tra hoạt động của ứng dụng web.

## License

MIT License 