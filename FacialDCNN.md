# DCNN: Nhận Diện Cảm Xúc từ Khuôn Mặt sử dụng DCNN
## Xử Lý Dữ Liệu
• Dữ liệu hình ảnh từ facial-expression-recognition-dataset.
• Hình ảnh được chuyển thành grayscale, kích thước 48x48 pixels.
## Kiến Trúc Mô Hình
• Block 1: 2 lớp Conv2D (64 filters, 5x5 kernel), BatchNormalization, MaxPooling2D, Dropout (0.4).
• Block 2: 2 lớp Conv2D (128 filters, 3x3 kernel), BatchNormalization, MaxPooling2D, Dropout (0.4).
• Block 3: 2 lớp Conv2D (256 filters, 3x3 kernel), BatchNormalization, MaxPooling2D, Dropout (0.5).
• Flatten, Dense(128), BatchNormalization, Dropout (0.6).
• Lớp đầu ra Dense với activation softmax.
## Huấn Luyện và Đánh Giá

• Mô hình được biên dịch với optimizer Adam và loss function là categorical_crossentropy.
• Đánh giá qua các biểu đồ độ chính xác và mất mát trên tập huấn luyện và kiểm thử.
• Dùng thuật toán lan truyền ngược (backpropagation) và thuật toán tối ưu hóa Adam để tìm các thông số cho block 1, block 2, block 3, flatten, dense, cuối cùng ta sẽ có đầu ra. Độ chỉnh xác khoảng 60% -
70%.

## Ý nghĩa
1. Dropout dùng để ngăn chặn overfitting, nó sẽ tắt ngẫu nhiên 1 số node trong quá trình huấn luyện, để các node còn lại phải học cách tổng quát hơn, và không chỉ học thuộc lòng dữ liệu huấn luyện.
2. BatchNormalization dùng để chuẩn hoá dữ liệu, để các giá trị nằm trong khoảng [-1,1], giúp cho việc huấn luyện nhanh hơn, và tránh overfitting.
3. MaxPooling2D dùng để giảm kích thước của ma trận, giúp cho việc huấn luyện nhanh hơn, và tránh overfitting.
4. Conv2D dùng để tìm ra các đặc trưng của dữ liệu, nó sẽ trượt qua từng dòng của dữ liệu, để tìm ra các đặc trưng của dữ liệu, đơn giản hoá, đây chính là ma trận 2 chiều
