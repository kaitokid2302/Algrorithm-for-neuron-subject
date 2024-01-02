# FaceCNN: Nhận Diện Cảm Xúc từ Khuôn Mặt sử dụng CNN
## Xử Lý Dữ Liệu

• Dữ liệu hình ảnh từ facial-expression-recognition-dataset .
• Ành chuyền thành grayscale, kích thước 48x48 pixels.
• Dùng ImageDataGenerator cho rescaling và augmentation.

## Kiến Trúc Mô Hình

• Block 1: 2 lớp Conv2D (64 filters, 5x5 kernel), BatchNormalization, MaxPooling2D ,
Dropout (0.4) .
• Block 2: 2 lớp Conv2D (128 filters, 3x3 kernel), BatchNormalization, MaxPooling2D,
Dropout (0.4) •
• Block 3: 2 lớp Conv2D (256 filters, 3x3 kernel), BatchNormalization, MaxPooling2D ,
Dropout (0.5) .
• Flatten , Dense(128) , BatchNormalization , Dropout (0.6) •
• Lớp đâu ra Dense với activation softmax .

## Huấn Luyện và Đánh Gia

• Mô hình được biên dịch với optimizer Adam và loss function là categorical_crossentropy .
• Đánh giá qua các biểu đồ độ chính xác và mất mát trên tập huấn luyện và kiềm thử.
• Dùng thuật toán lan truyền ngược(backpropagation) và thuật toán tối ưu hóa Adam để tìm các thông số cho block 1, block 2, block 3, flatten, dense, cuối cùng ta sẽ có đầu ra. Độ chỉnh xác khoảng 60% - 70%
