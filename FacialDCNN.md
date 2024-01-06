# DCNN: Nhận Diện Cảm Xúc Khuôn Mặt

DCNN, hay Deep Convolutional Neural Network, là một mô hình mạng nơ-ron tích chập sâu được sử dụng để nhận diện cảm xúc từ hình ảnh khuôn mặt. Dưới đây là cách chúng ta xây dựng và huấn luyện mô hình này.
Chuẩn Bị Dữ Liệu

## Chuẩn bị dữ liệu:

• Tải và xử lý dữ liệu từ bộ sưu tập facial-expression-recognition.
• Sử dụng ImageDataGenerator để thực hiện các biến đổi ảnh như xoay, zoom, và các phép biến đổi khác nhằm tăng cường đa dạng dữ liệu đầu vào.
• Chuyển đổi giá trị pixel từ khoảng (0, 255) về khoảng (0, 1) để tối ưu hóa việc học của mô hình.
• Resize hình ảnh về kích thước 48x48 và chuyển đổi thành màu xám để tập trung vào cấu trúc và đặc trưng của khuôn mặt.
Xây Dựng Mô Hình DCNN

## Mô hình

• Sử dụng lớp Conv2D với các bộ lọc kích thước 5x5 và 3x3, số lượng bộ lọc tăng dần từ 64, 128, đến 256. Các thông số của lớp này được tối ưu hóa bởi thuật toán backpropagation và Adam.
• Áp dụng Batch Normalization và MaxPooling để giảm overfitting và giảm kích thước ma trận đặc trưng.
• Sử dụng Dropout để ngăn chặn overfitting bằng cách loại bỏ ngẫu nhiên một số nút.
• Lớp Dense cuối cùng với 7 nút, mỗi nút tương ứng với một loại cảm xúc, sử dụng hàm kích hoạt softmax để phân loại.
Chi Tiết Kiến Trúc Mô Hình

• Block 1: 2 lớp Conv2D (64 filters, 5x5 kernel), BatchNormalization, MaxPooling2D, Dropout (0.4).
• Block 2: 2 lớp Conv2D (128 filters, 3x3 kernel), BatchNormalization, MaxPooling2D, Dropout (0.4).
• Block 3: 2 lớp Conv2D (256 filters, 3x3 kernel), BatchNormalization, MaxPooling2D, Dropout (0.5).
• Flatten, Dense(128), BatchNormalization, Dropout (0.6).
• Lớp đầu ra Dense với hàm kích hoạt softmax.
Huấn Luyện và Đánh Giá

## Huấn luyện
• Chia dữ liệu thành hai phần: tập huấn luyện và tập kiểm thử với tỉ lệ 80%-20%.
• Huấn luyện mô hình với các thông số tối ưu, sau đó đánh giá hiệu suất trên tập kiểm thử.
• Khi thực hiện, mô hình sẽ trả về một ma trận 7 cột, mỗi cột tương ứng với xác suất của một cảm xúc. Cảm xúc có xác suất cao nhất sẽ được chọn là kết quả cuối cùng.