# FaceCNN: Nhận Diện Cảm Xúc từ Khuôn Mặt sử dụng CNN
## Xử Lý Dữ Liệu

• Dữ liệu hình ảnh từ facial-expression-recognition-dataset .  
Ta sẽ sử dụng ImageDataGenerator của thư viện keras để lấy các đặc trưng của dữ liệu, hỗ trợ việc chia tỉ lệ, xoay, và các kiểu biến đổi ảnh khác để làm tăng cường nhiểu kiểu input, làm cho mô hình dự đoán tốt hơn  
• Biến đổi các pixel trong khoảng (0, 255) thành khoảng (0, 1) để cho giá trị nhỏ hơn, mô hình sẽ dự đoán tốt hơn
• Resize lại kích thước hình ảnh: 48x48. Biến đổi hình ảnh thành màu xám, mục đích ở đây là tập trung vào các cấu trúc, đặc trưng của khuôn mặt hơn là các màu sắc, vì màu sắc không ảnh hưởng nhiều đến cảm xúc của con người.



## Kiến Trúc Mô Hình

• Ta sẽ tìm các ma trận kernel cho các lớp Conv2D, thông số này sẽ được tìm bằng thuật toán lan truyền ngược(backpropagation) và thuật toán tối ưu hóa Adam. Sử dụng các bộ lọc (filters) với kích thước 5x5 cho lớp đầu tiên và 3x3 cho các lớp sau, tăng dần số lượng bộ lọc từ 64, 128, đến 256.  
• Tiếp theo, ta sử dụng Batch Normalization và MaxPooling, đây chủ yếu là các kỹ thuật để giảm overfitting. Bằng cách giảm số lượng tham số, và giảm kích thước ma trận.
• Tiếp theo, ta dùng dropout để giảm overfitting, mục đích cũng tương tự như Batch Normalization và MaxPooling, nhưng ở đây, ta sẽ bỏ đi một số node trong mạng để giảm hiện tượng overfitting.
• Cuối cùng, ta có 1 lớp Dense với 7 node, tương ứng với 7 cảm xúc, và dùng hàm softmax để phân loại. Ta cũng dùng thuật toán lan truyền ngược(backpropagation) và thuật toán tối ưu hóa Adam để tìm các thông số cho lớp này. Lưu ý, trước khi đưa vào lớp Dense, ta sẽ làm phẳng ma trận, tức là nối các hàng của ma trận lại với nhau, để phù hợp với đầu vào của lớp Dense.

### Cụ thể hơn về kiến trúc mô hình, ta có   
• Block 1: 2 lớp Conv2D (64 filters, 5x5 kernel), BatchNormalization, MaxPooling2D ,
Dropout (0.4) .
• Block 2: 2 lớp Conv2D (128 filters, 3x3 kernel), BatchNormalization, MaxPooling2D,
Dropout (0.4) •
• Block 3: 2 lớp Conv2D (256 filters, 3x3 kernel), BatchNormalization, MaxPooling2D ,
Dropout (0.5) .
• Flatten , Dense(128) , BatchNormalization , Dropout (0.6) •
• Lớp đâu ra Dense với activation softmax .

## Kết Quả
• Ta sẽ chia dữ liệu thành 2 tập, tập train và tập test, với tỉ lệ 80% - 20%. Và tìm các thông số cho mô hình, sau đó đánh giá mô hình bằng tập test.    
• Ban đầu, các thông số của mô hình sẽ được khởi tạo ngẫu nhiên. Sau khi dùng thuật toán lan truyền ngược (backpropagation) và thuật toán tối ưu hóa Adam để tìm các thông số tối ưu, mô hình sẽ được huấn luyện với các thông số này. Các thông số cần tìm ở đây là các ma trận kernel - ma trận 2 chiều, lớp Dense - ma trận 1 chiều, và các thông số khác.  
• Khi chạy mô hình, ta sẽ thu được 1 ma trận cảm xúc, với 7 cột, mỗi cột tương ứng với 1 cảm xúc, và ta sẽ lấy cảm xúc có xác xuất cao nhất, là cảm xúc của hình ảnh đó.  