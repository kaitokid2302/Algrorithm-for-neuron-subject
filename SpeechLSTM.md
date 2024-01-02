# Thuật toán CNN và LSTM cho bài toán nhận dạng tiếng nói
Dữ liệu được lấy từ bộ dataset facial-expression-recognition-dataset. Sử dụng ImageDataGenerator của Keras để tự động tải và tiền xử lý hình ảnh. Hình ảnh được chuyển đổi kích thước thành 48x48 pixels và chuyển thành dạng grayscale.
## Đọc dữ liệu
1. Ban đầu, dữ liệu sẽ được đọc từ file csv - đã được chuẩn hoá tín hiệu âm thanh thành các con số
2. Mỗi hàng tượng trưng cho 1 file âm thanh, có khá nhiều ô trống, vì thế ta sẽ chèn các ô trống này bằng 0, để làm cho toàn bộ file âm thanh có cùng độ dài, kích thước ma trận như nhau
3. Y=df['Emotion'] - Lấy cột cuối cùng - emotion - làm nhãn
4. Y=to_categorical(lb.fit_transform(Y)) - Chuyển nhãn về dạng one-hot vector, để cho CNN có thể hiêu được, ví dụ angry sẽ là [1,0,0,0,0,0,0,0,0,0], happy sẽ là [0,1,0,0,0,0,0,0,0,0]...
## Chuẩn hoá dữ liệu
1. Ta sẽ chia dữ liệu thành 2 tập, tập train và tập test
2. Chuẩn hoá bằng StandardScaler, để các giá trị nằm trong khoảng [-1,1]
3. np.expand_dims - Thêm chiều cho ma trận, để phù hợp với đầu vào của CNN
## Tìm các đặc trưng
1. Ta sẽ xây dựng mô hình CNN, dùng Conv1D, có:  
   1.1 Filter = 16  
   1.2 kernel = 3  

- Điều này mang ý nghĩa là ta sẽ có 16 bộ lọc, mỗi bộ lọc là 1 ma trận có kích thước 1x3, mỗi bộ lọc này sẽ trượt qua từng dòng(từng file âm thanh) của dữ liệu, để tìm ra các đặc trưng của dữ liệu.


- Ban đầu, các bộ lọc này sẽ có các giá trị ngẫu nhiên, sau khi trượt qua dữ liệu, các bộ lọc này sẽ được cập nhật lại, ở đây ta dùng thuật toán lan truyền ngược(backpropagation) và thuật toán tối ưu hóa Adam để cập nhật lại các bộ lọc này. Cuối dùng, ta sẽ tìm ra được 16 bộ lọc để phân biệt các đặc trưng của dữ liệu 1 cách tốt nhất.

2. Làm phẳng dữ liệu, để phù hợp với đầu vào của LSTM - Cụ thể ta sẽ thu được ma trận chỉ có duy nhất 1 hảng

## Thực hiện thuật toán phân loại cảm xúc
2. Dùng thuật toán LSTM (Long Short-Term Memory):
   
 - Về cơ bản, thuật toán LSTM chỉ là 1 dạng của RNN, về cơ bản, ta có thể hiểu RNN chạy từ trái qua phải, mỗi bước, nó sẽ nhận đầu vào là 1 vector, và trả ra 1 vector, vector này sẽ được truyền vào bước tiếp theo, và cứ như vậy, đến bước cuối cùng, nó sẽ trả ra 1 vector cuối cùng, vector này sẽ được dùng để phân loại.
  
- Ở mỗi layer, ta sẽ tìm các thông số:
  - Forget gate: Để xem các thông tin nào cần thiết, hoặc không cần thiết(thông tin thu được từ bước trước đó - layer trước đó)
  - Input gate: W(hh) - để biến đổi thông tin thu được từ bước trước đó, thành thông tin mới
  - W(hx) - để biển đổi thông tin hiện tại thành thông tin mới
  - Output Gate: W(yh) - để biến đổi thông tin hiện tại và thông tin trước đó thành thông tin mới, dùng cho layer tiếp theo

- Ta sẽ dùng thuật toán lan truyền ngược(backpropagation) và thuật toán tối ưu hóa Adam để tìm các thông số này.

3. Cuối cùng, ta sẽ thu được mô hình, có độ chính xác khoảng 55% - 60%.