# Thuật toán CNN cho bài toán nhận dạng tiếng nói
## Đọc và biến đổi dữ liệu
1. librosa. load(path, duration=duration, offset=offset) : Hàm này đọc file âm thanh từ đường dẫn của chúng ta và chuyển đổi nó thành một màng số, đồng thời trả về tốc độ lấy mẫu ( sr - sample rate). 
    - Tham số duration nghĩa là độ dài của âm thanh: Ta sẽ lấy 2,5 hoặc 3 giây là đủ.
    -  offset là khoảng thời gian bắt đầu đọc file âm thanh: ở đây, offset sẽ là 0,5, tức là ta sẽ bắt đầu đọc từ giây thứ 0,5. Khoảng thời gian trước 0,5 thường sẽ là tiếng ồn hoặc im lặng, nên ta sẽ bỏ qua.
    -  sr là tốc độ lấy mẫu của âm thanh, tức là số mẫu được lấy trong 1 giây
2. Chuyển đổi âm thanh thành dạng số: Bằng cách lấy các thông số zcr, rmse, mfcc, sau đó ghép lại thành mảng có 1 hàng duy nhất
3. Ta cũng sẽ biến đổi âm thanh thành các biến thể khác nhau, nhằm tạo ra sự đa dạng cho dữ liệu, cụ thể, ta sẽ làm cho tiếng nhỏ hơn, ồn hơn, thay đổi tốc độ âm thanh...
4. Số cột sau khi biến đổi sẽ là 2376, những hàng nào chưa đủ 2376 cột, ta sẽ thêm các cột 0 vào cuối hàng, để đủ 2376 cột
## Chuẩn hoá dữ liệu
1. Ta sẽ chia dữ liệu thành 2 tập, tập train và tập test
2. Chuẩn hoá bằng StandardScaler, để các giá trị nằm trong khoảng [-1,1]
3. np.expand_dims - Thêm chiều cho ma trận, để phù hợp với đầu vào của CNN
## Thực hiện thuật toán phân loại cảm xúc
1. Có nhiều layer, mỗi layer sẽ có số lượng bộ lọc(filter) là 256, và kích thước ma trận kernel là 1x5
2. Ở mỗi layer, ta sẽ tìm các thông số cho các ma trận kernel này, thông số này sẽ được tìm bằng thuật toán lan truyền ngược(backpropagation) và thuật toán tối ưu hóa Adam
3. Sau khi có được các thông số cho các ma trận kernel, ta sẽ dùng các ma trận này để nhân với ma trận dữ liệu, sau đó dùng maxpooling để giảm kích thước ma trận đi
4. Sau khi trải qua nhiều layer, ta sẽ làm phẳng ma trận
5. Cuối dùng, ta sẽ tiếp tục tìm ra lớp dense - lớp này sẽ có 7 node, tương ứng với 7 cảm xúc, và dùng hàm softmax để phân loại. Ta cũng dùng thuật toán lan truyền ngược(backpropagation) và thuật toán tối ưu hóa Adam để tìm các thông số cho lớp này.
6. Ta thu được mô hình, sau khi dùng 1 file âm thanh, ta sẽ thu được 1 ma trận 1x7, mỗi ô sẽ là xác xuất của cảm xúc, ô nào có xác xuất cao nhất, ô đó chính là cảm xúc của file âm thanh đó.