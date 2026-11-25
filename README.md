# Thực hành Học máy ứng dụng

Kho bài thực hành Python gồm chín chủ đề, từ hồi quy tuyến tính đến mạng nơ-ron nhiều lớp. Mỗi bài có tài liệu PDF hướng dẫn, mã nguồn và dữ liệu phù hợp với nội dung thực hành.

## Danh sách bài thực hành

| Bài | Chủ đề | Tài liệu | Chạy toàn bộ |
| --- | --- | --- | --- |
| 01 | Hồi quy tuyến tính | [PDF](bai01_hoi_quy/Lab01_Hoi_quy_tuyen_tinh.pdf) | [`bai01_run_all.py`](bai01_hoi_quy/scripts/bai01_run_all.py) |
| 02 | Hồi quy logistic | [PDF](bai02_hoi_quy_logistic/Lab02_Hoi_quy_logistic.pdf) | [`bai02_run_all.py`](bai02_hoi_quy_logistic/scripts/bai02_run_all.py) |
| 03 | Naive Bayes | [PDF](bai03_naive_bayes/Lab03_Naive_Bayes.pdf) | [`bai03_run_all.py`](bai03_naive_bayes/scripts/bai03_run_all.py) |
| 04 | K hàng xóm gần nhất (KNN) | [PDF](bai04_k-nearest_neighbors/Lab04_KNN.pdf) | [`bai04_run_all.py`](bai04_k-nearest_neighbors/scripts/bai04_run_all.py) |
| 05 | Cây quyết định và rừng ngẫu nhiên | [PDF](bai05_decision_tree_and_random_forest/Lab05_CayQuyetDinh_RungNgauNhien.pdf) | [`bai05_run_all.py`](bai05_decision_tree_and_random_forest/scripts/bai05_run_all.py) |
| 06 | Máy vectơ hỗ trợ (SVM) | [PDF](bai06_SVM/Lab06_SVM.pdf) | [`bai06_run_all.py`](bai06_SVM/scripts/bai06_run_all.py) |
| 07 | Đánh giá mô hình | [PDF](bai07_model_evaluation/Lab07_DanhGiaMoHinh.pdf) | [`bai07_run_all.py`](bai07_model_evaluation/scripts/bai07_run_all.py) |
| 08 | Phân cụm K-Means | [PDF](bai08_KMeans/Lab08_KMeans.pdf) | [`bai08_run_all.py`](bai08_KMeans/scripts/bai08_run_all.py) |
| 09 | Mạng nơ-ron nhiều lớp (MLP) | [PDF](bai09_MLP/Lab09_MLP.pdf) | [`bai09_run_all.py`](bai09_MLP/scripts/bai09_run_all.py) |

## Nội dung từng Lab

### Lab 01 · Hồi quy tuyến tính

Khám phá dữ liệu giá nhà và đo sai số dự đoán. Tính hệ số hồi quy bằng công thức, đối chiếu với scikit-learn và thực hành cập nhật hệ số bằng gradient descent. Đánh giá trên tập kiểm tra, xem đồ thị phần dư và so sánh mô hình một biến với mô hình nhiều biến.

### Lab 02 · Hồi quy logistic

Dự đoán sinh viên qua môn từ dữ liệu học tập. Khảo sát hàm sigmoid, huấn luyện mô hình, đọc ma trận nhầm lẫn và xem precision, recall thay đổi theo ngưỡng xác suất. Dùng hai đặc trưng để vẽ ranh giới quyết định.

### Lab 03 · Naive Bayes

Tính xác suất theo định lý Bayes, biểu diễn tin nhắn bằng túi từ và xây dựng bộ lọc thư rác. Dùng làm trơn Laplace khi gặp từ chưa xuất hiện trong một lớp và áp dụng Gaussian Naive Bayes cho dữ liệu số của sinh viên.

### Lab 04 · K hàng xóm gần nhất (KNN)

Tính khoảng cách và dự đoán theo các điểm lân cận. Khảo sát ảnh hưởng của chuẩn hóa, chọn số hàng xóm K, so sánh ranh giới dự đoán trên dữ liệu hai vầng trăng và dùng KNN hồi quy để dự đoán giá nhà.

### Lab 05 · Cây quyết định và rừng ngẫu nhiên

Tính độ bất thuần Gini để chọn phép chia và đọc cây quyết định trên dữ liệu khách vay. Khảo sát ảnh hưởng của độ sâu và chuẩn hóa, so sánh cây đơn với rừng ngẫu nhiên theo số cây, xem độ quan trọng của đặc trưng và áp dụng cây hồi quy cho giá nhà.

### Lab 06 · Máy vectơ hỗ trợ (SVM)

Xác định ranh giới có lề rộng nhất và các vectơ hỗ trợ. Khảo sát tham số C, chuẩn hóa và tác dụng của việc nâng chiều, rồi so sánh các kernel trên dữ liệu hai vầng trăng. Dò lưới C và gamma bằng kiểm định chéo và dùng SVM để nhận dạng chữ số viết tay.

### Lab 07 · Đánh giá mô hình

Đánh giá dự đoán sinh viên bỏ học trên dữ liệu lệch lớp. So sánh với mô hình luôn đoán lớp đông, tính ma trận nhầm lẫn, precision, recall và F1. Chọn ngưỡng bằng kiểm định chéo trên tập huấn luyện, so sánh ROC/AUC và đo ảnh hưởng của cân bằng lớp.

### Lab 08 · Phân cụm K-Means

Tự cài đặt bước gán cụm và cập nhật tâm bằng NumPy, sau đó phân khúc khách hàng. Dùng inertia và silhouette để chọn K. Khảo sát ảnh hưởng của thang đo, tâm ban đầu và hình dạng cụm trên dữ liệu hai vầng trăng, rồi giảm số màu của ảnh bằng K-Means.

### Lab 09 · Mạng nơ-ron nhiều lớp (MLP)

Tự tính lan truyền tiến bằng NumPy và dùng lớp ẩn để giải bài toán XOR. Khảo sát kích thước mạng, điều chuẩn và chuẩn hóa dữ liệu. Huấn luyện MLP nhận dạng chữ số viết tay và theo dõi đường mất mát cùng kết quả khi dừng sớm.

## Cài đặt và chạy

Các lệnh dưới đây dùng PowerShell. Ví dụ sử dụng Python 3.11 và chạy toàn bộ bài 6 từ thư mục gốc của repo.

```powershell
git clone https://github.com/andrew-caominhtri/Machine-Learning.git
Set-Location Machine-Learning
py -3.11 -m venv hocmay
.\hocmay\Scripts\python.exe -m pip install numpy pandas matplotlib scikit-learn pillow
.\hocmay\Scripts\python.exe .\bai01_hoi_quy\scripts\bai01_run_all.py
```

Trong VS Code, chọn `hocmay\Scripts\python.exe` qua **Python: Select Interpreter**, mở file `baiXX_run_all.py` của bài cần chạy, rồi chọn **Run Python File**. Bạn cũng có thể chạy riêng một bước bằng cách mở file tương ứng trong `code/`.

## Cấu trúc thư mục

```text
Machine-Learning/
├── bai01_hoi_quy/
│   ├── Lab01_Hoi_quy_tuyen_tinh.pdf
│   ├── code/
│   ├── data/
│   ├── figures/
│   ├── outputs/
│   └── scripts/
│       └── bai01_run_all.py
├── bai02_hoi_quy_logistic/
├── bai03_naive_bayes/
├── bai04_k-nearest_neighbors/
├── bai05_decision_tree_and_random_forest/
├── bai06_SVM/
├── bai07_model_evaluation/
├── bai08_KMeans/
└── bai09_MLP/
```

Bài 1 minh họa cách tổ chức bên trong một thư mục Lab.

| Vị trí | Nội dung |
| --- | --- |
| `Lab*.pdf` | Hướng dẫn, mã ví dụ và bài tập làm thêm |
| `code/` | Các file Python theo từng bước trong tài liệu |
| `data/` | Các bộ dữ liệu CSV dùng trong bài |
| `figures/` | Hình minh họa được lưu dưới dạng PNG |
| `outputs/` | Ảnh kết quả hiển thị trên terminal |
| `scripts/baiXX_run_all.py` | Chạy lần lượt các bước của bài |


## Người làm bài Lab

[Andrew-Cao Minh Trí](https://github.com/andrew-caominhtri)
