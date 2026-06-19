<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<h1 align="center"><b>NHẬP MÔN THỊ GIÁC MÁY TÍNH</b></h1>

---

# THÀNH VIÊN NHÓM

| STT | MSSV     | Họ và Tên            | Chức vụ     | Email                  |
|-----|----------|----------------------|-------------|------------------------|
| 1   | 23520899 | Nguyễn Thế Luân      | Nhóm trưởng | 23520899@gm.uit.edu.vn |
| 2   | 23520361  | Đặng Vân Duy | Thành viên | 23520361@gm.uit.edu.vn |

---

# GIỚI THIỆU MÔN HỌC

- **Tên môn học:** Nhập môn thị giác máy tính
- **Mã môn học:** CS231
- **Mã lớp:** CS231.Q23
- **Năm học:** Học kỳ 2 (2025 - 2026)
- **Giảng viên:** Tiến Sĩ Mai Tiến Dũng

---

# GIỚI THIỆU ĐỀ TÀI

- **Tên đề tài:** Phân loại bệnh da bằng mô hình học sâu
- **Bài toán:** Xây dựng hệ thống nhận diện ảnh da thuộc một trong sáu nhóm: Chickenpox, Cowpox, HFMD, Healthy, Measles và Monkeypox.
- **Hướng tiếp cận:** Huấn luyện và đánh giá các mô hình CNN dựa trên VGG16 và ResNet18, sau đó triển khai demo dự đoán bằng Streamlit.
- **Đầu vào:** Ảnh da định dạng `.jpg`, `.jpeg` hoặc `.png`.
- **Đầu ra:** Nhãn dự đoán, xác suất từng lớp và bảng soft label.

---

# CÔNG NGHỆ SỬ DỤNG

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Pillow

---

# CẤU TRÚC THƯ MỤC

```text
CS231.Q23/
|-- demo/
|   |-- app.py
|   |-- best_resnet18.pth
|   `-- best_vgg16.pth
|-- notebooks/
|   |-- enhanced-resnet18.ipynb
|   |-- enhanced-vgg16.ipynb
|   |-- resnet18.ipynb
|   |-- split-data.ipynb
|   |-- training-resnet18.ipynb
|   `-- vgg16.ipynb
|-- src/
|   |-- data/
|   |   |-- dataset.py
|   |   `-- split_dataset.py
|   |-- evaluation/
|   |   `-- evaluate.py
|   |-- models/
|   |   `-- vgg16_model.py
|   `-- training/
|       `-- train_vgg16.py
|-- requirements.txt
`-- README.md
```

---

# CÀI ĐẶT MÔI TRƯỜNG

## 1. Tạo môi trường ảo

```powershell
python -m venv .venv
```

## 2. Kích hoạt môi trường

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn kích hoạt môi trường, chạy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Cài đặt thư viện

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# CHẠY DEMO

Từ thư mục gốc của project, chạy:

```powershell
streamlit run demo/app.py
```

Sau đó mở địa chỉ Streamlit hiển thị trên terminal, thường là:

```text
http://localhost:8501
```

Trong giao diện demo:

1. Chọn mô hình VGG16 hoặc ResNet18 ở sidebar.
2. Tải ảnh da cần dự đoán.
3. Xem nhãn dự đoán, xác suất từng lớp và bảng kết quả.

Lưu ý: các file trọng số `.pth` thường không được push lên Git do kích thước lớn và đã được khai báo trong `.gitignore`. Để chạy demo, cần đặt các file sau trong thư mục `demo/`:

```text
demo/best_vgg16.pth
demo/best_resnet18.pth
```

---

# HUẤN LUYỆN VÀ ĐÁNH GIÁ

## Chia dữ liệu

Script chia dữ liệu nằm tại:

```text
src/data/split_dataset.py
```

Ví dụ chạy:

```powershell
python src/data/split_dataset.py --source data/raw --output data/splits --force
```

## Huấn luyện VGG16

```powershell
python src/training/train_vgg16.py
```

## Đánh giá mô hình

Các hàm đánh giá nằm tại:

```text
src/evaluation/evaluate.py
```

Các metric được sử dụng gồm Accuracy, Precision, Recall, F1-score, ROC-AUC, Sensitivity, Specificity, MCC và Confusion Matrix.

---

# NOTEBOOKS

- `split-data.ipynb`: chia dữ liệu thành train/validation/test.
- `training-resnet18.ipynb`: huấn luyện ResNet18.
- `resnet18.ipynb`: thử nghiệm và đánh giá ResNet18.
- `vgg16.ipynb`: thử nghiệm và đánh giá VGG16.
- `enhanced-resnet18.ipynb`: phiên bản cải tiến cho ResNet18.
- `enhanced-vgg16.ipynb`: phiên bản cải tiến cho VGG16.

---

# KẾT QUẢ

Project triển khai hai mô hình chính:

| Mô hình  | Mục đích |
|----------|----------|
| VGG16    | Phân loại ảnh bệnh da dựa trên kiến trúc CNN sâu |
| ResNet18 | Phân loại ảnh bệnh da với kiến trúc residual network |

Demo Streamlit cho phép so sánh nhanh kết quả dự đoán giữa hai mô hình trên cùng một ảnh đầu vào.

---

# HƯỚNG PHÁT TRIỂN

- Bổ sung thêm dữ liệu để tăng khả năng tổng quát hóa.
- Thử nghiệm thêm các kiến trúc như EfficientNet, DenseNet hoặc Vision Transformer.
- Tối ưu giao diện demo để trực quan hóa tốt hơn.
- Triển khai mô hình lên nền tảng cloud để người dùng có thể sử dụng trực tuyến.

---

# TÀI LIỆU THAM KHẢO

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Torchvision Models: https://pytorch.org/vision/stable/models.html
- Streamlit Documentation: https://docs.streamlit.io/
- Scikit-learn Documentation: https://scikit-learn.org/stable/
