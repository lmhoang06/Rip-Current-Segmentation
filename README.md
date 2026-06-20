# Phát hiện dòng chảy xa bờ sử dụng kiến trúc cải tiến YOLO26n-seg kết hợp khối chú ý CBAM

Kho lưu trữ này chứa toàn bộ mã nguồn, cấu hình thực nghiệm và tài liệu của đồ án môn học CS231 - Thị giác máy tính nâng cao. Dự án tập trung cải tiến mô hình phân đoạn thực thể một giai đoạn phục vụ bài toán cảnh báo nguy hiểm và an toàn sinh mạng bờ biển.

---

## Thành viên thực hiện

* **Họ và tên:** Lê Minh Hoàng
* **Mã số sinh viên (MSSV):** 24520542
* **Lớp:** CS231.Q21
* **Cơ quan chủ quản:** Trường Đại học Công nghệ Thông tin - ĐHQG-HCM

---

## Kiến trúc hệ thống & Đóng góp cốt lõi

* **Mô hình nền tảng:** YOLO26n-seg kết hợp bộ tối ưu hóa MuSGD và cơ chế huấn luyện NMS-free.
* **Giải pháp cải tiến:** Tích hợp khối chú ý kết hợp tuần tự **CBAM (Convolutional Block Attention Module)** trực tiếp vào sau các tầng đầu ra đặc trưng chiến lược **P3** và **P4** thuộc mạng Neck.
* **Bộ dữ liệu thực nghiệm:** Khảo sát trên bộ dữ liệu **RipVIS**, chia tách luồng video thủ công nhằm chống rò rỉ dữ liệu (data leakage) giữa các tập thực nghiệm.
* **Hiệu năng thực tế:** Mô hình cải tiến đạt bước nhảy vọt về khả năng thu hồi đặc trưng mờ nhạt:
  * **Recall:** Tăng thuần **13.50%** (từ 50.42% lên 63.92%).
  * **AP@50:** Tăng **9.19%** (từ 48.43% lên 57.62%).
  * **F2-score:** Đạt mốc **65.07%**, tối ưu hóa việc hạn chế bỏ sót hiểm họa nguy hiểm.

---

## Cấu trúc kho lưu trữ (Repository Structure)

```text
├── Báo cáo môn học/              # Chứa file báo cáo PDF và Slide thuyết trình (.pptx)
├── cs231_demo/                   # Ứng dụng Demo thời gian thực sử dụng Flask/Gunicorn
│   ├── static/                   # Các tệp giao diện tĩnh (CSS, JS)
│   ├── app.py                    # Khởi chạy dịch vụ ứng dụng web chính
│   ├── cbam_runtime.py           # Logic suy luận kiến trúc YOLO26 + CBAM
│   └── overlay.py                # Xử lý đồ họa đè mặt nạ pixel (Instance Mask)
├── model_weights/                # Thư mục lưu trữ trọng số mô hình huấn luyện (.pt)
│   ├── cs231_yolo26n_baseline.pt
│   └── cs231_yolo26n_cbam_p3p4.pt
├── notebooks/                    # Các kịch bản huấn luyện khảo sát thực nghiệm trên Kaggle
│   ├── cs231-yolo26n-baseline.ipynb
│   └── cs231-yolo26n-cbam-ablation-*.ipynb
├── build_yolo_dataset_from_ripvis.py # Script tiền xử lý ánh xạ bộ dữ liệu RipVIS sang định dạng YOLO
└── requirements-demo.txt         # Danh sách các thư viện cần thiết để chạy Demo

```

---

## Hướng dẫn cài đặt và Khởi chạy ứng dụng Demo

### 1. Cài đặt môi trường

Đảm bảo bạn đã cài đặt Python >= 3.9 trên hệ thống (Khuyến khích sử dụng WSL2 hoặc Linux). Tiến hành cài đặt các gói thư viện phụ thuộc:

```bash
pip install -r cs231_demo/requirements-demo.txt

```

### 2. Khởi chạy ứng dụng

Di chuyển vào thư mục demo và cấp quyền thực thi cho script khởi chạy tự động:

```bash
cd cs231_demo
chmod +x start.sh
./start.sh

```

Sau khi chạy, truy cập vào giao diện cục bộ tại địa chỉ `http://localhost:8000` để thử nghiệm tính năng phân vùng dòng chảy xa bờ real-time.

---

## Bản quyền và Giấy phép

Dự án này được mở rộng dựa trên mã nguồn mở của Ultralytics và các tài liệu thuộc cuộc thi NTIRE/AIM. Mã nguồn của đồ án tuân thủ theo các điều khoản bảo mật và giấy phép đính kèm trong tệp [LICENSE](https://www.google.com/search?q=LICENSE).
