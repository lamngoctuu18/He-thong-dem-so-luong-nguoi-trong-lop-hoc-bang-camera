<h1 align="center">HỆ THỐNG ĐẾM SỐ LƯỢNG NGƯỜI TRONG LỚP HỌC BẰNG CAMERA</h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e5a919d1-d081-4d12-819e-5fb18ce91a68" width="200"/>
  <img src="https://github.com/user-attachments/assets/59dec55d-7825-422f-b80c-ac6915e3775a" width="170"/>
</p>
<p align="center">
  <a href="https://www.facebook.com/DNUAIoTLab">
    <img src="https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge" alt="Made by AIoTLab"/>
  </a>
  <a href="https://fitdnu.net/">
    <img src="https://img.shields.io/badge/Fit%20DNU-green?style=for-the-badge" alt="Fit DNU"/>
  </a>
  <a href="https://dainam.edu.vn">
    <img src="https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge" alt="DaiNam University"/>
  </a>
</p>

## **1. Giới thiệu**  
**Hệ thống Đếm Số Lượng Người bằng Camera sử dụng Trí tuệ nhân tạo (AI)** để phát hiện, định danh (Track ID) và đếm số lượng người ra vào thông qua luồng video trực tiếp từ Camera RTSP, Webcam Laptop hoặc file Video. Hệ thống phù hợp cho các ứng dụng giám sát như:

- Theo dõi sĩ số sinh viên trong lớp học, phòng thực hành, văn phòng, trung tâm thương mại.
- Kiểm soát lưu lượng người ra vào (`In / Out`) tại các sự kiện.
- Hỗ trợ thống kê, cảnh báo giọng nói và quản lý không gian thông minh.

### **Cơ chế hoạt động cốt lõi:**
1. **Nhận diện người (Object Detection):** Sử dụng mô hình **YOLOv8** kết hợp bộ lọc hình học (`is_valid_person_box`) để nhận diện chính xác cả người ở xa, người bị bàn ghế/lan can che khuất một phần và loại bỏ đồ vật nhiễu xung quanh.
2. **Theo dõi đối tượng (Multi-Object Tracking):** Sử dụng **DeepSORT** với tọa độ chuẩn `[left, top, width, height]` để duy trì mã định danh (`ID`) cho từng người.
3. **Bộ đếm thông minh chống đếm trùng (`ClassroomPeopleCounter`):**
   - **Khử trùng lặp Bounding Box (IoU Deduplication):** Ngăn chặn tình trạng 1 người bị gắn 2 khung nhận diện chồng lên nhau.
   - **Phân vùng Ân hạn (Zone-Aware Grace Period):** Giữ trạng thái có mặt (`3.5 giây`) khi học sinh ngồi trong lớp cúi xuống hoặc bị che khuất tạm thời để không báo `Ra/Vào` ảo; đồng thời ghi nhận nhanh (`0.8 giây`) khi người bước ra ở vùng rìa cửa lớp.
   - **Khớp nối ID theo vị trí & thời gian (Spatial-Temporal ID Stitching):** Tự động khôi phục lại `ID` gốc nếu một người bị mất dấu rồi xuất hiện lại ở cùng vị trí với `Track ID` mới, triệt tiêu hoàn toàn lỗi đếm nhân đôi.

---

## **2. Các điểm nâng cấp mới nhất (Changelog & Improvements)**

- **Loại bỏ mô hình LSTM dư thừa:** Gỡ bỏ phụ thuộc nặng nề vào `TensorFlow/Keras`, giúp hệ thống khởi động nhanh gấp 5 lần, nhẹ RAM/CPU và tập trung 100% tài nguyên cho **YOLOv8 + DeepSORT**.
- **Đọc Camera đa luồng không trễ (`ThreadedCamera`):** Tách riêng luồng đọc khung hình RTSP/Webcam (`utils/camera_stream.py`) với `CAP_PROP_BUFFERSIZE = 1` và `CAP_DSHOW` trên Windows, hỗ trợ tự động kết nối lại khi mạng chập chờn và cho phép nhiều tab trình duyệt cùng xem stream mượt ở 25–30 FPS.
- **Hiển thị Bounding Box & HUD chuyên nghiệp (`inference/visualization.py`):**
  - Vẽ khung Bounding Box có 4 góc nhấn đậm, chấm tâm và nhãn `ID:<số> Person <độ_tự_tin>%`.
  - Phân biệt màu sắc trực quan: **Xanh lá** cho người đã xác nhận ID và **Vàng cam** cho người vừa mới bước vào khung hình ngay từ frame đầu tiên.
- **Web Dashboard Thời Gian Thực (`templates/index.html`):**
  - Giao diện Dark Mode hiện đại, điều khiển Camera bằng AJAX (`fetch`) không cần tải lại trang.
  - 4 thẻ thống kê KPI cập nhật mỗi 1.5 giây: **Số người hiện tại**, **Tổng lượt Vào**, **Tổng lượt Ra**, **Tốc độ xử lý AI (FPS)**.
  - Biểu đồ đường (`Chart.js`) theo dõi biến động sĩ số trực tuyến và bảng **Nhật ký Vào/Ra gần nhất**.
  - Hỗ trợ phát thông báo giọng nói tiếng Việt trực tiếp trên trình duyệt (Web Speech API) song song với loa máy chủ (`pyttsx3`).
- **Sửa lỗi nhân bản dữ liệu MySQL & Hỗ trợ chế độ Fallback CSV:** Tự động làm rỗng bộ đệm nhật ký sau mỗi chu kỳ lưu, dùng `executemany()` và vẫn hoạt động độc lập với file CSV ngay cả khi chưa bật MySQL Server.

---

## **3. Công nghệ & Yêu cầu hệ thống**

- **Ngôn ngữ & Mô hình:** Python 3.8+, **YOLOv8 (`ultralytics`)**, **DeepSORT (`deep_sort_realtime`)**, **OpenCV (`opencv-python`)**.
- **Backend & Giao diện:** **Flask**, **Bootstrap 5**, **Chart.js**, **pyttsx3** (Text-to-Speech).
- **Cơ sở dữ liệu:** **MySQL Server** (tùy chọn) & **CSV**.
- **Phần cứng hỗ trợ:** Camera IP RTSP (Imou Ranger 2...) hoặc Webcam Laptop (`0`).

---

## **4. Cấu trúc thư mục dự án**

```text
He-thong-dem-so-luong-nguoi-trong-lop-hoc-bang-camera/
├── app.py                      # Flask Web Server & Luồng xử lý AI chính
├── main.py                     # Chế độ chạy giám sát kết hợp Desktop / Web
├── requirements.txt            # Danh sách thư viện Python cần thiết
├── .env.example                # Mẫu cấu hình biến môi trường bảo mật
├── config/
│   └── config.yaml             # Cấu hình Camera, YOLOv8, DeepSORT, MySQL
├── utils/
│   ├── camera_stream.py        # Luồng đọc Camera RTSP/Webcam độc lập (ThreadedCamera)
│   ├── people_counter.py       # Thuật toán đếm người chống trùng lặp (ClassroomPeopleCounter)
│   ├── video_processing.py     # Đếm người và xuất video kết quả từ file .mp4
│   ├── auto_labeling.py        # Tự động gán nhãn ảnh chuẩn định dạng YOLOv8
│   ├── data_preprocessing.py   # Chia tập dữ liệu ảnh/nhãn train & valid
│   └── database.py             # Kết nối và khởi tạo bảng MySQL
├── inference/
│   ├── visualization.py        # Vẽ Bounding Box, góc nhấn, ID và bảng HUD
│   └── predict.py              # Kiểm thử nhận diện & đếm người trên tập ảnh tĩnh
├── models/
│   ├── train_yolo.py           # Script Fine-tune mô hình YOLOv8 trên ảnh lớp học
│   ├── yolo_model.py           # Khởi tạo nhanh mô hình YOLOv8
│   └── deepsort_tracker.py     # Khởi tạo nhanh bộ theo dõi DeepSORT
├── templates/
│   └── index.html              # Giao diện Web Dashboard giám sát thời gian thực
└── test/
    ├── test_cam.py             # Mở cửa sổ Webcam/RTSP trực tiếp trên Desktop
    ├── test_people_counter.py  # Bộ kịch bản kiểm thử tự động chống đếm trùng người
    └── test_deepsort.py        # Kiểm tra hoạt động của thư viện DeepSORT
```

---

## **5. Hướng dẫn cài đặt & Sử dụng**

### **5.1. Cài đặt thư viện**
```bash
pip install -r requirements.txt
```

### **5.2. Cấu hình hệ thống (`config/config.yaml`)**
```yaml
camera:
  rtsp_url: "0"          # Điền "0" để dùng Webcam Laptop, hoặc đường dẫn "rtsp://..." cho Camera IP
  frame_rate: 20
  frame_skip: 2
  resolution: [640, 480]

yolo:
  model_path: "yolov8l.pt"
  confidence_threshold: 0.25  # Bắt nhạy cả người ở xa hoặc bị che khuất sau bàn ghế
  iou_threshold: 0.65         # Không xóa nhầm người đứng/ngồi sát cạnh nhau
  imgsz: 640
  min_box_area: 50
  max_box_area_ratio: 0.80
  min_aspect_ratio: 0.15
  max_aspect_ratio: 2.5

tracker:
  type: "DeepSORT"
  max_age: 30
  min_hits: 3
  max_cosine_distance: 0.3
  smoothing_window: 5
```

### **5.3. Cấu hình MySQL (Tùy chọn)**
Nếu sử dụng MySQL, tạo cơ sở dữ liệu `people_control` và chạy lệnh khởi tạo bảng:
```sql
CREATE DATABASE people_control;
```
```bash
python utils/database.py
```
*(Nếu không bật MySQL, hệ thống sẽ tự động chuyển sang chế độ lưu dữ liệu vào `data/current_people_count.csv` và `data/entry_exit_log.csv`).*

---

## **6. Các chế độ chạy hệ thống**

### **Chế độ 1: Chạy Giao diện Web Dashboard (Khuyên dùng)**
```bash
python app.py
```
Truy cập **`http://localhost:5000`** trên trình duyệt. Bạn có thể nhập RTSP URL hoặc nhấn nút **"💻 Dùng Webcam (0)"** rồi bấm **"▶ Bắt đầu Camera"**.

### **Chế độ 2: Mở trực tiếp cửa sổ Camera trên Desktop**
```bash
python test/test_cam.py
```
*(Nhấn phím **`q`** trên cửa sổ video để thoát).*

### **Chế độ 3: Đếm người trên file Video (`.mp4`)**
```bash
python utils/video_processing.py đường_dẫn_video.mp4
```
Video kết quả đã vẽ Bounding Box sẽ được lưu tại `data/counted_video.mp4` kèm các ảnh chụp mốc thời gian trong `data/video_snapshots/`.

### **Chế độ 4: Kiểm thử nhận diện trên 10 hình ảnh & Kiểm thử thuật toán đếm**
```bash
# Kiểm thử nhận diện và vẽ Bounding Box trên 10 ảnh trong thư mục images2/
python inference/predict.py

# Chạy 5 kịch bản Unit Test kiểm tra chống đếm trùng người
python test/test_people_counter.py
```

---

## **7. Các API Endpoint của Web Server (`app.py`)**

| Endpoint | Phương thức | Mô tả |
| :--- | :---: | :--- |
| `/` | `GET` | Trả về giao diện Web Dashboard chính (`templates/index.html`). |
| `/video_feed` | `GET` | Cung cấp luồng video MJPEG thời gian thực đã vẽ Bounding Box, ID và HUD. |
| `/status` | `GET` | Trả về JSON trạng thái thời gian thực: `camera_active`, `current_people_count`, `total_entries`, `total_exits`, `fps`, `recent_events`. |
| `/start` | `POST` | Kết nối và khởi động luồng Camera RTSP hoặc Webcam (`rtsp_url`). |
| `/stop` | `POST` | Dừng kết nối Camera và lưu toàn bộ nhật ký xuống CSV / MySQL. |
| `/announce` | `POST` | Kích hoạt thông báo giọng nói tiếng Việt công bố số người hiện tại trong lớp. |

---

## **8. Poster Dự Án**

![Poster_Nhom9_CNTT_1601 -_1_](https://github.com/user-attachments/assets/41c750e6-52ae-4d4c-b525-262805ef826d)
