<h1 align="center">HỆ THỐNG ĐẾM SỐ LƯỢNG NGƯỜI TRONG LỚP HỌC BẰNG CAMERA </h1>

<div align="center">

<img src="https://github.com/user-attachments/assets/a7ad8c17-5216-4f9c-9bc5-715bfdc7283a" width="200px">
<img src="https://github.com/user-attachments/assets/d057713b-9362-4a1e-9423-89691f3ab44d" width="200px">

[![Made by AIoTLab](https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Fit DNU](https://img.shields.io/badge/Fit%20DNU-green?style=for-the-badge)](https://fitdnu.net/)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge)](https://dainam.edu.vn)

</div>
## **Giới thiệu**  
**Hệ thống Đếm Số Lượng Người bằng Camera sử dụng trí tuệ nhân tạo (AI)** để phát hiện và đếm số lượng người ra vào thông qua luồng video trực tiếp. Hệ thống phù hợp cho các ứng dụng giám sát như:

- Theo dõi số người trong lớp học, văn phòng, trung tâm thương mại.

- Kiểm soát lưu lượng người ra vào tại các sự kiện.

- Hỗ trợ thống kê và quản lý không gian thông minh.

**Hệ thống hoạt động theo cơ chế:**

- Nhận diện người trong video bằng mô hình YOLOv8.

- Theo dõi di chuyển của từng người bằng DeepSORT.

- Đếm số lượng người đi vào và đi ra khỏi khu vực giám sát.
---

**Công nghệ sử dụng:**
- **YOLOv8**: Nhận diện người từ camera. 
- **OpenCV**: Xử lý hình ảnh và video. 
- **DeepSORT**: Theo dõi ID của từng người.
- **LSTM (Long Short-Term Memory)**: train dữ liệu 
- **Flask**: API backend  
- **MySQL**, **CSV**: Lưu trữ dữ liệu đếm số người. 
---

## **Thiết bị sử dụng trong bài**
**Phần cứng**
- Camera Imou Ranger 2. 
- Laptop

**Phần mềm**
- Hệ điều hành, môi trường Python.
- Các thư viện xử lý ảnh ( OpenCV, YOLOv8,...)

---
## **Yêu cầu hệ thống**  
- **Python** 3.7 trở lên  
- **MySQL Server**  
- **OpenCV, TensorFlow**  
- Các thư viện Python cần thiết (**liệt kê trong `requirements.txt`**)  

---

## **Hướng dẫn cài đặt**  

### **1. Cài đặt các thư viện cần thiết**  
Chạy lệnh sau để cài đặt các thư viện Python yêu cầu:  
```
pip install -r requirements.txt
```

### **2. Hướng dẫn thực hiện**  
Sơ đồ cấu trúc:
![image](https://github.com/user-attachments/assets/e37715da-e690-4a1f-960b-ea656e188acf)
#### **2.1. Kiểm tra nguồn RTSP của camera (test/test_cam.py)**  
Định dạng nguồn: 
```
rtsp://[username]:[password]@[Địa-chỉ-IP]:554/cam/realmonitor?channel=1&subtype=0
```
Ví dụ:
```
rtsp://admin:L2688A29@192.168.199.44:554/cam/realmonitor?channel=1&subtype=0
```
*Lưu ý:* Mặc định username của hầu hết camera là admin 

#### **2.2. Tạo file config.yaml**
1. Điền nguồn RTSP của camera sau khi test thành công
```
camera:Cấu hình cho mô hình YOLO dùng để phát hiện đối tượng
  rtsp_url: "rtsp://admin:L2688A29@192.168.199.44:554/cam/realmonitor?channel=1&subtype=1"  # Updated camera source
  frame_rate: 20   # Số khung hình xử lý mỗi giây
  resolution: [480, 360]  # Increased resolution to improve frame size
```
2. Cấu hình cho mô hình YOLO dùng để phát hiện đối tượng
```
yolo:
  model_path: "yolov8l.pt"         # Đường dẫn đến file mô hình YOLO (phiên bản yolov8l)
  confidence_threshold: 0.5         # Ngưỡng độ tin cậy; chỉ các dự đoán có confidence >= 0.5 mới được chấp nhận
  iou_threshold: 0.4                # Ngưỡng IoU (Intersection over Union) dùng trong non-max suppression để loại bỏ các dự đoán trùng lặp
```
3. Cấu hình cho bộ theo dõi (tracker) sử dụng thuật toán DeepSORT
```
tracker:
  type: "DeepSORT"                # Loại bộ theo dõi được sử dụng (ở đây dùng DeepSORT)
  max_age: 30                     # Số khung hình tối đa theo dõi trước khi cho rằng mục tiêu đã mất (số khung hình trống)
  min_hits: 3                     # Số lần xác nhận liên tiếp cần có để xác nhận sự tồn tại của mục tiêu
```
4. Cấu hình cho mô hình LSTM dùng trong các tác vụ dự đoán chuỗi (sequence prediction)
```
lstm:
  sequence_length: 10             # Độ dài chuỗi dữ liệu đầu vào cho LSTM (số bước thời gian)
  epochs: 50                      # Số epoch huấn luyện cho mô hình LSTM
  batch_size: 16                  # Số mẫu dữ liệu được xử lý cùng lúc trong một batch
  learning_rate: 0.001            # Tốc độ học (learning rate) cho quá trình huấn luyện LSTM
```
5. Cấu hình đường dẫn tới dữ liệu và mô hình đã được huấn luyện
```
dataset:
  image_folder: "data/images/"           # Thư mục chứa các ảnh đầu vào
  label_folder: "data/labels/"             # Thư mục chứa các nhãn tương ứng của ảnh
  trained_model_path: "data/trained_model.h5"  # Đường dẫn lưu file mô hình đã được huấn luyện (định dạng .h5)
```
6. Cấu hình kết nối đến cơ sở dữ liệu MySQL
```
  host: "localhost"               # Địa chỉ host của MySQL server (ở đây là localhost)
  user: "root"                    # Tên người dùng để kết nối (user mặc định là root)
  password: "your_password"      # Mật khẩu tương ứng cho user
  database: "people_control"      # Tên cơ sở dữ liệu sẽ sử dụng
```
#### **2.3. Tạo cơ sở dữ liệu**  
Mở MySQL và chạy lệnh sau để tạo cơ sở dữ liệu **`people_control`**
```
CREATE DATABASE people_control;
USE people_control;
```

#### **2.4. Tạo các bảng cần thiết và kết nối với database (database.py)**  
```
def create_table_if_not_exists(cursor):
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS entry_exit_log (
        id INT AUTO_INCREMENT PRIMARY KEY,
        timestamp DATETIME,
        event VARCHAR(10),
        count_change INT
    )
    """)
```
Kết nối database bằng cú pháp:
```
python database.py
```

#### **2.5. Tự động lưu file CSV và MySQL (app.py)**  
Tự động lưu và CSV:
```
 try:
                with open('data/current_people_count.csv', 'a') as f:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{current_people_count}\n")
            except PermissionError as e:
                print(f"❌ Lỗi khi lưu dữ liệu vào CSV: {e}")
```

```
try:
                df_log = pd.DataFrame(entry_exit_log, columns=["timestamp", "event", "count_change"])
                df_log.to_csv('C:/Project_Structure/data/entry_exit_log.csv', index=False)
                print("📊 Dữ liệu đã được lưu vào 'C:/Project_Structure/data/entry_exit_log.csv'")
            except PermissionError as e:
                print(f"❌ Lỗi khi lưu dữ liệu vào CSV: {e}")
```
Tự động lưu vào MySQL:
```
for log in entry_exit_log:
                cursor.execute("INSERT INTO entry_exit_log (timestamp, event, count_change) VALUES (%s, %s, %s)", log)
            db.commit()
            print("📊 Dữ liệu đã được lưu vào MySQL")

```
#### **2.6. Tiến hành train mô hình** 
**Bước 1: Chuyền dataset đã thu thập được vào Folder (images)**

![image](https://github.com/user-attachments/assets/3b2c0942-6012-479d-be58-23cf56b69820)

Các link dataset:
- [CrowdHuman Crowd Detection CSV Labels](https://www.kaggle.com/datasets/permanalwep/crowd-human-csv-labels)
- [CrowdHuman Crowd Detection](https://www.kaggle.com/datasets/permanalwep/crowdhuman-crowd-detection)
- [CrowdHuman Face](https://www.kaggle.com/datasets/permanalwep/crowdhuman-face)

**Bước 2: Tiến hành tự động gán nhãn bằng mô hình yolov8 (auto_labeling.py)**

![image](https://github.com/user-attachments/assets/1f31a6d2-e086-43c7-aec7-f424de74a345)

```
python auto_labeling.py
```
**Bước 3: Kiểm tra sau khi tự động gán nhãn (labels/)**

![image](https://github.com/user-attachments/assets/22653595-baa2-4564-98da-f4c9cefdcd7b)

Có file .txt là đúng

**Bước 4: Bắt đầu train (models/train_lstm.py)**
```
python train_lstm.py
```
Kết quả sau khi train tốt nhất sẽ được lưu vào file **`trained_model.h5`**
![image](https://github.com/user-attachments/assets/c604b999-7b20-478b-907c-a9f710aa6b08)

#### **2.7. Chạy toàn bộ mô hình sau khi train xong (app.py)** 
```
python app.py
```
## **Các API Endpoint**  
| Endpoint                 | Phương thức | Mô tả |
|--------------------------|------------|-------|
| `/`                      | GET        | Trả về trang giao diện chính của ứng dụng (index.html), hiển thị giao diện web cho người dùng. |
| `/video_feed`            | GET        | Cung cấp stream video real-time dưới dạng JPEG từ hàm **`generate_frames()`**, cho phép hiển thị video trực tuyến trong trình duyệt. |
| `/start`               | POST       | Khởi động kết nối camera với RTSP URL được truyền qua request (tham số **`rtsp_url`**).|
| `/stop`        | POST       | Dừng kết nối camera và giải phóng các tài nguyên liên quan. |
| `/announce`          | POST       | Kích hoạt thông báo giọng nói (voice notification) để công bố số lượng người hiện tại. |
---

## **Ghi chú:**  
✅ Hãy kết hợp cả CPU+GPU để camera hoạt động tốt hơn. 

---
**Poster**

![Poster_Nhom9_CNTT_1601 -_1_](https://github.com/user-attachments/assets/41c750e6-52ae-4d4c-b525-262805ef826d)


