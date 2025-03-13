import os
import cv2
from ultralytics import YOLO

# Load mô hình YOLOv8
yolo_model = YOLO('yolov8l.pt')

image_folder = '../data/images/'  # Đường dẫn ảnh
label_folder = '../data/labels/'  # Đường dẫn nhãn
os.makedirs(label_folder, exist_ok=True)

def auto_label_images():
    if not os.path.exists(image_folder):
        print(f"❌ Thư mục {image_folder} không tồn tại!")
        return

    if not os.listdir(image_folder):
        print(f"⚠️ Thư mục {image_folder} trống. Hãy thêm ảnh trước!")
        return

    print(f"✅ Đang gán nhãn cho {len(os.listdir(image_folder))} ảnh...")

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        label_path = os.path.join(label_folder, img_name.replace('.jpg', '.txt'))

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Không thể đọc ảnh {img_name}, bỏ qua...")
            continue

        results = yolo_model(img, conf=0.5)  # Nhận diện ảnh bằng YOLOv8 với ngưỡng tự tin cao hơn

        with open(label_path, 'w') as f:
            for result in results[0].boxes.data.tolist():  # Lấy danh sách bounding box
                x1, y1, x2, y2, conf, cls = result
                if int(cls) == 0 and conf > 0.5:  # Chỉ nhận diện người (class = 0) với độ tin cậy cao
                    f.write(f"{int(cls)} {x1} {y1} {x2} {y2} {conf}\n")

        print(f"✅ Gán nhãn: {label_path}")

if __name__ == "__main__":
    auto_label_images()
