import os
import cv2
import yaml

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, "data", ".mplconfig"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE_DIR, "data", ".ultralytics"))

from ultralytics import YOLO
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

yolo_model = YOLO(config['yolo'].get('model_path', 'yolov8l.pt'))

image_folder = os.path.join(BASE_DIR, config['dataset'].get('image_folder', 'data/images/'))
label_folder = os.path.join(BASE_DIR, config['dataset'].get('label_folder', 'data/labels/'))
os.makedirs(label_folder, exist_ok=True)


def auto_label_frame(frame, conf_threshold=0.55):
    """Nhận diện người trên một khung hình (frame) và vẽ bounding box."""
    if frame is None:
        return frame, []

    results = yolo_model(frame, conf=conf_threshold, classes=[0], verbose=False)
    boxes = []
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        if int(cls) == 0 and conf >= conf_threshold:
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return frame, boxes


def auto_label_images(src_image_folder=image_folder, dst_label_folder=label_folder):
    """
    Tự động gán nhãn ảnh theo đúng định dạng chuẩn YOLOv8:
    <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm> (giá trị trong khoảng [0, 1]).
    """
    if not os.path.exists(src_image_folder):
        print(f"❌ Thư mục {src_image_folder} không tồn tại!")
        return

    image_files = [
        f for f in os.listdir(src_image_folder)
        if os.path.isfile(os.path.join(src_image_folder, f))
        and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not image_files:
        print(f"⚠️ Thư mục {src_image_folder} trống. Hãy thêm ảnh trước!")
        return

    os.makedirs(dst_label_folder, exist_ok=True)
    conf_thres = config['yolo'].get('confidence_threshold', 0.55)
    print(f"✅ Đang gán nhãn chuẩn YOLOv8 cho {len(image_files)} ảnh...")

    for img_name in image_files:
        img_path = os.path.join(src_image_folder, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(dst_label_folder, label_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Không thể đọc ảnh {img_name}, bỏ qua...")
            continue

        img_h, img_w = img.shape[:2]
        results = yolo_model(img, conf=conf_thres, classes=[0], verbose=False)

        with open(label_path, 'w', encoding='utf-8') as f:
            for result in results[0].boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = result
                if int(cls) == 0 and conf >= conf_thres:
                    # Chuyển đổi sang tọa độ chuẩn hóa [0, 1] của YOLOv8
                    x_center = ((x1 + x2) / 2.0) / img_w
                    y_center = ((y1 + y2) / 2.0) / img_h
                    w_norm = (x2 - x1) / img_w
                    h_norm = (y2 - y1) / img_h
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        print(f"✅ Gán nhãn: {label_path}")


if __name__ == "__main__":
    auto_label_images()
