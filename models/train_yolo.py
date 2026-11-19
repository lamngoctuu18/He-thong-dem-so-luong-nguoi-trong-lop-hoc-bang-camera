import os
import sys
import yaml

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, "data", ".mplconfig"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE_DIR, "data", ".ultralytics"))

from ultralytics import YOLO
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.data_preprocessing import split_data

CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


def train_yolo_person_detector(epochs=30, imgsz=640, batch=8):
    """
    Fine-tune mô hình YOLOv8 trên tập ảnh lớp học thực tế để nhận diện học sinh chính xác hơn
    và không nhận nhầm đồ vật xung quanh (balo, ghế, quạt, áo khoác...).
    """
    image_folder = os.path.join(BASE_DIR, config['dataset'].get('image_folder', 'data/images/'))
    label_folder = os.path.join(BASE_DIR, config['dataset'].get('label_folder', 'data/labels/'))

    split_data(image_folder, label_folder)

    train_img_dir = os.path.join(image_folder, 'train')
    val_img_dir = os.path.join(image_folder, 'valid')

    if not os.path.exists(train_img_dir) or not os.listdir(train_img_dir):
        print(f"❌ Không tìm thấy ảnh huấn luyện tại '{train_img_dir}'.")
        print("💡 Hãy đặt ảnh lớp học vào 'data/images/', gán nhãn bằng LabelImg hoặc 'utils/auto_labeling.py' trước.")
        return

    dataset_yaml_path = os.path.join(BASE_DIR, 'data', 'classroom_person.yaml')
    dataset_config = {
        'path': os.path.join(BASE_DIR, 'data'),
        'train': 'images/train',
        'val': 'images/valid',
        'nc': 1,
        'names': ['person']
    }
    with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, allow_unicode=True)

    base_model_path = config['yolo'].get('model_path', 'yolov8l.pt')
    print(f"🚀 Bắt đầu Fine-tune mô hình YOLOv8 từ '{base_model_path}'...")
    model = YOLO(base_model_path)
    model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=os.path.join(BASE_DIR, 'data', 'yolo_runs'),
        name='classroom_person'
    )
    print("🎉 Huấn luyện YOLOv8 hoàn tất! Trọng số tốt nhất được lưu trong 'data/yolo_runs/classroom_person/weights/best.pt'.")


if __name__ == "__main__":
    train_yolo_person_detector()

