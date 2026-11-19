import os
import sys
import subprocess

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, "data", ".mplconfig"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE_DIR, "data", ".ultralytics"))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    import ultralytics  # noqa: F401
except ModuleNotFoundError:
    _py39 = r"C:\Users\84916\AppData\Local\Programs\Python\Python39\python.exe"
    if os.path.exists(_py39) and os.path.normcase(sys.executable) != os.path.normcase(_py39):
        sys.exit(subprocess.call([_py39, *sys.argv]))
    raise

import time
import zipfile
import cv2
import yaml

from ultralytics import YOLO, settings
from inference.visualization import _draw_person_box

settings.update({'runs_dir': os.path.join(BASE_DIR, 'data', 'runs')})

CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

yolo_model = YOLO(os.path.join(BASE_DIR, config['yolo'].get('model_path', 'yolov8l.pt')))


def is_valid_person_box(w, h, frame_area):
    if w <= 0 or h <= 0:
        return False
    yolo_cfg = config.get('yolo', {})
    min_area = yolo_cfg.get('min_box_area', 50)
    max_area_ratio = yolo_cfg.get('max_box_area_ratio', 0.80)
    min_ar = yolo_cfg.get('min_aspect_ratio', 0.15)
    max_ar = yolo_cfg.get('max_aspect_ratio', 2.5)

    box_area = w * h
    if box_area < min_area or box_area > frame_area * max_area_ratio:
        return False
    aspect_ratio = w / float(h)
    return min_ar <= aspect_ratio <= max_ar


def predict_people_in_image(image_path, output_dir=None):
    """Nhận diện, vẽ Bounding Box và đếm số lượng người trên một ảnh tĩnh bằng YOLOv8."""
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy file ảnh: {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không thể đọc file ảnh: {image_path}")
        return None

    h, w = img.shape[:2]
    frame_area = w * h
    t0 = time.time()
    results = yolo_model(
        img,
        conf=config['yolo'].get('confidence_threshold', 0.25),
        iou=config['yolo'].get('iou_threshold', 0.65),
        imgsz=config['yolo'].get('imgsz', 960),
        classes=[0],
        project=os.path.join(BASE_DIR, 'data', 'runs'),
        verbose=False
    )
    infer_ms = (time.time() - t0) * 1000.0

    raw_boxes = results[0].boxes.data.tolist()
    valid_boxes = []
    confs = []

    for idx, result in enumerate(raw_boxes, start=1):
        x1, y1, x2, y2, conf, cls = result
        bw, bh = x2 - x1, y2 - y1
        if is_valid_person_box(bw, bh, frame_area):
            valid_boxes.append((x1, y1, x2, y2, conf))
            confs.append(conf)
            label = f"ID:{len(valid_boxes)} Person {int(round(conf * 100))}%"
            _draw_person_box(img, x1, y1, x2, y2, label, color=(0, 220, 60))

    # Vẽ HUD thông tin góc trái trên
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (240, 52), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.putText(
        img,
        f"Count: {len(valid_boxes)} people",
        (16, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 230, 255),
        2,
        cv2.LINE_AA
    )

    saved_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        saved_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
        cv2.imwrite(saved_path, img)

    avg_conf = (sum(confs) / len(confs) * 100.0) if confs else 0.0
    min_conf = (min(confs) * 100.0) if confs else 0.0
    max_conf = (max(confs) * 100.0) if confs else 0.0

    return {
        "image": os.path.basename(image_path),
        "resolution": f"{w}x{h}",
        "raw_count": len(raw_boxes),
        "filtered_out": len(raw_boxes) - len(valid_boxes),
        "valid_count": len(valid_boxes),
        "avg_conf": avg_conf,
        "min_conf": min_conf,
        "max_conf": max_conf,
        "infer_ms": infer_ms,
        "saved_path": saved_path
    }


def test_10_images():
    sample_dir = os.path.join(BASE_DIR, 'images2')
    output_dir = os.path.join(BASE_DIR, 'data', 'test_10_results')
    if not os.path.exists(sample_dir):
        print(f"❌ Không tìm thấy thư mục '{sample_dir}'")
        return

    images = sorted([
        f for f in os.listdir(sample_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])[:10]

    print(f"🚀 Đang chạy kiểm thử nhận diện người trên {len(images)} ảnh...\n")
    results = []
    for idx, img_name in enumerate(images, start=1):
        img_path = os.path.join(sample_dir, img_name)
        info = predict_people_in_image(img_path, output_dir=output_dir)
        if info:
            results.append(info)
            print(
                f"[{idx:02d}/10] {info['image']} ({info['resolution']}) -> "
                f"Phát hiện: {info['valid_count']} người "
                f"(Đã lọc nhiễu: {info['filtered_out']} box) | "
                f"Độ tự tin TB: {info['avg_conf']:.1f}% [{info['min_conf']:.1f}% - {info['max_conf']:.1f}%] | "
                f"Thời gian: {info['infer_ms']:.0f} ms"
            )

    if results:
        total_people = sum(r['valid_count'] for r in results)
        total_filtered = sum(r['filtered_out'] for r in results)
        mean_conf = sum(r['avg_conf'] for r in results if r['valid_count'] > 0) / max(1, sum(1 for r in results if r['valid_count'] > 0))
        mean_time = sum(r['infer_ms'] for r in results) / len(results)
        print("\n" + "=" * 75)
        print(f"📊 TỔNG KẾT KIỂM THỬ 10 ẢNH:")
        print(f"   • Tổng số người nhận diện hợp lệ : {total_people} người (Trung bình: {total_people / len(results):.1f} người/ảnh)")
        print(f"   • Tổng số box nhiễu đã lọc bỏ    : {total_filtered} box")
        print(f"   • Độ tự tin trung bình           : {mean_conf:.2f}%")
        print(f"   • Thời gian xử lý trung bình/ảnh : {mean_time:.1f} ms")
        print(f"   • Ảnh kết quả đã vẽ Bounding Box : {output_dir}")
        print("=" * 75)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_people_in_image(sys.argv[1], output_dir=os.path.join(BASE_DIR, 'data', 'test_10_results'))
    else:
        test_10_images()
