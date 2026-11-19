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

import cv2
import numpy as np
import yaml
import time
import threading

from ultralytics import YOLO, settings
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.camera_stream import ThreadedCamera
from utils.people_counter import ClassroomPeopleCounter
from inference.visualization import draw_tracks_and_overlay

settings.update({'runs_dir': os.path.join(BASE_DIR, 'data', 'runs')})

CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

print("🔄 Đang tải các mô hình (YOLOv8 + DeepSORT)...")

yolo_model = None
try:
    yolo_model = YOLO(os.path.join(BASE_DIR, config['yolo']['model_path']))
    print("✅ YOLO model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")

tracker = None
try:
    tracker = DeepSort(
        max_age=config['tracker']['max_age'],
        n_init=config['tracker'].get('min_hits', 3),
        max_cosine_distance=config['tracker'].get('max_cosine_distance', 0.3)
    )
    print("✅ DeepSort tracker loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load DeepSort tracker: {e}")


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


# Mặc định dùng camera trong config (đã đặt là "0" cho Webcam Laptop) hoặc truyền qua tham số dòng lệnh
cam_source = sys.argv[1] if len(sys.argv) > 1 else config['camera'].get('rtsp_url', 0)
print(f"📷 Đang mở nguồn camera: {cam_source} ...")

cam = ThreadedCamera(
    cam_source,
    resolution=tuple(config['camera']['resolution']),
    fps=config['camera'].get('frame_rate', 20)
)

if not cam.start(retries=5):
    print("❌ Không thể kết nối camera!")
    sys.exit(1)

people_counter = ClassroomPeopleCounter()
state_lock = threading.Lock()
shared_state = {
    "current_count": 0,
    "total_entries": 0,
    "total_exits": 0,
    "display_objects": [],
    "ai_fps": 0.0,
    "running": True
}


def ai_worker():
    """Luồng chạy YOLOv8 + DeepSORT độc lập để cửa sổ hiển thị Webcam luôn mượt 25-30 FPS."""
    last_id = -1
    t_prev = time.time()
    while shared_state["running"] and cam.is_running:
        ret, frame, fid = cam.read()
        if not ret or frame is None or fid == last_id:
            time.sleep(0.01)
            continue

        last_id = fid
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_w * frame_h
        detections = []

        if yolo_model is not None:
            results = yolo_model(
                frame,
                conf=config['yolo'].get('confidence_threshold', 0.25),
                iou=config['yolo'].get('iou_threshold', 0.65),
                imgsz=config['yolo'].get('imgsz', 640),
                classes=[0],
                project=os.path.join(BASE_DIR, 'data', 'runs'),
                verbose=False
            )
            for result in results[0].boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = result
                w, h = x2 - x1, y2 - y1
                if is_valid_person_box(w, h, frame_area):
                    detections.append(([x1, y1, w, h], conf, "person"))

        tracks = tracker.update_tracks(detections, frame=frame) if tracker is not None else []
        c_count, _, disp_objs = people_counter.update(tracks, detections, frame.shape)

        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now
        cur_fps = (1.0 / dt) if dt > 0 else 0.0

        with state_lock:
            shared_state["current_count"] = c_count
            shared_state["total_entries"] = people_counter.total_entries
            shared_state["total_exits"] = people_counter.total_exits
            shared_state["display_objects"] = disp_objs
            shared_state["ai_fps"] = (
                0.8 * shared_state["ai_fps"] + 0.2 * cur_fps
                if shared_state["ai_fps"] > 0 else cur_fps
            )


worker = threading.Thread(target=ai_worker, daemon=True)
worker.start()

print("✅ Cửa sổ Webcam đã sẵn sàng! Nhấn phím 'q' trên cửa sổ video để thoát.")

while cam.is_running:
    ret, live_frame, _ = cam.read()
    if not ret or live_frame is None:
        time.sleep(0.01)
        continue

    with state_lock:
        c_count = shared_state["current_count"]
        t_in = shared_state["total_entries"]
        t_out = shared_state["total_exits"]
        objs = list(shared_state["display_objects"])
        ai_fps = shared_state["ai_fps"]

    annotated = draw_tracks_and_overlay(
        live_frame,
        [],
        c_count,
        display_objects=objs,
        total_entries=t_in,
        total_exits=t_out,
        fps=ai_fps
    )

    cv2.imshow("Real-Time Monitoring (Laptop Camera - Press 'q' to quit)", annotated)

    if cv2.waitKey(15) & 0xFF == ord('q'):
        print("🛑 Chương trình đã dừng lại.")
        break

shared_state["running"] = False
cam.stop()
cv2.destroyAllWindows()
