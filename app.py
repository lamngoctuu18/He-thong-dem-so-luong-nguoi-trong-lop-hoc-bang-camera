import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, "data", ".mplconfig"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE_DIR, "data", ".ultralytics"))

# Tự động chuyển sang môi trường Python 3.9 (nơi đã cài đặt đầy đủ ultralytics, deep_sort_realtime, flask...)
# nếu người dùng gõ 'python app.py' đang trỏ mặc định vào Python 3.12 chưa cài thư viện.
try:
    import ultralytics  # noqa: F401
except ModuleNotFoundError:
    _py39 = r"C:\Users\84916\AppData\Local\Programs\Python\Python39\python.exe"
    if os.path.exists(_py39) and os.path.normcase(sys.executable) != os.path.normcase(_py39):
        sys.exit(subprocess.call([_py39, *sys.argv]))
    raise

import cv2
import numpy as np
from collections import deque

from ultralytics import YOLO, settings
from deep_sort_realtime.deepsort_tracker import DeepSort

settings.update({'runs_dir': os.path.join(BASE_DIR, 'data', 'runs')})
import pandas as pd
from datetime import datetime, timedelta
import yaml
import winsound
import time
from flask import Flask, render_template, Response, request, jsonify
from utils.database import get_db_connection, create_table_if_not_exists
from utils.camera_stream import ThreadedCamera
from utils.people_counter import ClassroomPeopleCounter
from inference.visualization import draw_tracks_and_overlay, create_placeholder_frame
import pyttsx3
import threading

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')

# Load configuration
with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Ensure data directory exists
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

CURRENT_COUNT_CSV = os.path.join(
    BASE_DIR,
    config.get('dataset', {}).get('current_count_csv', 'data/current_people_count.csv')
)
ENTRY_EXIT_CSV = os.path.join(
    BASE_DIR,
    config.get('dataset', {}).get('entry_exit_csv', 'data/entry_exit_log.csv')
)

print("🔄 Đang tải các mô hình (YOLOv8 + DeepSORT)...")

yolo_model = None
try:
    yolo_model = YOLO(config['yolo']['model_path'])
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

# Connect to MySQL (fallback to CSV if MySQL is unavailable)
db = get_db_connection()
cursor = None
if db is not None:
    cursor = db.cursor()
    create_table_if_not_exists(cursor)
else:
    print("⚠️ Không thể kết nối MySQL. Hệ thống sẽ hoạt động ở chế độ lưu file CSV.")

# Initialize state variables
cap = None  # Tham chiếu tới ThreadedCamera hiện tại
current_people_count = 0
total_entries = 0
total_exits = 0
current_fps = 0.0

smoothing_window = max(1, int(config.get('tracker', {}).get('smoothing_window', 5)))
count_window = deque(maxlen=smoothing_window)
people_counter = ClassroomPeopleCounter()
entry_exit_log = []
recent_events = deque(maxlen=15)  # Lưu 15 sự kiện Vào/Ra gần nhất để hiển thị trên Dashboard

state_lock = threading.Lock()
latest_jpeg_bytes = None
latest_display_objects = []
worker_running = False
worker_thread = None

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine_lock = threading.Lock()


def _encode_jpeg(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes() if ret else None


# Khởi tạo sẵn frame chờ khi chưa bật camera
latest_jpeg_bytes = _encode_jpeg(
    create_placeholder_frame(tuple(config['camera']['resolution']))
)


def voice_notification(message):
    with engine_lock:
        try:
            engine.say(message)
            engine.runAndWait()
        except Exception as e:
            print(f"❌ Lỗi thông báo giọng nói: {e}")


def save_periodic_logs():
    """Lưu số lượng người hiện tại và nhật ký ra/vào xuống CSV và MySQL mà không bị trùng lặp."""
    global entry_exit_log, current_people_count, db, cursor

    with state_lock:
        count_snapshot = current_people_count
        logs_to_save = list(entry_exit_log)
        entry_exit_log.clear()

    # 1. Lưu current_people_count vào CSV
    try:
        os.makedirs(os.path.dirname(CURRENT_COUNT_CSV), exist_ok=True)
        with open(CURRENT_COUNT_CSV, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{count_snapshot}\n")
    except Exception as e:
        print(f"❌ Lỗi khi lưu current_people_count vào CSV: {e}")

    if not logs_to_save:
        return

    # 2. Lưu các bản ghi mới vào MySQL
    if db is not None and cursor is not None:
        try:
            db.ping(reconnect=True, attempts=3, delay=1)
            cursor.executemany(
                "INSERT INTO entry_exit_log (timestamp, event, count_change) VALUES (%s, %s, %s)",
                logs_to_save
            )
            db.commit()
            print(f"📊 Đã lưu {len(logs_to_save)} bản ghi mới vào MySQL.")
        except Exception as e:
            print(f"❌ Lỗi khi lưu dữ liệu vào MySQL: {e}")

    # 3. Lưu nối tiếp (append) các bản ghi mới vào CSV
    try:
        os.makedirs(os.path.dirname(ENTRY_EXIT_CSV), exist_ok=True)
        file_exists = os.path.exists(ENTRY_EXIT_CSV) and os.path.getsize(ENTRY_EXIT_CSV) > 0
        df_log = pd.DataFrame(logs_to_save, columns=["timestamp", "event", "count_change"])
        df_log.to_csv(ENTRY_EXIT_CSV, mode='a', header=not file_exists, index=False, encoding='utf-8')
        print(f"📊 Đã lưu {len(logs_to_save)} bản ghi mới vào '{ENTRY_EXIT_CSV}'.")
    except Exception as e:
        print(f"❌ Lỗi khi lưu entry_exit_log vào CSV: {e}")


def is_valid_person_box(w, h, frame_area):
    """Lọc bỏ các bounding box có kích thước hoặc tỉ lệ bất thường (đồ vật xung quanh, nhiễu)."""
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
    if aspect_ratio < min_ar or aspect_ratio > max_ar:
        return False

    return True


def process_single_frame(frame):
    """
    Xử lý AI trên 1 khung hình:
    1. Nhận diện lớp 'person' (class=0) bằng YOLOv8 + Lọc hình học chống nhận nhầm đồ vật.
    2. Cập nhật DeepSORT tracker với tọa độ chuẩn [left, top, width, height].
    3. Lọc nhiễu số người bằng cửa sổ trung vị (Median Filter) trước khi ghi nhận Vào/Ra.
    4. Vẽ Bounding Box, ID, Count, In/Out và FPS lên frame.
    """
    global current_people_count, total_entries, total_exits

    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_w * frame_h
    detections = []

    if yolo_model is not None:
        results = yolo_model(
            frame,
            conf=config['yolo'].get('confidence_threshold', 0.25),
            iou=config['yolo'].get('iou_threshold', 0.65),
            imgsz=config['yolo'].get('imgsz', 640),
            classes=[0],  # Chỉ lọc class 0 (person) ngay trong NMS
            project=os.path.join(BASE_DIR, 'data', 'runs'),
            verbose=False
        )
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result
            w, h = x2 - x1, y2 - y1
            if is_valid_person_box(w, h, frame_area):
                detections.append(([x1, y1, w, h], conf, "person"))

    tracks = []
    if tracker is not None:
        tracks = tracker.update_tracks(detections, frame=frame)

    with state_lock:
        new_count, events, display_objects = people_counter.update(
            tracks,
            detections,
            frame.shape
        )
        current_people_count = new_count
        total_entries = people_counter.total_entries
        total_exits = people_counter.total_exits
        latest_display_objects[:] = display_objects

        if events:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for ev_type, change in events:
                entry_exit_log.append([timestamp, ev_type, change])
                recent_events.appendleft({
                    "timestamp": timestamp,
                    "event": ev_type,
                    "count_change": change,
                    "total_after": current_people_count
                })

    annotated = draw_tracks_and_overlay(
        frame.copy(),
        tracks,
        current_people_count,
        detections=detections,
        display_objects=display_objects,
        total_entries=total_entries,
        total_exits=total_exits,
        fps=current_fps
    )
    return annotated


def _inference_worker():
    """Luồng xử lý AI chạy ngầm, độc lập với số lượng tab trình duyệt đang mở."""
    global worker_running, latest_jpeg_bytes, current_fps

    frame_skip = max(1, int(config['camera'].get('frame_skip', 2)))
    alarm_interval = timedelta(minutes=1)
    next_alarm_time = datetime.now() + alarm_interval
    last_frame_id = -1
    t_prev = time.time()

    while worker_running and cap is not None:
        ret, frame, frame_id = cap.read()
        if not ret or frame is None:
            placeholder = create_placeholder_frame(
                tuple(config['camera']['resolution']),
                message="Reconnecting to Camera..."
            )
            encoded = _encode_jpeg(placeholder)
            if encoded:
                latest_jpeg_bytes = encoded
            time.sleep(0.2)
            continue

        # Bỏ qua nếu chưa có frame mới từ camera
        if frame_id == last_frame_id:
            time.sleep(0.01)
            continue

        last_frame_id = frame_id
        if frame_id % frame_skip != 0:
            time.sleep(0.005)
            continue

        t_now = time.time()
        dt = t_now - t_prev
        if dt > 0:
            current_fps = 0.8 * current_fps + 0.2 * (1.0 / dt) if current_fps > 0 else (1.0 / dt)
        t_prev = t_now

        annotated = process_single_frame(frame)
        encoded = _encode_jpeg(annotated)
        if encoded:
            latest_jpeg_bytes = encoded

        if datetime.now() >= next_alarm_time:
            next_alarm_time = datetime.now() + alarm_interval
            threading.Thread(target=winsound.Beep, args=(1000, 500), daemon=True).start()
            save_periodic_logs()
            threading.Thread(
                target=voice_notification,
                args=(f"Hiện tại có {current_people_count} người có mặt.",),
                daemon=True
            ).start()


def start_camera(rtsp_url=None):
    global cap, worker_running, worker_thread
    stop_camera()

    if not rtsp_url:
        rtsp_url = config['camera']['rtsp_url']

    resolution = tuple(config['camera']['resolution'])
    fps = config['camera'].get('frame_rate', 20)
    cam = ThreadedCamera(rtsp_url, resolution=resolution, fps=fps)

    if cam.start(retries=5):
        cap = cam
        count_window.clear()
        people_counter.reset()
        worker_running = True
        worker_thread = threading.Thread(target=_inference_worker, daemon=True)
        worker_thread.start()
        return True
    return False


def stop_camera():
    global cap, worker_running, worker_thread, latest_jpeg_bytes, current_fps
    worker_running = False
    if worker_thread is not None and worker_thread.is_alive():
        worker_thread.join(timeout=2.0)
    worker_thread = None

    if cap is not None:
        cap.stop()
        cap = None

    current_fps = 0.0
    placeholder = create_placeholder_frame(
        tuple(config['camera']['resolution']),
        message="Camera Stopped"
    )
    encoded = _encode_jpeg(placeholder)
    if encoded:
        latest_jpeg_bytes = encoded
    cv2.destroyAllWindows()


def generate_frames():
    """
    Phát luồng MJPEG mượt 20 FPS cho trình duyệt:
    Lấy trực tiếp khung hình mới nhất từ ThreadedCamera và vẽ chồng Bounding Box (latest_display_objects)
    từ luồng AI chạy ngầm để video webcam không bị khựng khi chạy trên CPU.
    """
    while True:
        if cap is not None and cap.is_running:
            ret, live_frame, _ = cap.read()
            if ret and live_frame is not None:
                with state_lock:
                    objs_snapshot = list(latest_display_objects)
                    c_count = current_people_count
                    t_in = total_entries
                    t_out = total_exits
                    fps_val = current_fps
                annotated = draw_tracks_and_overlay(
                    live_frame,
                    [],
                    c_count,
                    display_objects=objs_snapshot,
                    total_entries=t_in,
                    total_exits=t_out,
                    fps=fps_val
                )
                frame_bytes = _encode_jpeg(annotated)
                if frame_bytes is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.04)
                    continue

        frame_bytes = latest_jpeg_bytes
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)


@app.route('/')
def index():
    default_rtsp = os.environ.get('RTSP_URL', config['camera'].get('rtsp_url', ''))
    return render_template('index.html', default_rtsp_url=default_rtsp)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status', methods=['GET'])
def status():
    with state_lock:
        is_active = cap is not None and cap.is_connected
        return jsonify({
            "camera_active": is_active,
            "current_people_count": current_people_count,
            "total_entries": total_entries,
            "total_exits": total_exits,
            "fps": round(current_fps, 1),
            "recent_events": list(recent_events)
        }), 200


@app.route('/start', methods=['POST'])
def start():
    rtsp_url = request.form.get('rtsp_url', '').strip()
    if not rtsp_url and request.is_json:
        rtsp_url = (request.json or {}).get('rtsp_url', '').strip()
    rtsp_url = rtsp_url or os.environ.get('RTSP_URL', config['camera']['rtsp_url'])

    if start_camera(rtsp_url):
        return jsonify({"status": "ok", "message": "Đã kết nối và khởi động camera thành công!"}), 200
    else:
        return jsonify({"status": "error", "message": "Không thể kết nối tới nguồn camera. Vui lòng kiểm tra RTSP URL hoặc thiết bị."}), 500


@app.route('/stop', methods=['POST'])
def stop():
    stop_camera()
    save_periodic_logs()
    return jsonify({"status": "ok", "message": "Đã dừng camera và lưu dữ liệu thành công."}), 200


@app.route('/announce', methods=['POST'])
def announce():
    global current_people_count
    try:
        msg = f"Hiện tại có {current_people_count} người có mặt trong lớp học."
        threading.Thread(
            target=voice_notification,
            args=(msg,),
            daemon=True
        ).start()
        return jsonify({
            "status": "ok",
            "message": msg,
            "count": current_people_count
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
