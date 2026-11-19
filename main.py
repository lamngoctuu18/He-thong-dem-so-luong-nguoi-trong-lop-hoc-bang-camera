import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, "data", ".mplconfig"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE_DIR, "data", ".ultralytics"))

try:
    import ultralytics  # noqa: F401
except ModuleNotFoundError:
    _py39 = r"C:\Users\84916\AppData\Local\Programs\Python\Python39\python.exe"
    if os.path.exists(_py39) and os.path.normcase(sys.executable) != os.path.normcase(_py39):
        sys.exit(subprocess.call([_py39, *sys.argv]))
    raise

import cv2
import time
from datetime import datetime, timedelta
import winsound
import threading
from utils.camera_stream import ThreadedCamera
from app import (
    app,
    config,
    process_single_frame,
    save_periodic_logs,
)
import app as app_state


def monitor():
    """Chạy giám sát trực tiếp trên cửa sổ OpenCV (Desktop mode) sử dụng ThreadedCamera."""
    cam = ThreadedCamera(
        config['camera']['rtsp_url'],
        resolution=tuple(config['camera']['resolution']),
        fps=config['camera'].get('frame_rate', 20)
    )
    if not cam.start(retries=5):
        return

    frame_skip = max(1, int(config['camera'].get('frame_skip', 2)))
    alarm_interval = timedelta(minutes=5)
    next_alarm_time = datetime.now() + alarm_interval
    stop_time = datetime.now() + timedelta(minutes=5)
    last_frame_id = -1
    t_prev = time.time()

    while cam.is_running:
        try:
            ret, frame, frame_id = cam.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            if frame_id == last_frame_id or frame_id % frame_skip != 0:
                time.sleep(0.005)
                continue

            last_frame_id = frame_id
            t_now = time.time()
            dt = t_now - t_prev
            if dt > 0:
                app_state.current_fps = (
                    0.8 * app_state.current_fps + 0.2 * (1.0 / dt)
                    if app_state.current_fps > 0 else (1.0 / dt)
                )
            t_prev = t_now

            annotated = process_single_frame(frame)
            cv2.imshow("Real-Time Monitoring", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Chương trình đã dừng lại.")
                break

            if datetime.now() >= next_alarm_time:
                next_alarm_time = datetime.now() + alarm_interval
                threading.Thread(target=winsound.Beep, args=(1000, 500), daemon=True).start()
                save_periodic_logs()

            if datetime.now() >= stop_time:
                print("🛑 Chương trình đã dừng lại sau 5 phút.")
                break

        except Exception as e:
            print(f"❌ Lỗi trong quá trình xử lý: {e}")
            break

    cam.stop()
    cv2.destroyAllWindows()
    save_periodic_logs()


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
