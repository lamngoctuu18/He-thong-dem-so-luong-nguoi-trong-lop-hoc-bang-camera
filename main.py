import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import pandas as pd
from datetime import datetime, timedelta
import yaml
import os
import winsound
import time
from flask import Flask, render_template, Response
from utils.database import get_db_connection, create_table_if_not_exists

app = Flask(__name__)

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

print("🔄 Đang tải các mô hình...")

try:
    yolo_model = YOLO(config['yolo']['model_path'])
    print("✅ YOLO model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")

try:
    tracker = DeepSort(max_age=config['tracker']['max_age'], n_init=config['tracker']['min_hits'])
    print("✅ DeepSort tracker loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load DeepSort tracker: {e}")

lstm_model_path = config['dataset']['trained_model_path']
if os.path.exists(lstm_model_path):
    try:
        model = load_model(lstm_model_path, custom_objects={'mse': MeanSquaredError()})
        print("✅ LSTM model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LSTM model: {e}")
else:
    print(f"❌ LSTM model file not found at {lstm_model_path}")

# Connect to MySQL
db = get_db_connection()
if db is None:
    exit()
cursor = db.cursor()
create_table_if_not_exists(cursor)

# Cấu hình camera IMOU (RTSP)
RTSP_URL = config['camera']['rtsp_url']
cap = None
current_people_count = 0
people_count_history = []
entry_exit_log = []

def start_camera():
    global cap
    retries = 5
    while retries > 0:
        cap = cv2.VideoCapture(RTSP_URL)
        if cap.isOpened():
            print("✅ Kết nối camera thành công!")
            return True
        else:
            print(f"❌ Không thể kết nối camera! Thử lại... ({5 - retries}/5)")
            retries -= 1
            cap.release()
            cap = None
            time.sleep(2)
    return False

def generate_frames():
    global cap, current_people_count, people_count_history, entry_exit_log
    frame_skip = 5  # Process every 5th frame to reduce delay
    frame_count = 0
    alarm_interval = timedelta(minutes=1)  # Save data every 1 minute
    next_alarm_time = datetime.now() + alarm_interval

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (config['camera']['resolution'][0], config['camera']['resolution'][1]))

        results = yolo_model(frame, conf=config['yolo']['confidence_threshold'])
        detections = []
        
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result
            if int(cls) == 0:
                detections.append(([x1, y1, x2, y2], conf, "person"))

        tracks = tracker.update_tracks(detections, frame=frame)
        new_people_count = sum(1 for track in tracks if track.is_confirmed())

        if new_people_count != current_people_count:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if new_people_count > current_people_count:
                entry_exit_log.append([timestamp, "entry", new_people_count - current_people_count])
            else:
                entry_exit_log.append([timestamp, "exit", current_people_count - new_people_count])
            current_people_count = new_people_count

        people_count_history.append(current_people_count)
        if len(people_count_history) >= config['lstm']['sequence_length']:
            input_sequence = np.array(people_count_history[-config['lstm']['sequence_length']:]).reshape(1, config['lstm']['sequence_length'], 1)
            predicted_count = model.predict(input_sequence)[0][0]
            cv2.putText(frame, f"Predicted: {int(predicted_count)}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Count: {current_people_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if datetime.now() >= next_alarm_time:
            winsound.Beep(1000, 1000)  # Sound the alarm
            with open('data/current_people_count.csv', 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{current_people_count}\n")
            next_alarm_time = datetime.now() + alarm_interval

            # Save logs to MySQL
            for log in entry_exit_log:
                cursor.execute("INSERT INTO entry_exit_log (timestamp, event, count_change) VALUES (%s, %s, %s)", log)
            db.commit()
            print("📊 Dữ liệu đã được lưu vào MySQL")

            # Save logs to CSV
            try:
                df_log = pd.DataFrame(entry_exit_log, columns=["timestamp", "event", "count_change"])
                df_log.to_csv('C:/Project_Structure/data/entry_exit_log.csv', index=False)
                print("📊 Dữ liệu đã được lưu vào 'C:/Project_Structure/data/entry_exit_log.csv'")
            except PermissionError as e:
                print(f"❌ Lỗi khi lưu dữ liệu vào CSV: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def monitor():
    global cap, current_people_count, people_count_history, entry_exit_log
    if not start_camera():
        return

    frame_skip = 5  # Process every 5th frame to reduce delay
    frame_count = 0
    alarm_interval = timedelta(minutes=5)
    next_alarm_time = datetime.now() + alarm_interval
    stop_time = datetime.now() + timedelta(minutes=5)  # Stop after 5 minutes

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                print("❌ Không nhận được khung hình từ camera.")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = cv2.resize(frame, (config['camera']['resolution'][0], config['camera']['resolution'][1]))

            results = yolo_model(frame, conf=config['yolo']['confidence_threshold'])
            detections = []
            
            for result in results[0].boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = result
                if int(cls) == 0:
                    detections.append(([x1, y1, x2, y2], conf, "person"))

            tracks = tracker.update_tracks(detections, frame=frame)
            new_people_count = sum(1 for track in tracks if track.is_confirmed())

            if new_people_count != current_people_count:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if new_people_count > current_people_count:
                    entry_exit_log.append([timestamp, "entry", new_people_count - current_people_count])
                else:
                    entry_exit_log.append([timestamp, "exit", current_people_count - new_people_count])
                current_people_count = new_people_count

            people_count_history.append(current_people_count)
            if len(people_count_history) >= config['lstm']['sequence_length']:
                input_sequence = np.array(people_count_history[-config['lstm']['sequence_length']:]).reshape(1, config['lstm']['sequence_length'], 1)
                predicted_count = model.predict(input_sequence)[0][0]
                cv2.putText(frame, f"Predicted: {int(predicted_count)}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Count: {current_people_count}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("Real-Time Monitoring", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Chương trình đã dừng lại.")
                break

            # Check if it's time to sound the alarm and log the current count
            if datetime.now() >= next_alarm_time:
                winsound.Beep(1000, 1000)  # Sound the alarm
                with open('data/current_people_count.csv', 'a') as f:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{current_people_count}\n")
                next_alarm_time = datetime.now() + alarm_interval

                # Save logs to MySQL
                for log in entry_exit_log:
                    cursor.execute("INSERT INTO entry_exit_log (timestamp, event, count_change) VALUES (%s, %s, %s)", log)
                db.commit()
                print("📊 Dữ liệu đã được lưu vào MySQL")

                # Save logs to CSV
                df_log = pd.DataFrame(entry_exit_log, columns=["timestamp", "event", "count_change"])
                df_log.to_csv('C:/Project_Structure/data/entry_exit_log.csv', index=False)
                print("📊 Dữ liệu đã được lưu vào 'C:/Project_Structure/data/entry_exit_log.csv'")

            if datetime.now() >= stop_time:
                print("🛑 Chương trình đã dừng lại sau 5 phút.")
                break

        except Exception as e:
            print(f"❌ Lỗi trong quá trình xử lý: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save final logs to MySQL
    for log in entry_exit_log:
        cursor.execute("INSERT INTO entry_exit_log (timestamp, event, count_change) VALUES (%s, %s, %s)", log)
    db.commit()
    print("📊 Dữ liệu cuối cùng đã được lưu vào MySQL")

    # Save final logs to CSV
    df_log = pd.DataFrame(entry_exit_log, columns=["timestamp", "event", "count_change"])
    df_log.to_csv('c:/Project_Structure/data/entry_exit_log.csv', index=False)
    print("📊 Dữ liệu cuối cùng đã được lưu vào 'c:/Project_Structure/data/entry_exit_log.csv'")

    with open('data/current_people_count.csv', 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{current_people_count}\n")
    print("📊 Dữ liệu cuối cùng đã được lưu vào 'data/current_people_count.csv'")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
