import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import yaml
import os
from datetime import datetime

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
    tracker = DeepSort(max_age=config['tracker']['max_age'])
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

# Cấu hình camera IMOU (RTSP)
RTSP_URL = config['camera']['rtsp_url']
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("❌ Không thể kết nối camera!")
    exit()

print("✅ Kết nối camera thành công!")

frame_width, frame_height = config['camera']['resolution']
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, config['camera']['frame_rate'])

current_people_count = 0
people_count_history = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Không nhận được khung hình từ camera.")
        break

    frame = cv2.resize(frame, (frame_width, frame_height))

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

cap.release()
cv2.destroyAllWindows()
