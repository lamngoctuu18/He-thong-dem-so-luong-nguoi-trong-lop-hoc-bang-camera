import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from sklearn.metrics import r2_score
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Add the parent directory to the system path to resolve the utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_preprocessing import load_labeled_data, preprocess_data, create_sequences, split_data
from lstm_model import build_lstm_model
from utils.plot_metrics import plot_metrics  # Import the new plot_metrics function

# Split data into training and validation sets
image_folder = '../data/images/'
label_folder = '../data/labels/'
split_data(image_folder, label_folder)

# Load training and validation data
train_label_folder = os.path.join(label_folder, 'train')
valid_label_folder = os.path.join(label_folder, 'valid')
df_train = load_labeled_data(train_label_folder)
df_valid = load_labeled_data(valid_label_folder)

if df_train.empty or df_valid.empty:
    print("❌ Không có dữ liệu hợp lệ sau khi xử lý.")
    exit()

# Chuẩn hóa dữ liệu
df_train, scaler_train = preprocess_data(df_train)
df_valid, scaler_valid = preprocess_data(df_valid)

# Chia dữ liệu thành chuỗi thời gian
X_train, y_train = create_sequences(df_train['people_count'].values)
X_valid, y_valid = create_sequences(df_valid['people_count'].values)

# Load hoặc tạo mô hình mới
try:
    model = load_model('C:/Project_Structure/data/trained_model.h5', custom_objects={'mse': MeanSquaredError()})
    print("✅ Loaded existing model.")
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=['mae'])
except:
    model = build_lstm_model()
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=['mae'])
    print("🆕 Created new LSTM model.")

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_valid, y_valid), verbose=1)  # Reduced epochs to 50
model.save('data/trained_model.h5')

# Đánh giá mô hình
y_pred = model.predict(X_train)
mae = MeanAbsoluteError()(y_train, y_pred).numpy()
r2 = r2_score(y_train, y_pred)
print(f"📊 Độ lỗi trung bình tuyệt đối (MAE): {mae}")
print(f"✅ Hệ số xác định (R² Score): {r2}")

# Vẽ biểu đồ huấn luyện
plot_metrics(history, metric='loss')
plot_metrics(history, metric='mae')

print("🎉 Training hoàn tất! Mô hình đã lưu vào 'data/trained_model.h5'")

