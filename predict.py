import numpy as np
from tensorflow.keras.models import load_model

# Load mô hình đã train
model = load_model('c:/Project_Structure/models/data/trained_model.h5', compile=False)

# Tạo dữ liệu mẫu để dự đoán
sample_data = np.array([5, 6, 7, 8, 7, 6, 8, 9, 10, 11]).reshape(1, 10, 1)

# Dự đoán số người tiếp theo
predicted_count = model.predict(sample_data)[0][0]
print(f"📌 Dự đoán số người tiếp theo: {predicted_count}")
