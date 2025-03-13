import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import shutil

def load_labeled_data(label_folder):
    data = []
    for label_file in os.listdir(label_folder):
        label_path = os.path.join(label_folder, label_file)

        with open(label_path, 'r') as f:
            people_count = sum(1 for line in f if line.strip())

        timestamp = pd.to_datetime("2025-03-10", format="%Y-%m-%d")
        data.append([timestamp, people_count])

    df = pd.DataFrame(data, columns=['timestamp', 'people_count'])
    df = df.sort_values('timestamp')
    return df

def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['people_count'] = scaler.fit_transform(df[['people_count']])
    return df, scaler

def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def split_data(image_folder, label_folder, valid_size=0.2):
    image_files = os.listdir(image_folder)
    train_files, valid_files = train_test_split(image_files, test_size=valid_size, random_state=42)

    train_image_folder = os.path.join(image_folder, 'train')
    valid_image_folder = os.path.join(image_folder, 'valid')
    train_label_folder = os.path.join(label_folder, 'train')
    valid_label_folder = os.path.join(label_folder, 'valid')

    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(valid_image_folder, exist_ok=True)
    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(valid_label_folder, exist_ok=True)

    for file in train_files:
        shutil.move(os.path.join(image_folder, file), os.path.join(train_image_folder, file))
        label_file = file.replace('.jpg', '.txt')
        label_src = os.path.join(label_folder, label_file)
        label_dst = os.path.join(train_label_folder, label_file)
        if os.path.exists(label_src):
            shutil.move(label_src, label_dst)
        else:
            print(f"⚠️ Label file not found: {label_src}")

    for file in valid_files:
        shutil.move(os.path.join(image_folder, file), os.path.join(valid_image_folder, file))
        label_file = file.replace('.jpg', '.txt')
        label_src = os.path.join(label_folder, label_file)
        label_dst = os.path.join(valid_label_folder, label_file)
        if os.path.exists(label_src):
            shutil.move(label_src, label_dst)
        else:
            print(f"⚠️ Label file not found: {label_src}")

    print(f"✅ Split data into {len(train_files)} training and {len(valid_files)} validation images.")

if __name__ == "__main__":
    image_folder = '../data/images/'
    label_folder = '../data/labels/'
    split_data(image_folder, label_folder)
