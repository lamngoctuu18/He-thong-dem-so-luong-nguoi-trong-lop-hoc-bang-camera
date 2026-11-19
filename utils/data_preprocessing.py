import os
import random
import shutil

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def split_data(image_folder, label_folder, valid_size=0.2):
    """Chia tập dữ liệu ảnh và nhãn YOLOv8 thành 2 thư mục train/ và valid/."""
    if not os.path.exists(image_folder):
        print(f"⚠️ Thư mục ảnh '{image_folder}' chưa tồn tại.")
        return

    # Chỉ lấy các file ảnh, bỏ qua các thư mục con như 'train' và 'valid'
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder, f))
        and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not image_files:
        print("ℹ️ Không có ảnh mới cần chia tách (hoặc dữ liệu đã được chia vào train/valid).")
        return

    rng = random.Random(42)
    rng.shuffle(image_files)
    valid_count = max(1, int(len(image_files) * valid_size))
    valid_files = image_files[:valid_count]
    train_files = image_files[valid_count:]

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
        label_file = os.path.splitext(file)[0] + '.txt'
        label_src = os.path.join(label_folder, label_file)
        label_dst = os.path.join(train_label_folder, label_file)
        if os.path.exists(label_src):
            shutil.move(label_src, label_dst)
        else:
            print(f"⚠️ Label file not found: {label_src}")

    for file in valid_files:
        shutil.move(os.path.join(image_folder, file), os.path.join(valid_image_folder, file))
        label_file = os.path.splitext(file)[0] + '.txt'
        label_src = os.path.join(label_folder, label_file)
        label_dst = os.path.join(valid_label_folder, label_file)
        if os.path.exists(label_src):
            shutil.move(label_src, label_dst)
        else:
            print(f"⚠️ Label file not found: {label_src}")

    print(f"✅ Split data into {len(train_files)} training and {len(valid_files)} validation images.")


if __name__ == "__main__":
    default_image_folder = os.path.join(BASE_DIR, 'data', 'images')
    default_label_folder = os.path.join(BASE_DIR, 'data', 'labels')
    split_data(default_image_folder, default_label_folder)
