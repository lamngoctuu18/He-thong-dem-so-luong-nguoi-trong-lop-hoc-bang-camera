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
import cv2
import yaml
import numpy as np

from ultralytics import YOLO, settings
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.people_counter import ClassroomPeopleCounter
from inference.visualization import draw_tracks_and_overlay

settings.update({'runs_dir': os.path.join(BASE_DIR, 'data', 'runs')})

CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


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


def process_video(video_path, output_video_path=None, snapshot_dir=None, show_window=False, frame_skip=3):
    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy video: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / video_fps if video_fps > 0 else 0

    # Giữ tỉ lệ khung hình chuẩn (ví dụ 640x360 cho video 16:9) để xử lý nhanh và nét
    proc_w = 640
    proc_h = int(round(orig_h * (proc_w / float(orig_w)))) if orig_w > 0 else 360

    print(f"🎬 Thông tin Video: {orig_w}x{orig_h} @ {video_fps:.2f} FPS | Tổng: {total_frames} frames ({duration_sec:.2f} giây)")
    print("🔄 Đang nạp mô hình YOLOv8 + DeepSORT...")

    yolo_model = YOLO(os.path.join(BASE_DIR, config['yolo'].get('model_path', 'yolov8l.pt')))
    tracker = DeepSort(
        max_age=config['tracker'].get('max_age', 30),
        n_init=config['tracker'].get('min_hits', 2),
        max_cosine_distance=config['tracker'].get('max_cosine_distance', 0.3)
    )
    counter = ClassroomPeopleCounter(
        interior_grace_sec=2.5,
        border_grace_sec=0.6,
        stitch_timeout_sec=4.0,
        duplicate_iou_threshold=0.75
    )

    if output_video_path is None:
        output_video_path = os.path.join(BASE_DIR, 'data', 'counted_video.mp4')
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, video_fps, (proc_w, proc_h))

    if snapshot_dir is None:
        snapshot_dir = os.path.join(BASE_DIR, 'data', 'video_snapshots')
    os.makedirs(snapshot_dir, exist_ok=True)

    frame_idx = 0
    current_count = 0
    display_objects = []
    tracks = []
    detections = []
    counts_over_time = []
    per_second_counts = {}
    event_timeline = []
    snapshot_seconds = {1, 5, 9, 13}
    saved_snapshots = []

    t_start = time.time()
    frame_area = proc_w * proc_h

    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            break

        frame_idx += 1
        video_time_sec = frame_idx / video_fps
        frame = cv2.resize(raw_frame, (proc_w, proc_h))

        if frame_idx == 1 or frame_idx % frame_skip == 0:
            results = yolo_model(
                frame,
                conf=config['yolo'].get('confidence_threshold', 0.25),
                iou=config['yolo'].get('iou_threshold', 0.65),
                imgsz=640,
                classes=[0],
                project=os.path.join(BASE_DIR, 'data', 'runs'),
                verbose=False
            )
            detections = []
            for r in results[0].boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = r
                bw, bh = x2 - x1, y2 - y1
                if is_valid_person_box(bw, bh, frame_area):
                    detections.append(([x1, y1, bw, bh], conf, "person"))

            tracks = tracker.update_tracks(detections, frame=frame)
            current_count, events, display_objects = counter.update(
                tracks,
                detections,
                frame.shape,
                now=video_time_sec
            )

            for ev_type, change in events:
                event_timeline.append({
                    "time_sec": round(video_time_sec, 2),
                    "frame": frame_idx,
                    "event": ev_type,
                    "change": change,
                    "count_after": current_count
                })

        counts_over_time.append(current_count)
        sec_mark = int(video_time_sec)
        per_second_counts[sec_mark] = current_count

        elapsed = time.time() - t_start
        proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0

        annotated = draw_tracks_and_overlay(
            frame.copy(),
            tracks,
            current_count,
            detections=detections,
            display_objects=display_objects,
            total_entries=counter.total_entries,
            total_exits=counter.total_exits,
            fps=proc_fps
        )
        writer.write(annotated)

        if sec_mark in snapshot_seconds:
            snapshot_seconds.remove(sec_mark)
            snap_path = os.path.join(snapshot_dir, f"snapshot_{sec_mark:02d}s.jpg")
            cv2.imwrite(snap_path, annotated)
            saved_snapshots.append(snap_path)

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            print(
                f"⏳ Đã xử lý {frame_idx}/{total_frames} frames ({video_time_sec:.1f}s) | "
                f"Sĩ số hiện tại: {current_count} | In: {counter.total_entries} | Out: {counter.total_exits}"
            )

        if show_window:
            cv2.imshow("Video Processing", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    if show_window:
        cv2.destroyAllWindows()

    max_count = max(counts_over_time) if counts_over_time else 0
    min_count = min(counts_over_time) if counts_over_time else 0
    avg_count = float(np.mean(counts_over_time)) if counts_over_time else 0.0
    unique_people = counter.next_canonical_id - 1

    print("\n" + "=" * 75)
    print("📊 KẾT QUẢ ĐẾM NGƯỜI TRONG VIDEO:")
    print(f"   • Thời lượng video               : {duration_sec:.2f} giây ({total_frames} frames)")
    print(f"   • Tổng số người xuất hiện (In)   : {counter.total_entries} người (Số ID duy nhất: {unique_people})")
    print(f"   • Tổng số người rời khung (Out)  : {counter.total_exits} người")
    print(f"   • Số người có mặt cùng lúc tối đa: {max_count} người")
    print(f"   • Số người có mặt trung bình     : {avg_count:.1f} người (Cuối video: {current_count} người)")
    print(f"   • Video kết quả đã vẽ BoundingBox: {output_video_path}")
    print("=" * 75)

    print("\n📈 Biến động số người theo từng giây:")
    for sec in sorted(per_second_counts.keys()):
        print(f"   - Giây thứ {sec:02d}s: {per_second_counts[sec]} người")

    if event_timeline:
        print("\n🕒 Nhật ký sự kiện Vào / Ra trong video:")
        for ev in event_timeline:
            label = "Vào (+)" if ev['event'] == 'entry' else "Ra (-)"
            print(f"   - [Giây {ev['time_sec']:05.2f}s | Frame {ev['frame']:03d}] {label}{ev['change']} người -> Sĩ số: {ev['count_after']}")

    return {
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "total_entries": counter.total_entries,
        "total_exits": counter.total_exits,
        "unique_people": unique_people,
        "max_count": max_count,
        "min_count": min_count,
        "avg_count": avg_count,
        "final_count": current_count,
        "output_video_path": output_video_path,
        "snapshots": saved_snapshots
    }


if __name__ == "__main__":
    target_video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, 'data', 'input_video.mp4')
    process_video(target_video)
