import cv2
import numpy as np


def draw_boxes(frame, boxes, color=(0, 255, 0), thickness=2):
    """Vẽ danh sách các bounding box [x1, y1, x2, y2] lên khung hình."""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def put_text(frame, text, position=(20, 40), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, color=(255, 0, 0), thickness=2):
    cv2.putText(frame, text, position, font, font_scale, color, thickness)
    return frame


def _draw_person_box(frame, x1, y1, x2, y2, label_text, color=(0, 230, 64)):
    """Vẽ khung Bounding Box nổi bật có góc nhấn, tâm và nhãn nền màu cho từng người."""
    h_img, w_img = frame.shape[:2]
    x1 = max(0, min(w_img - 1, int(x1)))
    y1 = max(0, min(h_img - 1, int(y1)))
    x2 = max(0, min(w_img - 1, int(x2)))
    y2 = max(0, min(h_img - 1, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return

    # 1. Khung chữ nhật chính
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 2. Vẽ 4 góc đậm (Corner accents) giúp dễ quan sát trên nền lớp học
    corner_len = max(6, min(16, (x2 - x1) // 4, (y2 - y1) // 4))
    t = 3
    # Góc trên-trái
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, t)
    # Góc trên-phải
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, t)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, t)
    # Góc dưới-trái
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, t)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, t)
    # Góc dưới-phải
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, t)

    # 3. Chấm tâm đối tượng
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

    # 4. Nhãn tên + độ tự tin có nền màu phía trên box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thick = 1
    (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, font_thick)

    label_y1 = y1 - th - 8
    if label_y1 < 0:
        label_y1 = y1 + 2
        label_y2 = y1 + th + 8
        text_y = y1 + th + 4
    else:
        label_y2 = y1
        text_y = y1 - 4

    label_x2 = min(w_img - 1, x1 + tw + 10)
    cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(frame, label_text, (x1 + 5, text_y), font, font_scale, (0, 0, 0), font_thick + 1, cv2.LINE_AA)
    cv2.putText(frame, label_text, (x1 + 5, text_y), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)


def draw_tracks_and_overlay(frame, tracks, current_count, detections=None, display_objects=None, total_entries=0, total_exits=0, fps=None):
    """
    Vẽ Bounding Box cho tất cả người được phát hiện trong khung hình:
    - Nếu có display_objects (từ ClassroomPeopleCounter đã khử trùng lặp và nối ID), vẽ theo display_objects.
    - Ngược lại fallback sang tracks / detections.
    """
    drawn_boxes = []

    if display_objects is not None:
        for obj in display_objects:
            bx1, by1, bx2, by2 = obj['box']
            conf = obj.get('conf')
            conf_str = f" {int(round(conf * 100))}%" if conf is not None else ""
            if obj.get('confirmed') and obj.get('id') is not None:
                label = f"ID:{obj['id']} Person{conf_str}"
                box_color = (0, 220, 60)
            else:
                label = f"Person{conf_str}"
                box_color = (0, 200, 255)
            _draw_person_box(frame, bx1, by1, bx2, by2, label, color=box_color)
            drawn_boxes.append((bx1, by1, bx2, by2))
    elif tracks:
        for track in tracks:
            # Bỏ qua các track đã mất dấu quá 1 frame (tránh vẽ khung ma khi người đã đi khỏi)
            if getattr(track, 'time_since_update', 0) > 1:
                continue

            ltrb = track.to_ltrb(orig=True)
            if ltrb is None:
                ltrb = track.to_ltrb()
            bx1, by1, bx2, by2 = map(int, ltrb)

            conf = getattr(track, 'det_conf', None)
            conf_str = f" {int(round(conf * 100))}%" if conf is not None else ""

            if track.is_confirmed():
                label = f"ID:{track.track_id} Person{conf_str}"
                box_color = (0, 220, 60)  # Xanh lá cây cho người đã xác nhận ID
            else:
                label = f"Person{conf_str}"
                box_color = (0, 200, 255)  # Vàng cam cho người mới bước vào (frame đầu tiên)

            _draw_person_box(frame, bx1, by1, bx2, by2, label, color=box_color)
            drawn_boxes.append((bx1, by1, bx2, by2))

    # Nếu không có track nào được vẽ nhưng YOLOv8 có phát hiện người (detections), vẽ trực tiếp từ detections
    if not drawn_boxes and display_objects is None and detections:
        for det in detections:
            (x1, y1, w, h), conf, _ = det
            bx1, by1, bx2, by2 = int(x1), int(y1), int(x1 + w), int(y1 + h)
            label = f"Person {int(round(conf * 100))}%"
            _draw_person_box(frame, bx1, by1, bx2, by2, label, color=(0, 220, 60))

    # Bảng thông tin HUD bán trong suốt ở góc trên bên trái
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (235, 74), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(
        frame,
        f"Count: {current_count}",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 230, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"In: {total_entries} | Out: {total_exits}",
        (16, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 128),
        2,
        cv2.LINE_AA
    )

    if fps is not None:
        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (max(10, w - 115), 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    return frame


def create_placeholder_frame(resolution=(480, 360), message="Camera Offline - Press Start"):
    """Tạo khung hình chờ khi camera chưa khởi động hoặc đang kết nối."""
    w, h = resolution
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (35, 39, 43)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(message, font, 0.65, 2)[0]
    text_x = max(10, (w - text_size[0]) // 2)
    text_y = max(20, (h + text_size[1]) // 2)
    cv2.putText(frame, message, (text_x, text_y), font, 0.65, (200, 205, 210), 2)
    return frame
