import cv2

def draw_boxes(frame, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame

def put_text(frame, text, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 0, 0), thickness=2):
    cv2.putText(frame, text, position, font, font_scale, color, thickness)
    return frame
