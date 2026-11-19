import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.people_counter import ClassroomPeopleCounter


class DummyTrack:
    def __init__(self, track_id, box, confirmed=True, time_since_update=0, conf=0.88):
        self.track_id = track_id
        self._box = box
        self._confirmed = confirmed
        self.time_since_update = time_since_update
        self.det_conf = conf

    def is_confirmed(self):
        return self._confirmed

    def to_ltrb(self, orig=False, orig_strict=False):
        return self._box


def run_tests():
    counter = ClassroomPeopleCounter(
        interior_grace_sec=3.5,
        border_grace_sec=0.8,
        stitch_timeout_sec=5.0,
        duplicate_iou_threshold=0.55
    )
    frame_shape = (360, 480, 3)  # H=360, W=480
    t0 = 1000.0

    # 1. Hai học sinh bước vào ngồi ở giữa lớp
    tracks_step1 = [
        DummyTrack(1, [150, 120, 210, 260]),  # Giữa lớp
        DummyTrack(2, [260, 120, 320, 260]),  # Giữa lớp
    ]
    count, events, objs = counter.update(tracks_step1, [], frame_shape, now=t0)
    assert count == 2 and counter.total_entries == 2 and counter.total_exits == 0, f"Step 1 failed: {count}"
    print("✅ Kịch bản 1 (2 người vào lớp): Count=2, In=2, Out=0")

    # 2. DeepSORT sinh thêm 1 Track ID trùng lặp (ID=10) đè lên học sinh số 1 (IoU cao)
    tracks_step2 = [
        DummyTrack(1, [150, 120, 210, 260], conf=0.90),
        DummyTrack(10, [152, 122, 208, 258], conf=0.75),  # Trùng lên người 1
        DummyTrack(2, [260, 120, 320, 260], conf=0.88),
    ]
    count, events, objs = counter.update(tracks_step2, [], frame_shape, now=t0 + 0.5)
    assert count == 2 and counter.total_entries == 2 and len(objs) == 2, f"Step 2 failed: {count}"
    print("✅ Kịch bản 2 (Khử 2 khung chồng lấp trên 1 người): Không bị đếm nhân đôi (Count=2, In=2)")

    # 3. Học sinh số 1 cúi xuống nhặt bút (mất dấu tạm thời 2.0 giây ở giữa lớp)
    tracks_step3 = [
        DummyTrack(2, [260, 120, 320, 260]),
    ]
    count, events, objs = counter.update(tracks_step3, [], frame_shape, now=t0 + 2.5)
    assert count == 2 and counter.total_exits == 0, f"Step 3 failed: count={count}, exits={counter.total_exits}"
    print("✅ Kịch bản 3 (Bị che khuất tạm thời 2 giây ở giữa lớp): Giữ nguyên Count=2, không báo Exit ảo")

    # 4. Học sinh số 1 ngồi dậy sau 2.8 giây nhưng DeepSORT lại cấp một Track ID hoàn toàn mới (ID=99) ở cùng vị trí
    tracks_step4 = [
        DummyTrack(99, [155, 122, 215, 262]),  # Cùng vị trí người 1 nhưng bị đổi sang ID 99
        DummyTrack(2, [260, 120, 320, 260]),
    ]
    count, events, objs = counter.update(tracks_step4, [], frame_shape, now=t0 + 2.8)
    ids_on_screen = sorted([o['id'] for o in objs])
    assert count == 2 and counter.total_entries == 2 and counter.total_exits == 0, "Step 4 failed"
    assert ids_on_screen == [1, 2], f"Step 4 ID stitching failed: {ids_on_screen}"
    print(f"✅ Kịch bản 4 (DeepSORT đổi sang ID 99 sau khi bị khuất): Tự động nối lại ID gốc {ids_on_screen}, không cộng trùng lượt Vào (In=2, Out=0)")

    # 5. Học sinh số 2 đi ra phía rìa cửa lớp (x gần biên phải W=480) rồi bước ra khỏi khung hình
    tracks_step5a = [
        DummyTrack(99, [155, 122, 215, 262]),
        DummyTrack(2, [430, 120, 475, 260]),  # Di chuyển ra vùng rìa cửa lớp (cx = 452.5 > 480 * 0.9)
    ]
    counter.update(tracks_step5a, [], frame_shape, now=t0 + 3.5)

    # Sau 1.0 giây người số 2 biến mất khỏi rìa cửa lớp
    tracks_step5b = [
        DummyTrack(99, [155, 122, 215, 262]),
    ]
    count, events, objs = counter.update(tracks_step5b, [], frame_shape, now=t0 + 4.6)
    assert count == 1 and counter.total_entries == 2 and counter.total_exits == 1, f"Step 5 failed: count={count}, exits={counter.total_exits}"
    print("✅ Kịch bản 5 (1 người thực sự bước ra qua rìa khung hình): Ghi nhận chính xác Count=1, In=2, Out=1")
    print("\n🎉 TẤT CẢ 5 KỊCH BẢN KIỂM THỬ CHỐNG ĐẾM TRÙNG ĐỀU ĐẠT 100%!")


if __name__ == "__main__":
    run_tests()

