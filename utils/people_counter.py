import time
import math


def compute_iou(box_a, box_b):
    """Tính chỉ số Intersection over Union (IoU) giữa 2 box [x1, y1, x2, y2]."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return inter_area / float(area_a + area_b - inter_area)


class ClassroomPeopleCounter:
    """
    Bộ đếm người thông minh chống đếm trùng (Anti-Duplicate People Counter):
    1. Khử trùng lặp Bounding Box (IoU Deduplication): Ngăn 1 người bị gắn 2 Track ID chồng lên nhau.
    2. Phân vùng Ân hạn (Zone-Aware Grace Period):
       - Người ngồi giữa lớp bị che khuất tạm thời / cúi xuống được giữ trạng thái có mặt (3.5 giây),
         không bị tính nhầm là Đi ra rồi Đi vào lại.
       - Người đi ra ở vùng rìa khung hình (cửa/lối đi) được ghi nhận Đi ra nhanh chóng (0.8 giây).
    3. Khớp nối ID theo vị trí - thời gian (Spatial-Temporal ID Stitching):
       - Nếu một người bị mất dấu rồi xuất hiện lại gần vị trí cũ với một Track ID mới của DeepSORT,
         hệ thống tự động nối lại ID gốc, không cộng trùng vào tổng lượt Vào (total_entries).
    """

    def __init__(
        self,
        interior_grace_sec=3.5,
        border_grace_sec=0.8,
        stitch_timeout_sec=5.0,
        duplicate_iou_threshold=0.75,
        border_margin_ratio=0.10
    ):
        self.interior_grace_sec = interior_grace_sec
        self.border_grace_sec = border_grace_sec
        self.stitch_timeout_sec = stitch_timeout_sec
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.border_margin_ratio = border_margin_ratio

        self.reset()

    def reset(self):
        # map: raw_deepsort_id -> canonical_person_id
        self.id_alias = {}
        # map: canonical_person_id -> dict(box, last_seen, first_seen, is_at_border, conf)
        self.active_people = {}
        # map: canonical_person_id -> dict(box, lost_time) của người vừa mất dấu để nối lại nếu bị đổi ID
        self.recently_lost = {}
        self.next_canonical_id = 1
        self.current_count = 0
        self.total_entries = 0
        self.total_exits = 0

    def _is_near_border(self, box, frame_w, frame_h):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        margin_x = frame_w * self.border_margin_ratio
        margin_y = frame_h * self.border_margin_ratio
        return (
            cx <= margin_x
            or cx >= (frame_w - margin_x)
            or cy <= margin_y
            or cy >= (frame_h - margin_y)
        )

    def _box_distance(self, box_a, box_b):
        cx_a, cy_a = (box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0
        cx_b, cy_b = (box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0
        return math.hypot(cx_a - cx_b, cy_a - cy_b)

    def _find_matching_person(self, box, now, exclude_ids):
        """
        Tìm xem một Track ID mới của DeepSORT có thực chất là một người đang có mặt (bị mất dấu tạm thời)
        hoặc vừa mới mất dấu gần cùng vị trí hay không.
        """
        box_h = max(20.0, box[3] - box[1])
        max_dist = box_h * 1.15

        best_cid = None
        best_score = float('inf')

        # 1. Kiểm tra trong danh sách người đang trong thời gian ân hạn (chưa bị xóa khỏi active_people)
        for cid, info in self.active_people.items():
            if cid in exclude_ids:
                continue
            iou = compute_iou(box, info['box'])
            dist = self._box_distance(box, info['box'])
            if iou >= 0.30 or dist <= max_dist:
                score = dist - (iou * 50.0)
                if score < best_score:
                    best_score = score
                    best_cid = cid

        if best_cid is not None:
            return best_cid, "active"

        # 2. Kiểm tra trong danh sách người vừa mới bị tính là mất dấu (recently_lost) ở vùng giữa lớp
        for cid, info in list(self.recently_lost.items()):
            if now - info['lost_time'] > self.stitch_timeout_sec:
                del self.recently_lost[cid]
                continue
            iou = compute_iou(box, info['box'])
            dist = self._box_distance(box, info['box'])
            if iou >= 0.25 or dist <= max_dist * 0.9:
                score = dist - (iou * 50.0)
                if score < best_score:
                    best_score = score
                    best_cid = cid

        if best_cid is not None:
            return best_cid, "lost"

        return None, None

    def update(self, tracks, detections, frame_shape, now=None):
        """
        Cập nhật trạng thái đếm người trên mỗi frame.
        Trả về:
        - current_count: Số người hiện tại trong lớp
        - events: Danh sách sự kiện [('entry', n) hoặc ('exit', n)] phát sinh ở frame này
        - display_objects: Danh sách dict [{'id': canonical_id, 'box': (x1,y1,x2,y2), 'conf': conf, 'confirmed': bool}]
        """
        if now is None:
            now = time.time()

        frame_h, frame_w = frame_shape[:2]

        # Bước 1: Thu thập các track đang thực sự được phát hiện ở frame hiện tại (time_since_update == 0)
        raw_visible = []
        if tracks:
            for track in tracks:
                if getattr(track, 'time_since_update', 0) > 0:
                    continue
                ltrb = track.to_ltrb(orig=True)
                if ltrb is None:
                    ltrb = track.to_ltrb()
                box = [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])]
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                raw_visible.append({
                    'raw_id': str(track.track_id),
                    'box': box,
                    'conf': getattr(track, 'det_conf', None),
                    'confirmed': track.is_confirmed()
                })
        elif detections:
            for idx, det in enumerate(detections):
                (x1, y1, w, h), conf, _ = det
                raw_visible.append({
                    'raw_id': f"det_{idx}",
                    'box': [float(x1), float(y1), float(x1 + w), float(y1 + h)],
                    'conf': conf,
                    'confirmed': True
                })

        # Bước 2: Khử trùng lặp nếu 2 track chồng lấp lên cùng 1 người (IoU > duplicate_iou_threshold)
        # Ưu tiên track đã confirmed, đã có canonical_id lâu hơn hoặc độ tự tin cao hơn
        raw_visible.sort(
            key=lambda item: (
                1 if item['confirmed'] else 0,
                1 if item['raw_id'] in self.id_alias else 0,
                item['conf'] if item['conf'] is not None else 0.5
            ),
            reverse=True
        )
        dedup_visible = []
        for candidate in raw_visible:
            is_dup = False
            for kept in dedup_visible:
                if compute_iou(candidate['box'], kept['box']) >= self.duplicate_iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                dedup_visible.append(candidate)

        # Bước 3: Ánh xạ sang Canonical ID (định danh người duy nhất) và chống đếm trùng khi đổi ID
        seen_canonical_ids = set()
        display_objects = []
        newly_entered_count = 0
        revived_from_lost_count = 0

        for item in dedup_visible:
            raw_id = item['raw_id']
            box = item['box']
            conf = item['conf']
            confirmed = item['confirmed']

            if not confirmed:
                # Chỉ hiển thị box vàng cam cho người mới bước vào frame đầu, chưa tính vào sĩ số cho tới khi confirmed
                display_objects.append({
                    'id': None,
                    'box': tuple(map(int, box)),
                    'conf': conf,
                    'confirmed': False
                })
                continue

            cid = self.id_alias.get(raw_id)
            if cid is not None and cid in seen_canonical_ids:
                # Tránh trường hợp 2 raw_id cùng trỏ vào 1 cid trong cùng 1 frame
                continue

            if cid is not None and cid in self.active_people:
                # Người đã có trong danh sách đang hiện diện
                self.active_people[cid]['box'] = box
                self.active_people[cid]['last_seen'] = now
                self.active_people[cid]['conf'] = conf
                self.active_people[cid]['is_at_border'] = self._is_near_border(box, frame_w, frame_h)
                seen_canonical_ids.add(cid)
            else:
                # Track ID mới được confirm -> kiểm tra xem có phải là người cũ bị khuất/đổi ID không
                matched_cid, match_source = self._find_matching_person(box, now, seen_canonical_ids)
                if matched_cid is not None:
                    cid = matched_cid
                    self.id_alias[raw_id] = cid
                    if match_source == "lost":
                        # Khôi phục người vừa mới bị tính nhầm là exit cách đây vài giây
                        self.recently_lost.pop(cid, None)
                        revived_from_lost_count += 1
                    self.active_people[cid] = {
                        'box': box,
                        'last_seen': now,
                        'conf': conf,
                        'is_at_border': self._is_near_border(box, frame_w, frame_h)
                    }
                    seen_canonical_ids.add(cid)
                else:
                    # Đây thực sự là một người mới bước vào lớp
                    cid = self.next_canonical_id
                    self.next_canonical_id += 1
                    self.id_alias[raw_id] = cid
                    self.active_people[cid] = {
                        'box': box,
                        'last_seen': now,
                        'conf': conf,
                        'is_at_border': self._is_near_border(box, frame_w, frame_h)
                    }
                    seen_canonical_ids.add(cid)
                    newly_entered_count += 1

            display_objects.append({
                'id': cid,
                'box': tuple(map(int, box)),
                'conf': conf,
                'confirmed': True
            })

        # Nếu có người vừa bị tính là mất dấu (exit) ở giữa lớp nhưng xuất hiện lại ngay với ID mới,
        # hoàn tác lượt exit ảo đó để không làm tăng cả exit lẫn entry
        if revived_from_lost_count > 0:
            self.total_exits = max(0, self.total_exits - revived_from_lost_count)

        # Bước 4: Kiểm tra những người không xuất hiện ở frame này xem đã quá thời gian ân hạn chưa
        exited_count = 0
        for cid, info in list(self.active_people.items()):
            if cid in seen_canonical_ids:
                continue
            time_missing = now - info['last_seen']
            grace = self.border_grace_sec if info['is_at_border'] else self.interior_grace_sec
            if time_missing > grace:
                del self.active_people[cid]
                # Nếu người đó mất dấu ở giữa lớp (không phải rìa cửa), lưu vào recently_lost để phòng trường hợp họ cúi xuống lâu
                if not info['is_at_border']:
                    self.recently_lost[cid] = {
                        'box': info['box'],
                        'lost_time': now
                    }
                exited_count += 1

        # Bước 5: Cập nhật số lượng người và sự kiện Vào/Ra chính xác
        events = []
        if newly_entered_count > 0:
            self.total_entries += newly_entered_count
            events.append(("entry", newly_entered_count))

        if exited_count > 0:
            self.total_exits += exited_count
            events.append(("exit", exited_count))

        self.current_count = len(self.active_people)
        return self.current_count, events, display_objects

