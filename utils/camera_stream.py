import os
import cv2
import time
import threading


class ThreadedCamera:
    """
    Luồng đọc Camera độc lập (Threaded RTSP/Webcam Reader).
    Liên tục lấy frame mới nhất từ camera để tránh đọng buffer RTSP (giảm độ trễ)
    và tự động kết nối lại nếu luồng RTSP bị gián đoạn tạm thời.
    """

    def __init__(self, src, resolution=(640, 480), fps=20):
        self.src = int(src) if str(src).strip().isdigit() else src
        self.resolution = tuple(resolution)
        self.fps = fps
        self.cap = None
        self.frame = None
        self.frame_id = 0
        self.is_running = False
        self.is_connected = False
        self.lock = threading.Lock()
        self.thread = None

    def _open_capture(self):
        if isinstance(self.src, int) and os.name == 'nt':
            cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
        cap = cv2.VideoCapture(self.src)
        if isinstance(self.src, str) and self.src.lower().startswith("rtsp"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self, retries=5):
        if self.is_running:
            self.stop()

        attempt = 0
        while attempt < retries:
            attempt += 1
            cap = self._open_capture()

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                ret, first_frame = cap.read()
                if ret and first_frame is not None:
                    first_frame = cv2.resize(first_frame, self.resolution)
                with self.lock:
                    self.cap = cap
                    self.frame = first_frame
                    self.frame_id = 1 if first_frame is not None else 0
                    self.is_connected = True
                    self.is_running = True

                self.thread = threading.Thread(target=self._reader_loop, daemon=True)
                self.thread.start()
                print("✅ Kết nối camera thành công (Threaded Stream)!")
                return True

            print(f"❌ Không thể kết nối camera! Thử lại... ({attempt}/{retries})")
            cap.release()
            time.sleep(1.5)

        return False

    def _reader_loop(self):
        consecutive_failures = 0
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                self.is_connected = False
                if not self._reconnect():
                    time.sleep(2.0)
                    continue

            ret, raw_frame = self.cap.read()
            if not ret or raw_frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 15:
                    print("⚠️ Mất tín hiệu từ camera, đang thử kết nối lại...")
                    self.is_connected = False
                    self._reconnect()
                    consecutive_failures = 0
                else:
                    time.sleep(0.05)
                continue

            consecutive_failures = 0
            resized = cv2.resize(raw_frame, self.resolution)
            with self.lock:
                self.frame = resized
                self.frame_id += 1
                self.is_connected = True

            time.sleep(0.005)

    def _reconnect(self):
        if not self.is_running:
            return False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        cap = self._open_capture()
        if cap.isOpened():
            with self.lock:
                self.cap = cap
                self.is_connected = True
            print("🔄 Đã kết nối lại camera thành công!")
            return True
        cap.release()
        return False

    def read(self):
        """Trả về (ret, frame_copy, frame_id)."""
        with self.lock:
            if not self.is_running or self.frame is None:
                return False, None, self.frame_id
            return True, self.frame.copy(), self.frame_id

    def stop(self):
        self.is_running = False
        self.is_connected = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.frame = None
        print("🛑 Camera đã dừng lại.")

