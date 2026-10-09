import time
import math
import cv2
from PySide6.QtCore import QThread, Signal


class VideoCaptureThread(QThread):
    """Capture thread for webcam or local video file. Emits BGR frames."""
    raw_frame_ready = Signal(object)
    source_opened = Signal(dict)
    source_finished = Signal()
    source_error = Signal(str)

    def __init__(self):
        super().__init__()
        self.source = None
        self.is_camera = False
        self.is_running = False
        self.is_paused = False
        self.loop = True
        self.target_fps = 30.0
        self.total_frames = 0
        self.current_frame_idx = 0
        self.seek_requested_frame = -1

    def _begin(self, source, is_camera):
        self.stop()
        self.source = source
        self.is_camera = is_camera
        self.is_paused = False
        self.seek_requested_frame = -1
        self.current_frame_idx = 0
        self.is_running = True   # set before start() so an early stop() is honoured
        self.start()

    def open_camera(self, camera_index=0):
        self._begin(int(camera_index), True)

    def open_file(self, file_path):
        self._begin(str(file_path), False)

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def toggle_play(self):
        self.is_paused = not self.is_paused
        return self.is_paused

    def request_seek(self, frame_idx):
        if not self.is_camera and self.total_frames > 0:
            self.seek_requested_frame = max(0, min(frame_idx, self.total_frames - 1))

    def stop(self):
        self.is_running = False
        if self.isRunning():
            self.wait(5000)  # the thread releases its own capture device

    def run(self):
        cap = None
        try:
            if self.is_camera:
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(self.source)
            else:
                cap = cv2.VideoCapture(self.source)

            if not cap.isOpened():
                self.source_error.emit(f"无法打开: {self.source}")
                return

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or math.isnan(fps) or fps <= 0 or fps > 120:
                fps = 30.0
            self.target_fps = fps
            self.total_frames = 0 if self.is_camera else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self.source_opened.emit({
                "is_camera": self.is_camera, "total_frames": self.total_frames,
                "fps": fps, "width": w, "height": h, "source": str(self.source),
            })

            interval = 1.0 / fps
            while self.is_running:
                if self.seek_requested_frame >= 0 and not self.is_camera:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_requested_frame)
                    self.current_frame_idx = self.seek_requested_frame
                    self.seek_requested_frame = -1

                if self.is_paused and not self.is_camera:
                    time.sleep(0.03)
                    continue

                t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    if not self.is_camera and self.loop and self.total_frames > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame_idx = 0
                        continue
                    self.source_finished.emit()
                    break

                if not self.is_camera:
                    self.current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.raw_frame_ready.emit(frame)

                if self.is_camera:
                    time.sleep(0.001)
                else:
                    rest = interval - (time.perf_counter() - t0)
                    if rest > 0.002:
                        time.sleep(rest)
        except Exception as e:  # never die silently
            self.source_error.emit(f"视频线程异常: {e}")
        finally:
            if cap is not None:
                cap.release()
