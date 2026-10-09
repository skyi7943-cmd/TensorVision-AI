import time
import threading
from PySide6.QtCore import QThread, Signal


class InferenceWorker(QThread):
    """
    Runs AI inference + overlay rendering off the GUI thread.
    Only the LATEST submitted frame is kept (older ones are dropped), so the UI
    never lags behind the camera and never freezes while the GPU is busy.
    """
    result_ready = Signal(object, object)   # annotated_frame, results
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, engine, visualizer, get_params):
        super().__init__()
        self.engine = engine
        self.visualizer = visualizer
        self.get_params = get_params  # -> (task, conf, iou)
        self._frame = None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._running = True
        self.fps = 0.0
        self._hw = None
        self._hw_t = 0.0

    def submit(self, frame):
        with self._lock:
            self._frame = frame
        self._event.set()

    def shutdown(self):
        self._running = False
        self._event.set()
        self.wait(5000)

    def run(self):
        try:
            self.status.emit("正在加载 AI 模型 (首次启动约需数秒)...")
            self.engine.load_detect_model()
            self.engine.load_pose_model()
            self.status.emit("模型就绪")
        except Exception as e:
            self.failed.emit(f"模型加载失败: {e}")

        n, t_fps = 0, time.perf_counter()
        while self._running:
            self._event.wait(0.2)
            self._event.clear()
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None:
                continue
            try:
                task, conf, iou = self.get_params()
                res = self.engine.process_frame(frame, task=task, conf=conf, iou=iou)

                now = time.perf_counter()
                if now - self._hw_t > 1.0 or self._hw is None:
                    self._hw, self._hw_t = self.engine.get_hardware_info(), now
                n += 1
                if now - t_fps >= 0.5:
                    self.fps, n, t_fps = n / (now - t_fps), 0, now

                out = self.visualizer.render(frame, res, fps=self.fps, hw_info=self._hw)
                self.result_ready.emit(out, res)
            except Exception as e:
                self.failed.emit(f"推理出错: {e}")
                time.sleep(0.5)
