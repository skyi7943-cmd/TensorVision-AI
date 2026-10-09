import time
import cv2
from PySide6.QtCore import QThread, Signal


class VideoExportThread(QThread):
    progress = Signal(int, int, float, float)  # current, total, percent, eta_sec
    done = Signal(str)       # NB: not named "finished" - that would shadow QThread.finished
    error = Signal(str)

    def __init__(self, input_video_path, output_video_path, engine, visualizer,
                 task="detect", conf=0.35, iou=0.45):
        super().__init__()
        self.input_path = input_video_path
        self.output_path = output_video_path
        self.engine = engine
        self.visualizer = visualizer
        self.task = task
        self.conf = conf
        self.iou = iou
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        cap = writer = None
        try:
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                self.error.emit(f"无法打开输入视频: {self.input_path}")
                return

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or fps > 120:
                fps = 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            writer = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                self.error.emit(f"无法创建输出文件: {self.output_path}")
                return

            hw = self.engine.get_hardware_info()
            idx = 0
            t0 = time.perf_counter()
            while not self.is_cancelled:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                idx += 1
                res = self.engine.process_frame(frame, task=self.task, conf=self.conf, iou=self.iou)
                el = time.perf_counter() - t0
                pfps = idx / el if el > 0 else fps
                eta = (total - idx) / pfps if pfps > 0 and total > 0 else 0
                writer.write(self.visualizer.render(frame, res, fps=pfps, hw_info=hw))
                pct = idx / total * 100.0 if total > 0 else 0.0
                self.progress.emit(idx, total, pct, eta)

            if self.is_cancelled:
                self.error.emit("导出已取消。")
            else:
                self.done.emit(self.output_path)
        except Exception as e:
            self.error.emit(f"导出失败: {e}")
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
