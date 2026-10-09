import sys
import os
import faulthandler
import multiprocessing
import traceback
import datetime

# --- Windowed (no-console) builds have sys.stdout/stderr == None. torch/ultralytics/tqdm
# --- print to them and would crash, so give them a real sink + a persistent log file.
LOG_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "TensorVisionAI")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "app.log")
_log = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
if sys.stdout is None:
    sys.stdout = _log
if sys.stderr is None:
    sys.stderr = _log
faulthandler.enable(_log)

os.environ.setdefault("YOLO_AUTOINSTALL", "False")   # never pip-install at runtime
# YOLO_OFFLINE is left unset so ultralytics can auto-download models if missing


def _excepthook(exc_type, exc, tb):
    msg = "".join(traceback.format_exception(exc_type, exc, tb))
    _log.write(f"\n[{datetime.datetime.now()}] UNCAUGHT\n{msg}\n")
    try:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "TensorVision AI 错误", f"{msg[-1500:]}\n\n日志: {LOG_PATH}")
    except Exception:
        pass


sys.excepthook = _excepthook


def _t(m):
    _log.write(f"[{datetime.datetime.now():%H:%M:%S}] {m}\n")


def main():
    _t("start")
    multiprocessing.freeze_support()
    from PySide6.QtWidgets import QApplication, QLabel
    from PySide6.QtCore import Qt

    app = QApplication(sys.argv)
    app.setApplicationName("TensorVision AI")

    # Show feedback immediately: importing torch/ultralytics takes several seconds.
    splash = QLabel("TensorVision AI\n\n正在加载 CUDA / AI 引擎，请稍候...")
    splash.setAlignment(Qt.AlignmentFlag.AlignCenter)
    splash.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.WindowStaysOnTopHint)
    splash.setStyleSheet("background:#0b0e14;color:#38bdf8;font-size:18px;padding:40px;")
    splash.show()
    app.processEvents()

    _t("splash shown; importing ui")
    faulthandler.dump_traceback_later(25, repeat=False, file=_log)
    from ui.main_window import MainWindow
    faulthandler.cancel_dump_traceback_later()
    _t("ui imported; building window")
    window = MainWindow()
    _t("window built")
    window.show()
    _t("window shown")
    splash.close()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _excepthook(*sys.exc_info())
        raise
