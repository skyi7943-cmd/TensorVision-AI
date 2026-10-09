import os
import sys
import subprocess
import shutil

APP_NAME = "TensorVisionAI"
MAIN_SCRIPT = "main.py"
ICON_FILE = "app_icon.ico"

def build():
    print("=" * 60)
    print(f"  Starting Build for {APP_NAME} (PyInstaller .exe packaging)")
    print("=" * 60)

    # 1. Ensure icon exists
    if not os.path.exists(ICON_FILE):
        print("Generating icon...")
        import create_icon
        create_icon.create_app_icon(ICON_FILE)

    # 2. Prepare PyInstaller command
    # Using --onedir for instant startup and seamless CUDA DLL loading
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed", # No black console window
        "--name", APP_NAME,
        "--icon", ICON_FILE,
        # Collect data for ultralytics and PySide6
        "--collect-all", "ultralytics",
        "--collect-all", "cv2",
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
        "--hidden-import", "PySide6",
        "--hidden-import", "PySide6.QtCore",
        "--hidden-import", "PySide6.QtGui",
        "--hidden-import", "PySide6.QtWidgets",
        "--hidden-import", "models.engine",
        "--hidden-import", "core.video_stream",
        "--hidden-import", "core.video_exporter",
        "--hidden-import", "core.visualizer",
        "--hidden-import", "ui.main_window",
        "--hidden-import", "ui.video_widget",
        "--hidden-import", "ui.components",
        # Include project folders
        "--add-data", f"weights{os.pathsep}weights",
        "--add-data", f"ui{os.pathsep}ui",
        "--add-data", f"core{os.pathsep}core",
        "--add-data", f"models{os.pathsep}models",
        MAIN_SCRIPT
    ]

    print("Running command:")
    print(" ".join(cmd))
    print("-" * 60)

    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"Error: Build failed with exit code {res.returncode}")
        return False

    print("\n" + "=" * 60)
    print(f"Build completed successfully!")
    dist_dir = os.path.abspath(os.path.join("dist", APP_NAME))
    exe_path = os.path.join(dist_dir, f"{APP_NAME}.exe")
    print(f"Executable is located at:\n{exe_path}")
    print("=" * 60)
    return True

if __name__ == "__main__":
    build()
