import os
import sys
import urllib.request
from ultralytics import YOLO

def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading: {percent:.1f}% ({mb:.1f} MB / {total_mb:.1f} MB)")
        sys.stdout.flush()

def download():
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # 1. Download YOLOv8 models
    models = ["yolov8n.pt", "yolov8n-pose.pt", "yolov8s.pt", "yolov8s-pose.pt"]
    for m in models:
        target = os.path.join(weights_dir, m)
        if not os.path.exists(target):
            print(f"\n[1/2] Fetching Ultralytics model: {m} ...")
            model = YOLO(m)
            if os.path.exists(m) and m != target:
                import shutil
                shutil.move(m, target)
        print(f"✓ Model ready: {m}")

    # 2. Download YOLOPv2 road & lane model
    yolop_target = os.path.join(weights_dir, "yolopv2.pt")
    if not os.path.exists(yolop_target):
        yolop_url = "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
        print(f"\n[2/2] Fetching Panoptic Driving Model: yolopv2.pt (~150MB) ...")
        print(f"Source: {yolop_url}")
        try:
            urllib.request.urlretrieve(yolop_url, yolop_target, reporthook=reporthook)
            print("\n✓ YOLOPv2 model downloaded successfully!")
        except Exception as e:
            print(f"\n[!] Failed to auto-download YOLOPv2: {e}")
            print(f"Please manually download yolopv2.pt from: {yolop_url}")
            print(f"and save it into: {yolop_target}")
    else:
        print(f"✓ Model ready: yolopv2.pt")

if __name__ == "__main__":
    download()
