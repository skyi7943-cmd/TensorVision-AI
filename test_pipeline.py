import time
import cv2
from models.engine import VisionEngine
from core.visualizer import VisionVisualizer

def test_pipeline():
    print("=" * 60)
    print("Testing TensorVision AI Pipeline with NVIDIA CUDA & Tensor Cores")
    print("=" * 60)

    engine = VisionEngine()
    hw = engine.get_hardware_info()
    print("Hardware Info:", hw)
    assert hw.get("cuda_available") is True, "CUDA must be available!"

    print("\n--- 1. Testing Object Detection with CUDA & TensorCore FP16 ---")
    engine.load_detect_model("yolov8n.pt")
    
    # Read frame from demo_video.mp4
    cap = cv2.VideoCapture("demo_video.mp4")
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read frame from demo_video.mp4"

    # Warmup and timed inference
    for _ in range(5):
        engine.process_frame(frame, task="detect")

    t0 = time.perf_counter()
    N = 20
    for _ in range(N):
        res_det = engine.process_frame(frame, task="detect")
    elapsed = time.perf_counter() - t0
    avg_ms = (elapsed / N) * 1000
    fps = N / elapsed
    print(f"[Detect] Avg Latency: {avg_ms:.2f} ms | Throughput: {fps:.1f} FPS")

    print("\n--- 2. Testing Human Pose / Skeleton with CUDA & TensorCore FP16 ---")
    engine.load_pose_model("yolov8n-pose.pt")
    for _ in range(5):
        engine.process_frame(frame, task="pose")

    t0 = time.perf_counter()
    for _ in range(N):
        res_pose = engine.process_frame(frame, task="pose")
    elapsed = time.perf_counter() - t0
    avg_ms = (elapsed / N) * 1000
    fps = N / elapsed
    print(f"[Pose] Avg Latency: {avg_ms:.2f} ms | Throughput: {fps:.1f} FPS")

    print("\n--- 3. Testing Vehicle Detection ---")
    res_veh = engine.process_frame(frame, task="vehicle")
    print(f"[Vehicle] Detections count: {len(res_veh['detections'])}")

    print("\n--- 4. Testing Visualizer Rendering ---")
    vis = VisionVisualizer()
    annotated = vis.render(frame, res_det, fps=fps, hw_info=hw)
    cv2.imwrite("test_annotated_output.jpg", annotated)
    print("Saved test rendered output to: test_annotated_output.jpg")

    print("\nPipeline test passed with flying colors! 🚀")

if __name__ == "__main__":
    test_pipeline()
