import time
import os
import sys
import threading
import functools
import numpy as np

import cv2

# PyTorch optimization flags will be set when torch is imported
try:
    import torch
    if torch.cuda.is_available():
        # Enable cuDNN benchmark for auto-tuning convolution algorithms
        torch.backends.cudnn.benchmark = True
        # Enable Tensor Core TF32 & FP16 optimizations on Ampere/Ada Lovelace/Hopper
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
except ImportError:
    torch = None

try:
    import torchvision
except ImportError:
    torchvision = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def _letterbox_yolop(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def _make_grid_yolop(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def _split_for_trace_model(pred, anchor_grid):
    z = []
    st = [8, 16, 32]
    for i in range(3):
        bs, _, ny, nx = pred[i].shape
        p = pred[i].view(bs, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        y = p.sigmoid()
        gr = _make_grid_yolop(nx, ny).to(p.device)
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + gr) * st[i]
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]
        z.append(y.view(bs, -1, 85))
    return torch.cat(z, 1)


def _xywh2xyxy_yolop(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _non_max_suppression_yolop(prediction, conf_thres=0.3, iou_thres=0.45):
    if torchvision is None:
        raise RuntimeError("torchvision required for NMS")
    output = []
    for xi, x in enumerate(prediction):
        xc = x[..., 4] > conf_thres
        x = x[xc]
        if not x.shape[0]:
            output.append(torch.zeros((0, 6), device=prediction.device))
            continue
        x[:, 5:] *= x[:, 4:5]
        box = _xywh2xyxy_yolop(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if not x.shape[0]:
            output.append(torch.zeros((0, 6), device=prediction.device))
            continue
        boxes, scores = x[:, :4], x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        output.append(x[i])
    return output


def _scale_coords_yolop(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords


# COCO vehicle classes mapping
VEHICLE_CLASS_IDS = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    8: "Boat",
    15: "Train"
}

# 17 COCO Pose Keypoint names and pairs
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SKELETON_PAIRS = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Left Arm
    (5, 7), (7, 9),
    # Right Arm
    (6, 8), (8, 10),
    # Left Leg
    (11, 13), (13, 15),
    # Right Leg
    (12, 14), (14, 16)
]


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        with self._lock:
            return fn(self, *a, **k)
    return wrapper


class VisionEngine:
    """
    High-performance AI Vision Engine optimized for NVIDIA CUDA & Tensor Cores.
    Supports Object Detection, Vehicle Identification, and Human Skeleton/Pose.
    """
    def __init__(self, model_dir=None):
        if model_dir is None:
            if getattr(sys, 'frozen', False):
                base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
                model_dir = os.path.join(base_dir, "weights")
            else:
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")
        self._lock = threading.RLock()
        self.model_dir = os.path.abspath(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)

        self.device = "cuda:0" if (torch and torch.cuda.is_available()) else "cpu"
        self.use_fp16 = (self.device.startswith("cuda"))  # FP16 activates 4th Gen Tensor Cores
        
        self.current_task = "detect" # 'detect', 'pose', 'vehicle', 'fusion', 'road'
        self.detect_model = None
        self.pose_model = None
        self.road_model = None
        
        self.conf_threshold = 0.35
        self.iou_threshold = 0.45
        
        # Telemetry metrics
        self.last_latency_ms = 0.0
        self.last_preprocess_ms = 0.0
        self.last_inference_ms = 0.0
        self.last_postprocess_ms = 0.0

    def get_hardware_info(self):
        """Returns details about GPU, CUDA, and Tensor Core capabilities."""
        if not torch or not torch.cuda.is_available():
            return {
                "device_name": "CPU (CUDA Not Detected)",
                "cuda_available": False,
                "tensor_core_active": False,
                "vram_used_mb": 0,
                "vram_total_mb": 0,
                "compute_capability": "N/A"
            }
        
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        vram_alloc = torch.cuda.memory_allocated(0) / (1024 * 1024)
        vram_reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

        # Compute capability >= 7.0 supports Tensor Cores (Volta, Turing, Ampere, Ada Lovelace sm_89, Hopper)
        tensor_core_capable = (capability[0] >= 7)
        tensor_core_active = tensor_core_capable and self.use_fp16 and self.device.startswith("cuda")

        return {
            "device_name": gpu_name,
            "cuda_available": True,
            "tensor_core_capable": tensor_core_capable,
            "tensor_core_active": tensor_core_active,
            "vram_allocated_mb": round(vram_alloc, 1),
            "vram_reserved_mb": round(vram_reserved, 1),
            "vram_total_mb": round(vram_total, 1),
            "compute_capability": f"{capability[0]}.{capability[1]} (Tensor Core Gen {4 if capability[0]>=8 else 3})"
        }

    @_locked
    def set_device(self, use_cuda: bool):
        if use_cuda and torch and torch.cuda.is_available():
            self.device = "cuda:0"
            self.use_fp16 = True
        else:
            self.device = "cpu"
            self.use_fp16 = False
            
        if self.detect_model is not None:
            self.detect_model.to(self.device)
            if self.use_fp16:
                self.detect_model.model.half()
            else:
                self.detect_model.model.float()
        if self.pose_model is not None:
            self.pose_model.to(self.device)
            if self.use_fp16:
                self.pose_model.model.half()
            else:
                self.pose_model.model.float()
        if self.road_model is not None:
            self.road_model.to(self.device)
            if self.use_fp16:
                self.road_model.half()
            else:
                self.road_model.float()

    @_locked
    def set_fp16(self, enable: bool):
        """Toggle FP16 Half precision mode to explicitly control Tensor Core usage."""
        if self.device.startswith("cuda"):
            self.use_fp16 = enable
            if self.detect_model is not None:
                if enable:
                    self.detect_model.model.half()
                else:
                    self.detect_model.model.float()
            if self.pose_model is not None:
                if enable:
                    self.pose_model.model.half()
                else:
                    self.pose_model.model.float()
            if self.road_model is not None:
                if enable:
                    self.road_model.half()
                else:
                    self.road_model.float()
        else:
            self.use_fp16 = False

    @_locked
    def load_detect_model(self, model_name="yolov8n.pt"):
        """Load YOLO detection model with CUDA/TensorCore optimizations."""
        if YOLO is None:
            raise RuntimeError("Ultralytics library not installed.")
        model_path = os.path.join(self.model_dir, model_name)
        self.detect_model = YOLO(model_path)
        self.detect_model.to(self.device)
        if self.use_fp16 and self.device.startswith("cuda"):
            self.detect_model.model.half()
        # Warmup
        self._warmup(self.detect_model)
        return self.detect_model

    @_locked
    def load_pose_model(self, model_name="yolov8n-pose.pt"):
        """Load YOLO pose/skeleton estimation model."""
        if YOLO is None:
            raise RuntimeError("Ultralytics library not installed.")
        model_path = os.path.join(self.model_dir, model_name)
        self.pose_model = YOLO(model_path)
        self.pose_model.to(self.device)
        if self.use_fp16 and self.device.startswith("cuda"):
            self.pose_model.model.half()
        # Warmup
        self._warmup(self.pose_model)
        return self.pose_model

    @_locked
    def load_road_model(self, model_name="yolopv2.pt"):
        """Load YOLOPv2 panoptic driving model (Detection + Road Surface + Lanes) with CUDA/TensorCore."""
        if not torch:
            raise RuntimeError("PyTorch is required for YOLOPv2.")
        model_path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLOPv2 model weights not found at: {model_path}")
        self.road_model = torch.jit.load(model_path)
        self.road_model.to(self.device)
        if self.use_fp16 and self.device.startswith("cuda"):
            self.road_model.half()
        self.road_model.eval()
        self._warmup_road(self.road_model)
        return self.road_model

    def _warmup_road(self, model):
        """Warm up CUDA kernels for YOLOPv2."""
        try:
            dtype = torch.float16 if (self.use_fp16 and self.device.startswith("cuda")) else torch.float32
            dummy = torch.zeros(1, 3, 384, 640, device=self.device, dtype=dtype)
            with torch.no_grad():
                model(dummy)
            if torch and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Road model warmup warning: {e}")

    def _warmup(self, model):
        """Warm up CUDA kernels for smooth initial frame rates."""
        try:
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            model(dummy_img, device=self.device, verbose=False)
            if torch and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Warmup warning: {e}")

    @_locked
    def process_frame(self, frame_bgr, task=None, conf=None, iou=None):
        """
        Process a single BGR frame.
        task options:
            'detect': standard object detection (80 classes)
            'pose': human skeleton and keypoint estimation
            'vehicle': filter and highlight vehicles only
            'fusion': both detection and pose together
        """
        if task is None:
            task = self.current_task
        if conf is None:
            conf = self.conf_threshold
        if iou is None:
            iou = self.iou_threshold

        t0 = time.perf_counter()
        
        detections = []
        poses = []
        stats = {
            "total_objects": 0,
            "person_count": 0,
            "vehicle_count": 0,
            "other_count": 0
        }

        # 1. Human Skeleton / Pose Detection
        if task in ("pose", "fusion"):
            if self.pose_model is None:
                self.load_pose_model()
            
            p_results = self.pose_model(
                frame_bgr,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False
            )[0]

            if p_results.boxes is not None and len(p_results.boxes) > 0:
                boxes_xyxy = p_results.boxes.xyxy.cpu().numpy()
                boxes_conf = p_results.boxes.conf.cpu().numpy()
                
                # Keypoints (N, 17, 3) or (N, 17, 2)
                if p_results.keypoints is not None and p_results.keypoints.data is not None:
                    kpts = p_results.keypoints.data.cpu().numpy()
                else:
                    kpts = [None] * len(boxes_xyxy)

                for i, box in enumerate(boxes_xyxy):
                    kp = kpts[i] if i < len(kpts) else None
                    poses.append({
                        "box": [float(x) for x in box],
                        "conf": float(boxes_conf[i]),
                        "keypoints": kp.tolist() if kp is not None else None
                    })
                    stats["person_count"] += 1

        # 2. General Object Detection / Vehicle Detection
        if task in ("detect", "vehicle", "fusion"):
            if self.detect_model is None:
                self.load_detect_model()

            d_results = self.detect_model(
                frame_bgr,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False
            )[0]

            if d_results.boxes is not None and len(d_results.boxes) > 0:
                boxes_xyxy = d_results.boxes.xyxy.cpu().numpy()
                boxes_conf = d_results.boxes.conf.cpu().numpy()
                boxes_cls = d_results.boxes.cls.cpu().numpy().astype(int)
                names = d_results.names

                for i, box in enumerate(boxes_xyxy):
                    cls_id = int(boxes_cls[i])
                    cls_name = names.get(cls_id, str(cls_id))
                    is_vehicle = (cls_id in VEHICLE_CLASS_IDS)
                    is_person = (cls_id == 0)

                    # If vehicle mode, only keep vehicles
                    if task == "vehicle" and not is_vehicle:
                        continue

                    # If fusion mode, person boxes are already represented by pose
                    if task == "fusion" and is_person:
                        continue

                    if is_vehicle:
                        stats["vehicle_count"] += 1
                    elif is_person:
                        stats["person_count"] += 1
                    else:
                        stats["other_count"] += 1

                    detections.append({
                        "box": [float(x) for x in box],
                        "conf": float(boxes_conf[i]),
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "is_vehicle": is_vehicle
                    })

        # 3. YOLOPv2 Panoptic Driving Perception (Vehicle + Road Surface + Lane Lines)
        da_mask_out = None
        ll_mask_out = None
        if task == "road":
            if self.road_model is None:
                self.load_road_model()

            h0, w0 = frame_bgr.shape[:2]
            img_resized = cv2.resize(frame_bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)
            img_box, ratio, (dw, dh) = _letterbox_yolop(img_resized, 640, auto=True, stride=32)
            blob = np.ascontiguousarray(img_box[:, :, ::-1].transpose(2, 0, 1))

            dtype = torch.float16 if (self.use_fp16 and self.device.startswith("cuda")) else torch.float32
            img_t = torch.from_numpy(blob).to(self.device, dtype=dtype) / 255.0
            if img_t.ndimension() == 3:
                img_t = img_t.unsqueeze(0)

            with torch.no_grad():
                [pred, anchor_grid], seg, ll = self.road_model(img_t)
                pred_boxes = _split_for_trace_model(pred, anchor_grid)
                det = _non_max_suppression_yolop(pred_boxes, conf_thres=conf, iou_thres=iou)[0]

                # Segmentation masks
                da_predict = seg[:, :, 12:372, :]
                da_mask = torch.argmax(da_predict, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                ll_predict = ll[:, :, 12:372, :]
                ll_mask = (torch.sigmoid(ll_predict) > 0.5).squeeze().cpu().numpy().astype(np.uint8)

            da_mask_out = cv2.resize(da_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
            ll_mask_out = cv2.resize(ll_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

            if len(det):
                det[:, :4] = _scale_coords_yolop(img_box.shape[:2], det[:, :4], (720, 1280), (ratio, (dw, dh)))
                sx = w0 / 1280.0
                sy = h0 / 720.0
                det[:, [0, 2]] *= sx
                det[:, [1, 3]] *= sy

                for *xyxy, box_conf, cls_id in det:
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "conf": float(box_conf),
                        "class_id": 2,
                        "class_name": "Vehicle",
                        "is_vehicle": True
                    })
                    stats["vehicle_count"] += 1

        t1 = time.perf_counter()
        self.last_latency_ms = (t1 - t0) * 1000.0
        stats["total_objects"] = len(detections) + len(poses)

        return {
            "detections": detections,
            "poses": poses,
            "da_mask": da_mask_out,
            "ll_mask": ll_mask_out,
            "stats": stats,
            "latency_ms": self.last_latency_ms
        }

