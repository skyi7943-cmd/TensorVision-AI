import os
import time
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QFileDialog, QGroupBox, QRadioButton,
    QButtonGroup, QCheckBox, QSlider, QProgressBar, QMessageBox,
    QTabWidget, QSplitter, QFrame
)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QIcon, QFont

from ui.video_widget import VideoDisplayWidget
from ui.components import StatCard, LabeledSlider
from core.video_stream import VideoCaptureThread
from core.video_exporter import VideoExportThread
from core.inference_worker import InferenceWorker
from core.visualizer import VisionVisualizer
from models.engine import VisionEngine


DARK_THEME_QSS = """
QMainWindow {
    background-color: #0b0e14;
}
QWidget {
    color: #e2e8f0;
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #1f2737;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 14px;
    font-weight: bold;
    color: #90cdf4;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
    background-color: #0b0e14;
}
QTabWidget::pane {
    border: 1px solid #1f2737;
    border-radius: 8px;
    background: #111622;
    top: -1px;
}
QTabBar::tab {
    background: #161b26;
    color: #94a3b8;
    padding: 8px 16px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #1e293b;
    color: #38bdf8;
    font-weight: bold;
    border-bottom: 2px solid #38bdf8;
}
QPushButton {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 14px;
    color: #f8fafc;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #2b3952;
    border-color: #38bdf8;
}
QPushButton:pressed {
    background-color: #0f172a;
}
QPushButton#primaryBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0ea5e9);
    border: none;
    color: #ffffff;
}
QPushButton#primaryBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0369a1, stop:1 #0284c7);
}
QPushButton#stopBtn {
    background-color: #881337;
    border-color: #e11d48;
    color: #ffe4e6;
}
QPushButton#stopBtn:hover {
    background-color: #9f1239;
}
QComboBox {
    background-color: #161b26;
    border: 1px solid #283347;
    border-radius: 6px;
    padding: 6px 10px;
    color: #f1f5f9;
}
QComboBox:hover {
    border-color: #38bdf8;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #161b26;
    border: 1px solid #334155;
    selection-background-color: #0284c7;
    color: #f8fafc;
}
QCheckBox {
    spacing: 8px;
    color: #cbd5e1;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #3b4d66;
    background-color: #161b26;
}
QCheckBox::indicator:checked {
    background-color: #0ea5e9;
    border-color: #38bdf8;
}
QRadioButton {
    spacing: 8px;
    color: #cbd5e1;
}
QRadioButton::indicator {
    width: 15px;
    height: 15px;
    border-radius: 8px;
    border: 1px solid #3b4d66;
    background-color: #161b26;
}
QRadioButton::indicator:checked {
    background-color: #0ea5e9;
    border-color: #38bdf8;
}
QProgressBar {
    border: 1px solid #283347;
    border-radius: 6px;
    background-color: #161b26;
    text-align: center;
    color: #ffffff;
    font-size: 11px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #10b981);
    border-radius: 5px;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensorVision AI - 实时智能视觉识别系统 (CUDA & TensorCore 加速)")
        self.resize(1340, 850)
        self.setStyleSheet(DARK_THEME_QSS)

        # Core Engines
        self.engine = VisionEngine()
        self.visualizer = VisionVisualizer()
        self.capture_thread = VideoCaptureThread()
        self.export_thread = None
        self.worker = InferenceWorker(self.engine, self.visualizer, self._params)

        # FPS calculation
        self.fps_frame_counter = 0
        self.fps_last_time = time.perf_counter()
        self.current_fps = 0.0

        # State
        self.current_task = "detect"
        self.loaded_video_path = None
        self.is_video_playing = False

        self._init_ui()
        self._connect_signals()
        self._update_hardware_banner()
        self.worker.start()

    def _params(self):
        return self.current_task, self.slider_conf.value(), self.slider_iou.value()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(10)

        # --- Top Header ---
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #111622; border-radius: 8px; border: 1px solid #1f2737;")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(16, 10, 16, 10)

        # Title
        title_box = QVBoxLayout()
        title_lbl = QLabel("TensorVision AI 实时智能视觉系统")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #f8fafc;")
        sub_lbl = QLabel("多目标识别 · 人体骨骼姿态估计 · 车辆专属识别 · 离线视频分析")
        sub_lbl.setStyleSheet("font-size: 12px; color: #94a3b8;")
        title_box.addWidget(title_lbl)
        title_box.addWidget(sub_lbl)
        header_layout.addLayout(title_box)

        header_layout.addStretch()

        # Hardware acceleration badge
        self.hw_badge = QLabel("正在检测显卡与 Tensor Core...")
        self.hw_badge.setStyleSheet("""
            background-color: #1e293b;
            border: 1px solid #38bdf8;
            border-radius: 6px;
            padding: 6px 12px;
            color: #38bdf8;
            font-weight: bold;
            font-size: 12px;
        """)
        header_layout.addWidget(self.hw_badge)

        main_layout.addWidget(header_frame)

        # --- Main Body Splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #1f2737; width: 2px; }")

        # Left / Center Area: Video Display & Playback Controls
        video_area = QWidget()
        video_layout = QVBoxLayout(video_area)
        video_layout.setContentsMargins(0, 0, 8, 0)
        video_layout.setSpacing(8)

        self.display_widget = VideoDisplayWidget()
        video_layout.addWidget(self.display_widget, stretch=1)

        # Video Player Controls (for offline video)
        self.playback_bar = QFrame()
        self.playback_bar.setStyleSheet("background-color: #111622; border-radius: 8px; border: 1px solid #1f2737;")
        pb_layout = QHBoxLayout(self.playback_bar)
        pb_layout.setContentsMargins(12, 6, 12, 6)
        pb_layout.setSpacing(10)

        self.btn_play_pause = QPushButton("播放")
        self.btn_play_pause.setFixedWidth(75)
        self.btn_play_pause.clicked.connect(self._toggle_playback)

        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setRange(0, 100)
        self.video_slider.setEnabled(False)
        self.video_slider.sliderMoved.connect(self._on_slider_moved)

        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("color: #94a3b8; font-size: 11px;")

        self.chk_loop = QCheckBox("循环")
        self.chk_loop.setChecked(True)
        self.chk_loop.toggled.connect(lambda v: setattr(self.capture_thread, 'loop', v))

        self.btn_export = QPushButton("导出识别视频")
        self.btn_export.setStyleSheet("background-color: #065f46; border-color: #10b981; color: #a7f3d0;")
        self.btn_export.clicked.connect(self._on_export_clicked)

        pb_layout.addWidget(self.btn_play_pause)
        pb_layout.addWidget(self.video_slider, stretch=1)
        pb_layout.addWidget(self.lbl_time)
        pb_layout.addWidget(self.chk_loop)
        pb_layout.addWidget(self.btn_export)

        video_layout.addWidget(self.playback_bar)

        # Export progress bar (hidden by default)
        self.export_progress_box = QFrame()
        self.export_progress_box.setStyleSheet("background-color: #111622; border-radius: 8px; border: 1px solid #1f2737;")
        self.export_progress_box.setVisible(False)
        ep_layout = QHBoxLayout(self.export_progress_box)
        ep_layout.setContentsMargins(12, 6, 12, 6)

        self.lbl_export_status = QLabel("正在导出视频...")
        self.lbl_export_status.setStyleSheet("color: #38bdf8; font-size: 12px;")
        self.export_bar = QProgressBar()
        self.export_bar.setRange(0, 100)
        self.btn_cancel_export = QPushButton("取消")
        self.btn_cancel_export.setStyleSheet("background-color: #991b1b; color: #fee2e2;")
        self.btn_cancel_export.clicked.connect(self._cancel_export)

        ep_layout.addWidget(self.lbl_export_status)
        ep_layout.addWidget(self.export_bar, stretch=1)
        ep_layout.addWidget(self.btn_cancel_export)

        video_layout.addWidget(self.export_progress_box)

        # Bottom Metric Cards
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(8)
        self.card_fps = StatCard("实时帧率 (FPS)", "0.0", "fps", "#10b981")
        self.card_lat = StatCard("推理延迟 (Latency)", "0.0", "ms", "#38bdf8")
        self.card_total = StatCard("目标总数 (Total)", "0", "个", "#f59e0b")
        self.card_person = StatCard("人体/骨骼 (Skeleton)", "0", "人", "#a855f7")
        self.card_vehicle = StatCard("车辆总计 (Vehicle)", "0", "辆", "#ec4899")

        stats_layout.addWidget(self.card_fps)
        stats_layout.addWidget(self.card_lat)
        stats_layout.addWidget(self.card_total)
        stats_layout.addWidget(self.card_person)
        stats_layout.addWidget(self.card_vehicle)

        video_layout.addLayout(stats_layout)
        splitter.addWidget(video_area)

        # Right Area: Control Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(360)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 0, 0, 0)
        sidebar_layout.setSpacing(10)

        # 1. Source Tab (Webcam vs Video File)
        source_tabs = QTabWidget()
        
        # Tab Webcam
        tab_cam = QWidget()
        tab_cam_layout = QVBoxLayout(tab_cam)
        cam_row = QHBoxLayout()
        cam_lbl = QLabel("摄像头索引:")
        self.combo_cam = QComboBox()
        self.combo_cam.addItems(["摄像头 0 (默认内置/外接)", "摄像头 1", "摄像头 2"])
        cam_row.addWidget(cam_lbl)
        cam_row.addWidget(self.combo_cam, stretch=1)
        tab_cam_layout.addLayout(cam_row)

        self.btn_cam_toggle = QPushButton("开启摄像头实时识别")
        self.btn_cam_toggle.setObjectName("primaryBtn")
        self.btn_cam_toggle.clicked.connect(self._toggle_camera)
        tab_cam_layout.addWidget(self.btn_cam_toggle)
        tab_cam_layout.addStretch()

        source_tabs.addTab(tab_cam, "📷 实时摄像头")

        # Tab Video File
        tab_video = QWidget()
        tab_video_layout = QVBoxLayout(tab_video)
        self.btn_browse_video = QPushButton("📂 选择本地视频文件...")
        self.btn_browse_video.setObjectName("primaryBtn")
        self.btn_browse_video.clicked.connect(self._browse_video)
        
        self.lbl_selected_file = QLabel("未选择视频 (支持 MP4, AVI, MKV, MOV)")
        self.lbl_selected_file.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_selected_file.setWordWrap(True)

        tab_video_layout.addWidget(self.btn_browse_video)
        tab_video_layout.addWidget(self.lbl_selected_file)
        tab_video_layout.addStretch()

        source_tabs.addTab(tab_video, "🎬 本地离线视频")
        sidebar_layout.addWidget(source_tabs)

        # 2. AI Recognition Tasks
        group_tasks = QGroupBox("AI 识别核心功能")
        tasks_layout = QVBoxLayout(group_tasks)
        tasks_layout.setSpacing(8)

        self.task_group = QButtonGroup(self)
        self.rb_detect = QRadioButton("🎯 实时多目标识别 (COCO 80类)")
        self.rb_pose = QRadioButton("🦴 人类骨骼姿态识别 (17关键点)")
        self.rb_vehicle = QRadioButton("🚗 汽车/机动车专属识别 (高亮+统计)")
        self.rb_fusion = QRadioButton("⚡ 双模型协同 (目标框 + 人体骨骼)")
        self.rb_road = QRadioButton("🛣️ 全景智驾感知 (YOLOPv2 车+路面+车道线)")

        self.rb_detect.setChecked(True)
        self.task_group.addButton(self.rb_detect, 0)
        self.task_group.addButton(self.rb_pose, 1)
        self.task_group.addButton(self.rb_vehicle, 2)
        self.task_group.addButton(self.rb_fusion, 3)
        self.task_group.addButton(self.rb_road, 4)

        self.task_group.idClicked.connect(self._on_task_changed)

        tasks_layout.addWidget(self.rb_detect)
        tasks_layout.addWidget(self.rb_pose)
        tasks_layout.addWidget(self.rb_vehicle)
        tasks_layout.addWidget(self.rb_fusion)
        tasks_layout.addWidget(self.rb_road)
        sidebar_layout.addWidget(group_tasks)

        # 3. GPU Hardware & Tensor Core Acceleration
        group_accel = QGroupBox("显卡 CUDA & Tensor Core 加速引擎")
        accel_layout = QVBoxLayout(group_accel)
        accel_layout.setSpacing(8)

        self.chk_cuda = QCheckBox("启用 NVIDIA CUDA 硬件加速")
        self.chk_cuda.setChecked(True)
        self.chk_cuda.toggled.connect(self._on_device_toggle)

        self.chk_tensor_core = QCheckBox("启用 4th Gen Tensor Core 半精度 (FP16)")
        self.chk_tensor_core.setChecked(True)
        self.chk_tensor_core.toggled.connect(self._on_fp16_toggle)

        model_row = QHBoxLayout()
        model_lbl = QLabel("模型规模:")
        self.combo_model_size = QComboBox()
        self.combo_model_size.addItems([
            "Nano (极速高帧率 ~180FPS)",
            "Small (高精度标准版 ~120FPS)",
            "Medium (远距离增强推荐 ~80FPS)",
            "X-Large (旗舰极致远距离 ~45FPS)"
        ])
        self.combo_model_size.currentIndexChanged.connect(self._on_model_size_changed)
        model_row.addWidget(model_lbl)
        model_row.addWidget(self.combo_model_size, stretch=1)

        res_row = QHBoxLayout()
        res_lbl = QLabel("推理分辨率:")
        self.combo_resolution = QComboBox()
        self.combo_resolution.addItems([
            "640x640 (标准/近中距离)",
            "960x960 (清晰/中远距离增强)",
            "1280x1280 (高清/超远距离极限侦测)"
        ])
        self.combo_resolution.currentIndexChanged.connect(self._on_resolution_changed)
        res_row.addWidget(res_lbl)
        res_row.addWidget(self.combo_resolution, stretch=1)

        accel_layout.addWidget(self.chk_cuda)
        accel_layout.addWidget(self.chk_tensor_core)
        accel_layout.addLayout(model_row)
        accel_layout.addLayout(res_row)
        sidebar_layout.addWidget(group_accel)

        # 4. Parameters
        group_params = QGroupBox("检测精度调节与远距离强化")
        params_layout = QVBoxLayout(group_params)
        
        self.chk_distant_boost = QCheckBox("🔭 一键开启远距离骨骼/小目标强化")
        self.chk_distant_boost.setStyleSheet("color: #38bdf8; font-weight: bold; margin-bottom: 4px;")
        self.chk_distant_boost.toggled.connect(self._on_distant_boost_toggled)

        self.slider_conf = LabeledSlider("置信度阈值 (Confidence)", 0.05, 0.95, 0.35, 0.05)
        self.slider_conf.valueChanged.connect(lambda v: setattr(self.engine, 'conf_threshold', v))

        self.slider_kpt = LabeledSlider("骨骼关节灵敏度 (Keypoint Conf)", 0.05, 0.60, 0.20, 0.05)
        self.slider_kpt.valueChanged.connect(lambda v: setattr(self.visualizer, 'kpt_conf_threshold', v))

        self.slider_iou = LabeledSlider("交并比阈值 (IoU / NMS)", 0.1, 0.95, 0.45, 0.05)
        self.slider_iou.valueChanged.connect(lambda v: setattr(self.engine, 'iou_threshold', v))

        params_layout.addWidget(self.chk_distant_boost)
        params_layout.addWidget(self.slider_conf)
        params_layout.addWidget(self.slider_kpt)
        params_layout.addWidget(self.slider_iou)
        sidebar_layout.addWidget(group_params)

        # 5. Visual Toggles
        group_vis = QGroupBox("画面渲染图层控制")
        vis_layout = QVBoxLayout(group_vis)
        self.chk_boxes = QCheckBox("渲染物体识别边界框")
        self.chk_boxes.setChecked(True)
        self.chk_boxes.toggled.connect(lambda v: setattr(self.visualizer, 'show_boxes', v))

        self.chk_labels = QCheckBox("显示类别名称与置信度百分比")
        self.chk_labels.setChecked(True)
        self.chk_labels.toggled.connect(lambda v: setattr(self.visualizer, 'show_labels', v))

        self.chk_skeleton = QCheckBox("渲染骨骼关节连线 (Skeleton Limbs)")
        self.chk_skeleton.setChecked(True)
        self.chk_skeleton.toggled.connect(lambda v: setattr(self.visualizer, 'show_skeleton', v))

        self.chk_drivable = QCheckBox("渲染绿色可行驶路面 (Drivable Road)")
        self.chk_drivable.setChecked(True)
        self.chk_drivable.toggled.connect(lambda v: setattr(self.visualizer, 'show_drivable', v))

        self.chk_lanes = QCheckBox("渲染红色车道分割线 (Lane Lines)")
        self.chk_lanes.setChecked(True)
        self.chk_lanes.toggled.connect(lambda v: setattr(self.visualizer, 'show_lanes', v))

        self.chk_tactical = QCheckBox("🎯 战术火控瞄准锁定 (Tactical Lock)")
        self.chk_tactical.setChecked(True)
        self.chk_tactical.toggled.connect(lambda v: setattr(self.visualizer, 'show_tactical_lock', v))

        self.chk_cyber_profile = QCheckBox("🕵️ 赛博黑客全息身份档案 (Cyber Profiler)")
        self.chk_cyber_profile.setChecked(True)
        self.chk_cyber_profile.toggled.connect(lambda v: setattr(self.visualizer, 'show_cyber_profile', v))

        self.chk_bev_radar = QCheckBox("📡 上帝视角 3D 战术雷达 (BEV Radar)")
        self.chk_bev_radar.setChecked(True)
        self.chk_bev_radar.toggled.connect(lambda v: setattr(self.visualizer, 'show_bev_radar', v))

        self.chk_hud = QCheckBox("渲染画面顶部科技感 HUD 仪表板")
        self.chk_hud.setChecked(True)
        self.chk_hud.toggled.connect(lambda v: setattr(self.visualizer, 'show_hud', v))

        vis_layout.addWidget(self.chk_boxes)
        vis_layout.addWidget(self.chk_labels)
        vis_layout.addWidget(self.chk_skeleton)
        vis_layout.addWidget(self.chk_drivable)
        vis_layout.addWidget(self.chk_lanes)
        vis_layout.addWidget(self.chk_tactical)
        vis_layout.addWidget(self.chk_cyber_profile)
        vis_layout.addWidget(self.chk_bev_radar)
        vis_layout.addWidget(self.chk_hud)
        sidebar_layout.addWidget(group_vis)

        sidebar_layout.addStretch()
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        main_layout.addWidget(splitter)

        # Periodic timer for hardware telemetry
        self.telemetry_timer = QTimer(self)
        self.telemetry_timer.setInterval(1000)
        self.telemetry_timer.timeout.connect(self._update_hardware_banner)
        self.telemetry_timer.start()

    def _connect_signals(self):
        self.capture_thread.raw_frame_ready.connect(self.worker.submit)
        self.worker.result_ready.connect(self._on_result)
        self.worker.status.connect(lambda m: self.display_widget.clear(m))
        self.worker.failed.connect(lambda m: self.statusBar().showMessage(m, 8000))
        self.capture_thread.source_opened.connect(self._on_source_opened)
        self.capture_thread.source_finished.connect(self._on_source_finished)
        self.capture_thread.source_error.connect(self._on_source_error)

    def _update_hardware_banner(self):
        hw = self.engine.get_hardware_info()
        if hw.get("cuda_available"):
            dev_name = hw.get("device_name")
            vram_alloc = hw.get("vram_allocated_mb", 0)
            vram_tot = hw.get("vram_total_mb", 8000)
            tc_status = "4th-Gen Tensor Core FP16 已启用 🚀" if hw.get("tensor_core_active") else "CUDA FP32 运行中"
            self.hw_badge.setText(f"🟢 {dev_name} | {tc_status} | 显存: {int(vram_alloc)}MB / {int(vram_tot)}MB")
            self.hw_badge.setStyleSheet("background-color: #064e3b; border: 1px solid #10b981; color: #6ee7b7; border-radius: 6px; padding: 6px 12px; font-weight: bold;")
        else:
            self.hw_badge.setText("⚠️ 未检测到有效 CUDA 设备 (当前运行于 CPU 模式)")
            self.hw_badge.setStyleSheet("background-color: #451a03; border: 1px solid #f59e0b; color: #fde68a; border-radius: 6px; padding: 6px 12px; font-weight: bold;")

    def _on_task_changed(self, task_id):
        mapping = {0: "detect", 1: "pose", 2: "vehicle", 3: "fusion", 4: "road"}
        self.current_task = mapping.get(task_id, "detect")
        self.engine.current_task = self.current_task
        if self.current_task == "road" and self.engine.road_model is None:
            try:
                self.engine.load_road_model()
            except Exception as e:
                QMessageBox.warning(self, "模型加载", f"加载 YOLOPv2 模型失败: {e}")

    def _on_device_toggle(self, state):
        self.engine.set_device(bool(state))
        self._update_hardware_banner()

    def _on_fp16_toggle(self, state):
        self.engine.set_fp16(bool(state))
        self._update_hardware_banner()

    def _on_model_size_changed(self, index):
        models = [
            ("yolov8n.pt", "yolov8n-pose.pt"),
            ("yolov8s.pt", "yolov8s-pose.pt"),
            ("yolov8m.pt", "yolov8m-pose.pt"),
            ("yolov8x.pt", "yolov8x-pose.pt"),
        ]
        if index < 0 or index >= len(models):
            return
        model_det, model_pose = models[index]
        try:
            self.engine.load_detect_model(model_det)
            self.engine.load_pose_model(model_pose)
        except Exception as e:
            QMessageBox.warning(self, "模型加载", f"切换模型失败: {e}")

    def _on_resolution_changed(self, index):
        res_map = [640, 960, 1280]
        if 0 <= index < len(res_map):
            self.engine.set_imgsz(res_map[index])

    def _on_distant_boost_toggled(self, checked):
        if checked:
            # 开启远距离微小人体增强：
            # 1. 切换至 1280x1280 高清推理分辨率（特征图分辨率提升 2 倍，像素点多 4 倍）
            self.combo_resolution.setCurrentIndex(2)
            # 2. 如果当前是极速版 Nano 模型，自动切换为 Medium 远距离增强模型
            if self.combo_model_size.currentIndex() == 0:
                self.combo_model_size.setCurrentIndex(2)
            # 3. 调低检测和关节阈值，捕捉微小体态特征
            self.slider_conf.setValue(0.20)
            self.slider_kpt.setValue(0.15)
        else:
            self.combo_resolution.setCurrentIndex(0)
            self.slider_conf.setValue(0.35)
            self.slider_kpt.setValue(0.20)

    def _toggle_camera(self):
        if self.capture_thread.isRunning() and self.capture_thread.is_camera:
            self.capture_thread.stop()
            self.btn_cam_toggle.setText("开启摄像头实时识别")
            self.btn_cam_toggle.setObjectName("primaryBtn")
            self.btn_cam_toggle.setStyle(self.btn_cam_toggle.style())
            self.display_widget.clear("摄像头已停止")
        else:
            cam_idx = self.combo_cam.currentIndex()
            self.btn_cam_toggle.setText("停止摄像头")
            self.btn_cam_toggle.setObjectName("stopBtn")
            self.btn_cam_toggle.setStyle(self.btn_cam_toggle.style())
            self.playback_bar.setEnabled(False)
            self.capture_thread.open_camera(cam_idx)

    def _browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择本地视频文件",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv *.flv);;All Files (*)"
        )
        if file_path:
            self.loaded_video_path = file_path
            self.lbl_selected_file.setText(f"已选择: {os.path.basename(file_path)}")
            self.playback_bar.setEnabled(True)
            self.btn_play_pause.setText("暂停")
            self.is_video_playing = True
            self.capture_thread.open_file(file_path)

    def _toggle_playback(self):
        if not self.capture_thread.isRunning() and self.loaded_video_path:
            self.capture_thread.open_file(self.loaded_video_path)
            self.btn_play_pause.setText("暂停")
            self.is_video_playing = True
            return

        is_paused = self.capture_thread.toggle_play()
        self.is_video_playing = not is_paused
        self.btn_play_pause.setText("播放" if is_paused else "暂停")

    def _on_slider_moved(self, position):
        if self.capture_thread.total_frames > 0:
            target_frame = int((position / 100.0) * self.capture_thread.total_frames)
            self.capture_thread.request_seek(target_frame)

    @Slot(object, object)
    def _on_result(self, annotated_frame, results):
        self.current_fps = self.worker.fps
        self.display_widget.set_frame(annotated_frame)

        stats = results.get("stats", {})
        self.card_fps.set_value(f"{self.current_fps:.1f}")
        self.card_lat.set_value(f"{results.get('latency_ms', 0):.1f}")
        self.card_total.set_value(stats.get("total_objects", 0))
        self.card_person.set_value(stats.get("person_count", 0))
        self.card_vehicle.set_value(stats.get("vehicle_count", 0))

        # 6. Update video slider position
        if not self.capture_thread.is_camera and self.capture_thread.total_frames > 0:
            cur = self.capture_thread.current_frame_idx
            tot = self.capture_thread.total_frames
            pct = int((cur / tot) * 100)
            self.video_slider.blockSignals(True)
            self.video_slider.setValue(pct)
            self.video_slider.blockSignals(False)

            # Update timestamp
            fps = self.capture_thread.target_fps or 30.0
            cur_sec = int(cur / fps)
            tot_sec = int(tot / fps)
            self.lbl_time.setText(f"{cur_sec // 60:02d}:{cur_sec % 60:02d} / {tot_sec // 60:02d}:{tot_sec % 60:02d}")

    @Slot(dict)
    def _on_source_opened(self, meta):
        if not meta.get("is_camera"):
            self.video_slider.setEnabled(True)
        else:
            self.video_slider.setEnabled(False)
            self.lbl_time.setText("实时摄像头信号")

    @Slot()
    def _on_source_finished(self):
        self.btn_play_pause.setText("重放")
        self.is_video_playing = False

    @Slot(str)
    def _on_source_error(self, err_msg):
        QMessageBox.critical(self, "视频源错误", f"无法加载视频流: {err_msg}")
        self.btn_cam_toggle.setText("开启摄像头实时识别")
        self.btn_cam_toggle.setObjectName("primaryBtn")
        self.btn_cam_toggle.setStyle(self.btn_cam_toggle.style())

    def _on_export_clicked(self):
        if not self.loaded_video_path or not os.path.exists(self.loaded_video_path):
            QMessageBox.information(self, "导出提示", "请先选择一个本地视频文件再进行导出。")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存已识别渲染的视频",
            os.path.splitext(self.loaded_video_path)[0] + "_AI_Annotated.mp4",
            "MP4 Video (*.mp4)"
        )
        if not out_path:
            return

        # Pause playback
        self.capture_thread.pause()
        self.btn_play_pause.setText("播放")

        self.export_progress_box.setVisible(True)
        self.export_bar.setValue(0)
        self.lbl_export_status.setText("正在初始化导出...")

        self.export_thread = VideoExportThread(
            input_video_path=self.loaded_video_path,
            output_video_path=out_path,
            engine=self.engine,
            visualizer=self.visualizer,
            task=self.current_task,
            conf=self.slider_conf.value(),
            iou=self.slider_iou.value()
        )
        self.export_thread.progress.connect(self._on_export_progress)
        self.export_thread.done.connect(self._on_export_finished)
        self.export_thread.error.connect(self._on_export_error)
        self.export_thread.start()

    def _on_export_progress(self, current, total, pct, eta):
        self.export_bar.setValue(int(pct))
        self.lbl_export_status.setText(f"正在导出: {current}/{total} 帧 ({pct:.1f}%) | 预计剩余: {int(eta)}秒")

    def _on_export_finished(self, out_path):
        self.export_progress_box.setVisible(False)
        QMessageBox.information(self, "导出完成", f"视频已成功识别并保存至:\n{out_path}")

    def _on_export_error(self, err_msg):
        self.export_progress_box.setVisible(False)
        QMessageBox.warning(self, "导出终止", f"{err_msg}")

    def _cancel_export(self):
        if self.export_thread and self.export_thread.isRunning():
            self.export_thread.cancel()

    def closeEvent(self, event):
        self.capture_thread.stop()
        self.worker.shutdown()
        if self.export_thread and self.export_thread.isRunning():
            self.export_thread.cancel()
            self.export_thread.wait(3000)
        event.accept()
