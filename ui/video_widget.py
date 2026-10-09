import cv2
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QImage, QColor, QFont, QPen
from PySide6.QtCore import Qt, QRect


class VideoDisplayWidget(QWidget):
    """
    High-performance video rendering canvas preserving aspect ratio.
    Displays modern cyberpunk placeholder when idle.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setStyleSheet("background-color: #0c0f17; border-radius: 8px;")
        self.current_qimage = None
        self.status_message = "等待视频源输入 / 请选择摄像头或载入本地视频"

    def set_frame(self, frame_bgr):
        if frame_bgr is None:
            return
        
        # Convert BGR to RGB for Qt
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        self.current_qimage = QImage(
            rgb_frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        ).copy() # copy to avoid memory overlap
        
        self.update()

    def clear(self, message="等待视频源输入 / 请选择摄像头或载入本地视频"):
        self.current_qimage = None
        self.status_message = message
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Draw deep dark canvas background
        painter.fillRect(self.rect(), QColor("#0c0f17"))

        if self.current_qimage is not None and not self.current_qimage.isNull():
            # Calculate aspect ratio scaling
            widget_w = self.width()
            widget_h = self.height()
            img_w = self.current_qimage.width()
            img_h = self.current_qimage.height()

            scale = min(widget_w / img_w, widget_h / img_h)
            target_w = int(img_w * scale)
            target_h = int(img_h * scale)

            target_x = (widget_w - target_w) // 2
            target_y = (widget_h - target_h) // 2

            target_rect = QRect(target_x, target_y, target_w, target_h)
            painter.drawImage(target_rect, self.current_qimage)
            
            # Subtle accent border around active video area
            painter.setPen(QPen(QColor(0, 210, 255, 60), 1))
            painter.drawRect(target_rect)
        else:
            # Idle placeholder with grid & cyber reticle
            w = self.width()
            h = self.height()
            cx, cy = w // 2, h // 2

            # Grid lines
            painter.setPen(QPen(QColor(30, 40, 60, 100), 1, Qt.PenStyle.DashLine))
            painter.drawLine(0, cy, w, cy)
            painter.drawLine(cx, 0, cx, h)

            # Central reticle
            painter.setPen(QPen(QColor(0, 210, 255, 120), 2))
            painter.drawEllipse(cx - 40, cy - 40, 80, 80)
            painter.drawEllipse(cx - 20, cy - 20, 40, 40)
            painter.drawLine(cx - 50, cy, cx + 50, cy)
            painter.drawLine(cx, cy - 50, cx, cy + 50)

            # Text
            painter.setPen(QColor("#70809b"))
            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
            painter.drawText(self.rect().adjusted(0, 120, 0, 0), Qt.AlignmentFlag.AlignCenter, self.status_message)
