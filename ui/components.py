from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QCheckBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont


class StatCard(QFrame):
    """Modern dark dashboard metric card."""
    def __init__(self, title="Metric", value="0", unit="", accent_color="#00d2ff", parent=None):
        super().__init__(parent)
        self.accent_color = accent_color
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #161b26;
                border: 1px solid #232b3e;
                border-radius: 8px;
                padding: 4px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet("color: #7d8b9f; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;")
        
        val_layout = QHBoxLayout()
        val_layout.setSpacing(4)
        self.val_lbl = QLabel(value)
        self.val_lbl.setStyleSheet(f"color: {accent_color}; font-size: 20px; font-weight: bold;")
        
        self.unit_lbl = QLabel(unit)
        self.unit_lbl.setStyleSheet("color: #9aa7bc; font-size: 11px; margin-top: 6px;")
        
        val_layout.addWidget(self.val_lbl)
        val_layout.addWidget(self.unit_lbl)
        val_layout.addStretch()

        layout.addWidget(self.title_lbl)
        layout.addLayout(val_layout)

    def set_value(self, value, unit=None):
        self.val_lbl.setText(str(value))
        if unit is not None:
            self.unit_lbl.setText(unit)


class LabeledSlider(QFrame):
    """Custom slider with current value readout."""
    valueChanged = Signal(float)

    def __init__(self, label_text, min_val=0.1, max_val=1.0, default_val=0.35, step=0.05, parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.factor = 100.0

        self.setStyleSheet("background: transparent; border: none;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self.lbl_title = QLabel(label_text)
        self.lbl_title.setStyleSheet("color: #cfd7e6; font-size: 12px; font-weight: 500;")
        
        self.lbl_val = QLabel(f"{default_val:.2f}")
        self.lbl_val.setStyleSheet("color: #00d2ff; font-size: 12px; font-weight: bold;")
        
        header.addWidget(self.lbl_title)
        header.addStretch()
        header.addWidget(self.lbl_val)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_val * self.factor))
        self.slider.setMaximum(int(max_val * self.factor))
        self.slider.setValue(int(default_val * self.factor))
        self.slider.setSingleStep(int(step * self.factor))

        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #1f2737;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0077ff, stop:1 #00d2ff);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 2px solid #00d2ff;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #00d2ff;
            }
        """)

        self.slider.valueChanged.connect(self._on_change)

        layout.addLayout(header)
        layout.addWidget(self.slider)

    def _on_change(self, int_val):
        real_val = int_val / self.factor
        self.lbl_val.setText(f"{real_val:.2f}")
        self.valueChanged.emit(real_val)

    def value(self):
        return self.slider.value() / self.factor
