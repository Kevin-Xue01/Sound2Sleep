import math
import time
import json
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy, QLabel
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QPen, QFont, QColor

def dummy_connect_to_device(callback):
    QTimer.singleShot(10000, lambda: callback(True))

def dummy_data_quality_validation():
    return "yellow"

def dummy_sleep_staging_algorithm():
    pass

class StatusWidget(QWidget):
    def __init__(self, status="green", parent=None):
        super().__init__(parent)
        self.status = status
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.bullet = QLabel()
        self.bullet.setFixedSize(15, 15)
        self.bullet.setStyleSheet(f"background-color: {self.get_color()}; border-radius: 7px;")

        self.text = QLabel(self.get_text())
        self.text.setFont(QFont("Arial", 16))
        layout.addWidget(self.bullet)
        layout.addWidget(self.text)
        self.setLayout(layout)

    def get_color(self):
        if self.status == "green":
            return "green"
        elif self.status == "yellow":
            return "yellow"
        elif self.status == "red":
            return "red"
        else:
            return "gray"

    def get_text(self):
        if self.status == "green":
            return "Ready"
        elif self.status == "yellow":
            return "Wet Device"
        elif self.status == "red":
            return "Fix Headband"
        else:
            return ""

    def setStatus(self, status):
        self.status = status
        self.bullet.setStyleSheet(f"background-color: {self.get_color()}; border-radius: 7px;")
        self.text.setText(self.get_text())

class ConnectionLoadingScreen(QWidget):
    def __init__(self, duration=10, parent=None):
        super().__init__(parent)
        print("ConnectionLoadingScreen initialized")
        self.duration = duration  
        self.start_time = time.time()
        self.progress = 0 
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(50)  
        QTimer.singleShot(duration * 1000, self.connection_complete)
        self.on_complete_callback = None
        self.setFixedSize(800, 600)  

    def update_progress(self):
        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            elapsed = self.duration
        self.progress = (elapsed / self.duration) * 360
        self.update()

    def connection_complete(self):
        if self.on_complete_callback:
            self.on_complete_callback()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        center = rect.center()
        radius = min(rect.width(), rect.height()) // 4
        pen = QPen(Qt.white, 4)
        painter.setPen(pen)
        painter.drawEllipse(center, radius, radius)
        pen = QPen(Qt.green, 4)
        painter.setPen(pen)
        startAngle = 90 * 16  
        spanAngle = -int(self.progress * 16) 
        painter.drawArc(center.x()-radius, center.y()-radius, radius*2, radius*2, startAngle, spanAngle)
        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 14))
        text = "Connecting to headband"
        text_rect = painter.fontMetrics().boundingRect(text)
        text_rect.moveCenter(center)
        painter.drawText(text_rect, Qt.AlignCenter, text)

class EEGSimulationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase = 0
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_phase)
        self.timer.start(50) 

    def update_phase(self):
        self.phase += 0.1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(Qt.cyan, 2)
        painter.setPen(pen)
        width = self.width()
        height = self.height()
        mid_y = height // 2
        points = []
        for x in range(0, width, 2):
            y = mid_y + 50 * math.sin((x / width * 4 * math.pi) + self.phase)
            points.append((x, int(y)))
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])

class HeadbandConnectionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.loading_screen = ConnectionLoadingScreen(duration=10)
        dummy_connect_to_device(self.handle_device_connected)
        self.main_layout.addWidget(self.loading_screen, alignment=Qt.AlignCenter)
        self.setLayout(self.main_layout)

    def handle_device_connected(self, success):
        if success:
            self.start_data_quality_monitor()
        else:
            pass

    def start_data_quality_monitor(self):
        self.quality_timer = QTimer(self)
        self.quality_timer.timeout.connect(self.check_quality)
        self.quality_timer.start(1000) 

    def check_quality(self):
        try:
            with open("data_quality.json", "r") as f:
                data = json.load(f)
            quality = data.get("quality", "yellow")
        except Exception:
            print("Error reading data_quality.json")
            quality = "yellow"

        if hasattr(self, "status_widget"):
            self.status_widget.setStatus(quality)
            if quality == "green":
                self.sleep_button.setEnabled(True)
                self.sleep_button.setStyleSheet("background-color: #3A1D92; color: white; padding: 10px; border-radius: 10px;")
            else:
                self.sleep_button.setEnabled(False)
                self.sleep_button.setStyleSheet("background-color: gray; color: white; padding: 10px; border-radius: 10px;")
        else:
            self.switch_to_eeg_simulation(quality)

    def switch_to_eeg_simulation(self, quality):
        print("Switching to EEG simulation")
        self.main_layout.removeWidget(self.loading_screen)
        self.loading_screen.deleteLater()
        self.eeg_simulation = EEGSimulationWidget()
        self.main_layout.addWidget(self.eeg_simulation)
        self.main_layout.setStretchFactor(self.eeg_simulation, 3)

        control_container = QWidget()
        control_layout = QVBoxLayout(control_container)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(10)

        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(5)
        status_layout.addStretch(1)

        self.status_widget = StatusWidget(status=quality)
        status_layout.addWidget(self.status_widget, alignment=Qt.AlignCenter)
        status_layout.addStretch(1)
        control_layout.addLayout(status_layout)

        self.sleep_button = QPushButton("Start Sleep Algorithm")
        self.sleep_button.setFont(QFont("Arial", 16))

        if quality == "green":
            self.sleep_button.setEnabled(True)
            self.sleep_button.setStyleSheet("background-color: #3A1D92; color: white; padding: 10px; border-radius: 10px;")
        else:
            self.sleep_button.setEnabled(False)
            self.sleep_button.setStyleSheet("background-color: gray; color: white; padding: 10px; border-radius: 10px;")
        self.sleep_button.setMaximumWidth(400)
        self.sleep_button.clicked.connect(self.start_sleep_algorithm)

        control_layout.addWidget(self.sleep_button, alignment=Qt.AlignHCenter | Qt.AlignTop)

        self.main_layout.addWidget(control_container)

        self.main_layout.setStretchFactor(control_container, 1)

    def start_sleep_algorithm(self):
        dummy_sleep_staging_algorithm()
        self.sleep_button.setText("I'm Awake!")
        self.sleep_button.setStyleSheet("background-color: red; color: white; padding: 10px; border-radius: 10px;")
        if hasattr(self, "quality_timer"):
            self.quality_timer.stop()
        try:
            self.sleep_button.clicked.disconnect()
        except Exception:
            pass
        self.sleep_button.clicked.connect(lambda: self.window().stacked_widget.setCurrentWidget(self.window().mood_page))