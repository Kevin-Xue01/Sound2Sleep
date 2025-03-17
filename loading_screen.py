import os
import sys
import json
import random
import subprocess
from datetime import datetime, timedelta

import mne
import yasa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle, Patch
from mne.datasets.sleep_physionet.age import fetch_data
import io

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QFrame, QStackedWidget
)
from PyQt5.QtGui import QFont, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QPointF


#--------------------- LoadingWidget ---------------------
class LoadingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.progress = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(100)  # update every 100ms (approx 10 sec total)

    def update_progress(self):
        self.progress += 1
        if self.progress >= 100:
            self.progress = 100
            self.timer.stop()
            # Inform parent that loading is complete
            if hasattr(self.parent(), 'loading_finished'):
                self.parent().loading_finished()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        # Determine a square region; subtract 20 for padding
        size = min(rect.width(), rect.height()) - 20
        center = rect.center()
        radius = size // 2  # Use integer division

        # Draw a background circle (gray)
        pen = QPen(Qt.gray, 10)
        painter.setPen(pen)
        painter.drawEllipse(center.x() - radius, center.y() - radius, 2 * radius, 2 * radius)

        # Draw progress arc in white
        pen.setColor(Qt.white)
        painter.setPen(pen)
        # Start at -90 degrees, span is based on progress
        span_angle = int(360 * self.progress / 100)
        painter.drawArc(int(center.x() - radius), int(center.y() - radius),
                        int(2 * radius), int(2 * radius),
                        -90 * 16, -span_angle * 16)

        # Draw percentage text in the center
        painter.setPen(Qt.white)
        font = QFont("Arial", 24, QFont.Bold)
        painter.setFont(font)
        text = f"{self.progress}%"
        painter.drawText(rect, Qt.AlignCenter, text)


# --------------------- LoadingScreen ---------------------
import os
import sys
import json
import random
import subprocess
from datetime import datetime, timedelta

import mne
import yasa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle, Patch
from mne.datasets.sleep_physionet.age import fetch_data
import io

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QFrame, QStackedWidget
)
from PyQt5.QtGui import QFont, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QPointF


#--------------------- LoadingWidget ---------------------
class LoadingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.progress = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(100)  # update every 100ms (approx 10 sec total)

    def update_progress(self):
        self.progress += 1
        if self.progress >= 100:
            self.progress = 100
            self.timer.stop()
            # Inform parent that loading is complete
            if hasattr(self.parent(), 'loading_finished'):
                self.parent().loading_finished()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        # Determine a square region; subtract 20 for padding
        size = min(rect.width(), rect.height()) - 20
        center = rect.center()
        radius = size // 2  # Use integer division

        # Draw a background circle (gray)
        pen = QPen(Qt.gray, 10)
        painter.setPen(pen)
        painter.drawEllipse(center.x() - radius, center.y() - radius, 2 * radius, 2 * radius)

        # Draw progress arc in white
        pen.setColor(Qt.white)
        painter.setPen(pen)
        # Start at -90 degrees, span is based on progress
        span_angle = int(360 * self.progress / 100)
        painter.drawArc(int(center.x() - radius), int(center.y() - radius),
                        int(2 * radius), int(2 * radius),
                        -90 * 16, -span_angle * 16)

        # Draw percentage text in the center
        painter.setPen(Qt.white)
        font = QFont("Arial", 24, QFont.Bold)
        painter.setFont(font)
        text = f"{self.progress}%"
        painter.drawText(rect, Qt.AlignCenter, text)

class LoadingScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #1A0033;")
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Add the "Loading..." label at the top
        label = QLabel("Preparing Sleep Report...")
        label.setFont(QFont("Arial", 25, QFont.Bold))
        label.setStyleSheet("color: white;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        # Add the progress widget at the bottom
        self.loading_widget = LoadingWidget(self)
        layout.addWidget(self.loading_widget)