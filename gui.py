import sys
import threading
import time
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QScreen
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class AppState(Enum):
    DISCONNECTED = "Disconnected"
    CONNECTING = "Connecting"
    CONNECTED = "Connected"
    RECORDING = "Recording"
    NOT_RECORDING = "Not Recording"

class BluetoothConnectionThread(QThread):
    connection_established = pyqtSignal()
    connection_failed = pyqtSignal()

    def run(self):
        time.sleep(3)  # Simulating connection delay
        success = np.random.choice([True, False], p=[0.8, 0.2])  # Simulate success rate
        if success:
            self.connection_established.emit()
        else:
            self.connection_failed.emit()

class EEGPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        self.figure, self.axes = plt.subplots(4, 1, figsize=(8, 6))
        self.figure.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95) 
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        self.data = np.zeros((4, 100))  # Placeholder EEG data
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

    def start_plotting(self):
        self.timer.start(100)  # Update every 100ms

    def stop_plotting(self):
        self.timer.stop()

    def update_plot(self):
        self.data = np.roll(self.data, -1, axis=1)  # Simulate incoming data
        self.data[:, -1] = np.random.randn(4)  # Simulated EEG signal
        
        for i, ax in enumerate(self.axes):
            ax.clear()
            ax.plot(self.data[i, :])
            ax.set_ylim(-2, 2)
        
        self.canvas.draw()

class ControlPanel(QWidget):
    def __init__(self, eeg_plot):
        super().__init__()
        self.eeg_plot = eeg_plot
        self.state = AppState.DISCONNECTED
        self.elapsed_time = 0  # Elapsed time in seconds
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        # Connection Panel
        self.connection_label = QLabel("Disconnected")
        self.connection_button = QPushButton("Connect")
        self.connection_button.clicked.connect(self.toggle_connection)
        
        connection_layout = QHBoxLayout()
        connection_layout.addWidget(self.connection_label)
        connection_layout.addWidget(self.connection_button)
        
        # Elapsed Time Panel
        self.elapsed_time_label = QLabel("Elapsed Time: 0s")
        self.elapsed_time_timer = QTimer(self)  # Timer for updating elapsed time
        self.elapsed_time_timer.timeout.connect(self.update_elapsed_time)
        
        # Recording Control
        self.recording_button = QPushButton("Start Recording")
        self.recording_button.setEnabled(False)
        self.recording_button.clicked.connect(self.toggle_recording)
        
        recording_layout = QHBoxLayout()
        recording_layout.addWidget(self.elapsed_time_label)
        recording_layout.addWidget(self.recording_button)
        
        layout.addLayout(connection_layout)
        layout.addLayout(recording_layout)
        self.setLayout(layout)
    
    def toggle_connection(self):
        if self.state == AppState.DISCONNECTED:
            self.connection_button.setEnabled(False)
            self.connection_label.setText("Connecting")
            self.state = AppState.CONNECTING
            self.connection_thread = BluetoothConnectionThread()
            self.connection_thread.connection_established.connect(self.on_connection_success)
            self.connection_thread.connection_failed.connect(self.on_connection_failure)
            self.connection_thread.start()
        elif self.state == AppState.CONNECTED:
            self.state = AppState.DISCONNECTED
            self.connection_label.setText("Disconnected")
            self.connection_button.setText("Connect")
            self.recording_button.setEnabled(False)
            self.recording_button.setText("Start Recording")
            self.elapsed_time_label.setText("Elapsed Time: 0s")
            self.elapsed_time = 0  # Reset elapsed time
    
    def on_connection_success(self):
        self.state = AppState.CONNECTED
        self.connection_label.setText("Connected")
        self.connection_button.setText("Disconnect")
        self.connection_button.setEnabled(True)
        self.recording_button.setEnabled(True)

    def on_connection_failure(self):
        self.state = AppState.DISCONNECTED
        self.connection_label.setText("Disconnected")
        self.connection_button.setText("Connect")
        self.connection_button.setEnabled(True)
    
    def toggle_recording(self):
        if self.state == AppState.CONNECTED:
            self.state = AppState.RECORDING
            self.recording_button.setText("Stop Recording")
            self.eeg_plot.start_plotting()
            self.elapsed_time = 0  # Reset elapsed time on start
            self.elapsed_time_timer.start(1000)  # Update every second
        elif self.state == AppState.RECORDING:
            self.state = AppState.CONNECTED
            self.recording_button.setText("Start Recording")
            self.eeg_plot.stop_plotting()
            self.elapsed_time_timer.stop()  # Stop the timer
            self.elapsed_time_label.setText(f"Elapsed Time: {self.elapsed_time}s")
    
    def update_elapsed_time(self):
        self.elapsed_time += 1
        self.elapsed_time_label.setText(f"Elapsed Time: {self.elapsed_time}s")

            

class ConfigPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QGridLayout()
        
        self.params = {}
        labels = ["Sampling Rate (Hz):", "Filter Low Cut (Hz):", "Filter High Cut (Hz):", "Window Size (s):"]
        
        for i, label in enumerate(labels):
            layout.addWidget(QLabel(label), i, 0)
            entry = QLineEdit()
            layout.addWidget(entry, i, 1)
            self.params[label] = entry
        
        self.setLayout(layout)

class EEGApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        screen = QApplication.primaryScreen().geometry()
        width, height = int(screen.width() * 0.9), int(screen.height() * 0.9)
        self.setGeometry(
            (screen.width() - width) // 2, 
            (screen.height() - height) // 2, 
            width, 
            height
        )

        main_layout = QHBoxLayout()
        right_panel = QVBoxLayout()
        
        self.eeg_plot = EEGPlot()
        self.control_panel = ControlPanel(self.eeg_plot)
        self.config_panel = ConfigPanel()
        
        self.eeg_plot.setFixedWidth(int(width * 0.65))
        
        right_panel.addWidget(self.control_panel)
        right_panel.addWidget(self.config_panel)
        
        main_layout.addWidget(self.eeg_plot)
        main_layout.addLayout(right_panel)
        
        self.setLayout(main_layout)
        self.setWindowTitle("EEG Recording Application")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EEGApp()
    window.show()
    sys.exit(app.exec())
