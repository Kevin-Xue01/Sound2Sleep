import ctypes
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pydantic import BaseModel, ValidationError
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from utils.config import EEGSessionConfig
from utils.constants import AppState, ExperimentMode


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
        self.ymin = -2
        self.ymax = 2

    def initUI(self):
        layout = QVBoxLayout()
        
        self.figure, self.axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
        self.figure.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95) 
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        self.data = np.zeros((4, 100))  # Placeholder EEG data
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(800, 600)
        self.setMaximumWidth(1100)

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
            ax.set_ylim(self.ymin, self.ymax)

            # Calculate variance for the current signal (data row)
            variance = np.var(self.data[i, :])
            
            # Display the variance on the top right of the plot
            ax.text(0.95, 0.95, f"Variance: {variance:.4f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=10, color="red")
        
        self.canvas.draw()

    def keyPressEvent(self, event):
        """Handle key press events to adjust the vertical axis scale."""
        if event.key() == Qt.Key.Key_Plus:  # Increase y-axis range
            self.ymin *= 1.2
            self.ymax *= 1.2
        elif event.key() == Qt.Key.Key_Minus:  # Decrease y-axis range
            self.ymin /= 1.2
            self.ymax /= 1.2
    
    def clear_plots(self):
        """Clears all plots and resets EEG data."""
        self.data = np.zeros((4, 100))  # Reset EEG data
        for ax in self.axes:
            ax.clear()
            ax.set_ylim(-2, 2)  # Keep the y-axis limits consistent
        self.canvas.draw()


class ControlPanel(QWidget):
    def __init__(self, eeg_plot: EEGPlot):
        super().__init__()
        self.save_folder = None
        self.eeg_plot = eeg_plot
        self.state = AppState.DISCONNECTED
        self.elapsed_time = 0  # Elapsed time in seconds
        self.subjects = ["Kevin", "Vicki", "Jaeyoung", "Sean"]
        self.selected_subject = self.subjects[0]  # Default to the first subject
        self.selected_experiment_mode = ExperimentMode.DISABLED  # Default mode

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

        # Save Folder Selection
        self.folder_button = QPushButton("Choose Save Folder")
        self.folder_button.clicked.connect(self.choose_save_folder)
        
        # Subject Selection Horizontal Layout (QLabel + QComboBox)
        subject_layout = QHBoxLayout()
        self.subject_label = QLabel("Select Subject:")
        self.subject_dropdown = QComboBox()
        self.subject_dropdown.addItems(self.subjects)
        self.subject_dropdown.setCurrentIndex(0)  # Default to first subject
        # self.subject_dropdown.currentIndexChanged.connect(self.update_selected_subject)

        # Experiment Mode Selection (Horizontally Aligned)
        experiment_layout = QHBoxLayout()
        self.experiment_label = QLabel("Experiment Mode:")
        self.experiment_dropdown = QComboBox()
        self.experiment_dropdown.addItems([mode.value for mode in ExperimentMode])
        self.experiment_dropdown.setCurrentIndex(0)  # Default to first mode (Disabled)
        self.experiment_dropdown.currentIndexChanged.connect(self.update_experiment_mode)

        experiment_layout.addWidget(self.experiment_label)
        experiment_layout.addWidget(self.experiment_dropdown)

        subject_layout.addWidget(self.subject_label)
        subject_layout.addWidget(self.subject_dropdown)

        layout.addLayout(connection_layout)
        layout.addWidget(self.folder_button)
        layout.addLayout(recording_layout)
        layout.addLayout(subject_layout)
        layout.addLayout(experiment_layout)
        
        self.setLayout(layout)

    def update_experiment_mode(self):
        self.selected_experiment_mode = ExperimentMode(self.experiment_dropdown.currentText())
    
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
        elif self.state == AppState.RECORDING:
            self.show_disconnect_warning()

    def show_disconnect_warning(self):
        print('Test')
        reply = QMessageBox.warning(
            self, 'Warning', 
            "You are currently recording data. Disconnecting will stop the recording. Are you sure you want to disconnect?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            # TODO: save data before disconnecting
            self.state = AppState.DISCONNECTED
            self.eeg_plot.stop_plotting()
            self.elapsed_time_timer.stop()  # Stop the timer
            self.elapsed_time = 0  # Reset elapsed time
            self.connection_label.setText("Disconnected")
            self.connection_button.setText("Connect")
            self.recording_button.setEnabled(False)
            self.recording_button.setText("Start Recording")
            self.elapsed_time_label.setText(f"Elapsed Time: {self.elapsed_time}s")
            self.recording_button.setText("Start Recording")

    def on_connection_success(self):
        self.state = AppState.CONNECTED
        self.connection_label.setText("Connected")
        self.connection_button.setText("Disconnect")
        self.connection_button.setEnabled(True)
        self.recording_button.setEnabled(True)
        self.update_recording_button_state()

    def on_connection_failure(self):
        self.state = AppState.DISCONNECTED
        self.connection_label.setText("Disconnected")
        self.connection_button.setText("Connect")
        self.connection_button.setEnabled(True)
    
    def toggle_recording(self):
        if self.state == AppState.CONNECTED and self.save_folder:
            self.state = AppState.RECORDING
            self.recording_button.setText("Stop Recording")
            self.eeg_plot.clear_plots()
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

    def choose_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.save_folder = folder
            self.folder_button.setText(f"{folder}")
            print(f"Selected Save Folder: {self.save_folder}")
        self.update_recording_button_state()

    def update_recording_button_state(self):
        self.recording_button.setEnabled(self.state == AppState.CONNECTED and bool(self.save_folder))
            

class ConfigPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.config = EEGSessionConfig()
        config_json = self.config.model_dump_json(indent=4)
        self.param_config_editor.setText(config_json)

    def initUI(self):
        layout = QVBoxLayout()
        
        self.params = {}

        self.param_config_label = QLabel("Current Parameter Config:")
        layout.addWidget(self.param_config_label)
        self.param_config_editor = QTextEdit()
        self.param_config_editor.setAcceptRichText(False)
        self.param_config_editor.setReadOnly(False)  # Allow editing the JSON
        layout.addWidget(self.param_config_editor)

        self.save_button = QPushButton("Update Config")
        self.save_button.clicked.connect(self.save_config)
        layout.addWidget(self.save_button)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red;")
        self.error_label.hide()

        layout.addWidget(self.error_label)
        self.setLayout(layout)

    def save_config(self):
        """Parse the JSON from the editor and update the config model."""
        self.error_label.hide()
        try:
            updated_config = EEGSessionConfig(**json.loads(self.param_config_editor.toPlainText()))
            print(self.config._key)
            print(self.config._created_at)
            # print(updated_config.model_dump())
            if updated_config != self.config:
                self.config = updated_config
                print(self.config._key)
                print(self.config._created_at)
                print("Config updated successfully:", self.config)
        except json.JSONDecodeError as e:
            self.error_label.setText("Invalid JSON")  # Display the first error message
            self.error_label.show()
        except ValidationError as e:
            self.error_label.setText(str("\n".join([f'{k}: {v}' for k, v in e.errors()[0].items() if k != "url"])))  # Display the first error message
            self.error_label.show()

class EEGApp(QWidget):
    def __init__(self):
        super().__init__()
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
        
        right_panel.addWidget(self.control_panel)
        right_panel.addWidget(self.config_panel)
        
        main_layout.addWidget(self.eeg_plot, stretch=1)
        main_layout.addLayout(right_panel)
        
        self.setLayout(main_layout)
        self.setWindowTitle("EEG Recording Application")

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x0000040
        
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)

    try:
        app = QApplication(sys.argv)
        window = EEGApp()
        window.show()
        exit_code = app.exec()
    finally:
        if sys.platform.startswith("win"):
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    sys.exit(exit_code)
