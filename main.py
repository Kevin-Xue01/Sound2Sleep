import ctypes
import json
import sys
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import psutil
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pydantic import ValidationError
from PyQt5.QtCore import Qt, QThread, QThreadPool, QTimer, pyqtSignal
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

from utils import (  # EEGProcessor,
    CHANNEL_NAMES,
    CHUNK_SIZE,
    SAMPLING_RATE,
    AppState,
    Audio,
    BlueMuse,
    DisplayConfig,
    EEGProcessor,
    ExperimentMode,
    FileWriter,
    Logger,
    MuseDataType,
    SessionConfig,
)


class EEGApp(QWidget):
    pool = QThreadPool.globalInstance()

    def __init__(self):
        super().__init__()
        self.app_state = AppState.DISCONNECTED
        self.elapsed_time = 0  # Elapsed time in seconds

        self.config = SessionConfig()
        self.logger = Logger(self.config._session_key, self.__class__.__name__)
        self.audio = Audio(self.config._audio)

        self.init_ui()

    def init_ui(self):
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
        
        config_panel_widget = QWidget()
        config_panel_layout = QVBoxLayout(config_panel_widget)
        param_config_label = QLabel("Current Parameter Config:")
        config_panel_layout.addWidget(param_config_label)
        self.param_config_editor = QTextEdit()
        self.param_config_editor.setAcceptRichText(False)
        self.param_config_editor.setReadOnly(False)  # Allow editing the JSON
        self.param_config_editor.setText(self.config.model_dump_json())
        config_panel_layout.addWidget(self.param_config_editor)

        save_button = QPushButton("Update Config")
        save_button.clicked.connect(self.save_config)
        config_panel_layout.addWidget(save_button)

        self.config_panel_error_label = QLabel("")
        self.config_panel_error_label.setStyleSheet("color: red;")
        self.config_panel_error_label.hide()
        config_panel_layout.addWidget(self.config_panel_error_label)
        #
        self.window_len_n = int(self.config._display.window_len * SAMPLING_RATE[MuseDataType.EEG])
        self.eeg_timestamps = np.zeros(self.window_len_n)
        self.eeg_data = np.zeros((self.window_len_n, len(CHANNEL_NAMES[MuseDataType.EEG])))
        self.eeg_plot_widget = pg.GraphicsLayoutWidget()
        self.eeg_plot_widget.addPlot(title="EEG Data")
        self.eeg_plot_widget.setYRange(-1500, 1500)
        self.eeg_plot_widget_curves = [self.eeg_plot_widget.plot(pen=pg.mkPen(color)) for color in ['r', 'g', 'b', 'y']]
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_eeg_plot)
        self.timer.start(33)  # ~30 FPS

        # self.eeg_plot = EEGPlot(self.config._display)
        control_panel_widget = QWidget()
        control_panel_layout = QVBoxLayout(control_panel_widget)
        
        self.connection_label = QLabel("Disconnected")
        self.connection_button = QPushButton("Connect")
        self.connection_button.clicked.connect(self.on_toggle_connection_button)
        
        connection_layout = QHBoxLayout()
        connection_layout.addWidget(self.connection_label)
        connection_layout.addWidget(self.connection_button)
        
        self.elapsed_time_label = QLabel("Elapsed Time: 0s")
        def update_elapsed_time():
            self.elapsed_time += 1
            self.elapsed_time_label.setText(f"Elapsed Time: {self.elapsed_time}s")
        self.elapsed_time_timer = QTimer(self)  # Timer for updating elapsed time
        self.elapsed_time_timer.timeout.connect(update_elapsed_time)
        
        # Recording Control
        self.record_button = QPushButton("Start Recording")
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self.on_toggle_record_button)
        
        recording_layout = QHBoxLayout()
        recording_layout.addWidget(self.elapsed_time_label)
        recording_layout.addWidget(self.record_button)

        # Experiment Mode Selection (Horizontally Aligned)
        experiment_layout = QHBoxLayout()
        self.experiment_label = QLabel("Experiment Mode:")
        self.experiment_dropdown = QComboBox()
        self.experiment_dropdown.addItems([mode.value for mode in ExperimentMode])
        self.experiment_dropdown.setCurrentIndex(0)  # Default to first mode (Disabled)
        self.experiment_dropdown.currentIndexChanged.connect(self.update_experiment_mode)

        experiment_layout.addWidget(self.experiment_label)
        experiment_layout.addWidget(self.experiment_dropdown)

        control_panel_layout.addLayout(connection_layout)
        control_panel_layout.addLayout(recording_layout)
        control_panel_layout.addLayout(experiment_layout)

        self.connection_timeout_error_label = QLabel("")
        self.connection_timeout_error_label.setStyleSheet("color: red;")
        self.connection_timeout_error_label.hide()

        right_panel.addWidget(self.connection_timeout_error_label)
        right_panel.addWidget(control_panel_widget)
        right_panel.addWidget(config_panel_widget)
        
        # main_layout.addWidget(self.eeg_plot, stretch=1)
        main_layout.addLayout(right_panel)
        
        self.setLayout(main_layout)
        self.setWindowTitle("Sound2Sleep: CLAS at Home")

    def update_eeg_plot(self):
        for i, curve in enumerate(self.eeg_plot_widget_curves):
            curve.setData(self.eeg_timestamps, self.eeg_data[::2, i])

    def update_experiment_mode(self, index):
        selected_experiment_mode = ExperimentMode(self.experiment_dropdown.itemText(index))
        if self.config.experiment_mode != selected_experiment_mode:
            self.on_config_update(SessionConfig(**self.config.model_dump(), experiment_mode=selected_experiment_mode))
    
    def on_toggle_connection_button(self):
        if self.app_state == AppState.DISCONNECTED:
            self.connection_button.setEnabled(False)
            self.record_button.setEnabled(False)
            self.experiment_dropdown.setEnabled(False)
            self.param_config_editor.setEnabled(False)
            self.connection_label.setText("Connecting")
            self.app_state = AppState.CONNECTING
            self.start_bluemuse()
        elif self.app_state == AppState.CONNECTED:
            self.stop_bluemuse()
        elif self.app_state == AppState.RECORDING:
            reply = QMessageBox.warning(
                self, 'Warning', 
                "You are currently recording data. Disconnecting will stop the recording. Are you sure you want to disconnect?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.stop_bluemuse()

    def on_connected(self):
        self.app_state = AppState.CONNECTED
        self.connection_label.setText("Connected")
        self.connection_button.setText("Disconnect")
        self.connection_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.experiment_dropdown.setEnabled(True)
        self.param_config_editor.setEnabled(True)

    def on_disconnected(self):
        self.app_state = AppState.DISCONNECTED
        self.connection_label.setText("Disconnected")
        self.connection_button.setText("Connect")
        self.record_button.setEnabled(False)
        self.record_button.setText("Start Recording")
        self.elapsed_time_label.setText("Elapsed Time: 0s")
        self.elapsed_time = 0  # Reset elapsed time
    
    def on_toggle_record_button(self):
        if self.app_state == AppState.CONNECTED:
            self.app_state = AppState.RECORDING
            self.record_button.setText("Stop Recording")
            self.elapsed_time = 0  # Reset elapsed time on start
            self.elapsed_time_timer.start(1000)  # Update every second
            self.file_writer = FileWriter(self.config._session_key)
            self.blue_muse.eeg_data_ready.connect(self.file_writer.write_eeg_data)

        elif self.app_state == AppState.RECORDING:
            self.blue_muse.eeg_data_ready.disconnect(self.file_writer.write_eeg_data)
            self.app_state = AppState.CONNECTED
            self.record_button.setText("Start Recording")
            self.elapsed_time_timer.stop()  # Stop the timer
            self.elapsed_time_label.setText(f"Elapsed Time: {self.elapsed_time}s")

    def on_connection_timeout(self):
        self.connection_timeout_error_label.setText("Connection Timeout")
        self.connection_timeout_error_label.show()

    def on_config_update(self, config: SessionConfig):
        self.config = config
        self.logger.update_session_key(self.config._session_key)
        self.file_writer.update_session_key(config._session_key)

        if self.app_state == AppState.CONNECTED or self.app_state == AppState.RECORDING:
            self.stop_bluemuse()
            QTimer.singleShot(3000, self.start_bluemuse)

    def save_config(self):
        """Parse the JSON from the editor and update the config model."""
        self.config_panel_error_label.hide()
        try:
            updated_config = SessionConfig(**json.loads(self.param_config_editor.toPlainText()))
            if updated_config != self.config:
                print("Config updated successfully:", self.config)
                self.config = updated_config
                self.on_config_update()
        except json.JSONDecodeError as e:
            self.config_panel_error_label.setText("Invalid JSON")  # Display the first error message
            self.config_panel_error_label.show()
        except ValidationError as e:
            self.config_panel_error_label.setText(str("\n".join([f'{k}: {v}' for k, v in e.errors()[0].items() if k != "url"])))  # Display the first error message
            self.config_panel_error_label.show()

    def start_bluemuse(self):
        self.blue_muse = BlueMuse(self.config)
        self.eeg_processor = EEGProcessor(self.config)

        self.blue_muse_thread = QThread()
        self.eeg_processor_thread = QThread()

        self.blue_muse.moveToThread(self.blue_muse_thread)
        self.eeg_processor.moveToThread(self.eeg_processor_thread)

        self.blue_muse_thread.started.connect(partial(self.blue_muse.run, self.config._session_key))

        self.blue_muse.connected.connect(self.on_connected)
        self.blue_muse.connected.connect(lambda: self.connection_timeout_error_label.hide())
        self.blue_muse.disconnected.connect(self.on_disconnected)
        self.blue_muse.connection_timeout.connect(self.on_connection_timeout)
        self.blue_muse.eeg_data_ready.connect(self.eeg_processor.process_data)
        # self.blue_muse.eeg_data_ready.connect(self.eeg_plot.update_plot)
        # self.blue_muse.eeg_data_ready.connect(self.eeg_plot.update_data)
        self.blue_muse.eeg_data_ready.connect(self.update_eeg_data)

        # self.eeg_processor.stim.connect(self.eeg_plot.plot_stim)

        self.blue_muse_thread.start()
        self.eeg_processor_thread.start()

    def update_eeg_data(self, timestamps, data):
        self.eeg_timestamps = np.concatenate([self.eeg_timestamps, timestamps])
        self.eeg_timestamps = self.eeg_timestamps[-self.window_len_n:]
        self.eeg_data = np.vstack([self.eeg_data, data])
        self.eeg_data = self.eeg_data[-self.window_len_n:]

    def stop_bluemuse(self):
        if self.blue_muse_thread.isRunning():
            self.blue_muse.stop()
            self.blue_muse_thread.quit()
            self.blue_muse_thread.wait()
            self.blue_muse = None
            self.blue_muse_thread = None

        if self.eeg_processor_thread.isRunning():
            self.eeg_processor.stop()
            self.eeg_processor_thread.quit()
            self.eeg_processor_thread.wait()
            self.eeg_processor = None
            self.eeg_processor_thread = None

    def play_audio(self):
        self.pool.start(self.audio.run)


class EEGPlot(pg.GraphicsLayoutWidget):
    def __init__(self, config: DisplayConfig):
        super().__init__()
        self.config = config
        self.window_len_n = int(self.config.window_len * SAMPLING_RATE[MuseDataType.EEG])
        self.ymin, self.ymax = -3000, 3000

        self.init_ui()

    def init_ui(self):
        self.plot_widget = self.addPlot(title="EEG Data")
        self.plot_widget.setYRange(self.ymin, self.ymax)
        self.curves = [self.plot_widget.plot(pen=pg.mkPen(color)) for color in ['r', 'g', 'b', 'y']]
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(33)  # ~30 FPS

        self.timestamps = np.zeros(self.window_len_n)
        self.data = np.zeros((self.window_len_n, len(CHANNEL_NAMES[MuseDataType.EEG])))

    def update_data(self, timestamps, data):
        """Append new data efficiently"""
        # self.timestamps = np.roll(self.timestamps, -len(timestamps))
        # self.timestamps[-len(timestamps):] = timestamps

        # self.data = np.roll(self.data, -len(timestamps), axis=0)
        # self.data[-len(timestamps):, :] = data
        self.timestamps = np.concatenate([self.timestamps, timestamps])
        self.timestamps = self.timestamps[-self.window_len_n:]
        self.data = np.vstack([self.data, data])
        self.data = self.data[-self.window_len_n:]

    def update_plot(self):
        """Efficiently update plots"""
        for i, curve in enumerate(self.curves):
            curve.setData(self.timestamps, self.data[::2, i])


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        print('Platform: Windows')
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
            for p in psutil.process_iter(['name']):
                if p.info['name'] == 'BlueMuse.exe': p.kill()
    sys.exit(exit_code)