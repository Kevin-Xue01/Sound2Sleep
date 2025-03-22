import ctypes
import json
import queue
import random
import subprocess
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timedelta
from functools import partial
from math import ceil, floor, isnan, nan, pi
from multiprocessing import Process, Queue, shared_memory
from threading import Thread, Timer
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
import pyqtgraph as pg
import scipy.signal as signal
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pydantic import ValidationError
from pylsl import StreamInfo, StreamInlet, resolve_byprop
from PyQt5.QtCore import (
    QObject,
    QRunnable,
    Qt,
    QThread,
    QThreadPool,
    QTimer,
    pyqtSignal,
)
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import butter, filtfilt

# from muselsl.constants import LSL_SCAN_TIMEOUT
from utils import (
    CHANNEL_NAMES,
    CHUNK_SIZE,
    DELAYS,
    DISPLAY_WINDOW_LEN_N,
    DISPLAY_WINDOW_LEN_S,
    EEG_PLOTTING_SHARED_MEMORY,
    LSL_SCAN_TIMEOUT,
    NUM_CHANNELS,
    SAMPLING_RATE,
    TIMESTAMPS,
    AppState,
    Audio,
    EEGProcessorOutput,
    ExperimentMode,
    FileWriter,
    Logger,
    MuseDataType,
    SessionConfig,
)

LAG_THRESHOLD = 3 * DELAYS[MuseDataType.EEG]

class DatastreamWorker(QObject):
    results_ready = pyqtSignal(np.ndarray, np.ndarray)  # Signal that will emit results once processing is done
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: SessionConfig, muse_data_type: MuseDataType):
        super().__init__()
        self.config = config
        self.muse_data_type = muse_data_type
        self.running = False
        self.parent_app = None

    def set_app(self, app: 'EEGApp'):
        """Set reference to main app to access necessary data."""
        self.parent_app = app

    def set_config(self, config: SessionConfig):
        """Set the session configuration."""
        self.config = config

    def stop(self):
        """Stops the process."""
        self.running = False
    
    def run(self):
        self.running = True
        no_data_counter = 0
        while self.running:
            time.sleep(DELAYS[self.muse_data_type])

            try:
                data, timestamps = self.parent_app.stream_inlet[self.muse_data_type].pull_chunk(timeout=DELAYS[self.muse_data_type], max_samples=CHUNK_SIZE[self.muse_data_type])
                if timestamps and len(timestamps) == CHUNK_SIZE[self.muse_data_type]:
                    timestamps = TIMESTAMPS[self.muse_data_type] + np.float64(time.time())
                    self.last_timestamp = timestamps[-1]
                    data = np.array(data).astype(np.float32)

                    self.results_ready.emit(data, timestamps)
                else:
                    no_data_counter += 1

                    if no_data_counter >= 10:
                        self.error.emit(f'No {self.muse_data_type} data received for 10 consecutive attempts')

            except Exception as ex:
                self.error.emit(traceback.format_exc())

        self.finished.emit()


class EEGApp(QWidget):
    def __init__(self):
        super().__init__()
        self.config = SessionConfig()
        self.audio = Audio(self.config.audio)
        self.file_writer = FileWriter(self.config)
        self.logger = Logger(self.config, self.__class__.__name__)

        self.app_state = AppState.DISCONNECTED
        self.recording_elapsed_time = 0  # Elapsed time in seconds
        self.reset_attempt_count = 0
        
        self.last_stim = 0.0
        self.processor_elapsed_time = 0.0
        self.target_phase = self.config.target_phase
        self.second_stim_start = nan
        self.second_stim_end = nan

        self.stream_info: dict[MuseDataType, Union[StreamInfo, None]] = {
            MuseDataType.EEG: None,
            MuseDataType.ACC: None,
            MuseDataType.PPG: None
        }
        self.stream_inlet: dict[MuseDataType, Union[StreamInlet, None]] = {
            MuseDataType.EEG: None,
            MuseDataType.ACC: None,
            MuseDataType.PPG: None
        }

        self.eeg_worker = DatastreamWorker(self.config, MuseDataType.EEG)
        self.eeg_thread = QThread()
        self.eeg_worker.moveToThread(self.eeg_thread)
        self.eeg_thread.started.connect(self.eeg_worker.run)
        self.eeg_worker.finished.connect(self.eeg_thread.quit)
        self.eeg_worker.results_ready.connect(self.handle_eeg_data)
        self.eeg_worker.error.connect(self.handle_eeg_error)
        self.eeg_worker.set_app(self)
        
        self.processing_window_len_n = int(SAMPLING_RATE[MuseDataType.EEG] * self.config.processing_window_len_s)

        self.amp_buffer = np.zeros(self.config.amp_buffer_len)
        self.hl_ratio_buffer = np.zeros(self.config.hl_ratio_buffer_len)

        self.sos_low = signal.butter(self.config.bpf_order, self.config.low_bpf_cutoff, btype = 'bandpass', output = 'sos', fs = SAMPLING_RATE[MuseDataType.EEG])
        self.sos_high = signal.butter(self.config.bpf_order, self.config.high_bpf_cutoff, btype = 'bandpass', output = 'sos', fs = SAMPLING_RATE[MuseDataType.EEG])
        self.zi_low = signal.sosfilt_zi(self.sos_low)
        self.zi_high = signal.sosfilt_zi(self.sos_high)

        self.wavelet_freqs = np.linspace(self.config.truncated_wavelet.low, self.config.truncated_wavelet.high, self.config.truncated_wavelet.n)
        trunc_wavelet_len = self.processing_window_len_n * 2 # double the length of the signal
        self.trunc_wavelets = [signal.morlet2(trunc_wavelet_len, self.config.truncated_wavelet.w * SAMPLING_RATE[MuseDataType.EEG] / (2 * f * np.pi), w = self.config.truncated_wavelet.w)[:trunc_wavelet_len // 2] for f in self.wavelet_freqs]
        self.selected_channel_ind = 1 # AF7
        self.switch_channel_counter = 0
        self.switch_channel_counter_max = int(self.config.switch_channel_period_s * SAMPLING_RATE[MuseDataType.EEG] / CHUNK_SIZE[MuseDataType.EEG])

        self.rolling_avg_len = 10
        self.eeg_rolling_avg = 0.0  # Rolling average for selected channel across chunks
        self.rolling_avg_buffer = np.zeros(self.rolling_avg_len)  # Buffer to store mean values of past chunks
        self.rolling_avg_index = 0  # Index to track position in circular buffer
        self.rolling_avg_sum = 0.0  # Scalar sum of the rolling window for efficiency

        self.init_ui()

        self.init_phase = True  # Track whether we're in the initialization phase
        self.init_data_count = 0  # Counter for how many chunks have been received
        self.init_data_count_max = DISPLAY_WINDOW_LEN_N / CHUNK_SIZE[MuseDataType.EEG]

        self.plotter_shm = shared_memory.SharedMemory(name=EEG_PLOTTING_SHARED_MEMORY, create=True, size=DISPLAY_WINDOW_LEN_N * NUM_CHANNELS[MuseDataType.EEG] * 4 + DISPLAY_WINDOW_LEN_N * 8)  # float32 EEG + float64 timestamps
        self.plotter_eeg_data = np.ndarray((DISPLAY_WINDOW_LEN_N, NUM_CHANNELS[MuseDataType.EEG]), dtype=np.float32, buffer=self.plotter_shm.buf[:DISPLAY_WINDOW_LEN_N * NUM_CHANNELS[MuseDataType.EEG] * 4])
        self.plotter_timestamps = np.ndarray((DISPLAY_WINDOW_LEN_N,), dtype=np.float64, buffer=self.plotter_shm.buf[DISPLAY_WINDOW_LEN_N * NUM_CHANNELS[MuseDataType.EEG] * 4:])
    
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
            self.recording_elapsed_time += 1
            self.elapsed_time_label.setText(f"Elapsed Time: {self.recording_elapsed_time}s")
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
        self.experiment_dropdown.setCurrentIndex([mode.value for mode in ExperimentMode].index(self.config.experiment_mode.value))  # Default to first mode (Disabled)
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
        
        # main_layout.addWidget(self.eeg_plot_widget)
        main_layout.addLayout(right_panel)
        
        self.setLayout(main_layout)
        self.setWindowTitle("Sound2Sleep: CLAS at Home")

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
        self.connection_label.setText("Connected")
        self.connection_button.setText("Disconnect")
        self.connection_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.experiment_dropdown.setEnabled(True)
        self.param_config_editor.setEnabled(True)
        self.connection_timeout_error_label.hide()
        self.app_state = AppState.RECORDING
        self.record_button.setText("Stop Recording")
        self.recording_elapsed_time = 0  # Reset elapsed time on start
        self.elapsed_time_timer.start(1000)  # Update every second

    def on_disconnected(self):
        self.app_state = AppState.DISCONNECTED
        self.connection_label.setText("Disconnected")
        self.connection_button.setText("Connect")
        self.record_button.setEnabled(False)
        self.record_button.setText("Start Recording")
        self.elapsed_time_label.setText("Elapsed Time: 0s")
        self.recording_elapsed_time = 0  # Reset elapsed time
        self.elapsed_time_timer.stop()
    
    def on_toggle_record_button(self):
        if self.app_state == AppState.CONNECTED:
            self.app_state = AppState.RECORDING
            self.record_button.setText("Stop Recording")
            self.recording_elapsed_time = 0  # Reset elapsed time on start
            self.elapsed_time_timer.start(1000)  # Update every second

        elif self.app_state == AppState.RECORDING:
            self.app_state = AppState.CONNECTED
            self.record_button.setText("Start Recording")
            self.elapsed_time_timer.stop()  # Stop the timer

    def on_connection_timeout(self):
        self.connection_timeout_error_label.setText("Connection Timeout")
        self.connection_timeout_error_label.show()

    def on_config_update(self, config: SessionConfig):
        self.config = config
        self.logger.update_session_key(self.config._session_key)
        self.file_writer.update_session_key(config._session_key)

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

    def play_audio(self, delay):
        QTimer.singleShot(delay * 1000, self.audio.run)

    # def screenoff(self):
    #     ''' Darken the screen by starting the blank screensaver '''
    #     try:
    #         subprocess.call(['C:\Windows\System32\scrnsave.scr', '/start'])
    #     except Exception as ex:
    #         self.logger.critical(traceback.format_exception(type(ex), ex, ex.__traceback__))

    # def handle_eeg_data(self, data: np.ndarray, timestamp: np.ndarray):
    #     if np.isnan(data).any():
    #         self.logger.critical(f"NaN found in data: {data.tolist()}")

    #     selected_channel_mean = np.mean(data[:, self.selected_channel_ind])
    #     oldest_value = self.rolling_avg_buffer[self.rolling_avg_index]
    #     self.rolling_avg_sum += selected_channel_mean - oldest_value
    #     self.rolling_avg_buffer[self.rolling_avg_index] = selected_channel_mean
    #     self.rolling_avg_index = (self.rolling_avg_index + 1) % self.rolling_avg_window_len
    #     self.eeg_rolling_avg = self.rolling_avg_sum / self.rolling_avg_window_len
    #     data[:, self.selected_channel_ind] -= self.eeg_rolling_avg

    #     self.eeg_timestamp = np.concatenate([self.eeg_timestamp, timestamp])[-DISPLAY_WINDOW_LEN_N:]
    #     self.eeg_data = np.vstack([self.eeg_data, data]) if self.eeg_data is not None else data
    #     self.eeg_data = self.eeg_data[-DISPLAY_WINDOW_LEN_N:]

    #     if len(self.eeg_timestamp) >= DISPLAY_WINDOW_LEN_N: 

    #         result, time_to_target, phase, freq, amp, amp_buffer_mean = self.process_eeg_step_1()
    #         self.logger.info(f"Result: {result}, Time to target: {time_to_target}, Phase: {phase}, Freq: {freq}, Amp: {amp}, Amp Buffer Mean: {amp_buffer_mean}")
    #         if (result == EEGProcessorOutput.STIM) or (result == EEGProcessorOutput.STIM2):
    #             time_to_target = time_to_target - self.config.time_to_target_offset
    #             self.process_eeg_step_2(time_to_target)

    #             stim_time = self.processor_elapsed_time + time_to_target
    #             if isnan(stim_time): self.logger.critical(f"Stim Time is NaN. EEG Timestamp: {self.eeg_timestamp[-1]}")
    #             self.file_writer.write_stim(stim_time)

    #     self.file_writer.write_chunk(data, timestamp)
    def handle_eeg_data(self, eeg_chunk: np.ndarray, timestamp_chunk: np.ndarray):
        self.logger.debug(f"Start EEG Handle: {time.time()}")
        self.plotter_eeg_data[:-CHUNK_SIZE[MuseDataType.EEG]] = self.plotter_eeg_data[CHUNK_SIZE[MuseDataType.EEG]:]
        self.plotter_timestamps[:-CHUNK_SIZE[MuseDataType.EEG]] = self.plotter_timestamps[CHUNK_SIZE[MuseDataType.EEG]:]

        self.plotter_eeg_data[-CHUNK_SIZE[MuseDataType.EEG]:] = eeg_chunk
        self.plotter_timestamps[-CHUNK_SIZE[MuseDataType.EEG]:] = timestamp_chunk

        if self.init_phase:
            self.init_data_count += 1
            if self.init_data_count >= self.init_data_count_max:
                self.init_phase = False
                print("Initialization complete. Processing EEG data.")
            return

        self.switch_channel_counter += 1
        if self.switch_channel_counter == self.switch_channel_counter_max:
            self.switch_channel()
            self.switch_channel_counter = 0

        selected_channel_mean = np.mean(eeg_chunk[:, self.selected_channel_ind])
        oldest_value = self.rolling_avg_buffer[self.rolling_avg_index]
        self.rolling_avg_sum += selected_channel_mean - oldest_value
        self.rolling_avg_buffer[self.rolling_avg_index] = selected_channel_mean
        self.rolling_avg_index = (self.rolling_avg_index + 1) % self.rolling_avg_len
        self.eeg_rolling_avg = self.rolling_avg_sum / self.rolling_avg_len

        self.process_eeg_data()
        self.file_writer.write_chunk(eeg_chunk, timestamp_chunk)
        self.logger.debug(f"End EEG Handle: {time.time()}")

    def process_eeg_data(self):
        result, time_to_target, phase, freq, amp, amp_buffer_mean = self.process_eeg_step_1()
        if (result == EEGProcessorOutput.STIM) or (result == EEGProcessorOutput.STIM2):
            self.logger.info(f"Result: {result}, Time to target: {time_to_target}, Phase: {phase}, Freq: {freq}, Amp: {amp}, Amp Buffer Mean: {amp_buffer_mean}")
            time_to_target = time_to_target - self.config.time_to_target_offset
            self.process_eeg_step_2(time_to_target)

            stim_time = self.processor_elapsed_time + time_to_target
            if isnan(stim_time): self.logger.critical(f"Stim Time is NaN. EEG Timestamp: {self.plotter_timestamps[-1]}")
            self.file_writer.write_stim(stim_time)

    def switch_channel(self):
        selected_channel_ind = np.argmin(np.sqrt(np.mean(self.plotter_eeg_data**2, axis=0)))
        if selected_channel_ind != self.selected_channel_ind:
            self.selected_channel_ind = selected_channel_ind
            self.logger.warning(f"Channel Switched: {self.selected_channel_ind}")

    def get_hl_ratio(self, selected_channel_data):
        lp_signal, self.zi_low = signal.sosfilt(self.sos_low, selected_channel_data, zi = self.zi_low)
        hp_signal, self.zi_high = signal.sosfilt(self.sos_high, selected_channel_data, zi = self.zi_high)

        envelope_lp = np.abs(signal.hilbert(lp_signal[SAMPLING_RATE[MuseDataType.EEG]:]))
        power_lf = envelope_lp**2

        envelope_hf = np.abs(signal.hilbert(hp_signal[SAMPLING_RATE[MuseDataType.EEG]:]))
        power_hf = envelope_hf**2

        hl_ratio = np.mean(power_hf) / np.mean(power_lf)
        hl_ratio = np.log10(hl_ratio)
        return hl_ratio
    
    def estimate_phase(self, selected_channel): 
        conv_vals = [np.dot(selected_channel, w) for w in self.trunc_wavelets]
        max_idx = np.argmax(np.abs(conv_vals))
        # amp = conv_vals[max_idx] / 2
        freq = self.wavelet_freqs[max_idx]
        phase = np.angle(conv_vals[max_idx]) % (2 * pi)
        amp = np.nanmax(np.abs(selected_channel[self.processing_window_len_n // 2:]))
        return phase, freq, amp
    
    def process_eeg_step_1(self):
        if self.second_stim_end < self.plotter_timestamps[-1]:
            self.second_stim_start = nan
            self.second_stim_end = nan

        phase, freq, amp = self.estimate_phase(self.plotter_eeg_data[-self.processing_window_len_n:, self.selected_channel_ind] - self.eeg_rolling_avg)
        hl_ratio = self.get_hl_ratio(self.plotter_eeg_data[-self.processing_window_len_n:, self.selected_channel_ind] - self.eeg_rolling_avg)
        self.amp_buffer[:-1] = self.amp_buffer[1:]
        self.amp_buffer[-1] = amp
        amp_buffer_mean = self.amp_buffer.mean()

        self.hl_ratio_buffer[:-1] = self.hl_ratio_buffer[1:]
        self.hl_ratio_buffer[-1] = hl_ratio
        hl_ratio_buffer_mean = self.hl_ratio_buffer.mean()
        self.processor_elapsed_time = self.plotter_timestamps[-1]

        if self.config.experiment_mode == ExperimentMode.DISABLED:
            self.logger.info(f"Phase: {phase}, Freq: {freq}, Amp: {amp}, Amp Buffer Mean: {amp_buffer_mean}")
            return EEGProcessorOutput.NOT_RUNNING, 0, phase, freq, amp, amp_buffer_mean

        # check if we're waiting for the 2nd stim
        # if NOT, run normal checks
        if isnan(self.second_stim_start):
            ### check backoff criteria ###
            if ((self.last_stim + self.config.backoff_time) > (self.processor_elapsed_time + self.config.stim1_prediction_limit_sec)):
                return EEGProcessorOutput.BACKOFF, 0, phase, freq, amp, amp_buffer_mean

            ### check amplitude criteria ###
            if (amp_buffer_mean < self.config.amp_buffer_mean_min) or (amp_buffer_mean > self.config.amp_buffer_mean_max):
                return EEGProcessorOutput.AMPLITUDE, 0, phase, freq, amp, amp_buffer_mean

            if hl_ratio_buffer_mean > self.config.hl_ratio_buffer_mean_threshold or hl_ratio > self.config.hl_ratio_latest_threshold:
                return EEGProcessorOutput.HL_RATIO, 0, phase, freq, amp, amp_buffer_mean

        # if we are waiting for 2nd stim, but before the backoff window, only use phase targeting
        if self.processor_elapsed_time < self.second_stim_start: # self.second_stim_start could be nan (in which case, the condition will be False)
            return EEGProcessorOutput.BACKOFF2, 0, phase, freq, amp, amp_buffer_mean

        ### perform forward prediction ###
        delta_t = ((self.target_phase - phase) % (2 * pi)) / (freq * 2 * pi)

        # cue a stim for the next target phase
        if isnan(self.second_stim_start):
            if delta_t > self.config.stim1_prediction_limit_sec:
                return EEGProcessorOutput.FUTURE, delta_t, phase, freq, amp, amp_buffer_mean

            self.last_stim = self.processor_elapsed_time + delta_t
            self.second_stim_start = self.last_stim + self.config.stim2_start_delay
            self.second_stim_end = self.last_stim + self.config.stim2_end_delay

            return EEGProcessorOutput.STIM, delta_t, phase, freq, amp, amp_buffer_mean

        else:
            if delta_t > self.config.stim2_prediction_limit_sec:
                return EEGProcessorOutput.FUTURE2, delta_t, phase, freq, amp, amp_buffer_mean

            self.second_stim_start = nan
            self.second_stim_end = nan

            if self.config.experiment_mode == ExperimentMode.RANDOM_PHASE_AUDIO_OFF or self.config.experiment_mode == ExperimentMode.RANDOM_PHASE_AUDIO_ON:
                self.randomize_phase()

            return EEGProcessorOutput.STIM2, delta_t, phase, freq, amp, amp_buffer_mean
    
    def process_eeg_step_2(self, time_to_target):
        if self.config.experiment_mode == ExperimentMode.CLAS_AUDIO_ON or self.config.experiment_mode == ExperimentMode.RANDOM_PHASE_AUDIO_ON: self.play_audio(time_to_target)

    def randomize_phase(self):
        self.target_phase = random.uniform(0.0, 2*np.pi)

    def handle_eeg_error(self, error_msg):
        self.eeg_worker.running = False
        self.logger.error(f"EEG Error: {error_msg}")
        if not hasattr(self, '_reset_in_progress') or not self._reset_in_progress:
            self._reset_in_progress = True
            QTimer.singleShot(2000, self._perform_lsl_reset)

    def _perform_lsl_reset(self):
        try:
            self.lsl_reset_stream_step1()
        finally:
            self._reset_in_progress = False
    
    def lsl_reload(self):
        eeg_ok = False
        
        for stream in MuseDataType:
            self.stream_info[stream] = resolve_byprop('type', stream.value, timeout=LSL_SCAN_TIMEOUT)

            if self.stream_info[stream]:
                self.stream_info[stream] = self.stream_info[stream][0]
                self.stream_inlet[stream] = StreamInlet(self.stream_info[stream])
                self.logger.info(f'{stream.name} OK.')
                if stream == MuseDataType.EEG: eeg_ok = True
            else:
                self.logger.warning(f'{stream.name} not found.')
        return eeg_ok
    
    def lsl_reset_stream_step1(self):
        self.on_connection_timeout()
        self.logger.error('Resetting stream step 1')
        try:
            for stream_type, _stream_inlet in self.stream_inlet.items():
                if stream_type == MuseDataType.EEG and _stream_inlet is not None:
                    _stream_inlet.close_stream()
        except Exception as ex:
            self.logger.critical(str(ex))
        subprocess.call('start bluemuse://stop?stopall', shell=True)
        time.sleep(3)
        self.lsl_reset_stream_step2()


    def lsl_reset_stream_step2(self):
        self.logger.error('Resetting stream step 2')
        subprocess.call('start bluemuse://start?startall', shell=True)
        time.sleep(3)
        self.lsl_reset_stream_step3()

    def lsl_reset_stream_step3(self):
        self.logger.error('Resetting stream step 3')
        reset_success = self.lsl_reload()

        if not reset_success:
            self.logger.warning('LSL stream reset unsuccessful')
            self.reset_attempt_count += 1
            if self.reset_attempt_count <= 5:
                self.logger.error('Resetting Attempt: ' + str(self.reset_attempt_count))
                if not hasattr(self, '_reset_in_progress') or not self._reset_in_progress:
                    self._reset_in_progress = True
                    self._perform_lsl_reset()
            else:
                self.reset_attempt_count = 0

                for p in psutil.process_iter(['name']):
                    if p.info['name'] == 'BlueMuse.exe':
                        self.logger.critical('Killing BlueMuse')
                        p.kill()

                if not hasattr(self, '_reset_in_progress') or not self._reset_in_progress:
                    self._reset_in_progress = True
                    QTimer.singleShot(3000, self._perform_lsl_reset())
        else:
            self.reset_attempt_count = 0
            self.logger.warning('LSL stream reset successful. Starting threads')
            time.sleep(3)
            subprocess.call('start bluemuse://start?streamfirst=true', shell=True)
            self.on_connected()

            # self._disconnect_worker_signals()
            # self._recreate_workers()
            # self._connect_worker_signals()

            if self.stream_inlet[MuseDataType.EEG] is not None:
                if not self.eeg_thread.isRunning():
                    self.eeg_thread.start()
            
    def start_bluemuse(self):
        subprocess.call('start bluemuse:', shell=True)
        subprocess.call('start bluemuse://setting?key=primary_timestamp_format!value=BLUEMUSE', shell=True)
        subprocess.call('start bluemuse://setting?key=channel_data_type!value=float32', shell=True)
        subprocess.call('start bluemuse://setting?key=eeg_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=accelerometer_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=gyroscope_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=ppg_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://start?streamfirst=true', shell=True)

        time.sleep(4)
        while not self.lsl_reload():
            self.logger.error(f"LSL streams not found, retrying in 4 seconds") 
            time.sleep(4)
        self.on_connected()
        self.eeg_worker.running = True
        
        if self.stream_inlet[MuseDataType.EEG] is not None:
            if not self.eeg_thread.isRunning():
                self.eeg_thread.start()
        
        # if self.stream_inlet[MuseDataType.ACC] is not None:
        #     if not self.acc_thread.isRunning():
        #         self.acc_thread.start()
        
        # if self.stream_inlet[MuseDataType.PPG] is not None:
        #     if not self.ppg_thread.isRunning():
        #         self.ppg_thread.start()
        
    def stop_bluemuse(self):
        self.eeg_worker.running = False
        if self.eeg_thread.isRunning():
            self.eeg_thread.quit()
            if not self.eeg_thread.wait(3000):  # 3 second timeout
                self.eeg_thread.terminate()
                self.logger.warning(f"Had to terminate thread forcefully")
        # if self.acc_thread.isRunning():
        #     self.acc_thread.quit()
        #     self.acc_thread.wait()
        
        # if self.ppg_thread.isRunning():
        #     self.ppg_thread.quit()
        #     self.ppg_thread.wait()

        try:
            for stream_type, _stream_inlet in self.stream_inlet.items():
                if stream_type == MuseDataType.EEG and _stream_inlet is not None:
                    _stream_inlet.close_stream()
        except Exception as ex:
            self.logger.critical(str(ex))

        subprocess.call('start bluemuse://stop?stopall', shell=True)
        subprocess.call('start bluemuse://shutdown', shell=True)

        for p in psutil.process_iter(['name']):
            if p.info['name'] == 'BlueMuse.exe':
                self.logger.critical('Killing BlueMuse')
                p.kill()
        self.on_disconnected()

    def closeEvent(self, event):
        self.eeg_worker.running = False
        if self.eeg_thread.isRunning():
            self.eeg_thread.quit()
            if not self.eeg_thread.wait(3000):  # 3 second timeout
                self.eeg_thread.terminate()
                self.logger.warning(f"Had to terminate thread forcefully")

        try:
            for stream_type, _stream_inlet in self.stream_inlet.items():
                if _stream_inlet is not None:
                    _stream_inlet.close_stream()
        except Exception as ex:
            self.logger.critical(str(ex))
        
        # Close shared memory
        if hasattr(self, 'plotter_shm'):
            self.plotter_shm.close()
            self.plotter_shm.unlink()
        
        event.accept()

# def plotter():
#     total_samples = SAMPLING_RATE[MuseDataType.EEG] * DISPLAY_WINDOW_LEN_S
#     shm_data = shared_memory.SharedMemory(name=EEG_PLOTTING_SHARED_MEMORY)
#     data_array = np.ndarray((DISPLAY_WINDOW_LEN_N, NUM_CHANNELS[MuseDataType.EEG]), dtype=np.float32, buffer=shm_data.buf[:DISPLAY_WINDOW_LEN_N * NUM_CHANNELS[MuseDataType.EEG] * 4])
#     timestamps = np.ndarray((DISPLAY_WINDOW_LEN_N,), dtype=np.float64, buffer=shm_data.buf[DISPLAY_WINDOW_LEN_N * NUM_CHANNELS[MuseDataType.EEG] * 4:])
#     # data_array = np.ndarray((TOTAL_SAMPLES, NUM_CHANNELS), dtype=np.float32, buffer=shm_data.buf[:TOTAL_SAMPLES * NUM_CHANNELS * 4])
#     # timestamps = np.ndarray((TOTAL_SAMPLES,), dtype=np.float64, buffer=shm_data.buf[TOTAL_SAMPLES * NUM_CHANNELS * 4:])
    
#     plt.ion()
#     fig, ax = plt.subplots()
#     lines = []
#     impedances = np.zeros(NUM_CHANNELS[MuseDataType.EEG])
#     time_axis = np.arange(-DISPLAY_WINDOW_LEN_S, 0, 1.0 / SAMPLING_RATE[MuseDataType.EEG])[::2]
    
#     for ii in range(NUM_CHANNELS[MuseDataType.EEG]):
#         line, = ax.plot(time_axis, np.zeros(DISPLAY_WINDOW_LEN_N // 2) - ii, lw=1)
#         lines.append(line)
    
#     ax.set_ylim(-NUM_CHANNELS[MuseDataType.EEG], 0.0)
#     ax.set_xlabel('Time (s)')
#     ax.set_yticks(np.arange(0, -NUM_CHANNELS[MuseDataType.EEG], -1))
#     ax.xaxis.grid(False)
    
#     try:
#         while True:
#             latest_time = timestamps[-1]
#             time_axis = timestamps[::2] - latest_time  # Convert to seconds relative to latest timestamp
#             plot_data = data_array - data_array.mean(axis=0)
#             impedances = np.std(data_array, axis=0)  # Recalculate impedances
#             ax.set_yticklabels([f'{label} - {impedance:.2f}' for label, impedance in zip(CHANNEL_NAMES[MuseDataType.EEG], impedances)])
            
#             for i, line in enumerate(lines):
#                 # norm_data = data_array[:, i] - np.mean(data_array[:, i])  # Normalize to mean 0
#                 line.set_xdata(time_axis)
#                 line.set_ydata(plot_data[::2, i] / 100 - i)  # Offset each channel
            
#             plt.pause(0.01)  # Update plot
#     except KeyboardInterrupt:
#         pass
#     finally:
#         shm_data.close()
#         shm_data.unlink()

# === EEG Processing Worker (Multiprocessing) ===
def process_eeg(data_queue, result_queue):
    """Worker function for EEG signal processing using a 2-second sliding window."""
    fs = 256  # Assume 256 Hz sampling rate
    window_size = fs * 2  # 2 seconds worth of data
    num_channels = 4  # Assume 4-channel EEG

    # Sliding buffer for 2s window
    timestamps_buffer = deque(maxlen=window_size)
    eeg_buffer = deque(maxlen=window_size)

    # Bandpass filter (1-40 Hz)
    b, a = butter(4, [1, 40], btype='bandpass', fs=fs)

    while True:
        timestamps, eeg_data = data_queue.get()
        if timestamps is None:  # Stop signal
            break

        # Append new data to the buffer
        timestamps_buffer.extend(timestamps)
        eeg_buffer.extend(eeg_data)

        # Process only when we have a full 2s window
        if len(eeg_buffer) == window_size:
            eeg_array = np.array(eeg_buffer)  # Convert to NumPy array
            filtered_data = filtfilt(b, a, eeg_array, axis=0)  # Apply filter

            result_queue.put((timestamps_buffer[-12:], filtered_data[-12:]))  # Send last chunk for display

# === Real-Time EEG GUI (PyQt5 + pyqtgraph) ===
class EEGViewer(QMainWindow):
    def __init__(self, result_queue):
        super().__init__()
        self.setWindowTitle("Real-Time EEG")
        self.graph = pg.PlotWidget()
        self.setCentralWidget(self.graph)
        self.curve = self.graph.plot(pen='g')

        self.result_queue = result_queue
        self.data = np.zeros(256)  # Buffer for display
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # Update every 50ms

    def update_plot(self):
        if not self.result_queue.empty():
            _, new_data = self.result_queue.get()
            self.data = np.roll(self.data, -new_data.shape[0])
            self.data[-new_data.shape[0]:] = new_data[:, 0]  # Display first channel
            self.curve.setData(self.data)

class LSLReceiver(Thread):
    def __init__(self, muse_data_type: MuseDataType=MuseDataType.EEG):
        super().__init__()
        self.daemon = True  # Ensures it closes when the main thread exits
        self.queue = queue.Queue(maxsize=10)  # Thread-safe queue
        self.running = True
        self.muse_data_type = muse_data_type

        subprocess.call('start bluemuse:', shell=True)
        subprocess.call('start bluemuse://setting?key=primary_timestamp_format!value=BLUEMUSE', shell=True)
        subprocess.call('start bluemuse://setting?key=channel_data_type!value=float32', shell=True)
        subprocess.call('start bluemuse://setting?key=eeg_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=accelerometer_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=gyroscope_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=ppg_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://start?streamfirst=true', shell=True)

        time.sleep(4)
        self.stream_info = resolve_byprop('type', self.muse_data_type.value, timeout=LSL_SCAN_TIMEOUT)

        if self.stream_info:
            self.stream_info = self.stream_info[0]
            self.stream_inlet = StreamInlet(self.stream_info)
        else:
            raise Exception()
        
    def stop(self):
        self.running = False
    
    def run(self):
        no_data_counter = 0
        while self.running:
            time.sleep(DELAYS[self.muse_data_type])

            data, timestamps = self.stream_inlet.pull_chunk(timeout=DELAYS[self.muse_data_type], max_samples=CHUNK_SIZE[self.muse_data_type])
            if timestamps and len(timestamps) == CHUNK_SIZE[self.muse_data_type]:
                timestamps = TIMESTAMPS[self.muse_data_type] + np.float64(time.time())
                data = np.array(data).astype(np.float32)

                self.queue.put((data, timestamps))
            else:
                no_data_counter += 1

                if no_data_counter >= 10:
                    print(f'No {self.muse_data_type} data received for 10 consecutive attempts')
                    raise Exception()


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x0000040
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)

    try:
        # lsl_thread = LSLReceiver()
        # lsl_thread.start()
        # data_queue = Queue()
        # result_queue = Queue()

        # proc = Process(target=process_eeg, args=(data_queue, result_queue))
        # proc.start()

        app = QApplication(sys.argv)
        window = EEGApp()
        # window = EEGViewer(result_queue)
        window.show()
        exit_code = app.exec()
        # try:
        #     while True:
        #         if not lsl_thread.queue.empty():
        #             timestamps, eeg_data = lsl_thread.queue.get()
        #             data_queue.put((timestamps, eeg_data))  # Send data to processor
                
        #         app.processEvents()  # Keep GUI responsive
        # except KeyboardInterrupt:
        #     print("Stopping...")
        #     lsl_thread.stop()
        #     data_queue.put((None, None))  # Stop processing worker
        #     proc.join()
    finally:
        if sys.platform.startswith("win"):
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            for p in psutil.process_iter(['name']):
                if p.info['name'] == 'BlueMuse.exe': p.kill()
