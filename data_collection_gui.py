import asyncio
import ctypes
import datetime
import json
import math
import random
import subprocess
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from functools import partial
from math import ceil, floor, isnan, nan, pi
from multiprocessing import Event, Process, Queue, queues, shared_memory
from multiprocessing.synchronize import Event as EventType

# from multiprocessing.synchronize import Event
from queue import Empty
from threading import Thread, Timer
from typing import Union

import bleak.exc
import matplotlib
import matplotlib.pyplot as plt
import muselsl.backends
import numpy as np
import psutil
import pyqtgraph as pg
import scipy.signal as signal
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from muselsl.muse import Muse
from muselsl.stream import find_muse  # Adjust this import as needed
from pydantic import ValidationError
from pylsl import StreamInfo, StreamInlet, resolve_byprop
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (
    QEvent,
    QObject,
    QRunnable,
    Qt,
    QThread,
    QThreadPool,
    QTimer,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QFont, QMovie, QPainter, QPen
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
from pyqtgraph import DateAxisItem
from scipy import signal

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

sys.coinit_flags = 0  # 0 means MTA


class ConnectionQuality(Enum):
    HIGH = 'High'
    MEDIUM = 'Medium'
    LOW = 'Low'

CONNECTION_QUALITY_COLORS = {
    ConnectionQuality.HIGH: 'green',
    ConnectionQuality.MEDIUM: 'yellow',
    ConnectionQuality.LOW: 'red',
}

CONNECTION_QUALITY_LABELS = {
    ConnectionQuality.HIGH: 'Ready',
    ConnectionQuality.MEDIUM: 'Wet Device',
    ConnectionQuality.LOW: 'Fix Headband',
}

class StatusWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.connection_quality = ConnectionQuality.LOW
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.indicator = QLabel()
        self.indicator.setFixedSize(15, 15)
        self.indicator.setStyleSheet(f"background-color: {CONNECTION_QUALITY_COLORS[self.connection_quality]}; border-radius: 7px;")

        self.text = QLabel(CONNECTION_QUALITY_LABELS[self.connection_quality])
        self.text.setFont(QFont("Arial", 16))
        self.text.setStyleSheet("color: white;")  # Set display text to white
        layout.addWidget(self.indicator)
        layout.addWidget(self.text)
        self.setLayout(layout)

    def setStatus(self, status):
        self.connection_quality = status
        self.indicator.setStyleSheet(f"background-color: {CONNECTION_QUALITY_COLORS[self.connection_quality]}; border-radius: 7px;")
        self.text.setText(CONNECTION_QUALITY_LABELS[self.connection_quality])

# Custom DateAxisItem that formats tick values as "Hour:Minute:Second"
class CustomDateAxis(DateAxisItem):
    def tickStrings(self, values, scale, spacing):
        out = []
        last_label = None
        for value in values:
            label = datetime.fromtimestamp(value).strftime("%H:%M:%S")
            if label == last_label:
                out.append("")
            else:
                out.append(label)
                last_label = label
        return out

class ConnectionWidget(QWidget):
    _on_connected = pyqtSignal()
    def __init__(self, parent, config: SessionConfig):
        super().__init__(parent)
        self._parent = parent
        self.config = config
        self.file_writer = FileWriter(self.config._session_key)
        self.logger = Logger(self.config._session_key, self.__class__.__name__)
        self.audio = Audio(self.config._audio)

        self.running_clas_algo = False

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.loading_screen = QLabel(self)
        self.loading_movie = QMovie("loading.gif")  # Replace with your GIF file
        self.loading_screen.setMovie(self.loading_movie)
        self.loading_screen.setVisible(True)
        self.main_layout.addWidget(self.loading_screen, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(self.main_layout)

        # Configuration
        self.running = False
        self.reset_attempt_count = 0
        
        self.last_stim = 0.0
        self.processor_elapsed_time = 0.0
        self.target_phase = self.config.target_phase
        self.second_stim_start = nan
        self.second_stim_end = nan

        self.selected_channel_ind = 1 # AF7
        self.switch_channel_counter = 0
        self.switch_channel_counter_max = int(self.config.switch_channel_period_s * SAMPLING_RATE[MuseDataType.EEG] / CHUNK_SIZE[MuseDataType.EEG])

        self.window_len_s = max(self.config.processing_window_len_s, self.config.mean_subtraction_window_len_s)
        self.window_len_n = int(SAMPLING_RATE[MuseDataType.EEG] * self.window_len_s)
        self.processing_window_len_n = int(SAMPLING_RATE[MuseDataType.EEG] * self.config.processing_window_len_s)
        self.display_window_len_n = int(SAMPLING_RATE[MuseDataType.EEG] * DISPLAY_WINDOW_LEN_S)
        self.amp_buffer = np.zeros(self.config.amp_buffer_len)
        self.hl_ratio_buffer = np.zeros(self.config.hl_ratio_buffer_len)

        self.sos_low = signal.butter(self.config.bpf_order, self.config.low_bpf_cutoff, btype = 'bandpass', output = 'sos', fs = SAMPLING_RATE[MuseDataType.EEG])
        self.sos_high = signal.butter(self.config.bpf_order, self.config.high_bpf_cutoff, btype = 'bandpass', output = 'sos', fs = SAMPLING_RATE[MuseDataType.EEG])
        self.zi_low = signal.sosfilt_zi(self.sos_low)
        self.zi_high = signal.sosfilt_zi(self.sos_high)

        trunc_wavelet_len = self.processing_window_len_n * 2 # double the length of the signal
        self.wavelet_freqs = np.linspace(self.config.truncated_wavelet.low, self.config.truncated_wavelet.high, self.config.truncated_wavelet.n)
        self.trunc_wavelets = [signal.morlet2(trunc_wavelet_len, self.config.truncated_wavelet.w * SAMPLING_RATE[MuseDataType.EEG] / (2 * f * np.pi), w = self.config.truncated_wavelet.w)[:trunc_wavelet_len // 2] for f in self.wavelet_freqs]
        
        low_freqs = np.linspace(0.5, 4, 5)
        self.low_freq_wavelets = [signal.morlet2(trunc_wavelet_len, self.config.truncated_wavelet.w * SAMPLING_RATE[MuseDataType.EEG] / (2 * f * np.pi), w = 5)[:trunc_wavelet_len // 2] for f in low_freqs]
        high_freqs = np.linspace(8, 12, 5)
        self.high_freq_wavelets = [signal.morlet2(trunc_wavelet_len, self.config.truncated_wavelet.w * SAMPLING_RATE[MuseDataType.EEG] / (2 * f * np.pi), w = 5)[:trunc_wavelet_len // 2] for f in high_freqs]

        # Rolling mean buffer
        self.rolling_buffer = np.zeros((self.window_len_n, 4))
        self.rolling_idx = 0
        self.buffer_filled = False
        
        self.eeg_data = np.array([])
        self.timestamps = np.array([])

        # Create a queue for communication between processes
        self._queue = Queue()
        self.connected_flag = Event()
        # Start the Muse LSL recording process
        self.recording_process = Process(
            target=ConnectionWidget.record,
            args=(self._queue, self.connected_flag),
            daemon=True
        )
        self.recording_process.start()
        self.loading_movie.start()
        self.connection_check_timer = QtCore.QTimer()
        self.connection_check_timer.timeout.connect(self.check_connection)
        self.connection_check_timer.start(100)  # Check every 100ms
        
        self.initially_connected = False
        self.display_every_counter = 0
        self.display_every_counter_max = 5

    def __init_plotting__(self):
        self.plot_widget = pg.PlotWidget()

        # Create a custom DateAxis for the x-axis to display time in HH:MM:SS format
        date_axis = CustomDateAxis(orientation='bottom')
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': date_axis})
        self.plot_widget.setYRange(-50, 50)

        self.lowcut = 1.0
        self.highcut = 4.0
        self.order = 4
        nyq = 0.5 * SAMPLING_RATE[MuseDataType.EEG]
        low = self.lowcut / nyq
        high = self.highcut / nyq
        self.b, self.a = signal.butter(self.order, [low, high], btype='band')

        # Create the curve (line) for the plot.
        self.curve = self.plot_widget.plot(pen='c')
        self.main_layout.addWidget(self.plot_widget)


    def play_audio(self, time_to_target):
        self.audio.play(time_to_target)
        # Run the audio playback in a separate thread to avoid blocking the UI
        # Thread(target=self.audio.play, args=(time_to_target,), daemon=True).start()

    def check_connection(self):
        # Non-blocking check for connection flag
        if self.connected_flag.is_set():
            self.connection_check_timer.stop()
            self.on_connected()

    def on_connected(self):
        self.loading_movie.stop()
        self.loading_screen.setVisible(False)

        # Create and add StatusWidget
        self.status_widget = StatusWidget(self)
        self.main_layout.addWidget(self.status_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Set initial status to checking
        self.status_widget.setStatus(ConnectionQuality.LOW)
        
        # Initialize quality check variables
        self.quality_check_enabled = True
        self.quality_check_start_time = time.time()
        self.processing_enabled = False
        self.min_quality_check_duration = 30  # Minimum 30 seconds of quality checking
        
        # Create start button (initially disabled)
        self.CLAS_button = QPushButton("Start Processing")
        self.CLAS_button.setFixedSize(200, 40)
        self.CLAS_button.setEnabled(False)
        self.CLAS_button.clicked.connect(self.on_CLAS_button_clicked)

        self.CLAS_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                border: none;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QPushButton:hover:!disabled {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)
        
        # Add button to layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.CLAS_button)
        button_layout.addStretch()
        self.main_layout.addLayout(button_layout)
        
        # Setup the timer for real-time updates
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.update_timer.start(20)  # 50 fps update rate
        
        # Install event filter for key press events
        self.installEventFilter(self)
        self.__init_plotting__()

    def on_CLAS_button_clicked(self):
        if not self.processing_enabled:
            self.quality_check_enabled = False
            self.processing_enabled = True
            
            self.CLAS_button.setEnabled(True)
            self.CLAS_button.setText("I'm awake")
            
            if self.status_widget.connection_quality == ConnectionQuality.HIGH:
                self.logger.info(f"Starting EEG processing with {self.status_widget.connection_quality.value} signal quality.")
            else:
                self.logger.warning(f"Starting EEG processing with {self.status_widget.connection_quality.value} signal quality.")
        else:
            self.quality_check_enabled = False
            self.processing_enabled = False
            if self.recording_process.is_alive():
                self.recording_process.terminate()
                print("Recording process terminated")
            self.update_timer.stop()
            self._parent.stacked_widget.setCurrentWidget(self._parent.mood_page)

    def get_hl_ratio(self, selected_channel_data):
        # lp_signal, self.zi_low = signal.sosfilt(self.sos_low, selected_channel_data, zi = self.zi_low)
        # hp_signal, self.zi_high = signal.sosfilt(self.sos_high, selected_channel_data, zi = self.zi_high)

        # envelope_lp = np.abs(signal.hilbert(lp_signal[SAMPLING_RATE[MuseDataType.EEG]:]))
        # power_lf = envelope_lp**2

        # envelope_hf = np.abs(signal.hilbert(hp_signal[SAMPLING_RATE[MuseDataType.EEG]:]))
        # power_hf = envelope_hf**2

        # hl_ratio = np.mean(power_hf) / np.mean(power_lf)
        # hl_ratio = np.log10(hl_ratio)
        # return hl_ratio
        low_conv_vals = [np.dot(selected_channel_data, w) for w in self.low_freq_wavelets]
        high_conv_vals = [np.dot(selected_channel_data, w) for w in self.high_freq_wavelets]

        low_amp = np.nanmean(np.abs(low_conv_vals) **2)
        high_amp = np.nanmean(np.abs(high_conv_vals) **2)
        hl_ratio = high_amp / low_amp
        hl_ratio = np.log10(hl_ratio)
        return hl_ratio

    
    def estimate_phase(self, selected_channel): 
        conv_vals = [np.dot(selected_channel, w) for w in self.trunc_wavelets]
        max_idx = np.argmax(np.abs(conv_vals))
        amp = np.abs(conv_vals[max_idx] / 2)
        freq = self.wavelet_freqs[max_idx]
        phase = (-np.angle(conv_vals[max_idx])) % (2 * pi)
        
        return phase, freq, amp
    
    def switch_channel(self):
        selected_channel_ind = np.argmin(np.sqrt(np.mean(self.eeg_data**2, axis=0)))
        if self.selected_channel_ind != selected_channel_ind:
            self.logger.info(f"Switching channels from [{self.selected_channel_ind}] to [{selected_channel_ind}]")
            self.selected_channel_ind = selected_channel_ind
            
    def process_eeg_step_1(self, mean_to_subtract):
        self.switch_channel_counter += 1
        if self.switch_channel_counter == self.switch_channel_counter_max:
            self.switch_channel()
            self.switch_channel_counter = 0

        if self.second_stim_end < self.timestamps[-1]:
            self.second_stim_start = nan
            self.second_stim_end = nan

        phase, freq, amp = self.estimate_phase(self.eeg_data[-self.processing_window_len_n:, self.selected_channel_ind] - mean_to_subtract[self.selected_channel_ind])
        hl_ratio = self.get_hl_ratio(self.eeg_data[-self.processing_window_len_n:, self.selected_channel_ind] - mean_to_subtract[self.selected_channel_ind])
        self.amp_buffer[:-1] = self.amp_buffer[1:]
        self.amp_buffer[-1] = amp
        amp_buffer_mean = self.amp_buffer.mean()

        self.hl_ratio_buffer[:-1] = self.hl_ratio_buffer[1:]
        self.hl_ratio_buffer[-1] = hl_ratio
        hl_ratio_buffer_mean = self.hl_ratio_buffer.mean()
        self.processor_elapsed_time = self.timestamps[-1]

        if self.config.experiment_mode == ExperimentMode.DISABLED:
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
        if self.config.experiment_mode == ExperimentMode.CLAS_AUDIO_ON or self.config.experiment_mode == ExperimentMode.RANDOM_PHASE_AUDIO_ON: 
            self.play_audio(time_to_target)

    def randomize_phase(self):
        self.target_phase = random.uniform(0.0, 2*np.pi)

    def update_plot(self):
        while not self._queue.empty():
            
            try:
                new_samples, new_timestamps = self._queue.get_nowait()

                if len(self.eeg_data) == 0:
                    self.eeg_data = new_samples
                    self.timestamps = new_timestamps
                else:
                    self.eeg_data = np.vstack((self.eeg_data, new_samples))
                    self.timestamps = np.append(self.timestamps, new_timestamps)

                self.eeg_data = self.eeg_data[-self.window_len_n:, :]
                self.timestamps = self.timestamps[-self.window_len_n:]

                # If we're still in quality check period
                if self.quality_check_enabled:
                    self.check_signal_quality()
                    self.update_button_state()
                    continue

                mean_to_subtract = np.mean(self.eeg_data, axis=0)
                self.file_writer.write_chunk(new_samples, new_timestamps)

                result, time_to_target, phase, freq, amp, amp_buffer_mean = self.process_eeg_step_1(mean_to_subtract)
                self.logger.info(f"Result {result}, Time to target: {time_to_target}, Phase: {phase}, Freq: {freq}, Amp: {amp}, Amp Buffer Mean: {amp_buffer_mean}")

                if (result == EEGProcessorOutput.STIM) or (result == EEGProcessorOutput.STIM2):
                    time_to_target = time_to_target - self.config.time_to_target_offset
                    self.process_eeg_step_2(time_to_target)
                    self.file_writer.write_stim(self.processor_elapsed_time + time_to_target)
                
                if self.display_every_counter == self.display_every_counter_max:
                    eeg_data = self.eeg_data[-self.display_window_len_n:, 1]
                    timestamps = self.timestamps[-self.display_window_len_n:]
                    eeg_data = (eeg_data - np.mean(eeg_data))[::2]
                    timestamps = timestamps[::2]

                    filtered_signal = signal.filtfilt(self.b, self.a, eeg_data)

                    # Update the plot using the downsampled data.
                    self.curve.setData(timestamps, filtered_signal)
                else:
                    self.display_every_counter += 1
            except Empty:
                break
    
    def check_signal_quality(self):
        if len(self.eeg_data) < self.processing_window_len_n:  return
            
        channel_stds = np.std(self.eeg_data[-self.processing_window_len_n:, :], axis=0)
        channel_std = np.min(channel_stds)
        
        # Update status based on minimum channel variance
        if channel_std > 250:
            self.status_widget.setStatus(ConnectionQuality.LOW)
        elif channel_std > 150:
            self.status_widget.setStatus(ConnectionQuality.MEDIUM)
        else:
            self.status_widget.setStatus(ConnectionQuality.HIGH)
        
    def update_button_state(self):
        elapsed_time = time.time() - self.quality_check_start_time
        quality = self.status_widget.connection_quality
        
        self.CLAS_button.setText(CONNECTION_QUALITY_LABELS[quality])
        
        # Enable button based on quality and elapsed time
        enable_button = (quality == ConnectionQuality.HIGH or (quality == ConnectionQuality.MEDIUM and elapsed_time >= self.min_quality_check_duration))
        if enable_button and not self.initially_connected:
            self.initially_connected = True
            self._on_connected.emit()
        self.CLAS_button.setEnabled(enable_button)
            
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            # Handle key press events
            # Add your key handling code here
            return True
        return super().eventFilter(obj, event)

    @classmethod
    def record(cls, _queue: Queue, _connected_flag: EventType):
        found_muse = None
        while not found_muse:
            found_muse = find_muse()
        address = found_muse["address"]
        print(f'Connecting to Muse: {address}')

        def save_eeg(new_samples: np.ndarray, new_timestamps: np.ndarray):
            new_samples = new_samples.transpose()[:, :-1].astype(np.float32) # IGNORE RIGHT AUX Final shape of queue input = ((12, 4), (12,))
            _queue.put((new_samples, new_timestamps)) 

        muse = Muse(address, save_eeg)
        while not muse.connect():
            pass

        muse.start()
        t_init = time.time()
        print(f"Start recording at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")
        last_update = t_init
        _connected_flag.set()
        while True:
            try:
                try:
                    while True:
                        muselsl.backends.sleep(1) # NOTE: this is not time.sleep(), it is asyncio.sleep(). Therefore, it is non-blocking
                        if time.time() - last_update > 10:
                            last_update = time.time()
                            muse.keep_alive()
                except bleak.exc.BleakError:
                    print('Disconnected. Attempting to reconnect...')
                    while True:
                        muse.connect(retries=3)
                        try:
                            muse.resume()
                        except bleak.exc.BleakDBusError:
                            print('DBus error occurred. Reconnecting.')
                            muse.disconnect()
                            continue
                        else:
                            break
                    print('Connected. Continuing with data collection...')
            except KeyboardInterrupt:
                print('Interrupt received. Exiting data collection.')
                break

        muse.stop()
        muse.disconnect()

    def __del__(self):
        if self.recording_process.is_alive():
            self.recording_process.terminate()
            print("Recording process terminated")
        super().__del__()