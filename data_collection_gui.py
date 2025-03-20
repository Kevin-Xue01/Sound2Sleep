import asyncio
import ctypes
import datetime
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import partial
from math import ceil, floor, isnan, nan, pi
from multiprocessing import Event, Manager, Process, Queue, queues, shared_memory
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
import pandas as pd
import psutil
import pyqtgraph as pg
import scipy.signal as signal
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
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
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph import DateAxisItem
from scipy import signal
from scipy.signal import firwin, lfilter, lfilter_zi

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

class ConnectionMode(Enum):
    GENERATED = auto()
    PLAYBACK = auto()
    REALTIME = auto()

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
    def __init__(self, parent, config: SessionConfig, connection_mode: ConnectionMode.GENERATED):

        super().__init__(parent)
        self.connection_mode = connection_mode
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
        
        self.simulation_params = Manager().dict({
            'pure_amp': 30,
            'pure_freq': 1,
            'pure_noise': 0.0
        })

        # Start the Muse LSL recording process
        self.recording_process = Process(
            target=ConnectionWidget.record,
            args=(self._queue, self.connected_flag, self.connection_mode),
            daemon=True
        )
        self.recording_process.start()
        self.loading_movie.start()
        self.connection_check_timer = QtCore.QTimer()
        self.connection_check_timer.timeout.connect(self.check_connection)
        self.connection_check_timer.start(100)  # Check every 100ms
        
        self.initially_connected = False
        self.display_every_counter = 0
        self.display_every_counter_max = 2

    def create_parameter_controls(self):
        param_layout = QHBoxLayout()
        
        # Amplitude control
        amp_layout = QVBoxLayout()
        amp_label = QLabel("Amplitude:")
        self.amp_slider = QSlider(Qt.Orientation.Horizontal)
        self.amp_slider.setMinimum(1)
        self.amp_slider.setMaximum(100)
        self.amp_slider.setValue(int(self.simulation_params['pure_amp']))
        self.amp_value = QLabel(f"{self.simulation_params['pure_amp']}")
        self.amp_slider.valueChanged.connect(self.update_amplitude)
        amp_layout.addWidget(amp_label)
        amp_layout.addWidget(self.amp_slider)
        amp_layout.addWidget(self.amp_value)
        
        # Frequency control
        freq_layout = QVBoxLayout()
        freq_label = QLabel("Frequency (Hz):")
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setMinimum(1)
        self.freq_slider.setMaximum(20)
        self.freq_slider.setValue(int(self.simulation_params['pure_freq']))
        self.freq_value = QLabel(f"{self.simulation_params['pure_freq']}")
        self.freq_slider.valueChanged.connect(self.update_frequency)
        freq_layout.addWidget(freq_label)
        freq_layout.addWidget(self.freq_slider)
        freq_layout.addWidget(self.freq_value)
        
        # Noise control
        noise_layout = QVBoxLayout()
        noise_label = QLabel("Noise:")
        self.noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_slider.setMinimum(0)
        self.noise_slider.setMaximum(100)
        self.noise_slider.setValue(int(self.simulation_params['pure_noise'] * 100))
        self.noise_value = QLabel(f"{self.simulation_params['pure_noise']:.2f}")
        self.noise_slider.valueChanged.connect(self.update_noise)
        noise_layout.addWidget(noise_label)
        noise_layout.addWidget(self.noise_slider)
        noise_layout.addWidget(self.noise_value)
        
        # Add controls to parameter layout
        param_layout.addLayout(amp_layout)
        param_layout.addLayout(freq_layout)
        param_layout.addLayout(noise_layout)
        
        # Add parameter controls to main layout
        param_widget = QWidget()
        param_widget.setLayout(param_layout)
        self.main_layout.addWidget(param_widget)

    def update_amplitude(self, value):
        self.simulation_params['pure_amp'] = value
        self.amp_value.setText(f"{value}")
        
    def update_frequency(self, value):
        self.simulation_params['pure_freq'] = value
        self.freq_value.setText(f"{value}")
        
    def update_noise(self, value):
        noise_value = value / 100.0
        self.simulation_params['pure_noise'] = noise_value
        self.noise_value.setText(f"{noise_value:.2f}")

    def __init_plotting__(self):
        sns.set(style="whitegrid")
        self.scale = 100
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_layout.addWidget(self.canvas)

        self.eeg_data_f = np.zeros((DISPLAY_WINDOW_LEN_N, 4))
        self.times = np.arange(-DISPLAY_WINDOW_LEN_S, 0, 1. / SAMPLING_RATE[MuseDataType.EEG])
        impedances = np.std(self.eeg_data_f, axis=0)
        lines = []

        for ii in range(4):
            line, = self.ax.plot(self.times[::2], self.eeg_data_f[::2, ii] - ii, lw=1)
            lines.append(line)
        self.lines = lines

        self.ax.set_ylim(-4 + 0.5, 0.5)
        ticks = np.arange(0, -4, -1)

        self.ax.set_xlabel('Time (s)')
        self.ax.xaxis.grid(False)
        self.ax.set_yticks(ticks)

        ticks_labels = ['%s - %.1f' % (CHANNEL_NAMES[MuseDataType.EEG][ii], impedances[ii]) for ii in range(4)]
        self.ax.set_yticklabels(ticks_labels)

        self.bf = firwin(32, np.array([1, 40]) / (SAMPLING_RATE[MuseDataType.EEG] / 2.), width=0.05, pass_zero=False)
        self.af = [1.0]

        zi = lfilter_zi(self.bf, self.af)
        self.filt_state = np.tile(zi, (4, 1)).transpose()

        # Add parameter tuning UI elements
        self.create_parameter_controls()

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
        self.status_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)  # Minimize space usage
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
        self.CLAS_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)  # Prevent expanding
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
        
        
        
        # Setup the timer for real-time updates
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_plot)

        self.update_timer.start(45)  # 50 fps update rate
        
        # Install event filter for key press events
        self.installEventFilter(self)
        self.__init_plotting__()
        # Add button to layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.CLAS_button)
        button_layout.addStretch()
        self.main_layout.addLayout(button_layout)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:  # Handle both '+' and '=' keys
                self.scale *= 1.1  # Increase scale by 10%
                print(f"Scale increased to: {self.scale}")
            elif event.key() == Qt.Key.Key_Minus:
                self.scale /= 1.1  # Decrease scale by 10%
                print(f"Scale decreased to: {self.scale}")
            return True  # Indicate that the event was handled
        return super().eventFilter(obj, event)

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

    def update_button_state(self):
        elapsed_time = time.time() - self.quality_check_start_time
        quality = self.status_widget.connection_quality
        
        self.CLAS_button.setText(CONNECTION_QUALITY_LABELS[quality])
        
        # Enable button based on quality and elapsed time
        enable_button = (quality == ConnectionQuality.HIGH or (quality == ConnectionQuality.MEDIUM and elapsed_time >= self.min_quality_check_duration))
        self.CLAS_button.setEnabled(enable_button)

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

                filt_samples, self.filt_state = lfilter(self.bf, self.af, new_samples, axis=0, zi=self.filt_state)

                if len(self.eeg_data) == 0:
                    self.eeg_data = new_samples
                    self.eeg_data_f = filt_samples
                    self.timestamps = new_timestamps
                else:
                    self.eeg_data = np.vstack((self.eeg_data, new_samples))
                    self.eeg_data_f = np.vstack((self.eeg_data_f, filt_samples))
                    self.timestamps = np.append(self.timestamps, new_timestamps)

                self.eeg_data = self.eeg_data[-self.window_len_n:, :]
                self.eeg_data_f = self.eeg_data_f[-DISPLAY_WINDOW_LEN_N:]
                self.timestamps = self.timestamps[-self.window_len_n:]

                # If we're still in quality check period
                if self.quality_check_enabled:
                    self.check_signal_quality()
                    self.update_button_state()
                    continue

                mean_to_subtract = np.mean(self.eeg_data, axis=0)
                self.file_writer.write_chunk(new_samples, new_timestamps)

                result, time_to_target, phase, freq, amp, amp_buffer_mean = self.process_eeg_step_1(mean_to_subtract)
                # self.logger.info(f"Result {result}, Time to target: {time_to_target}, Phase: {phase}, Freq: {freq}, Amp: {amp}, Amp Buffer Mean: {amp_buffer_mean}")

                if (result == EEGProcessorOutput.STIM) or (result == EEGProcessorOutput.STIM2):
                    time_to_target = time_to_target - self.config.time_to_target_offset
                    self.process_eeg_step_2(time_to_target)
                    self.file_writer.write_stim(self.processor_elapsed_time + time_to_target)
                
                if self.display_every_counter == self.display_every_counter_max:
                    for ii in range(4):
                        self.lines[ii].set_xdata(self.timestamps[-DISPLAY_WINDOW_LEN_N:][::2] - self.timestamps[-DISPLAY_WINDOW_LEN_N:][-1])
                        self.lines[ii].set_ydata(self.eeg_data_f[::2, ii] / self.scale - ii)
                        impedances = np.std(self.eeg_data_f, axis=0)

                    ticks_labels = ['%s - %.2f' % (CHANNEL_NAMES[MuseDataType.EEG][ii], impedances[ii]) for ii in range(4)]
                    self.ax.set_yticklabels(ticks_labels)
                    self.ax.set_xlim(-5, 0)
                    self.figure.canvas.draw()
                    self.display_every_counter = 0

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
            
    @classmethod
    def record(cls, _queue: Queue, _connected_flag: EventType, connection_mode: ConnectionMode):
        if connection_mode == ConnectionMode.REALTIME:
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
        elif connection_mode == ConnectionMode.GENERATED:
            _connected_flag.set()

            current_time = 0.0  # Maintain a continuously increasing time variable

            pure_amp = 30
            pure_freq = 1
            pure_noise = 0.0
            channel_phase_offsets = [i * (np.pi / 4) for i in range(NUM_CHANNELS[MuseDataType.EEG])]

            def simulate_pure_sine(_current_time, num_samples=CHUNK_SIZE[MuseDataType.EEG], _pure_amp=pure_amp, _pure_freq=pure_freq, _pure_noise=pure_noise, _channel_phase_offsets=channel_phase_offsets):

                timestamps = np.arange(CHUNK_SIZE[MuseDataType.EEG]) / SAMPLING_RATE[MuseDataType.EEG] + _current_time
                
                signals = []
                for i in range(NUM_CHANNELS[MuseDataType.EEG]):
                    phase_offset = _channel_phase_offsets[i]
                    signal = _pure_amp * np.sin(2 * np.pi * _pure_freq * timestamps + phase_offset)

                    if _pure_noise > 0:
                        signal += _pure_amp * np.random.normal(0, _pure_noise / 2, len(timestamps))

                    signals.append(signal)

                output = np.array(signals).T
                current_time = timestamps[-1] + 1 / SAMPLING_RATE[MuseDataType.EEG]  # Ensure continuous timestamps
                return output, timestamps, current_time

            while True:
                new_eeg_data, new_timestamps, current_time = simulate_pure_sine(current_time)
                _queue.put((new_eeg_data, new_timestamps)) 
                
                time.sleep(DELAYS[MuseDataType.EEG])
        else:
            pass


    def __del__(self):
        if self.recording_process.is_alive():
            self.recording_process.terminate()
            print("Recording process terminated")
        super().__del__()