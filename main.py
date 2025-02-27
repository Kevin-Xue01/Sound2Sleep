import ctypes
import json
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from functools import partial
from math import ceil, floor, isnan, nan, pi
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
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from muselsl.constants import LSL_SCAN_TIMEOUT
from utils import (
    CHANNEL_NAMES,
    CHUNK_SIZE,
    DELAYS,
    SAMPLING_RATE,
    TIMESTAMPS,
    AppState,
    Audio,
    BlueMuse,
    DisplayConfig,
    EEGProcessor,
    EEGProcessorOutput,
    ExperimentMode,
    FileWriter,
    Logger,
    MuseDataType,
    SessionConfig,
)

# class DataWorker(QRunnable):
#     def __init__(self, task_func):
#         super().__init__()
#         self.task_func = task_func

#     def run(self):
#         self.task_func()


class EEGApp(QWidget):
    def __init__(self):
        super().__init__()
        self.threadpool = QThreadPool.globalInstance()
        self.app_state = AppState.DISCONNECTED
        self.elapsed_time = 0  # Elapsed time in seconds
        self.last_stim_line = None
        self.reset_attempt_count = 0
        self.processor_time_elapsed = 0.0
        self.config = SessionConfig()
        self.logger = Logger(self.config._session_key, self.__class__.__name__)
        self.audio = Audio(self.config._audio)
        self.target_phase = self.config.target_phase

        
        self.init_ui()

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

        self.running_stream = False
        self.file_writer = FileWriter(self.config._session_key)

        self.amp_buffer = np.zeros(self.config.amp_buffer_len)
        self.hl_ratio_buffer = np.zeros(self.config.hl_ratio_buffer_len)

        self.window_len_n = int(SAMPLING_RATE[MuseDataType.EEG] * self.config.window_len_s)
        self.times = np.zeros(self.window_len_n)
        self.data = np.zeros((self.window_len_n, len(CHANNEL_NAMES[MuseDataType.EEG])))

        self.second_stim_start = nan
        self.second_stim_end = nan

        self.sos_low = signal.butter(self.config.bpf_order, self.config.low_bpf_cutoff, btype = 'bandpass', output = 'sos', fs = SAMPLING_RATE[MuseDataType.EEG])
        self.sos_high = signal.butter(self.config.bpf_order, self.config.high_bpf_cutoff, btype = 'bandpass', output = 'sos', fs = SAMPLING_RATE[MuseDataType.EEG])

        self.zi_low = signal.sosfilt_zi(self.sos_low)
        self.zi_high = signal.sosfilt_zi(self.sos_high)

        self.wavelet_freqs = np.linspace(self.config.truncated_wavelet.low, self.config.truncated_wavelet.high, self.config.truncated_wavelet.n)

        trunc_wavelet_len = self.window_len_n * 2 # double the length of the signal
        self.trunc_wavelets = [signal.morlet2(trunc_wavelet_len, self.config.truncated_wavelet.w * SAMPLING_RATE[MuseDataType.EEG] / (2 * f * np.pi), w = self.config.truncated_wavelet.w)[:trunc_wavelet_len // 2] for f in self.wavelet_freqs]

        self.selected_channel_ind = 0  # Default to channel 0
        self.switch_channel_counter = 0
        self.switch_channel_counter_max = int(self.config.switch_channel_period_s * SAMPLING_RATE[MuseDataType.EEG] / CHUNK_SIZE[MuseDataType.EEG])

    

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
        
        matplotlib.use('QtAgg')
        sns.set_theme(style="whitegrid")
        sns.despine(left=True)

        self.fig, self.axes = plt.subplots(1, 1, figsize=[15, 6], sharex=True)
        self.eeg_nchan = len(CHANNEL_NAMES[MuseDataType.EEG])
        self.eeg_window_len_n = int(SAMPLING_RATE[MuseDataType.EEG] * self.config._display.window_len_s)
        self.times = np.arange(-self.config._display.window_len_s, 0, 1. / SAMPLING_RATE[MuseDataType.EEG])
        self.eeg_data = np.zeros((self.eeg_window_len_n, self.eeg_nchan))
        self.eeg_timestamps = np.linspace(-self.config._display.window_len_s, 0, self.eeg_window_len_n)
        self.impedances = np.std(self.eeg_data, axis=0)
        self.lines = []

        for ii in range(self.eeg_nchan):
            line, = self.axes.plot(self.times[::2], self.eeg_data[::2, ii] - ii, lw=1)
            self.lines.append(line)

        self.axes.set_ylim(-self.eeg_nchan + 0.5, 0.5)
        ticks = np.arange(0, -self.eeg_nchan, -1)

        self.axes.set_xlabel('Time (s)')
        self.axes.xaxis.grid(False)
        self.axes.set_yticks(ticks)

        self.axes.set_yticklabels([f'{label} - {impedance:2f}' for label, impedance in zip(CHANNEL_NAMES[MuseDataType.EEG], self.impedances)])

        self.display_every = 5
        self.eeg_plot_widget = FigureCanvas(self.fig)
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
        # main_layout.addWidget(self.eeg_plot_layout_widget)
        # main_layout.addWidget(self.eeg_plot_widget)
        main_layout.addLayout(right_panel)
        
        self.setLayout(main_layout)
        self.setWindowTitle("Sound2Sleep: CLAS at Home")

    # def update_eeg_plot(self):
    #     for i, curve in enumerate(self.eeg_plot_widget_curves):
    #         curve.setData(self.eeg_timestamps[::2], self.eeg_data[::2, i])

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
        self.connection_timeout_error_label.hide()

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

        elif self.app_state == AppState.RECORDING:
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

    def screenoff(self):
        ''' Darken the screen by starting the blank screensaver '''
        try:
            subprocess.call(['C:\Windows\System32\scrnsave.scr', '/start'])
        except Exception as ex:
            self.logger.critical(traceback.format_exception(type(ex), ex, ex.__traceback__))

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

    def eeg_callback(self):
        no_data_counter = 0
        display_every_counter = 0
        while self.running_stream:
            time.sleep(DELAYS[MuseDataType.EEG])
            try:
                data, timestamps = self.stream_inlet[MuseDataType.EEG].pull_chunk(timeout=DELAYS[MuseDataType.EEG], max_samples=CHUNK_SIZE[MuseDataType.EEG])
                if timestamps and len(timestamps) == CHUNK_SIZE[MuseDataType.EEG]:
                    self.processor_time_elapsed = np.float64(time.time())
                    timestamps = TIMESTAMPS[MuseDataType.EEG] + self.processor_time_elapsed
                    data = np.array(data).astype(np.float32)
                    self.times = np.concatenate([self.times, timestamps])
                    self.times = self.times[-self.eeg_window_len_n:]
                    self.eeg_data = np.vstack([self.eeg_data, data])
                    self.eeg_data = self.eeg_data[-self.eeg_window_len_n:]
                    result, time_to_target, phase, freq, amp, amp_buffer_mean = self.process_eeg_step_1()
                    if (result == EEGProcessorOutput.STIM) or (result == EEGProcessorOutput.STIM2):
                        time_to_target = time_to_target - self.config.time_to_target_offset
                        self.process_eeg_step_2(time_to_target)
                        self.file_writer.write_stim(self.processor_time_elapsed + time_to_target)
                    self.file_writer.write_chunk(data, timestamps)
                    # if display_every_counter == self.config._display.display_every:
                    #     plot_data = self.eeg_data - self.eeg_data.mean(axis=0)
                    #     for ii in range(4):
                    #         self.lines[ii].set_xdata(self.times[::4] - self.times[-1])
                    #         self.lines[ii].set_ydata(plot_data[::4, ii] / 100 - ii)
                    #         self.impedances = np.std(plot_data, axis=0)

                    #     self.axes.set_yticklabels([f'{label} - {impedance:2f}' for label, impedance in zip(CHANNEL_NAMES[MuseDataType.EEG], self.impedances)])
                    #     self.axes.set_xlim(-self.config._display.window_len_s, 0)
                    #     self.eeg_plot_widget.draw()
                    #     display_every_counter = 0
                    # else:
                    #     display_every_counter += 1
                else:
                    no_data_counter += 1

                    if no_data_counter > 64:
                        QTimer.singleShot(2000, self.lsl_reset_stream_step1)
                        self.running_stream = False

            except Exception as ex:
                self.logger.critical(traceback.format_exception(type(ex), ex, ex.__traceback__))

        self.logger.info('EEG thread stopped')

    def acc_callback(self):
        no_data_counter = 0
        while self.running_stream:
            time.sleep(DELAYS[MuseDataType.ACC])
            try:
                data, timestamps = self.stream_inlet[MuseDataType.ACC].pull_chunk(timeout=DELAYS[MuseDataType.ACC], max_samples=CHUNK_SIZE[MuseDataType.ACC])
                if timestamps and len(timestamps) == CHUNK_SIZE[MuseDataType.ACC]:
                    timestamps = TIMESTAMPS[MuseDataType.ACC] + np.float64(time.time())
                    data = np.array(data)

                    self.process_acc(timestamps, data)
                else:
                    no_data_counter += 1

                    if no_data_counter > 64:
                        QTimer.singleShot(2000, self.lsl_reset_stream_step1)
                        self.running_stream = False

            except Exception as ex:
                self.logger.critical(traceback.format_exception(type(ex), ex, ex.__traceback__))

        self.logger.info('ACC thread stopped')

    def ppg_callback(self):
        no_data_counter = 0
        while self.running_stream:
            time.sleep(DELAYS[MuseDataType.PPG])
            try:
                data, timestamps = self.stream_inlet[MuseDataType.PPG].pull_chunk(timeout=DELAYS[MuseDataType.PPG], max_samples=CHUNK_SIZE[MuseDataType.PPG])
                if timestamps and len(timestamps) == CHUNK_SIZE[MuseDataType.PPG]:
                    timestamps = TIMESTAMPS[MuseDataType.PPG] + np.float64(time.time())
                    data = np.array(data)

                    self.process_ppg(timestamps, data)
                else:
                    no_data_counter += 1

                    if no_data_counter > 64:
                        QTimer.singleShot(2000, self.lsl_reset_stream_step1)
                        self.running_stream = False

            except Exception as ex:
                self.logger.critical(traceback.format_exception(type(ex), ex, ex.__traceback__))

        self.logger.info('PPG thread stopped')

    # def process_eeg(self, timestamps: np.ndarray, data: np.ndarray):
    #     self.logger.info(timestamps)
    #     self.logger.info(data)

    def switch_channel(self):
        self.selected_channel_ind = np.argmin(np.sqrt(np.mean(self.eeg_data**2, axis=0)))

    def get_hl_ratio(self, selected_channel_data):
        lp_signal, self.zi_low = signal.sosfilt(self.sos_low, selected_channel_data, zi = self.zi_low)
        hp_signal, self.zi_high = signal.sosfilt(self.sos_high, selected_channel_data, zi = self.zi_high)

        # compute the lp envelope of the signal
        envelope_lp = np.abs(signal.hilbert(lp_signal[SAMPLING_RATE[MuseDataType.EEG]:]))
        power_lf = envelope_lp**2

        # compute the hf envelope of the signal
        envelope_hf = np.abs(signal.hilbert(hp_signal[SAMPLING_RATE[MuseDataType.EEG]:]))
        power_hf = envelope_hf**2
        # compute ratio and store
        hl_ratio = np.mean(power_hf) / np.mean(power_lf)
        hl_ratio = np.log10(hl_ratio)
        return hl_ratio
    
    def estimate_phase(self, selected_channel): 
        conv_vals = [np.dot(selected_channel, w) for w in self.trunc_wavelets]
        max_idx = np.argmax(np.abs(conv_vals))
        amp = conv_vals[max_idx] / 2
        freq = self.wavelet_freqs[max_idx]
        phase = np.angle(conv_vals[max_idx]) % (2 * pi)
        
        return phase, freq, amp
    
    def process_eeg_step_1(self):
        self.switch_channel_counter += 1
        if self.switch_channel_counter == self.switch_channel_counter_max:
            self.switch_channel()
            self.switch_channel_counter = 0
        if self.second_stim_end < self.processor_time_elapsed:
            self.second_stim_start = nan
            self.second_stim_end = nan

        phase, freq, amp = self.estimate_phase(self.eeg_data[:, self.selected_channel_ind])
        hl_ratio = self.get_hl_ratio(self.eeg_data[:, self.selected_channel_ind])
        self.amp_buffer[:-1] = self.amp_buffer[1:]
        self.amp_buffer[-1] = amp
        amp_buffer_mean = self.amp_buffer.mean()

        self.hl_ratio_buffer[:-1] = self.hl_ratio_buffer[1:]
        self.hl_ratio_buffer[-1] = hl_ratio
        hl_ratio_buffer_mean = self.hl_ratio_buffer.mean()

        # print(phase, freq, amp, hl_ratio)

        if self.config.experiment_mode == ExperimentMode.DISABLED:
            return EEGProcessorOutput.NOT_RUNNING, 0, phase, freq, amp, amp_buffer_mean

        # check if we're waiting for the 2nd stim
        # if NOT, run normal checks
        if isnan(self.second_stim_start):
            ### check backoff criteria ###
            if ((self.last_stim + self.config.backoff_time) > (self.processor_time_elapsed + self.config.stim1_prediction_limit_sec)):
                return EEGProcessorOutput.BACKOFF, 0, phase, freq, amp, amp_buffer_mean

            ### check amplitude criteria ###
            if (amp_buffer_mean < self.config.amp_buffer_mean_min) or (amp_buffer_mean > self.config.amp_buffer_mean_max):
                return EEGProcessorOutput.AMPLITUDE, 0, phase, freq, amp, amp_buffer_mean

            if hl_ratio_buffer_mean > self.config.hl_ratio_buffer_mean_max or hl_ratio > self.config.hl_ratio_latest_max:
                return EEGProcessorOutput.HL_RATIO, 0, phase, freq, amp, amp_buffer_mean

        # if we are waiting for 2nd stim, but before the backoff window, only use phase targeting
        if self.processor_time_elapsed < self.second_stim_start:
            return EEGProcessorOutput.BACKOFF2, 0, phase, freq, amp, amp_buffer_mean

        ### perform forward prediction ###
        delta_t = ((self.target_phase - phase) % (2 * pi)) / (freq * 2 * pi)

        # cue a stim for the next target phase
        if isnan(self.second_stim_start):
            if delta_t > self.config.stim1_prediction_limit_sec:
                return EEGProcessorOutput.FUTURE, delta_t, phase, freq, amp, amp_buffer_mean

            self.last_stim = self.processor_time_elapsed + delta_t
            self.second_stim_start = self.last_stim + self.config.stim2_start_delay
            self.second_stim_end = self.last_stim + self.config.stim2_end_delay

            return EEGProcessorOutput.STIM, delta_t, phase, freq, amp, amp_buffer_mean

        else:
            if delta_t > self.config.stim2_prediction_limit_sec:
                return EEGProcessorOutput.FUTURE2, delta_t, phase, freq, amp, amp_buffer_mean

            self.second_stim_start = nan
            self.second_stim_end = nan

            if self.config.experiment_mode == ExperimentMode.RANDOM_PHASE:
                self.randomize_phase()

            return EEGProcessorOutput.STIM2, delta_t, phase, freq, amp, amp_buffer_mean
    
    def process_eeg_step_2(self, time_to_target):
        if self.config.experiment_mode == ExperimentMode.CLAS: self.play_audio(time_to_target)

    def randomize_phase(self):
        self.target_phase = random.uniform(0.0, 2*np.pi)
    
    def process_acc(self, timestamps: np.ndarray, data: np.ndarray):
        self.logger.info(timestamps)
        self.logger.info(data)

    def process_ppg(self, timestamps: np.ndarray, data: np.ndarray):
        self.logger.info(timestamps)
        self.logger.info(data)

    def lsl_reset_stream_step1(self):
        self.on_connection_timeout()
        self.logger.info('Resetting stream step 1')
        subprocess.call('start bluemuse://stop?stopall', shell=True)
        time.sleep(3)
        self.lsl_reset_stream_step2()


    def lsl_reset_stream_step2(self):
        self.logger.info('Resetting stream step 2')
        subprocess.call('start bluemuse://start?startall', shell=True)
        time.sleep(3)
        self.lsl_reset_stream_step3()

    def lsl_reset_stream_step3(self):
        self.logger.info('Resetting stream step 3')
        reset_success = self.lsl_reload()

        if not reset_success:
            self.logger.info('LSL stream reset successful. Starting threads')
            self.reset_attempt_count += 1
            if self.reset_attempt_count <= 3:
                self.logger.info('Resetting Attempt: ' + str(self.reset_attempt_count))
                self.lsl_reset_stream_step1() 
            else:
                self.reset_attempt_count = 0

                for p in psutil.process_iter(['name']):
                    print(p.info)
                    if p.info['name'] == 'BlueMuse.exe':
                        self.logger.info('Killing BlueMuse')
                        p.kill()

                time.sleep(2)
                self.lsl_reset_stream_step1()
        else:
            self.reset_attempt_count = 0
            self.running_stream = True
            self.logger.info('LSL stream reset successful. Starting threads')
            time.sleep(3)
            subprocess.call('start bluemuse://start?streamfirst=true', shell=True)
            self.on_connected()

            for stream in MuseDataType:
                if self.stream_inlet[stream] is not None:
                    curr_thread = QThread(self)
                    curr_thread.started.connect(self.eeg_callback)
                    curr_thread.finished.connect(partial(self.logger.info, f"{str(stream)} thread stopped"))
                    curr_thread.start()
        
    def start_bluemuse(self):
        subprocess.call('start bluemuse:', shell=True)
        subprocess.call('start bluemuse://setting?key=primary_timestamp_format!value=BLUEMUSE', shell=True)
        subprocess.call('start bluemuse://setting?key=channel_data_type!value=float32', shell=True)
        subprocess.call('start bluemuse://setting?key=eeg_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=accelerometer_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=gyroscope_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=ppg_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://start?streamfirst=true', shell=True)

        time.sleep(3)
        while not self.lsl_reload():
            self.logger.error(f"LSL streams not found, retrying in 3 seconds") 
            time.sleep(3)
        self.on_connected()
        self.running_stream = True
        for stream in MuseDataType:
            if self.stream_inlet[stream] is not None:
                curr_thread = QThread(self)
                curr_thread.started.connect(self.eeg_callback)
                curr_thread.finished.connect(partial(self.logger.info, f"{str(stream)} thread stopped"))
                curr_thread.start()
        
    def stop_bluemuse(self):
        self.running_stream = False
        try:
            for _stream_inlet in self.stream_inlet.values():
                if _stream_inlet is not None:
                    _stream_inlet.close_stream()
        except Exception as ex:
            self.logger.critical(str(ex))

        subprocess.call('start bluemuse://stop?stopall', shell=True)
        subprocess.call('start bluemuse://shutdown', shell=True)

        for p in psutil.process_iter(['name']):
            if p.info['name'] == 'BlueMuse.exe':
                self.logger.info('Killing BlueMuse')
                p.kill()
        self.on_disconnected()

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
            for p in psutil.process_iter(['name']):
                if p.info['name'] == 'BlueMuse.exe': p.kill()
    sys.exit(exit_code)