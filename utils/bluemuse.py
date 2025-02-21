import csv
import ctypes
import subprocess
import time
import traceback
from datetime import datetime
from functools import partial
from multiprocessing import Process
from threading import Thread, Timer

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
from muselsl.constants import LSL_SCAN_TIMEOUT, VIEW_SUBSAMPLE
from pylsl import StreamInfo, StreamInlet, resolve_byprop
from PyQt5.QtCore import QObject, pyqtSignal
from scipy.signal import firwin, lfilter, lfilter_zi

from .config import EEGSessionConfig
from .constants import (
    CHANNEL_NAMES,
    CHUNK_SIZE,
    NB_CHANNELS,
    SAMPLING_RATE,
    TIMESTAMPS,
    MuseDataType,
)
from .logger import Logger


class BlueMuse(QObject):
    eeg_data_ready = pyqtSignal(np.ndarray, np.ndarray)
    acc_data_ready = pyqtSignal(np.ndarray, np.ndarray)
    ppg_data_ready = pyqtSignal(np.ndarray, np.ndarray)

    connection_timeout = pyqtSignal()
    connected = pyqtSignal()
    disconnected = pyqtSignal()

    no_data_count = 0
    reset_attempt_count = 0
    
    def init_EEG_UI(self):
        pass
        # matplotlib.use('TkAgg')
        # sns.set_theme(style="whitegrid")
        # sns.despine(left=True)

        # self.ui_window_s = 5

        # self.fig, self.axes = plt.subplots(1, 1, figsize=[15, 6], sharex=True)
        # self.fig.canvas.mpl_connect('close_event', self.stop_streaming)
        # help_str = """
        #             toggle filter : d
        #             toogle full screen : f
        #             zoom out : /
        #             zoom in : *
        #             increase time scale : -
        #             decrease time scale : +
        #         """
        # print(help_str)
        # self.eeg_nchan = NB_CHANNELS[MuseDataType.EEG]
        # self.eeg_ui_samples = int(self.ui_window_s * SAMPLING_RATE[MuseDataType.EEG])
        # self.data = np.zeros((self.eeg_ui_samples, self.eeg_nchan))
        # self.times = np.arange(-self.ui_window_s, 0, 1. / SAMPLING_RATE[MuseDataType.EEG])
        # self.impedances = np.std(self.data, axis=0)
        # self.lines = []

        # for ii in range(self.eeg_nchan):
        #     line, = self.axes.plot(self.times[::VIEW_SUBSAMPLE], self.data[::VIEW_SUBSAMPLE, ii] - ii, lw=1)
        #     self.lines.append(line)

        # self.axes.set_ylim(-self.eeg_nchan + 0.5, 0.5)
        # ticks = np.arange(0, -self.eeg_nchan, -1)

        # self.axes.set_xlabel('Time (s)')
        # self.axes.xaxis.grid(False)
        # self.axes.set_yticks(ticks)

        # self.axes.set_yticklabels([f'{label} - {impedance:2f}' for label, impedance in zip(CHANNEL_NAMES[MuseDataType.EEG], self.impedances)])

        # self.display_every = 5

        # self.bf = firwin(32, np.array([1, 40]) / (SAMPLING_RATE[MuseDataType.EEG] / 2.), width=0.05, pass_zero=False)
        # self.af = [1.0]

        # zi = lfilter_zi(self.bf, self.af)
        # self.filt_state = np.tile(zi, (self.eeg_nchan, 1)).transpose()
        # self.data_f = np.zeros((self.eeg_ui_samples, self.eeg_nchan))

    def __init__(self):
        super().__init__()
        self.stream_info: dict[MuseDataType, StreamInfo] = dict()
        self.stream_inlet: dict[MuseDataType, StreamInlet] = dict()

    def screenoff(self):
        ''' Darken the screen by starting the blank screensaver '''
        try:
            subprocess.call(['C:\Windows\System32\scrnsave.scr', '/start'])
        except Exception as ex:
            self.logger.critical(traceback.format_exception(type(ex), ex, ex.__traceback__))

    def lsl_reload(self):
        allok = True
        self.stream_info = dict()
        self.stream_inlet = dict()
        for stream in MuseDataType:
            self.stream_info[stream] = resolve_byprop('type', stream.value, timeout=LSL_SCAN_TIMEOUT)

            if self.stream_info[stream]:
                self.stream_info[stream] = self.stream_info[stream][0]
                self.logger.info(f'{stream.name} OK.')
            else:
                self.logger.warning(f'{stream.name} not found.')
                allok = False
        if allok: self.connected.emit()
        return allok


    # def eeg_callback(self):
    #     with open(f'data/kevin/eeg_data_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.csv', mode='a', newline='') as file:
    #         writer = csv.writer(file)
    #         if file.tell() == 0:
    #             writer.writerow(['Timestamp'] + CHANNEL_NAMES[MuseDataType.EEG])
            
    #         display_every_counter = 0
    #         no_data_counter = 0
    #         while self.run_eeg_thread:
    #             time.sleep(CHUNK_SIZE[MuseDataType.EEG] / SAMPLING_RATE[MuseDataType.EEG])
    #             try:
    #                 data, timestamps = self.stream_inlet[MuseDataType.EEG].pull_chunk(timeout=1.0, max_samples=CHUNK_SIZE[MuseDataType.EEG])
    #                 if timestamps and len(timestamps) == CHUNK_SIZE[MuseDataType.EEG]:
    #                     timestamps = TIMESTAMPS[MuseDataType.EEG] + time.time() + 1. / SAMPLING_RATE[MuseDataType.EEG]

    #                     for t, s in zip(timestamps, data):
    #                         writer.writerow([t] + list(s))

    #                     self.times = np.concatenate([self.times, timestamps])
    #                     self.n_samples = int(SAMPLING_RATE[MuseDataType.EEG] * self.processing_window_s)
    #                     self.times = self.times[-self.n_samples:]
    #                     self.data = np.vstack([self.data, data])
    #                     self.data = self.data[-self.n_samples:]
    #                     filt_samples, self.filt_state = lfilter(self.bf, self.af, data, axis=0, zi=self.filt_state)
    #                     self.data_f = np.vstack([self.data_f, filt_samples])
    #                     self.data_f = self.data_f[-self.n_samples:]

    #                     display_every_counter += 1
    #                     if display_every_counter == self.display_every:
    #                         print('Displaying data')
    #                         for ii in range(NB_CHANNELS[MuseDataType.EEG]):
    #                             self.lines[ii].set_xdata(self.times[::VIEW_SUBSAMPLE] - self.times[-1])
    #                             self.lines[ii].set_ydata(self.data_f[::VIEW_SUBSAMPLE, ii] - ii)
    #                             self.impedances = np.std(self.data_f, axis=0)

    #                         self.axes.set_yticklabels([f'{label} - {impedance:2f}' for label, impedance in zip(CHANNEL_NAMES[MuseDataType.EEG], self.impedances)])
    #                         self.axes.set_xlim(-self.ui_window_s, 0)
    #                         self.fig.canvas.draw()
    #                         display_every_counter = 0
    #                 else:
    #                     no_data_counter += 1

    #                     if no_data_counter > 20:
    #                         self.run_eeg_thread = False
    #                         self.run_acc_thread = False
    #                         self.run_ppg_thread = False
    #                         Timer(1, self.lsl_reset_stream_step1).start()

    #             except Exception as ex:
    #                 self.logger.critical(traceback.format_exception(type(ex), ex, ex.__traceback__))

    #         self.logger.info('EEG thread stopped')
    def eeg_callback(self):
        no_data_counter = 0
        while self.run_eeg_thread:
            time.sleep(CHUNK_SIZE[MuseDataType.EEG] / SAMPLING_RATE[MuseDataType.EEG])
            try:
                data, timestamps = self.stream_inlet[MuseDataType.EEG].pull_chunk(timeout=1.0, max_samples=CHUNK_SIZE[MuseDataType.EEG])
                if timestamps and len(timestamps) == CHUNK_SIZE[MuseDataType.EEG]:
                    timestamps = TIMESTAMPS[MuseDataType.EEG] + time.time() + 1. / SAMPLING_RATE[MuseDataType.EEG]

                    # self.times = np.concatenate([self.times, timestamps])
                    # self.n_samples = int(SAMPLING_RATE[MuseDataType.EEG] * self.processing_window_s)
                    # self.times = self.times[-self.n_samples:]
                    # self.data = np.vstack([self.data, data])
                    # self.data = self.data[-self.n_samples:]
                    # filt_samples, self.filt_state = lfilter(self.bf, self.af, data, axis=0, zi=self.filt_state)
                    # self.data_f = np.vstack([self.data_f, filt_samples])
                    # self.data_f = self.data_f[-self.n_samples:]


                    self.eeg_data_ready.emit(np.random.rand(12).astype(np.float64), np.random.rand(12, 4).astype(np.float32))
                    # display_every_counter += 1
                    # if display_every_counter == self.display_every:
                    #     print('Displaying data')
                    #     for ii in range(NB_CHANNELS[MuseDataType.EEG]):
                    #         self.lines[ii].set_xdata(self.times[::VIEW_SUBSAMPLE] - self.times[-1])
                    #         self.lines[ii].set_ydata(self.data_f[::VIEW_SUBSAMPLE, ii] - ii)
                    #         self.impedances = np.std(self.data_f, axis=0)

                    #     self.axes.set_yticklabels([f'{label} - {impedance:2f}' for label, impedance in zip(CHANNEL_NAMES[MuseDataType.EEG], self.impedances)])
                    #     self.axes.set_xlim(-self.ui_window_s, 0)
                    #     self.fig.canvas.draw()
                    #     display_every_counter = 0
                else:
                    no_data_counter += 1

                    if no_data_counter > 20:
                        self.run_eeg_thread = False
                        self.run_acc_thread = False
                        self.run_ppg_thread = False
                        Timer(2, self.lsl_reset_stream_step1).start()

            except Exception as ex:
                self.logger.critical(traceback.format_exception(type(ex), ex, ex.__traceback__))

        self.logger.info('EEG thread stopped')

    def acc_callback(self):
        no_data_counter = 0
        while self.run_acc_thread:
            time.sleep(CHUNK_SIZE[MuseDataType.ACCELEROMETER] / SAMPLING_RATE[MuseDataType.ACCELEROMETER])
            try:
                data, timestamps = self.stream_inlet[MuseDataType.ACCELEROMETER].pull_chunk(timeout=1.0, max_samples=CHUNK_SIZE[MuseDataType.ACCELEROMETER])
                if timestamps and len(timestamps) == CHUNK_SIZE[MuseDataType.ACCELEROMETER]:
                    timestamps = TIMESTAMPS[MuseDataType.ACCELEROMETER] + time.time() + 1. / SAMPLING_RATE[MuseDataType.ACCELEROMETER]

                    # self.times = np.concatenate([self.times, timestamps])
                    # self.n_samples = int(SAMPLING_RATE[MuseDataType.EEG] * self.processing_window_s)
                    # self.times = self.times[-self.n_samples:]
                    # self.data = np.vstack([self.data, data])
                    # self.data = self.data[-self.n_samples:]
                    # filt_samples, self.filt_state = lfilter(self.bf, self.af, data, axis=0, zi=self.filt_state)
                    # self.data_f = np.vstack([self.data_f, filt_samples])
                    # self.data_f = self.data_f[-self.n_samples:]


                    self.acc_data_ready.emit(np.random.rand(12).astype(np.float64), np.random.rand(12, 4).astype(np.float32))
                    # display_every_counter += 1
                    # if display_every_counter == self.display_every:
                    #     print('Displaying data')
                    #     for ii in range(NB_CHANNELS[MuseDataType.EEG]):
                    #         self.lines[ii].set_xdata(self.times[::VIEW_SUBSAMPLE] - self.times[-1])
                    #         self.lines[ii].set_ydata(self.data_f[::VIEW_SUBSAMPLE, ii] - ii)
                    #         self.impedances = np.std(self.data_f, axis=0)

                    #     self.axes.set_yticklabels([f'{label} - {impedance:2f}' for label, impedance in zip(CHANNEL_NAMES[MuseDataType.EEG], self.impedances)])
                    #     self.axes.set_xlim(-self.ui_window_s, 0)
                    #     self.fig.canvas.draw()
                    #     display_every_counter = 0
                else:
                    no_data_counter += 1

                    if no_data_counter > 20:
                        self.run_eeg_thread = False
                        self.run_acc_thread = False
                        self.run_ppg_thread = False
                        Timer(2, self.lsl_reset_stream_step1).start()

            except Exception as ex:
                self.logger.critical(traceback.format_exception(type(ex), ex, ex.__traceback__))

        self.logger.info('ACC thread stopped')

    def ppg_callback(self):
        no_data_counter = 0
        while self.run_ppg_thread:
            time.sleep(CHUNK_SIZE[MuseDataType.PPG] / SAMPLING_RATE[MuseDataType.PPG])
            try:
                data, timestamps = self.stream_inlet[MuseDataType.PPG].pull_chunk(timeout=1.0, max_samples=CHUNK_SIZE[MuseDataType.PPG])
                if timestamps and len(timestamps) == CHUNK_SIZE[MuseDataType.PPG]:
                    timestamps = TIMESTAMPS[MuseDataType.PPG] + time.time() + 1. / SAMPLING_RATE[MuseDataType.PPG]

                    # self.times = np.concatenate([self.times, timestamps])
                    # self.n_samples = int(SAMPLING_RATE[MuseDataType.EEG] * self.processing_window_s)
                    # self.times = self.times[-self.n_samples:]
                    # self.data = np.vstack([self.data, data])
                    # self.data = self.data[-self.n_samples:]
                    # filt_samples, self.filt_state = lfilter(self.bf, self.af, data, axis=0, zi=self.filt_state)
                    # self.data_f = np.vstack([self.data_f, filt_samples])
                    # self.data_f = self.data_f[-self.n_samples:]


                    self.ppg_data_ready.emit(np.random.rand(12).astype(np.float64), np.random.rand(12, 4).astype(np.float32))
                    # display_every_counter += 1
                    # if display_every_counter == self.display_every:
                    #     print('Displaying data')
                    #     for ii in range(NB_CHANNELS[MuseDataType.EEG]):
                    #         self.lines[ii].set_xdata(self.times[::VIEW_SUBSAMPLE] - self.times[-1])
                    #         self.lines[ii].set_ydata(self.data_f[::VIEW_SUBSAMPLE, ii] - ii)
                    #         self.impedances = np.std(self.data_f, axis=0)

                    #     self.axes.set_yticklabels([f'{label} - {impedance:2f}' for label, impedance in zip(CHANNEL_NAMES[MuseDataType.EEG], self.impedances)])
                    #     self.axes.set_xlim(-self.ui_window_s, 0)
                    #     self.fig.canvas.draw()
                    #     display_every_counter = 0
                else:
                    no_data_counter += 1

                    if no_data_counter > 20:
                        self.run_eeg_thread = False
                        self.run_acc_thread = False
                        self.run_ppg_thread = False
                        Timer(2, self.lsl_reset_stream_step1).start()

            except Exception as ex:
                self.logger.critical(traceback.format_exception(type(ex), ex, ex.__traceback__))

        self.logger.info('PPG thread stopped')

    def lsl_reset_stream_step1(self):
        self.connection_timeout.emit()
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
                    if p.info['name'] == 'BlueMuse.exe':
                        self.logger.info('Killing BlueMuse')
                        p.kill()

                time.sleep(2)
                self.lsl_reset_stream_step1()
        else:
            # if all streams have resolved, start polling data again!
            self.reset_attempt_count = 0
            self.logger.info('LSL stream reset successful. Starting threads')
            time.sleep(3)
            subprocess.call('start bluemuse://start?streamfirst=true', shell=True)

            # start the selected steram
            for stream in MuseDataType:
                self.stream_inlet[stream] = StreamInlet(self.stream_info[stream])
            
            self.start_threads()


    def start_threads(self):
        self.run_eeg_thread = True
        self.run_acc_thread = True
        self.run_ppg_thread = True

        self.eeg_thread = Thread(target=self.eeg_callback, daemon=True)
        self.acc_thread = Thread(target=self.acc_callback, daemon=True)
        self.ppg_thread = Thread(target=self.ppg_callback, daemon=True)

        self.eeg_thread.start()
        self.acc_thread.start()
        self.ppg_thread.start()

    def run(self, session_key: str):
        subprocess.call('start bluemuse:', shell=True)
        subprocess.call('start bluemuse://setting?key=primary_timestamp_format!value=BLUEMUSE', shell=True)
        subprocess.call('start bluemuse://setting?key=channel_data_type!value=float32', shell=True)
        subprocess.call('start bluemuse://setting?key=eeg_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=accelerometer_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=gyroscope_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=ppg_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://start?streamfirst=true', shell=True)
        self.session_key = session_key
        self.logger = Logger(self.session_key, self.__class__.__name__)
        time.sleep(3)
        while not self.lsl_reload():
            self.logger.error(f"LSL streams not found, retrying in 3 seconds") 
            time.sleep(3)
        self.connected.emit()
        for stream in MuseDataType:
            self.stream_inlet[stream] = StreamInlet(self.stream_info[stream])
        
        self.start_threads()

    def stop(self):
        self.run_eeg_thread = False
        self.run_acc_thread = False
        self.run_ppg_thread = False

        for stream in MuseDataType:
            try:
                self.stream_inlet[stream].close_stream()
            except Exception as ex:
                self.logger.critical(str(ex))
        
        subprocess.call('start bluemuse://stop?stopall', shell=True)
        subprocess.call('start bluemuse://shutdown', shell=True)
        for p in psutil.process_iter(['name']):
            if p.info['name'] == 'BlueMuse.exe':
                self.logger.info('Killing BlueMuse')
                p.kill()
        self.disconnected.emit()
        self.stream_info: dict[MuseDataType, StreamInfo] = dict()
        self.stream_inlet: dict[MuseDataType, StreamInlet] = dict()


# if __name__ == "__main__":
#     ES_CONTINUOUS = 0x80000000
#     ES_SYSTEM_REQUIRED = 0x00000001
#     ES_AWAYMODE_REQUIRED = 0x0000040
    
#     ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)

#     try:
#         clas = BlueMuse()
#         clas.init_EEG_UI()
#         # clas.start_streaming()
#         plt.show()
#     finally:
#         ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)