import subprocess
import time
import traceback
from threading import Thread, Timer

import numpy as np
import psutil
from muselsl.constants import LSL_SCAN_TIMEOUT
from pylsl import StreamInfo, StreamInlet, resolve_byprop
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from scipy.signal import firwin, lfilter, lfilter_zi

from .config import SessionConfig
from .constants import CHUNK_SIZE, DELAYS, TIMESTAMPS, MuseDataType
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

    def __init__(self, config: SessionConfig):
        super().__init__()
        self.logger = Logger(config._session_key, self.__class__.__name__)
        self.stream_info: dict[MuseDataType, StreamInfo] = dict()
        self.stream_inlet: dict[MuseDataType, StreamInlet] = dict()

        self.run_eeg_thread = False
        self.run_acc_thread = False
        self.run_ppg_thread = False

    def screenoff(self):
        ''' Darken the screen by starting the blank screensaver '''
        try:
            subprocess.call(['C:\Windows\System32\scrnsave.scr', '/start'])
        except Exception as ex:
            self.logger.critical(traceback.format_exception(type(ex), ex, ex.__traceback__))

    def lsl_reload(self):
        eeg_ok = False
        self.run_eeg_thread = False
        self.run_acc_thread = False
        self.run_ppg_thread = False
        
        for stream in MuseDataType:
            self.stream_info[stream] = resolve_byprop('type', stream.value, timeout=LSL_SCAN_TIMEOUT)

            if self.stream_info[stream]:
                self.stream_info[stream] = self.stream_info[stream][0]
                self.logger.info(f'{stream.name} OK.')
                if stream == MuseDataType.EEG:
                    eeg_ok = True
                    self.run_eeg_thread = True
                elif stream == MuseDataType.ACCELEROMETER: self.run_acc_thread = True
                elif stream == MuseDataType.PPG: self.run_ppg_thread = True
            else:
                self.logger.warning(f'{stream.name} not found.')
        if eeg_ok: self.connected.emit()
        return eeg_ok

    def eeg_callback(self):
        no_data_counter = 0
        while self.run_eeg_thread:
            time.sleep(DELAYS[MuseDataType.EEG])
            try:
                data, timestamps = self.stream_inlet[MuseDataType.EEG].pull_chunk(timeout=1.0, max_samples=CHUNK_SIZE[MuseDataType.EEG])
                if timestamps and len(timestamps) == CHUNK_SIZE[MuseDataType.EEG]:
                    timestamps = TIMESTAMPS[MuseDataType.EEG] + np.float64(time.time())

                    self.eeg_data_ready.emit(timestamps, np.array(data))
                else:
                    no_data_counter += 1

                    if no_data_counter > 64:
                        Timer(2, self.lsl_reset_stream_step1).start()
                        break

            except Exception as ex:
                self.logger.critical(traceback.format_exception(type(ex), ex, ex.__traceback__))

        self.logger.info('EEG thread stopped')

    def acc_callback(self):
        no_data_counter = 0
        while self.run_acc_thread:
            time.sleep(DELAYS[MuseDataType.ACCELEROMETER])
            try:
                data, timestamps = self.stream_inlet[MuseDataType.ACCELEROMETER].pull_chunk(timeout=1.0, max_samples=CHUNK_SIZE[MuseDataType.ACCELEROMETER])
                if timestamps and len(timestamps) == CHUNK_SIZE[MuseDataType.ACCELEROMETER]:
                    timestamps = TIMESTAMPS[MuseDataType.ACCELEROMETER] + np.float64(time.time())

                    self.acc_data_ready.emit(timestamps, np.array(data))
                else:
                    no_data_counter += 1

                    if no_data_counter > 64:
                        Timer(2, self.lsl_reset_stream_step1).start()
                        break

            except Exception as ex:
                self.logger.critical(traceback.format_exception(type(ex), ex, ex.__traceback__))

        self.logger.info('ACC thread stopped')

    def ppg_callback(self):
        no_data_counter = 0
        while self.run_ppg_thread:
            time.sleep(DELAYS[MuseDataType.PPG])
            try:
                data, timestamps = self.stream_inlet[MuseDataType.PPG].pull_chunk(timeout=1.0, max_samples=CHUNK_SIZE[MuseDataType.PPG])
                if timestamps and len(timestamps) == CHUNK_SIZE[MuseDataType.PPG]:
                    timestamps = TIMESTAMPS[MuseDataType.PPG] + np.float64(time.time())

                    self.acc_data_ready.emit(timestamps, np.array(data))
                else:
                    no_data_counter += 1

                    if no_data_counter > 64:
                        Timer(2, self.lsl_reset_stream_step1).start()
                        break

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
                    print(p.info)
                    if p.info['name'] == 'BlueMuse.exe':
                        self.logger.info('Killing BlueMuse')
                        p.kill()

                time.sleep(2)
                self.lsl_reset_stream_step1()
        else:
            self.reset_attempt_count = 0
            self.logger.info('LSL stream reset successful. Starting threads')
            time.sleep(3)
            subprocess.call('start bluemuse://start?streamfirst=true', shell=True)

            if self.run_eeg_thread: self.stream_inlet[MuseDataType.EEG] = StreamInlet(self.stream_info[MuseDataType.EEG])
            if self.run_acc_thread: self.stream_inlet[MuseDataType.ACCELEROMETER] = StreamInlet(self.stream_info[MuseDataType.ACCELEROMETER])
            if self.run_ppg_thread: self.stream_inlet[MuseDataType.PPG] = StreamInlet(self.stream_info[MuseDataType.PPG])
            
            self.start_threads()

    def start_threads(self):
        self.eeg_thread = Thread(target=self.eeg_callback, daemon=True)
        self.acc_thread = Thread(target=self.acc_callback, daemon=True)
        self.ppg_thread = Thread(target=self.ppg_callback, daemon=True)

        self.eeg_thread.start()
        # self.acc_thread.start()
        # self.ppg_thread.start()

    def run(self, session_key: str):
        subprocess.call('start bluemuse:', shell=True)
        subprocess.call('start bluemuse://setting?key=primary_timestamp_format!value=BLUEMUSE', shell=True)
        subprocess.call('start bluemuse://setting?key=channel_data_type!value=float32', shell=True)
        subprocess.call('start bluemuse://setting?key=eeg_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=accelerometer_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=gyroscope_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=ppg_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://start?streamfirst=true', shell=True)
        self.logger.update_session_key(session_key)
        time.sleep(3)
        while not self.lsl_reload():
            self.logger.error(f"LSL streams not found, retrying in 3 seconds") 
            time.sleep(3)
        self.connected.emit()
        if self.run_eeg_thread: self.stream_inlet[MuseDataType.EEG] = StreamInlet(self.stream_info[MuseDataType.EEG])
        if self.run_acc_thread: self.stream_inlet[MuseDataType.ACCELEROMETER] = StreamInlet(self.stream_info[MuseDataType.ACCELEROMETER])
        if self.run_ppg_thread: self.stream_inlet[MuseDataType.PPG] = StreamInlet(self.stream_info[MuseDataType.PPG])

        self.start_threads()

    def stop(self):
        try:
            if self.run_eeg_thread: self.stream_inlet[MuseDataType.EEG].close_stream()
            if self.run_acc_thread: self.stream_inlet[MuseDataType.ACCELEROMETER].close_stream()
            if self.run_ppg_thread: self.stream_inlet[MuseDataType.PPG].close_stream()
        except Exception as ex:
            self.logger.critical(str(ex))

        subprocess.call('start bluemuse://stop?stopall', shell=True)
        subprocess.call('start bluemuse://shutdown', shell=True)

        self.run_eeg_thread = False
        self.run_acc_thread = False
        self.run_ppg_thread = False

        for p in psutil.process_iter(['name']):
            if p.info['name'] == 'BlueMuse.exe':
                self.logger.info('Killing BlueMuse')
                p.kill()
        self.disconnected.emit()