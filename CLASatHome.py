import csv
import ctypes
import subprocess
import time
import traceback
from datetime import datetime
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
from scipy.signal import firwin, lfilter, lfilter_zi

from constants import (
    CHANNEL_NAMES,
    CHUNK_SIZE,
    NB_CHANNELS,
    SAMPLING_RATE,
    TIMESTAMPS,
    Config,
    DataStream,
)


# HELPER STATIC FUNCTIONS
def find_procs_by_name(name) -> list[Process]:
    ls = []
    for p in psutil.process_iter(['name']):
        if p.info['name'] == name:
            ls.append(p)
    return ls

class CLASatHome:
    no_data_count = 0
    reset_attempt_count = 0
    
    def init_EEG_UI(self):
        matplotlib.use('TkAgg')
        sns.set_theme(style="whitegrid")
        sns.despine(left=True)

        self.ui_window_s = 5
        self.scale = 100

        self.fig, self.axes = plt.subplots(1, 1, figsize=[15, 6], sharex=True)
        self.fig.canvas.mpl_connect('close_event', self.stop_streaming)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        help_str = """
                    toggle filter : d
                    toogle full screen : f
                    zoom out : /
                    zoom in : *
                    increase time scale : -
                    decrease time scale : +
                """
        print(help_str)
        self.eeg_nchan = NB_CHANNELS[DataStream.EEG]
        self.eeg_ui_samples = int(self.ui_window_s * SAMPLING_RATE[DataStream.EEG])
        self.data = np.zeros((self.eeg_ui_samples, self.eeg_nchan))
        self.times = np.arange(-self.ui_window_s, 0, 1. / SAMPLING_RATE[DataStream.EEG])
        self.impedances = np.std(self.data, axis=0)
        self.lines = []

        for ii in range(self.eeg_nchan):
            line, = self.axes.plot(self.times[::VIEW_SUBSAMPLE], self.data[::VIEW_SUBSAMPLE, ii] - ii, lw=1)
            self.lines.append(line)

        self.axes.set_ylim(-self.eeg_nchan + 0.5, 0.5)
        ticks = np.arange(0, -self.eeg_nchan, -1)

        self.axes.set_xlabel('Time (s)')
        self.axes.xaxis.grid(False)
        self.axes.set_yticks(ticks)

        self.axes.set_yticklabels([f'{label} - {impedance:2f}' for label, impedance in zip(CHANNEL_NAMES[DataStream.EEG], self.impedances)])

        self.display_every = 5

        self.bf = firwin(32, np.array([1, 40]) / (SAMPLING_RATE[DataStream.EEG] / 2.), width=0.05, pass_zero=False)
        self.af = [1.0]

        zi = lfilter_zi(self.bf, self.af)
        self.filt_state = np.tile(zi, (self.eeg_nchan, 1)).transpose()
        self.data_f = np.zeros((self.eeg_ui_samples, self.eeg_nchan))

    def init_BlueMuse(self):
        subprocess.call('start bluemuse:', shell=True)
        subprocess.call('start bluemuse://setting?key=primary_timestamp_format!value=BLUEMUSE', shell=True)
        subprocess.call('start bluemuse://setting?key=channel_data_type!value=float32', shell=True)
        subprocess.call('start bluemuse://setting?key=eeg_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=accelerometer_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=gyroscope_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=ppg_enabled!value=true', shell=True)

    def __init__(self, filt=True):
        self.stream_info: dict[DataStream, StreamInfo] = dict()
        self.stream_inlet: dict[DataStream, StreamInlet] = dict()

        self.filt = filt
        self.processing_window_s = 2

    def screenoff(self):
        ''' Darken the screen by starting the blank screensaver '''
        try:
            subprocess.call(['C:\Windows\System32\scrnsave.scr', '/start'])
        except Exception as ex:
            # construct traceback
            tbstring = traceback.format_exception(type(ex), ex, ex.__traceback__)
            print('\n'.join(tbstring))

    def lsl_reload(self):
        print('Reloading LSL streams')

        allok = True
        self.stream_info = dict()
        self.stream_inlet = dict()
        for stream in DataStream:
            self.stream_info[stream] = resolve_byprop('type', stream.value, timeout=LSL_SCAN_TIMEOUT)

            if self.stream_info[stream]:
                self.stream_info[stream] = self.stream_info[stream][0]
                print(f'{stream.name} OK.')
            else:
                print(f'{stream.name} not found.')
                allok = False

        return allok


    def eeg_callback(self):
        with open(f'data/kevin/eeg_data_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['Timestamp'] + CHANNEL_NAMES[DataStream.EEG])
            
            display_every_counter = 0
            no_data_counter = 0
            while self.run_eeg_thread:
                time.sleep(CHUNK_SIZE[DataStream.EEG] / SAMPLING_RATE[DataStream.EEG])
                try:
                    data, timestamps = self.stream_inlet[DataStream.EEG].pull_chunk(timeout=1.0, max_samples=CHUNK_SIZE[DataStream.EEG])
                    if timestamps and len(timestamps) == CHUNK_SIZE[DataStream.EEG]:
                        timestamps = TIMESTAMPS[DataStream.EEG] + time.time() + 1. / SAMPLING_RATE[DataStream.EEG]

                        for t, s in zip(timestamps, data):
                            writer.writerow([t] + list(s))

                        self.times = np.concatenate([self.times, timestamps])
                        self.n_samples = int(SAMPLING_RATE[DataStream.EEG] * self.processing_window_s)
                        self.times = self.times[-self.n_samples:]
                        self.data = np.vstack([self.data, data])
                        self.data = self.data[-self.n_samples:]
                        filt_samples, self.filt_state = lfilter(self.bf, self.af, data, axis=0, zi=self.filt_state)
                        self.data_f = np.vstack([self.data_f, filt_samples])
                        self.data_f = self.data_f[-self.n_samples:]

                        display_every_counter += 1
                        # if display_every_counter == self.display_every:
                        #     print('Displaying data')
                        #     if self.filt:
                        #         plot_data = self.data_f
                        #     elif not self.filt:
                        #         plot_data = self.data - self.data.mean(axis=0)
                        #     for ii in range(NB_CHANNELS[DataStream.EEG]):
                        #         self.lines[ii].set_xdata(self.times[::Config.UI.subsample] -
                        #                                 self.times[-1])
                        #         self.lines[ii].set_ydata(plot_data[::Config.UI.subsample, ii] /
                        #                                 self.scale - ii)
                        #         self.impedances = np.std(plot_data, axis=0)

                        #     self.axes.set_yticklabels([f'{label} - {impedance:2f}' for label, impedance in zip(CHANNEL_NAMES[DataStream.EEG], self.impedances)])
                        #     self.axes.set_xlim(-self.ui_window_s, 0)
                        #     self.fig.canvas.draw()
                        #     display_every_counter = 0
                    else:
                        no_data_counter += 1

                        if no_data_counter > 20:
                            self.run_eeg_thread = False
                            self.run_acc_thread = False
                            self.run_ppg_thread = False
                            Timer(1, self.lsl_reset_stream_step1).start()

                except Exception as ex:
                    tbstring = traceback.format_exception(type(ex), ex, ex.__traceback__)
                    print('\n'.join(tbstring))
            print('EEG thread stopped')

    def acc_callback(self):
        while self.run_acc_thread:
            print('ACC callback')
            time.sleep(3)
        print('ACC thread stopped')

    def ppg_callback(self):
        while self.run_ppg_thread:
            print('PPG callback')
            time.sleep(3)
        print('PPG thread stopped')

    def lsl_reset_stream_step1(self):
        print('Resetting stream step 1')
        subprocess.call('start bluemuse://stop?stopall', shell=True)
        time.sleep(3)
        self.lsl_reset_stream_step2()


    def lsl_reset_stream_step2(self):
        print('Resetting stream step 2')
        subprocess.call('start bluemuse://start?startall', shell=True)
        time.sleep(3)
        self.lsl_reset_stream_step3()

    def lsl_reset_stream_step3(self):
        print('Resetting stream step 3')
        reset_success = self.lsl_reload()

        if not reset_success:
            self.reset_attempt_count += 1
            if self.reset_attempt_count < Config.Connection.reset_attempt_count_max:
                print('Resetting Attempt: ' + str(self.reset_attempt_count))
                self.lsl_reset_stream_step1() 
            else:
                self.reset_attempt_count = 0

                # if the stream really isn't working.. kill bluemuse
                for p in find_procs_by_name('BlueMuse.exe'):
                    print('Killing BlueMuse')
                    p.kill()

                # try the reset process again
                print('Resetting stream again')
                # Timer(3, self.lsl_reset_stream_step1)
                time.sleep(2)
                self.lsl_reset_stream_step1()
        else:
            # if all streams have resolved, start polling data again!
            self.reset_attempt_count = 0
            print('Starting threads')
            time.sleep(3)
            subprocess.call('start bluemuse://start?streamfirst=true', shell=True)

            # start the selected steram
            for stream in DataStream:
                self.stream_inlet[stream] = StreamInlet(self.stream_info[stream])
            
            self.start_threads()


    def on_key_press(self, event):
        if event.key == '/':
            self.scale *= 1.2
        elif event.key == '*':
            self.scale /= 1.2
        elif event.key == '+':
            self.ui_window_s += 1
        elif event.key == '-':
            if self.ui_window_s > 1:
                self.ui_window_s -= 1
        elif event.key == 'd':
            self.filt = not(self.filt)

    def start_threads(self):
        self.run_eeg_thread = True
        self.run_acc_thread = True
        self.run_ppg_thread = True

        self.eeg_thread = Thread(target=self.eeg_callback)
        self.acc_thread = Thread(target=self.acc_callback)
        self.ppg_thread = Thread(target=self.ppg_callback)

        self.eeg_thread.daemon = True
        self.acc_thread.daemon = True
        self.ppg_thread.daemon = True

        self.eeg_thread.start()
        self.acc_thread.start()
        self.ppg_thread.start()

    def start_streaming(self):
        subprocess.call('start bluemuse://start?streamfirst=true', shell=True)
        while not self.lsl_reload(): 
            time.sleep(3)

        # start the selected steram
        for stream in DataStream:
            self.stream_inlet[stream] = StreamInlet(self.stream_info[stream])
        
        self.start_threads()


    def stop_streaming(self):
        self.run_eeg_thread = False
        self.run_acc_thread = False
        self.run_ppg_thread = False

        for stream in DataStream:
            try:
                self.stream_inlet[stream].close_stream()
            except Exception as ex:
                print('\n' + str(ex))

        subprocess.call('start bluemuse://stop?stopall', shell=True)



if __name__ == "__main__":
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_AWAYMODE_REQUIRED = 0x0000040
    
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)

    try:
        clas = CLASatHome()
        clas.init_EEG_UI()
        clas.init_BlueMuse()
        clas.start_streaming()
        plt.show()
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)