import csv
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
from muselsl.constants import LSL_SCAN_TIMEOUT
from pylsl import StreamInfo, StreamInlet, resolve_byprop, resolve_streams
from scipy.signal import firwin, lfilter, lfilter_zi

from constants import (
    CHANNEL_NAMES,
    CHUNK_SIZE,
    NB_CHANNELS,
    SAMPLING_RATE,
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

# class CLASatHome:
#     # stream information
#     lsl = dict()
#     inlet = dict()

#     def __init__(self):
#         # start bluemuse if not already started
#         self.lsl: dict[DataStream, StreamInlet] = dict()
#         subprocess.call('start bluemuse:', shell=True)
#         subprocess.call('start bluemuse://setting?key=primary_timestamp_format!value=BLUEMUSE', shell=True)
#         subprocess.call('start bluemuse://setting?key=channel_data_type!value=float32', shell=True)
#         subprocess.call('start bluemuse://setting?key=eeg_enabled!value=true', shell=True)
#         subprocess.call('start bluemuse://setting?key=accelerometer_enabled!value=true', shell=True)
#         subprocess.call('start bluemuse://setting?key=gyroscope_enabled!value=false', shell=True)
#         subprocess.call('start bluemuse://setting?key=ppg_enabled!value=true', shell=True)

#     def screenoff(self):
#         ''' Darken the screen by starting the blank screensaver '''
#         try:
#             subprocess.call(['C:\Windows\System32\scrnsave.scr', '/start'])
#         except Exception as ex:
#             # construct traceback
#             tbstring = traceback.format_exception(type(ex), ex, ex.__traceback__)
#             tbstring.insert(0, '=== ' + datetime.datetime.now().toISOString() + ' ===')

#             # print to screen and error log file
#             print('\n'.join(tbstring))
#             self.files['err'].writelines(tbstring)

#     def lsl_reload(self):
#         allok = True
#         self.lsl = dict()
#         for stream in DataStream:
#             self.lsl[stream] = resolve_byprop('type', stream.value, timeout=LSL_SCAN_TIMEOUT)

#             if self.lsl[stream]:
#                 self.lsl[stream] = self.lsl[stream][0]
#                 print(f'{stream.name} OK.')
#             else:
#                 print(f'{stream.name} not found.')
#                 allok = False

#         return allok

#     def lsl_timer_callback(self):
#         for stream in DataStream:
#             try:
#                 chunk, times = self.inlet[d].pull_chunk()
#                 chunk = np.array(chunk)

#                 if len(times) > 0:
#                     self.no_data_count = 0

#                     # store the data
#                     self.plots[d].add_data(chunk)

#                     # submit EEG data to the PLL
#                     # if d == 'EEG':
#                     #     _, ts_ref, ts_lockbin = self.pll.process_block(chunk[:, 0])
#                     #     self.plots['PLL'].add_data(np.stack((ts_ref, ts_lockbin), axis=1))

#                     self.files[d].write('NCHK'.encode('ascii'))
#                     self.files[d].write(chunk.dtype.char.encode('ascii'))
#                     self.files[d].write(np.array(chunk.shape).astype(np.uint32).tobytes())
#                     self.files[d].write(np.array(times).astype(np.double).tobytes())
#                     self.files[d].write('TTTT'.encode('ascii'))
#                     self.files[d].write(chunk.tobytes(order='C'))

#                 else:
#                     self.no_data_count += 1

#                     # if no data after 2 seconds, attempt to reset and recover
#                     if self.no_data_count > 20:
#                         self.lsl_reset_stream_step1()

#             except Exception as ex:
#                 print(ex)

#     def lsl_reset_stream_step1(self):
#         self.no_data_count = 0  # reset no data counter
#         self.lsl_timer.stop()  # stop data pulling loop

#         # restart bluemuse streaming, wait, and restart
#         subprocess.call('start bluemuse://stop?stopall', shell=True)
#         Timer(3, self.lsl_reset_stream_step2)

#     def lsl_reset_stream_step2(self):
#         ''' 
#         Try to restart streams the lsl pull timer.
#         Part of interrupted stream restart process.
#         '''
#         subprocess.call('start bluemuse://start?streamfirst=true', shell=True)
#         Timer(3, self.lsl_reset_stream_step3)

#     def lsl_reset_stream_step3(self):
#         reset_success = self.lsl_reload()

#         if not reset_success:
#             # if we can't get the streams up, try again
#             self.reset_attempt_count = self.reset_attempt_count + 1
#             if self.reset_attempt_count < 3:
#                 self.lsl_reset_stream_step1()
#             else:
#                 self.reset_attempt_count = 0

#                 # if the stream really isn't working.. kill bluemuse
#                 for p in find_procs_by_name('BlueMuse.exe'):
#                     p.kill()

#                 # try the reset process again
#                 QTimer.singleShot(3000, self.lsl_reset_stream_step1)
#         else:
#             # if all streams have resolved, start polling data again!
#             self.reset_attempt_count = 0
#             self.lsl_timer.start()

#     def start_streaming(self):
#         ''' 
#         Callback for "Start" button
#         Start bluemuse, streams, initialize recording files
#         '''

#         # initialize bluemuse and try to resolve LSL streams
#         subprocess.call('start bluemuse://start?streamfirst=true', shell=True)
#         if not self.lsl_reload():
#             self.status.setStyleSheet("background-color: yellow")
#             self.status.setText('Unable to connect to Muse S...')
#             return

#         # initialize metadata file
#         fileroot = uuid.uuid4().hex
#         starttime = datetime.now()
#         self.meta = {
#             "start_time": starttime.isoformat(),
#             "data": {},
#             "fs": {},
#             "nchan": {},
#             "error_log": fileroot + '_err.txt'
#         }

#         # start the selected steram
#         self.inlet = dict()
#         for k in self.datastreams:
#             self.inlet[k] = StreamInlet(self.lsl[k])
#             self.plots[k].init_data(fsample=self.lsl[k].nominal_srate(),
#                                     history_time=8,
#                                     nchan=self.lsl[k].channel_count())

#             # include details in metadata
#             self.meta['data'][k] = fileroot + '_' + k + '.dat'
#             self.meta['fs'][k] = self.lsl[k].nominal_srate()
#             self.meta['nchan'][k] = self.lsl[k].channel_count()

#             self.files[k] = open(os.path.join('output', self.meta['data'][k]), 'wb')

#         # save the metafile
#         with open(os.path.join('output', 'cah_%s.json' % starttime.strftime('%Y%m%dT%H%M%S')), 'w') as f:
#             json.dump(self.meta, f)

#         # initialize the error log
#         self.files['err'] = open(os.path.join('output', self.meta['error_log']), 'w')

#         # initialize the PLL
#         self.plots['PLL'].init_data(fsample=self.lsl['EEG'].nominal_srate(), history_time=8, nchan=2)
#         self.pll = PhaseLockedLoop(fs=self.lsl['EEG'].nominal_srate())

#         # initialize the data stream timer
#         self.lsl_timer = QTimer()
#         self.lsl_timer.timeout.connect(self.lsl_timer_callback)
#         self.lsl_timer.start(100)

#         # initialize the plot refresh timer
#         self.draw_timer = QTimer()
#         self.draw_timer.timeout.connect(self.draw_timer_callback)
#         self.draw_timer.start(500)

#         # set button state
#         self.status.setStyleSheet("background-color: green")
#         self.status.setText('Streaming...')

#     def stop_streaming(self):
#         ''' 
#         Callback for "Stop" button
#         Stop lsl chunk timers, GUI update timers, stop streams
#         '''
#         if self.lsl_timer is not None:
#             self.lsl_timer.stop()
#             self.lsl_timer = None

#         if self.draw_timer is not None:
#             self.draw_timer.stop()
#             self.draw_timer = None

#         for k in self.inlet:
#             try:
#                 self.inlet[k].close_stream()
#             except Exception as ex:
#                 # construct traceback
#                 tbstring = traceback.format_exception(type(ex), ex, ex.__traceback__)
#                 tbstring.insert(0, '=== ' + datetime.datetime.now().toISOString() + ' ===')

#                 # print to screen and error log file
#                 print('\n'.join(tbstring))
#                 self.files['err'].writelines(tbstring)

#         for k in self.files:
#             self.files[k].close()

#         # set button state
#         self.status.setStyleSheet("background-color: white")
#         self.status.setText('Ready.')

#         subprocess.call('start bluemuse://stop?stopall', shell=True)



class CLASatHome:
    no_data_count = 0
    reset_attempt_count = 0
    
    def init_EEG_UI(self, ui_window_s=Config.UI.window_s, scale=Config.UI.scale):
        matplotlib.use('TkAgg')
        sns.set_theme(style="whitegrid")
        sns.despine(left=True)

        self.figsize = np.int16(Config.UI.figure.split('x'))
        self.ui_window_s = ui_window_s
        self.scale = scale

        self.fig, self.axes = plt.subplots(1, 1, figsize=self.figsize, sharex=True)
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
        self.eeg_nchan = len(CHANNEL_NAMES[DataStream.EEG])
        self.eeg_ui_samples = int(self.ui_window_s * SAMPLING_RATE[DataStream.EEG])
        self.data = np.zeros((self.eeg_ui_samples, self.eeg_nchan))
        self.times = np.arange(-self.ui_window_s, 0, 1. / SAMPLING_RATE[DataStream.EEG])
        self.impedances = np.std(self.data, axis=0)
        self.lines = []

        for ii in range(self.eeg_nchan):
            line, = self.axes.plot(self.times[::Config.UI.subsample], self.data[::Config.UI.subsample, ii] - ii, lw=1)
            self.lines.append(line)

        self.axes.set_ylim(-self.eeg_nchan + 0.5, 0.5)
        ticks = np.arange(0, -self.eeg_nchan, -1)

        self.axes.set_xlabel('Time (s)')
        self.axes.xaxis.grid(False)
        self.axes.set_yticks(ticks)

        self.axes.set_yticklabels([f'{label} - {impedance:2f}' for label, impedance in zip(CHANNEL_NAMES[DataStream.EEG], self.impedances)])

        self.display_every = int(0.2 / (12 / SAMPLING_RATE[DataStream.EEG]))

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
            self.stream_info[stream] = resolve_byprop('type', stream.value, timeout=2*LSL_SCAN_TIMEOUT)

            if self.stream_info[stream]:
                self.stream_info[stream] = self.stream_info[stream][0]
                print(f'{stream.name} OK.')
            else:
                print(f'{stream.name} not found.')
                allok = False

        return allok

    def eeg_callback(self):
        with open(f'data/kevin/eeg_data_{datetime.now().strftime("%Y-%m-%d")}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['Timestamp'] + CHANNEL_NAMES[DataStream.EEG])
            
            display_every_counter = 0
            no_data_counter = 0
            while self.run_eeg_thread:
                time.sleep(0.05)
                try:
                    data, timestamps = self.stream_inlet[DataStream.EEG].pull_chunk(timeout=1.0, max_samples=CHUNK_SIZE[DataStream.EEG])
                    # data = np.array(data)
                    # timestamps = np.array(timestamps)
                    # print(f"Data shape: {data.shape}, Timestamps shape: {timestamps.shape}")
                    if timestamps:
                        if Config.UI.dejitter:
                            timestamps = np.float64(np.arange(len(timestamps))) # TODO: change to static call
                            timestamps /= SAMPLING_RATE[DataStream.EEG]
                            timestamps += time.time() + 1. / SAMPLING_RATE[DataStream.EEG]

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

                        # if no data after 2 seconds, attempt to reset and recover
                        if no_data_counter > 20:
                            self.lsl_reset_stream_step1()

                except Exception as ex:
                    tbstring = traceback.format_exception(type(ex), ex, ex.__traceback__)
                    print('\n'.join(tbstring))

    def acc_callback(self):
        while True:
            print('ACC callback')
            time.sleep(3)


    def ppg_callback(self):
        while True:
            print('PPG callback')
            time.sleep(3)

    def lsl_reset_stream_step1(self):
        # restart bluemuse streaming, wait, and restart
        print('Resetting stream step 1 at ' + str(datetime.now()))
        subprocess.call('start bluemuse://stop?stopall', shell=True)
        Timer(3, self.lsl_reset_stream_step2)
        print('Resetting stream step 1 done at' + str(datetime.now()))


    def lsl_reset_stream_step2(self):
        print('Resetting stream step 2')
        subprocess.call('start bluemuse://start?streamfirst=true', shell=True)
        Timer(3, self.lsl_reset_stream_step3)

    def lsl_reset_stream_step3(self):
        print('Resetting stream step 3')
        reset_success = self.lsl_reload()

        if not reset_success:
            # if we can't get the streams up, try again
            self.reset_attempt_count = self.reset_attempt_count + 1
            if self.reset_attempt_count < Config.Connection.reset_attempt_count_max:
                self.lsl_reset_stream_step1()
            else:
                self.reset_attempt_count = 0

                # if the stream really isn't working.. kill bluemuse
                for p in find_procs_by_name('BlueMuse.exe'):
                    p.kill()

                # try the reset process again
                print('Resetting stream again')
                Timer(3, self.lsl_reset_stream_step1)
        else:
            # if all streams have resolved, start polling data again!
            self.reset_attempt_count = 0
            self.start_streaming()


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
        ''' 
        Callback for "Start" button
        Start bluemuse, streams, initialize recording files
        '''

        # initialize bluemuse and try to resolve LSL streams
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
    clas = CLASatHome()
    clas.init_EEG_UI()
    clas.init_BlueMuse()
    clas.start_streaming()
    plt.show()
    # time.sleep(100)