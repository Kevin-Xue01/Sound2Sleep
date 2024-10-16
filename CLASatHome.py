from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtCore import QTimer
import sys
from pylsl import StreamInlet, resolve_byprop, resolve_streams
import os
import os.path
import json
import uuid
import subprocess
import traceback
import psutil

import numpy as np
import matplotlib, matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from scipy import signal
from datetime import datetime

matplotlib.use('QT5Agg')
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

# HELPER STATIC FUNCTIONS
def find_procs_by_name(name):
    ls = []
    for p in psutil.process_iter(['name']):
        if p.info['name'] == name:
            ls.append(p)
    return ls

# MAIN CLASS
class CLASatHome(QtWidgets.QMainWindow):
    datastreams = ['EEG', 'Accelerometer', 'PPG']

    no_data_count = 0
    reset_attempt_count = 0

    lsl_timer = None
    draw_timer = None

    # stream information
    plots = dict()
    lsl = dict()
    inlet = dict()
    files = dict()

    def __init__(self):
        super().__init__()

        # load the UI
        # uic.loadUi('CLASatHome.ui', self)
        uic.loadUi('CLASatHome.ui', self)

        # bind buttons and stuff
        self.btn_start.clicked.connect(self.start_streaming)
        self.btn_stop.clicked.connect(self.stop_streaming)
        self.btn_screenoff.clicked.connect(self.screenoff)

        # set status indicator state
        self.status.setStyleSheet("background-color: white")

        # plots
        self.plots = dict()
        self.plots['EEG'] = TimeseriesPlot(parent=self.timeseries_widget)
        self.plots['Accelerometer'] = SmallPlot(parent=self.accl_widget)
        self.plots['PPG'] = SmallPlot(parent=self.pleth_widget, filter=0.01)

        # start bluemuse if not already started
        subprocess.call('start bluemuse:', shell=True)
        subprocess.call('start bluemuse://setting?key=primary_timestamp_format!value=BLUEMUSE', shell=True)
        subprocess.call('start bluemuse://setting?key=channel_data_type!value=float32', shell=True)
        subprocess.call('start bluemuse://setting?key=eeg_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=accelerometer_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=gyroscope_enabled!value=false', shell=True)
        subprocess.call('start bluemuse://setting?key=ppg_enabled!value=true', shell=True)

        # display the window
        self.show()

    def screenoff(self):
        ''' Darken the screen by starting the blank screensaver '''
        try:
            subprocess.call(['C:\Windows\System32\scrnsave.scr', '/start'])
        except Exception as ex:
            # construct traceback
            tbstring = traceback.format_exception(type(ex), ex, ex.__traceback__)
            tbstring.insert(0, '=== ' + datetime.datetime.now().toISOString() + ' ===')

            # print to screen and error log file
            print('\n'.join(tbstring))
            self.files['err'].writelines(tbstring)

    def lsl_reload(self):
        ''' 
        Resolve all 3 LSL streams from the Muse S.
        This function blocks for up to 10 seconds.
        '''
        print('\n\n=== ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ===\n')
        allok = True
        self.lsl = dict()
        for t in self.datastreams:
            self.lsl[t] = resolve_byprop('type', t, timeout=10)

            if self.lsl[t]:
                self.lsl[t] = self.lsl[t][0]
                print('%s OK.' % t)
            else:
                print('%s not found.' % t)
                allok = False

        return allok

    def lsl_timer_callback(self):
        ''' 
        Get data from LSL streams and route it to the right place (plot, files, phase tracker).
        Callback for lsl_timer.
        '''
        for d in self.datastreams:
            try:
                # chunk, times = self.inlet[d].pull_chunk()
                # chunk = np.array(chunk)

                # if len(times) > 0:
                #     self.no_data_count = 0

                #     # store the data
                #     self.plots[d].add_data(chunk)

                #     # submit EEG data to the PLL
                #     # if d == 'EEG':
                #     #     _, ts_ref, ts_lockbin = self.pll.process_block(chunk[:, 0])
                #     print(f"Type: {d}, shape: {chunk.shape}")

                #     self.files[d].write('NCHK'.encode('ascii'))
                #     self.files[d].write(chunk.dtype.char.encode('ascii'))
                #     self.files[d].write(np.array(chunk.shape).astype(np.uint32).tobytes())
                #     self.files[d].write(np.array(times).astype(np.double).tobytes())
                #     self.files[d].write('TTTT'.encode('ascii'))
                #     self.files[d].write(chunk.tobytes(order='C'))
                #     print(chunk)

                # else:
                #     self.no_data_count += 1

                #     # if no data after 2 seconds, attempt to reset and recover
                #     if self.no_data_count > 20:
                #         self.lsl_reset_stream_step1()
                sample, times = self.inlet[d].pull_sample()

                if times:
                    print(sample)

                else:
                    self.no_data_count += 1

                    # if no data after 2 seconds, attempt to reset and recover
                    if self.no_data_count > 20:
                        self.lsl_reset_stream_step1()

            except Exception as ex:
                # construct traceback
                tbstring = traceback.format_exception(type(ex), ex, ex.__traceback__)
                tbstring.insert(0, '=== ' + datetime.datetime.now().toISOString() + ' ===')

                # print to screen and error log file
                print('\n'.join(tbstring))
                self.files['err'].writelines(tbstring)

    def lsl_reset_stream_step1(self):
        self.no_data_count = 0  # reset no data counter
        self.lsl_timer.stop()  # stop data pulling loop

        # restart bluemuse streaming, wait, and restart
        subprocess.call('start bluemuse://stop?stopall', shell=True)
        QTimer.singleShot(3000, self.lsl_reset_stream_step2)

    def lsl_reset_stream_step2(self):
        ''' 
        Try to restart streams the lsl pull timer.
        Part of interrupted stream restart process.
        '''
        subprocess.call('start bluemuse://start?streamfirst=true', shell=True)
        QTimer.singleShot(3000, self.lsl_reset_stream_step3)

    def lsl_reset_stream_step3(self):
        reset_success = self.lsl_reload()

        if not reset_success:
            # if we can't get the streams up, try again
            self.reset_attempt_count = self.reset_attempt_count + 1
            if self.reset_attempt_count < 3:
                self.lsl_reset_stream_step1()
            else:
                self.reset_attempt_count = 0

                # if the stream really isn't working.. kill bluemuse
                for p in find_procs_by_name('BlueMuse.exe'):
                    p.kill()

                # try the reset process again
                QTimer.singleShot(3000, self.lsl_reset_stream_step1)
        else:
            # if all streams have resolved, start polling data again!
            self.reset_attempt_count = 0
            self.lsl_timer.start()

    def draw_timer_callback(self):
        ''' 
        Tell all the plots to redraw.
        Callback for draw_timer.
        '''

        # redraw the plots on timeout
        for k in self.plots:
            try:
                self.plots[k].redraw()
            except Exception as ex:
                # construct traceback
                tbstring = traceback.format_exception(type(ex), ex, ex.__traceback__)
                tbstring.insert(0, '=== ' + datetime.datetime.now().toISOString() + ' ===')

                # print to screen and error log file
                print('\n'.join(tbstring))
                self.files['err'].writelines(tbstring)

    def start_streaming(self):
        ''' 
        Callback for "Start" button
        Start bluemuse, streams, initialize recording files
        '''

        # initialize bluemuse and try to resolve LSL streams
        subprocess.call('start bluemuse://start?streamfirst=true', shell=True)
        if not self.lsl_reload():
            self.status.setStyleSheet("background-color: yellow")
            self.status.setText('Unable to connect to Muse S...')
            return

        # initialize metadata file
        fileroot = uuid.uuid4().hex
        starttime = datetime.now()
        self.meta = {
            "start_time": starttime.isoformat(),
            "data": {},
            "fs": {},
            "nchan": {},
            "error_log": fileroot + '_err.txt'
        }

        # start the selected steram
        self.inlet = dict()
        for k in self.datastreams:
            self.inlet[k] = StreamInlet(self.lsl[k])
            self.plots[k].init_data(fsample=self.lsl[k].nominal_srate(),
                                    history_time=8,
                                    nchan=self.lsl[k].channel_count())

            # include details in metadata
            self.meta['data'][k] = fileroot + '_' + k + '.dat'
            self.meta['fs'][k] = self.lsl[k].nominal_srate()
            self.meta['nchan'][k] = self.lsl[k].channel_count()
            curr_path = os.path.join('output', self.meta['data'][k])
            # os.makedirs(curr_path, exist_ok=True)
            self.files[k] = open(curr_path, 'wb')

        # save the metafile
        with open(os.path.join('output', 'cah_%s.json' % starttime.strftime('%Y%m%dT%H%M%S')), 'w') as f:
            json.dump(self.meta, f)

        # initialize the error log
        self.files['err'] = open(os.path.join('output', self.meta['error_log']), 'w')

        # initialize the data stream timer
        self.lsl_timer = QTimer()
        self.lsl_timer.timeout.connect(self.lsl_timer_callback)
        self.lsl_timer.start(100)

        # initialize the plot refresh timer
        self.draw_timer = QTimer()
        self.draw_timer.timeout.connect(self.draw_timer_callback)
        self.draw_timer.start(500)

        # set button state
        self.status.setStyleSheet("background-color: green")
        self.status.setText('Streaming...')

    def stop_streaming(self):
        ''' 
        Callback for "Stop" button
        Stop lsl chunk timers, GUI update timers, stop streams
        '''
        if self.lsl_timer is not None:
            self.lsl_timer.stop()
            self.lsl_timer = None

        if self.draw_timer is not None:
            self.draw_timer.stop()
            self.draw_timer = None

        for k in self.inlet:
            try:
                self.inlet[k].close_stream()
            except Exception as ex:
                # construct traceback
                tbstring = traceback.format_exception(type(ex), ex, ex.__traceback__)
                tbstring.insert(0, '=== ' + datetime.datetime.now().toISOString() + ' ===')

                # print to screen and error log file
                print('\n'.join(tbstring))
                self.files['err'].writelines(tbstring)

        for k in self.files:
            self.files[k].close()

        # set button state
        self.status.setStyleSheet("background-color: white")
        self.status.setText('Ready.')

        subprocess.call('start bluemuse://stop?stopall', shell=True)


class TimeseriesPlot(FigureCanvasQTAgg):

    def __init__(self, parent, dpi=96):
        wsize = parent.size()
        self.fig = matplotlib.figure.Figure(figsize=(wsize.width() / dpi, wsize.height() / dpi), dpi=dpi)

        self.axes = self.fig.add_subplot(111)
        self.axes.set_position((0.1, 0.1, 0.85, 0.85))

        s = super(TimeseriesPlot, self)
        s.__init__(self.fig)
        self.setParent(parent)

        s.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        s.updateGeometry()

    def init_data(self, fsample, history_time, nchan=4):
        self.fsample = fsample
        self.history_time = history_time
        self.data = np.zeros((int(self.fsample * self.history_time), nchan))
        self.time = np.arange(-1 * self.history_time * self.fsample, 0) / self.fsample

        self.ylims = 0.1

        self.compute_initial_figure()

    def compute_initial_figure(self):
        self.axes.clear()
        self.plt = self.axes.plot(self.time, self.data, alpha=0.6)
        self.axes.set_xlim(-1 * self.history_time, 0)
        self.draw()

    def add_data(self, x):
        n_new_samp = x.shape[0]
        self.data[:-n_new_samp, :] = self.data[n_new_samp:, :]
        self.data[-n_new_samp:, :] = x

        self.ylims = (0.9 * self.ylims) + (0.1 * np.abs(self.data).max() * 1.05)

    def redraw(self):
        for i, p in enumerate(self.plt):
            p.set_ydata(self.data[:, i])
        self.axes.set_ylim(-1 * self.ylims, self.ylims)
        self.draw()


class SmallPlot(FigureCanvasQTAgg):
    filt_sos = None
    filt_zi = None

    def __init__(self, parent, zerocenter=False, filter=None, dpi=60):
        wsize = parent.size()
        self.fig = matplotlib.figure.Figure(figsize=(wsize.width() / dpi, wsize.height() / dpi), dpi=dpi)

        self.axes = self.fig.add_subplot(111)
        # plt.tight_layout()
        self.axes.set_position((0.05, 0.18, 0.92, 0.77))

        s = super(SmallPlot, self)
        s.__init__(self.fig)
        self.setParent(parent)

        s.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        s.updateGeometry()

        self.zerocenter = zerocenter

        if filter is not None:
            self.filt_sos = signal.butter(2, filter, btype='highpass', output='sos')

    def init_data(self, fsample, history_time, nchan):
        self.fsample = fsample
        self.history_time = history_time
        self.data = np.zeros((int(self.fsample * self.history_time), nchan))
        self.time = np.arange(-1 * self.history_time * self.fsample, 0) / self.fsample

        self.ylims = 0.1

        self.compute_initial_figure()

        if self.filt_sos is not None:
            # self.filt_zi = np.stack([signal.sosfilt_zi(self.filt_sos)] * nchan, axis=2)
            self.filt_zi = np.zeros((1, 2, nchan))

    def compute_initial_figure(self):
        self.axes.clear()
        self.plt = self.axes.plot(self.time, self.data)
        self.axes.set_xlim(-1 * self.history_time, 0)
        self.draw()

    def add_data(self, x):
        if self.zerocenter is True:
            x = x - self.data.mean(axis=0)[None, :]

        if self.filt_sos is not None:
            x, self.filt_zi = signal.sosfilt(self.filt_sos, x, axis=0, zi=self.filt_zi)

        n_new_samp = x.shape[0]
        self.data[:-n_new_samp, :] = self.data[n_new_samp:, :]
        self.data[-n_new_samp:, :] = x

        self.ylims = (0.9 * self.ylims) + (0.1 * np.abs(self.data).max() * 1.05)

    def redraw(self):
        for i, p in enumerate(self.plt):
            p.set_ydata(self.data[:, i])
        self.axes.set_ylim(-1 * self.ylims, self.ylims)
        self.draw()


if __name__ == "__main__":
    App = QtWidgets.QApplication(sys.argv)
    window = CLASatHome()
    sys.exit(App.exec())