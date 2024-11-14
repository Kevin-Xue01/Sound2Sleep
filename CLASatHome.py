from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtCore import QTimer, QThread
import sys
from pylsl import StreamInlet, resolve_byprop
import os
import os.path
import pyqtgraph as pg

import numpy as np
import matplotlib, matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from CLASAlgo import CLASAlgo
from scipy import signal
from datetime import datetime
from utils import screenoff, BlueMuseSignal, StreamType
from blue_muse import BlueMuse
from data_controller import DataWriter

matplotlib.use('QT5Agg')
QtWidgets.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)


# MAIN CLASS
class CLASatHome(QtWidgets.QMainWindow):
    draw_timer = None

    plots = dict()

    def init_UI(self):
        uic.loadUi('CLASatHome.ui', self)
        # bind buttons and stuff
        self.btn_start.clicked.connect(self.start_streaming)
        self.btn_stop.clicked.connect(self.stop_streaming)
        self.btn_screenoff.clicked.connect(screenoff)

        # set status indicator state
        self.status.setStyleSheet("background-color: white")

    def init_BlueMuse(self):
        self.blue_muse_signal = BlueMuseSignal()
        self.blue_muse_signal.update_data.connect(self.write_data)
        self.blue_muse_worker = BlueMuse(self.blue_muse_signal)
        self.blue_muse_thread = QThread()
        self.blue_muse_worker.moveToThread(self.blue_muse_thread)
        self.blue_muse_thread.started.connect(self.blue_muse_worker.run)  # Ensure run is the main task

    def __init__(self):
        super().__init__()
        self.init_UI()
        self.init_BlueMuse()
        
        output_file = os.path.join(os.getcwd(), 'data', 'kevin', 'output.h5')
        self.eeg_data_writer = DataWriter(output_file, 4, 12)

        self.clas_algo = CLASAlgo(100, 'params.json')

        

        # # plots
        # self.plots = dict()
        # self.plots['EEG'] = TimeseriesPlot(parent=self.timeseries_widget)

        # display the window
        self.show()

    def write_data(self, streamtype, timestamps, data):
        if streamtype == StreamType.EEG:
            print(data.shape)
            self.eeg_data_writer.write_data(timestamps, data)
            # data_mean = np.mean(data, axis=1)
            # combined = np.vstack((timestamps, data_mean)).T
            # if not os.path.exists(self.output_file):
            #     df = pd.DataFrame(columns=['timestamp', 'data'])
            #     df.to_csv(self.output_file, index=False)
            # df = pd.DataFrame(combined, columns=['timestamp', 'data'])
            # df.to_csv(self.output_file, mode='a', header=not os.path.exists(self.output_file), index=False)

    # def draw_timer_callback(self):
    #     ''' 
    #     Tell all the plots to redraw.
    #     Callback for draw_timer.
    #     '''

    #     # redraw the plots on timeout
    #     for k in self.plots:
    #         try:
    #             self.plots[k].redraw()
    #         except Exception as ex:
    #             # construct traceback
    #             tbstring = traceback.format_exception(type(ex), ex, ex.__traceback__)
    #             tbstring.insert(0, '=== ' + datetime.datetime.now().toISOString() + ' ===')

    #             # print to screen and error log file
    #             print('\n'.join(tbstring))
    #             self.files['err'].writelines(tbstring)

    def start_streaming(self):
        ''' 
        Callback for "Start" button
        Start bluemuse, streams, initialize recording files
        '''
        # self.blue_muse_worker.start_streaming()
        self.blue_muse_thread.start()

        # # initialize metadata file
        # fileroot = uuid.uuid4().hex
        # starttime = datetime.now()
        # self.meta = {
        #     "start_time": starttime.isoformat(),
        #     "data": {},
        #     "fs": {},
        #     "nchan": {},
        #     "error_log": fileroot + '_err.txt'
        # }

        # start the selected steram
        # self.stream_inlet
        # for streamtype in StreamType:
            # self.plots[k].init_data(fsample=self.lsl[k].nominal_srate(),
            #                         history_time=8,
            #                         nchan=self.lsl[k].channel_count())

            # # include details in metadata
            # self.meta['data'][k] = fileroot + '_' + k + '.dat'
            # self.meta['fs'][k] = self.lsl[k].nominal_srate()
            # self.meta['nchan'][k] = self.lsl[k].channel_count()
            # curr_path = os.path.join('output', self.meta['data'][k])
            # # os.makedirs(curr_path, exist_ok=True)
            # self.files[k] = open(curr_path, 'wb')

        # # save the metafile
        # with open(os.path.join('output', 'cah_%s.json' % starttime.strftime('%Y%m%dT%H%M%S')), 'w') as f:
        #     json.dump(self.meta, f)

        # # initialize the plot refresh timer
        # self.draw_timer = QTimer()
        # self.draw_timer.timeout.connect(self.draw_timer_callback)
        # self.draw_timer.start(500)

        # # set button state
        # self.status.setStyleSheet("background-color: green")
        # self.status.setText('Streaming...')

    def stop_streaming(self):
        ''' 
        Callback for "Stop" button
        Stop lsl chunk timers, GUI update timers, stop streams
        '''
        if self.draw_timer is not None:
            self.draw_timer.stop()
            self.draw_timer = None

        # for k in self.files:
        #     self.files[k].close()

        # # set button state
        # self.status.setStyleSheet("background-color: white")
        # self.status.setText('Ready.')

        # subprocess.call('start bluemuse://stop?stopall', shell=True)


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



if __name__ == "__main__":
    App = QtWidgets.QApplication(sys.argv)
    window = CLASatHome()
    sys.exit(App.exec())