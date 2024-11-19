from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QSizePolicy, QHBoxLayout, QPushButton
from PyQt5.QtCore import QTimer, QThread, Qt
import sys
import os
import pyqtgraph as pg

import numpy as np
import matplotlib, matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from CLASAlgo import CLASAlgo
from scipy import signal
from datetime import datetime
from utils import BlueMuseSignal, StreamType
from blue_muse import BlueMuse
from data_controller import DataWriter

# matplotlib.use('QT5Agg')
# QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)

class EEGPlotterWidget(QWidget):
    MAX_BUFFER_SIZE = 1000  # Maximum number of data points to retain for each channel
    def __init__(self, eeg_channels=['TP9', 'AF7', 'AF8', 'TP10']):
        super().__init__()
        self.eeg_channels = eeg_channels
        self.plot_data = [[] for _ in range(len(self.eeg_channels))]  # Buffer to hold data for each channel
        self.time_data = []  # Time data for each channel

        # Initialize the GUI layout
        self.layout = QVBoxLayout(self)
        self.plots = []
        self.curves = []

        # Create time series plots for each channel
        for i, channel in enumerate(self.eeg_channels):
            plot = pg.PlotWidget(title=channel)
            plot.setLabel('bottom', 'time', units='s')
            plot.setLabel('left', 'EEG Amplitude', units='μV')
            plot.showGrid(x=True, y=True)  # Show grid lines
            plot.enableAutoRange(axis=pg.ViewBox.YAxis)  # Enable auto-scaling
            self.layout.addWidget(plot)  # Add plot to the layout

            curve = plot.plot([], pen=pg.mkPen('w'))  # Plot line initialized with an empty array
            self.plots.append(plot)
            self.curves.append(curve)

    def update_plots(self, streamtype: StreamType, times: np.ndarray, data: np.ndarray):
        """
        Update EEG data plots with new data chunks.

        Parameters:
        - data (np.ndarray): Array with shape (n_samples, 4) for 4 EEG channels.
        - times (np.ndarray): Array of timestamps for the EEG samples.
        """
        if streamtype == StreamType.EEG:
            self.time_data.extend(times.tolist())
            if len(self.time_data) > self.MAX_BUFFER_SIZE:
                self.time_data = self.time_data[-self.MAX_BUFFER_SIZE:]

            for i in range(data.shape[1]):
                # Append the new data to the buffer
                self.plot_data[i].extend(data[:, i].tolist())
                print(f'Max: {max(data[:, i])}, Min: {min(data[:, i])}')
                # Keep the buffer within the maximum size
                if len(self.plot_data[i]) > self.MAX_BUFFER_SIZE:
                    self.plot_data[i] = self.plot_data[i][-self.MAX_BUFFER_SIZE:]

                # Update the plot with new data
                self.curves[i].setData(self.time_data, self.plot_data[i])  # X-axis as time, Y-axis as EEG data

class CLASatHome(QMainWindow):
    draw_timer = None

    plots = dict()

    def init_UI(self):
        self.setWindowTitle('CLAS At Home')
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.eeg_plotter = EEGPlotterWidget()
        self.layout.addWidget(self.eeg_plotter)

        # Create a horizontal layout for the buttons below the EEG plot
        self.button_layout = QHBoxLayout()

        # Create "Start Streaming" button
        self.btn_start = QPushButton('Start Streaming')
        self.btn_start.clicked.connect(self.start_streaming)
        self.button_layout.addWidget(self.btn_start)

        # Create "Stop Streaming" button
        self.btn_stop = QPushButton('Stop Streaming')
        self.btn_stop.clicked.connect(self.stop_streaming)
        self.button_layout.addWidget(self.btn_stop)

        # Add the horizontal layout of buttons to the main layout
        self.layout.addLayout(self.button_layout)
        self.show()

    def init_BlueMuse(self):
        self.blue_muse_signal = BlueMuseSignal()
        # self.blue_muse_signal.update_data.connect(self.write_data)
        # self.blue_muse_signal.update_data.connect(self.eeg_plotter.update_plots)
        self.blue_muse = BlueMuse(self.blue_muse_signal)
        self.blue_muse.start_timer_signal.connect(self.start_timer)
        self.blue_muse.stop_timer_signal.connect(self.stop_timer)
        self.blue_muse_thread = QThread()
        self.blue_muse.moveToThread(self.blue_muse_thread)
        self.blue_muse_thread.started.connect(self.blue_muse.start_streaming)
        self.blue_muse_thread.finished.connect(self.blue_muse.stop_streaming)

    def __init__(self):
        super().__init__()
        self.init_UI()
        self.init_BlueMuse()
        self.lsl_timer = QTimer()
        self.lsl_timer.timeout.connect(self.blue_muse.lsl_timer_callback)

        output_file = os.path.join(os.getcwd(), 'data', 'kevin', 'output.h5')
        self.eeg_data_writer = DataWriter(output_file, 4, 12)

        # # self.clas_algo = CLASAlgo(100, 'params.json')

    def write_data(self, streamtype, timestamps, data):
        if streamtype == StreamType.EEG:
            self.eeg_data_writer.write_data(timestamps, data)

    def start_timer(self):
        if not self.lsl_timer.isActive():
            self.lsl_timer.start(200)

    def stop_timer(self):
        if self.lsl_timer.isActive():
            self.lsl_timer.stop()
    
    def start_streaming(self):
        if not self.blue_muse_thread.isRunning():
            self.blue_muse_thread.start()

    def stop_streaming(self):
        if self.blue_muse_thread.isRunning():
            self.blue_muse_thread.quit()  # Gracefully stop the thread
            self.blue_muse_thread.wait()  # Wait for the thread to fully finish

class TimeseriesPlot(FigureCanvasQTAgg):

    def __init__(self, parent, dpi=96):
        wsize = parent.size()
        self.fig = matplotlib.figure.Figure(figsize=(wsize.width() / dpi, wsize.height() / dpi), dpi=dpi)

        self.axes = self.fig.add_subplot(111)
        self.axes.set_position((0.1, 0.1, 0.85, 0.85))

        s = super(TimeseriesPlot, self)
        s.__init__(self.fig)
        self.setParent(parent)

        s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
    App = QApplication(sys.argv)
    window = CLASatHome()
    sys.exit(App.exec())