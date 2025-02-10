'''
GUI for running the Closed Loop Auditory Stimulation experiment - Take Hope
Author: Simeon Wong
'''

# Misc
import datetime
import io
import json
import logging
import os
import struct
import sys
import time
import traceback
from enum import Enum

# Qt Framework
from typing import Callable, Optional

# Plot stuff
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt

# Math stuff
import numpy as np
import psutil
import qdarkstyle

# for Pushover
import requests

# triggers
import serial
import serial.tools.list_ports
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtCore import QPoint, QRunnable, Qt, QThreadPool, QTimer
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen, QStaticText

# EEG streaming interface
from QOpenBCI import ConnectionState, QEEGStreamer

# CLAS algorithm
import QCLASAlgo
from CLASQtWidgets import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class CLASGUI(QtWidgets.QMainWindow):
    BUFFER_LEN = 30  # data to keep in memory, mostly for plotting

    pool = QThreadPool.globalInstance()

    mode_info: Optional[list] = None
    eeg_start_time: datetime.datetime

    reset_button_timer: QTimer = QTimer()

    s = {}

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger()

        # load strings
        with open('strings.yml', 'r') as f:
            self.strings = yaml.safe_load(f)


        ################# GUI STUFF #################
        # set darkmode
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        plt.style.use('dark_background')


        self.mainlayout = QtWidgets.QVBoxLayout()
        self.mainlayout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(QtWidgets.QWidget())
        self.centralWidget().setLayout(self.mainlayout)

        self.header = QtWidgets.QWidget()
        self.header.setFixedHeight(225)
        self.header.setStyleSheet("background: none;")
        self.screens_header(w=self.header)
        self.mainlayout.addWidget(self.header)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.setStyleSheet("background: none;")
        self.mainlayout.addWidget(self.stack)

        # render screens
        self.stack.addWidget(self.screens_home())
        self.stack.addWidget(self.screens_qc())
        self.stack.setCurrentIndex(1)

        # set full screen
        self.showFullScreen()

        # set window title
        self.setWindowTitle('CLAS')


        ################# INITIALIZE STATES #################
        self.update_status_lights()

        # show
        self.show()


    def screens_home(self):
        self.s['home'] = QtWidgets.QWidget()
        w = self.s['home']
        w.setStyleSheet(self.strings['takehome']['styles']['base_widget'])

        # title
        w.night_label = QtWidgets.QLabel(f'Night 1 of 2', w)
        w.night_label.setStyleSheet(self.strings['takehome']['styles']['title'])
        w.night_label.setAlignment(Qt.AlignCenter)
        w.night_label.setGeometry(275, 150, 1366, 160)

        # buttons
        w.start_button = QtWidgets.QPushButton('Start setup for sleep', w)
        w.start_button.setStyleSheet(self.strings['takehome']['styles']['button'])
        w.start_button.setGeometry(435, 375, 1050, 275)

        w.pvt_button = QtWidgets.QPushButton(self.strings['takehome']['buttons']['pvt'], w)
        w.pvt_button.setStyleSheet(self.strings['takehome']['styles']['button_sm'])
        w.pvt_button.setGeometry(435, 740, 1050, 150)

        return w

    def screens_qc(self):
        self.s['qc'] = QtWidgets.QWidget()
        w = self.s['qc']
        w.setStyleSheet(self.strings['takehome']['styles']['base_widget'])

        # night indicator
        w.night_label = QtWidgets.QLabel(f'Night 1 of 2', w)
        w.night_label.setStyleSheet(self.strings['takehome']['styles']['h3'])
        w.night_label.setAlignment(Qt.AlignLeft)
        w.night_label.setGeometry(60, 50, 300, 60)

        # title
        w.title_label = QtWidgets.QLabel(f'Put on your headband', w)
        w.title_label.setStyleSheet(self.strings['takehome']['styles']['h2'])
        w.title_label.setAlignment(Qt.AlignCenter)
        w.title_label.setGeometry(520, 50, 877, 60)

        # subtitle
        w.subtitle_label = QtWidgets.QLabel(f"Adjust until you're comfortable\nand we have a good signal", w)
        w.subtitle_label.setStyleSheet(self.strings['takehome']['styles']['h3'])
        w.subtitle_label.setAlignment(Qt.AlignCenter)
        w.subtitle_label.setGeometry(520, 150, 877, 120)

        # QWidget for matplotlib
        w.plot_widget = QtWidgets.QWidget(w)
        w.plot_widget.setGeometry(872, 333, 710, 435)
        w.plot_widget.setStyleSheet("background: none;")
        w.plot_class = QCPlot(parent=w.plot_widget)
        w.plot_class.plot(np.linspace(0, 1, 100), np.sin(np.linspace(0, 4*np.pi, 100)))

        # ready to sleep button
        w.ready_button = QtWidgets.QPushButton("I'm ready to sleep", w)
        w.ready_button.setStyleSheet(self.strings['takehome']['styles']['button_sm'])
        w.ready_button.setGeometry(695, 850, 534, 130)

        return w

    def screens_header(self, w:QtWidgets.QWidget):
        w.setStyleSheet(self.strings['takehome']['styles']['base_widget'])

        # eeg status light + label
        w.eeg_status = QStatusLight(size_px=42, parent=w)
        w.eeg_status.move(1370, 80)
        w.eeg_status_label = QtWidgets.QLabel(self.strings['takehome']['status']['eeg'], w)
        w.eeg_status_label.setStyleSheet('color: white; font-size: 30px; font-family: "DM Sans";')
        w.eeg_status_label.setGeometry(1436, 89, 300, 30)

        # algo status light + label
        w.algo_status = QStatusLight(size_px=42, parent=w)
        w.algo_status.move(1370, 135)
        w.algo_status_label = QtWidgets.QLabel(self.strings['takehome']['status']['algo'], w)
        w.algo_status_label.setStyleSheet('color: white; font-size: 30px; font-family: "DM Sans";')
        w.algo_status_label.setGeometry(1436, 145, 300, 30)

        # reset button
        w.reset_button = QResetButton(text='Triple\ntap to\nRESET', parent=w)
        w.reset_button.setGeometry(1750, 35, 145, 160)
        w.reset_button.reset.connect(self.reset_button_pressed)

    def reset_button_pressed(self):
        print('reset button pressed')
        pass

    def update_status_lights(self):
        # update eeg status
        if False: #self.eeg_streamer.connection_state == ConnectionState.CONNECTED:
            self.header.eeg_status.set_state(QStatusLight.State.OK)
            self.header.eeg_status_label.setText(self.strings['takehome']['status']['eeg'] + ' ' + self.strings['takehome']['status']['running'])
        else:
            self.header.eeg_status.set_state(QStatusLight.State.STOPPED)
            self.header.eeg_status_label.setText(self.strings['takehome']['status']['eeg'] + ' ' + self.strings['takehome']['status']['stopped'])

        # update algo status
        if False: #self.algo_state == QCLASAlgo.AlgoState.RUNNING:
            self.header.algo_status.set_state(QStatusLight.State.OK)
            self.header.algo_status_label.setText(self.strings['takehome']['status']['algo'] + ' ' + self.strings['takehome']['status']['running'])
        else:
            self.header.algo_status.set_state(QStatusLight.State.STOPPED)
            self.header.algo_status_label.setText(self.strings['takehome']['status']['algo'] + ' ' + self.strings['takehome']['status']['stopped'])





def excepthook(exc_type, exc_value, exc_tb):
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print('################\n## escapehook ##\n################\n' + tb +
          '################')
    # send_error(tb)

if __name__ == "__main__":
    sys.excepthook = excepthook
    App = QtWidgets.QApplication(sys.argv)
    window = CLASGUI()
    sys.exit(App.exec())
