import sys
import traceback

# Qt Framework
from typing import Callable
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QRunnable, QTimer, Qt, QThreadPool, QPoint
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QStaticText
from PyQt5.QtWidgets import QComboBox, QDateTimeEdit, QWidget, QFileDialog
import qdarkstyle

import os
import io
import json
import struct
import time
from enum import Enum
from typing import Optional
import logging
import psutil
import yaml

# Plot stuff
import matplotlib, matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# Math stuff
import numpy as np

# EEG streaming interface
from QOpenBCI import QEEGStreamer, ConnectionState

# triggers
import serial
import serial.tools.list_ports

# CLAS algorithm
import QCLASAlgo

# Misc
import datetime

# for Pushover
import requests
import yaml


logger = logging.getLogger()
logger.setLevel(logging.INFO)


with open('pushover.yml', 'r') as f:
    pushover_params = yaml.safe_load(f)

def send_error(msg):
    try:
        requests.post('https://api.pushover.net/1/messages.json',
                      data = {
                          "token": pushover_params['token'],
                          "user": pushover_params['user'],
                          "message":msg
                      })
    except:
        print('Pushover error')


class ToggleButtonState(Enum):
    DISABLED = 0
    OFF = 1
    ON = 2


class CLASGUI(QtWidgets.QMainWindow):
    BUFFER_LEN = 30  # data to keep in memory, mostly for plotting

    pool = QThreadPool.globalInstance()

    mode_info: Optional[list] = None
    eeg_start_time: datetime.datetime

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger()

        # load the UI
        uic.loadUi('CLASGUI.ui', self)

        # load config
        with open('config.yml', 'r') as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)

        # set darkmode
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        plt.style.use('dark_background')

        # files and stuff
        self.meta_path = ''
        self.meta_data = {}
        self.data_eeg = io.BytesIO()
        self.data_phase = io.BytesIO()
        self.data_intent = io.BytesIO()

        self.eeg = QEEGStreamer()


        ### BIND BUTTONS ###
        # main control
        self.btn_startstreaming.clicked.connect(
            self.btn_startstreaming_clicked)

        self.btn_startclas.clicked.connect(self.btn_startclas_clicked)
        self.btn_startclas_running = False

        self.btn_browseforparams.clicked.connect(
            self.btn_browseforparams_clicked)

        # sound and trigger tests
        self.btn_soundtest_startper.clicked.connect(
            self.soundtest_start_periodic)
        self.btn_soundtest_startcont.clicked.connect(
            self.soundtest_start_continuous)
        self.btn_soundtest_stop.clicked.connect(self.soundtest_stop)

        self.btn_triggertest_seq.clicked.connect(self.triggertest_start)
        self.btn_triggertest_max.clicked.connect(self.triggertest_max)

        self.triggertest_state = -1

        ### Crossover controls ###
        # put all 4 combo boxes in a list
        self.combo_modes = [
            self.combo_mode1, self.combo_mode2, self.combo_mode3,
            self.combo_mode4
        ]
        self.time_modes = [
            self.time_mode1, self.time_mode2, self.time_mode3, self.time_mode4
        ]

        # combo list items to set, and their corresp ExperimentMode
        self.combo_mode_items = {
            'label':
            ['-- Stop --', 'CLAS', 'SHAM-Muted', 'SHAM-Phase', 'SHAM-Delay'],
            'value': [
                QCLASAlgo.ExperimentMode.DISABLED,
                QCLASAlgo.ExperimentMode.CLAS,
                QCLASAlgo.ExperimentMode.SHAM_MUTED,
                QCLASAlgo.ExperimentMode.SHAM_PHASE,
                QCLASAlgo.ExperimentMode.SHAM_DELAY
            ]
        }

        for cb in self.combo_modes:
            for cbl, cbv in zip(self.combo_mode_items['label'],
                                self.combo_mode_items['value']):
                cb.addItem(cbl, cbv)

        # when guessing sensible stop times
        # use today, if in the evening. if it's still night, use yesterday as base
        basedate = datetime.datetime.now()
        if basedate.time().hour < 4:
            basedate = basedate - datetime.timedelta(days=1)
        basedate = basedate.date()

        self.time_mode1.setDateTime(
            datetime.datetime.combine(basedate, datetime.time(hour=22)))
        self.time_mode2.setDateTime(
            datetime.datetime.combine(basedate + datetime.timedelta(days=1),
                                      datetime.time(hour=3)))
        self.time_mode3.setDateTime(
            datetime.datetime.combine(basedate + datetime.timedelta(days=1),
                                      datetime.time(hour=8)))
        self.time_mode4.setDateTime(
            datetime.datetime.combine(basedate + datetime.timedelta(days=1),
                                      datetime.time(hour=12)))

        ### INITIALIZE PLOTS ###
        self.tsplot = TSPlot(self.timeseries)
        self.hplot = HistoricalPlot(self.historical)

        ### TRIGGER OUTPUT ###
        comports = serial.tools.list_ports.comports()
        comports = [p for p in comports if p.vid in [9025, 10755]]

        if len(comports) == 0:
            raise Exception('No matching ports found')

        if len(comports) > 1:
            raise Exception('Multiple trigger interfaces detected')

        ### Show figure ###
        self.show()

        ### Preinitialize other variables ###
        self.status_report = QTimer(self)
        self.triggertest_timer = None
        self.soundtest_timer = None

        # guess the clas params file
        guessed_path = os.path.join(os.getcwd(), 'clas_params_gamma.json')
        if os.path.exists(guessed_path):
            self.txt_clasparampath.setText(guessed_path)

        self.update_gui_statuslabel_buttons()

    def update_debug_metadata(self, **kwargs):
        for k in kwargs:
            self.meta_data[k] = kwargs[k]

        with open(self.meta_path, 'w') as f:
            json.dump(self.meta_data, f)

    def btn_startstreaming_clicked(self):
        ''' Start or stop EEG streaming, depending on toggle button's current state. '''
        print(datetime.datetime.now().isoformat() +
              ' → CLASGUI.btn_startstreaming_clicked() | Previous state: ' +
              str(self.eeg.state))

        if self.eeg.state != ConnectionState.DISCONNECTED:
            self.stop_streaming()
        else:
            self.start_streaming()

    def btn_startclas_clicked(self):
        ''' Start or stop CLAS algorithm, depending on toggle button state. '''
        print(datetime.datetime.now().isoformat() +
              ' → CLASGUI.btn_startclas_clicked() | Previous state: ' +
              str(self.btn_startclas_running))

        if self.btn_startclas_running is True:
            self.stop_algo()
        else:
            self.start_algo()

    def btn_browseforparams_clicked(self):
        print(datetime.datetime.now().isoformat() +
              ' → CLASGUI.btn_browseforparams_clicked()')

        filename, _ = QFileDialog.getOpenFileName(
            self, 'Select CLAS parameter file', os.getcwd(),
            'JSON file (*.json)')
        self.txt_clasparampath.setText(filename)

        self.update_gui_statuslabel_buttons()

    def update_gui_statuslabel_buttons(self):
        ''' Read the algorithm state and EEG state and set all GUI elements to match the internal state. '''
        print(datetime.datetime.now().isoformat() +
              ' → CLASGUI.update_gui_statuslabel_buttons()')

        # initialize button state variables
        # the logic will set these, then we update the status at the end
        btn_eeg = ToggleButtonState.DISABLED
        btn_clas = ToggleButtonState.DISABLED

        if self.eeg.state != ConnectionState.DISCONNECTED:  # is eeg streaming
            if self.btn_startclas_running:  # has "start CLAS" been clicked
                # eeg streaming, algo has been cued to start
                if self.runner.ca.experiment_mode == QCLASAlgo.ExperimentMode.DISABLED:
                    # algo cued to start, but not yet...
                    status_text = 'Waiting for start...'
                else:
                    # algo cued to start and is currently running
                    status_text = 'Running ' + str(
                        self.runner.ca.experiment_mode).split('.')[-1]

                btn_clas = ToggleButtonState.ON
                btn_eeg = ToggleButtonState.DISABLED

            else:
                # eeg streaming, but algo is disabled
                if self.eeg.state == ConnectionState.STREAMING:
                    status_text = 'EEG streaming'
                    btn_eeg = ToggleButtonState.ON
                    btn_clas = ToggleButtonState.OFF

                else:
                    status_text = 'EEG connecting: ' + str(self.eeg.state)
                    btn_eeg = ToggleButtonState.DISABLED
                    btn_clas = ToggleButtonState.DISABLED

        else:
            # eeg not streaming
            status_text = 'Disconnected'

            # eeg control toggle button state
            btn_eeg = ToggleButtonState.OFF
            btn_clas = ToggleButtonState.DISABLED

        ### BUTTON STATES
        # eeg button
        if btn_eeg == ToggleButtonState.DISABLED:
            self.btn_startstreaming.setText('-')
            self.btn_startstreaming.setStyleSheet('background-color: none')
            self.btn_startstreaming.setEnabled(False)

        elif btn_eeg == ToggleButtonState.OFF:
            self.btn_startstreaming.setText('Start streaming')
            self.btn_startstreaming.setStyleSheet('background-color: none')

            if len(self.txt_clasparampath.text()) > 0:
                self.btn_startstreaming.setEnabled(True)
            else:
                self.btn_startstreaming.setEnabled(False)

        elif btn_eeg == ToggleButtonState.ON:
            self.btn_startstreaming.setText('Stop streaming')
            self.btn_startstreaming.setStyleSheet('background-color: orange')
            self.btn_startstreaming.setEnabled(True)

        # clas buttons
        if btn_clas == ToggleButtonState.DISABLED:
            self.btn_startclas.setText('-')
            self.btn_startclas.setStyleSheet('background-color: none')
            self.btn_startclas.setEnabled(False)

            for el_c, el_t in zip(self.combo_modes, self.time_modes):
                el_c.setEnabled(True)
                el_t.setEnabled(True)

        elif btn_clas == ToggleButtonState.OFF:
            self.btn_startclas.setText('Start CLAS algo')
            self.btn_startclas.setStyleSheet('background-color: none')
            self.btn_startclas.setEnabled(True)

            for el_c, el_t in zip(self.combo_modes, self.time_modes):
                el_c.setEnabled(True)
                el_t.setEnabled(True)

        elif btn_clas == ToggleButtonState.ON:
            self.btn_startclas.setText('Stop CLAS algo')
            self.btn_startclas.setStyleSheet('background-color: orange')
            self.btn_startclas.setEnabled(True)

            for el_c, el_t in zip(self.combo_modes, self.time_modes):
                el_c.setEnabled(False)
                el_t.setEnabled(False)

        # trigger test
        if self.triggertest_timer is None and not self.btn_startclas_running:
            self.btn_triggertest_seq.setEnabled(True)
            self.btn_triggertest_max.setEnabled(True)
        else:
            self.btn_triggertest_seq.setEnabled(False)
            self.btn_triggertest_max.setEnabled(False)

        if self.triggertest_timer is not None:
            status_text = 'Trigger test'  # override status text with test

        # sound test
        if self.soundtest_timer is None and not self.btn_startclas_running:
            self.btn_soundtest_startper.setEnabled(True)
            self.btn_soundtest_startcont.setEnabled(True)
            self.btn_soundtest_stop.setEnabled(False)
        elif self.soundtest_timer is not None:
            self.btn_soundtest_startper.setEnabled(False)
            self.btn_soundtest_startcont.setEnabled(False)
            self.btn_soundtest_stop.setEnabled(True)
        else:
            self.btn_soundtest_startper.setEnabled(False)
            self.btn_soundtest_startcont.setEnabled(False)
            self.btn_soundtest_stop.setEnabled(False)


        if self.soundtest_timer is not None:
            status_text = 'Sound test'  # override status text with test

        self.txt_status.setText(status_text)

    def start_algo(self):
        ''' Start the CLAS algorithm. '''
        print(datetime.datetime.now().isoformat() + ' → CLASGUI.start_algo()')

        # if self.eeg.state != ConnectionState.STREAMING:
        #     msg = '!!! EEG state is ' + str(self.eeg.state) + '. Cannot start algo.'
        #     print(msg)
        #     send_error(msg)
        #     return

        self.runner.set_experiment_mode(QCLASAlgo.ExperimentMode.DISABLED)

        print('===============')

        # parse datetime and options, set timers
        mode_info = []
        for cm, tm in zip(self.combo_modes, self.time_modes):
            mode_sel_idx = cm.currentIndex()
            mode_sel_val = QCLASAlgo.ExperimentMode(
                cm.itemData(mode_sel_idx, Qt.UserRole))
            mode_time = tm.dateTime().toPyDateTime()

            msec_until_event = int(
                (mode_time - datetime.datetime.now()).total_seconds() * 1000)
            print('Switch to {} at {} ({:.0f} s)'.format(
                str(mode_sel_val), mode_time.isoformat(),
                msec_until_event / 1000))

            if msec_until_event <= 0:
                # run it now
                self.algo_set_mode(mode_sel_val)

            mode_info.append({
                'mode': mode_sel_val,
                'sched_time': mode_time,
                'delay_ms': msec_until_event
            })

        print('===============')

        self.mode_info = mode_info

        # status reporting timer
        self.status_report = QTimer(self)
        self.status_report.timeout.connect(self.check_and_print_status)
        self.status_report.start(1 * 60 * 1000)
        self.check_and_print_status()

        # write crossover info to debug log
        mode_info_print = mode_info.copy()
        for kk in range(len(mode_info_print)):
            citm = mode_info_print[kk].copy()
            citm['sched_time'] = citm['sched_time'].isoformat(
            )  # can't JSON datetime
            mode_info_print[kk] = citm
        self.update_debug_metadata(crossover=mode_info_print)

        self.btn_startclas_running = True
        self.update_gui_statuslabel_buttons()

    def stop_algo(self):
        ''' Stop the algorithm. '''
        print(datetime.datetime.now().isoformat() + ' → CLASGUI.stop_algo()')

        self.status_report.stop()
        self.algo_set_mode(QCLASAlgo.ExperimentMode.DISABLED)
        self.btn_startclas_running = False
        self.update_gui_statuslabel_buttons()

    def algo_set_mode(self, mode: QCLASAlgo.ExperimentMode):
        '''
        Switch experiment modes.
          1. Update CLAS algorithm module with the intended experiment mode
          2. Update the status box in the GUI
          3. Signal the change in experiment mode using the parallel port
        
        Called when starting CLAS algo, or in a timer at the specified time.
        '''
        print('Setting experiment mode to {} at {}'.format(
            str(mode),
            datetime.datetime.now().isoformat()))

        self.runner.set_experiment_mode(mode)
        self.send_pp_mode_train(mode)
        self.update_gui_statuslabel_buttons()

    def start_streaming(self):
        ''' Initiate streaming when the button is pressed '''
        print(datetime.datetime.now().isoformat() +
              ' → CLASGUI.start_streaming()')

        # reinitialize plot
        self.tsplot.clear()

        # setup output files
        filepath = QFileDialog.getSaveFileName(
            self,
            caption="Destination save file",
            filter="JSON file (*.debug.json)")[0]

        if filepath is None or len(filepath) == 0:
            print('No file selected')
            return

        # get base file root
        save_dir, save_basename = os.path.split(filepath)
        save_basename = os.path.splitext(save_basename)[0]
        save_basename = os.path.join(save_dir, save_basename)

        # files
        self.meta_path = save_basename + '.json'
        self.meta_data = {'type': 'curry_debug_v1'}
        self.data_eeg = open(save_basename + '.eeg.bin', 'wb')
        self.data_phase = open(save_basename + '.phase.bin', 'wb')
        self.data_intent = open(save_basename + '.intent.bin', 'wb')

        self.update_debug_metadata()

        # increase process priority
        process = psutil.Process(os.getpid())
        process.nice(psutil.HIGH_PRIORITY_CLASS)

        # initialize EEG system
        self.eeg = QEEGStreamer()
        self.eeg.dataReceived.connect(self.eeg_data_received)  #type:ignore
        self.eeg.initialized.connect(self.eeg_connected)  #type:ignore

        self.eeg.connectToHost(self.config['port_eeg'])

        self.update_gui_statuslabel_buttons()

    def stop_streaming(self):
        ''' Stop streaming, close EEG connection. '''
        print(datetime.datetime.now().isoformat() +
              ' → CLASGUI.stop_streaming()')

        self.eeg.close_connection()
        self.plot_timer.stop()

        def close_files():
            self.data_eeg.close()
            self.data_phase.close()
            self.data_intent.close()
        QTimer.singleShot(1000, close_files)  # schedule timer to close files in 1 second, after the rest of the data has been written

        # reset process priority
        process = psutil.Process(os.getpid())
        process.nice(psutil.NORMAL_PRIORITY_CLASS)

        self.update_gui_statuslabel_buttons()

    def eeg_connected(self):
        ''' Callback for eegConnected signal.
        Display active channel information, initialize timeseries plots, initialize buffers.
        '''
        print(datetime.datetime.now().isoformat() +
              ' → CLASGUI.eeg_connected()')

        # print channel information to the textbox
        info_list_as_text = [
            '%d - %s: %d' % (x['id'], x['chanLabel'], x['deviceType'])
            for x in self.eeg.info_list
        ]
        info_list_as_text = '\n'.join(info_list_as_text)
        self.chInfo.setPlainText(info_list_as_text)

        # initialize circular buffer
        self.fsample = self.eeg.basic_info['sampleRate']
        self.connected_time = time.time_ns()
        self.update_debug_metadata(
            fsample=self.fsample,
            acq_start_time=datetime.datetime.now().isoformat(),
            connected_time=self.connected_time)

        nsamples = self.fsample * (self.BUFFER_LEN)
        self.data_buffer = np.zeros((nsamples))
        self.time_values = np.arange(-1 * nsamples, 0) / self.fsample

        # initialize algorithm
        param_path = self.txt_clasparampath.text()
        self.runner = QCLASAlgo.EEGProcessor(fsample=self.fsample,
                                             param_file=param_path)
        self.runner.signals.cue_stim.connect(self.cue_stim)  #type:ignore
        self.runner.signals.datavals.connect(self.get_internals)  #type:ignore

        self.update_debug_metadata(clas_params=self.runner.params)

        # initial plot
        self.tsplot.plot(self.time_values, self.data_buffer)

        # tell CURRY NetStreamer to start sending data
        self.eeg.start_streaming()

        # start a timer to refresh the plot
        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.plot_eeg_data)
        self.plot_timer.start(500)

        # store current time
        self.eeg_start_time = datetime.datetime.now()

        self.update_gui_statuslabel_buttons()

    def get_internals(self, internals: dict):
        self.data_phase.write("BLKBLKBLK".encode('utf8'))
        self.data_phase.write(
            self.eeg_data_time.to_bytes(8, byteorder='little', signed=False))
        self.data_phase.write(
            struct.pack('<fffff', internals['phase'], internals['freq'],
                        internals['amp'], internals['meanamp'],
                        internals['quadrature']))

        self.txt_amplitude.setText('{:.2f}'.format(internals['amp']))
        self.txt_quadrature.setText('{:.4f}'.format(internals['quadrature']))

    def plot_eeg_data(self):
        ''' Called when plot refresh timer times out. Tell the MPL subclass to update data and redraw plot. '''
        self.tsplot.update_data(self.data_buffer)
        sys.stdout.flush()

    def eeg_data_received(self, sample_start: int, data: np.ndarray, rawdata: int):
        ''' 
        Callback for dataReceived signal. 
        Add data to circular buffer, check if the data should be sent for analysis. 
        '''
        self.eeg_data_time = time.time_ns()

        nsamples_received = data.shape[1]
        nsamples_remain = self.data_buffer.shape[0] - nsamples_received

        # store sample number of ending sample
        self.latest_sample_num = sample_start + nsamples_received

        # roll the array
        self.data_buffer[:nsamples_remain] = self.data_buffer[
            -nsamples_remain:]

        # add new data
        self.data_buffer[-nsamples_received:] = data[0, :]

        # run CLAS in separate thread
        self.runner.replace_data((datetime.datetime.now() - self.eeg_start_time).total_seconds(),
                                 self.data_buffer)
        self.pool.start(self.runner)

        # write stuff to file
        self.data_eeg.write("$".encode('utf8'))
        self.data_eeg.write(
            self.eeg_data_time.to_bytes(8, byteorder='little', signed=False))
        self.data_eeg.write(
            sample_start.to_bytes(1, byteorder='little', signed=False))

        npdata = data.astype('float32').tobytes()
        self.data_eeg.write(npdata)
        self.data_eeg.write(rawdata.to_bytes(4, byteorder='little', signed=True))

    def soundtest_start_periodic(self):
        ''' Play periodic auditory tones for a sound test. '''
        print(datetime.datetime.now().isoformat() +
              ' → CLASGUI.soundtest_start_periodic()')

        if self.soundtest_timer is not None:
            self.soundtest_timer.stop()

        self.soundtest_timer = QTimer(self)
        self.soundtest_timer.timeout.connect(self.soundtest_callback_periodic)
        self.soundtest_callback_periodic()
        self.soundtest_timer.start(1000)
        self.txt_soundtest.setText('Periodic tones playing...')

        self.update_gui_statuslabel_buttons()

    def soundtest_callback_periodic(self):
        ''' Called by periodic soundtest timer '''
        print(datetime.datetime.now().isoformat() +
              ' → CLASGUI.soundtest_callback_periodic()')
        QCLASAlgo.pink_noise.play()

    def soundtest_start_continuous(self):
        print(datetime.datetime.now().isoformat() +
              ' → CLASGUI.soundtest_start_continuous()')

        if self.soundtest_timer is not None:
            self.soundtest_timer.stop()

        self.soundtest_timer = QTimer(self)
        self.soundtest_timer.timeout.connect(
            self.soundtest_callback_continuous)
        self.txt_soundtest.setText('Continuous tone playing...')

        self.soundtest_continous_wav = QCLASAlgo.PinkNoiseGenerator.generate_noise(
            length=1, ramp=0)

        self.soundtest_callback_continuous()
        self.soundtest_timer.start(950)

        self.update_gui_statuslabel_buttons()

    def soundtest_callback_continuous(self):
        print(datetime.datetime.now().isoformat() +
              ' → CLASGUI.soundtest_callback_continuous()')
        self.soundtest_continous_wav.play()

    def soundtest_stop(self):
        print(datetime.datetime.now().isoformat() +
              ' → CLASGUI.soundtest_stop()')
        if self.soundtest_timer is not None:
            self.soundtest_timer.stop()
            self.soundtest_timer = None

        self.txt_soundtest.setText('Not running')

        self.update_gui_statuslabel_buttons()

    def triggertest_start(self):
        print(datetime.datetime.now().isoformat() +
              ' → CLASGUI.triggertest_start()')

        # reset timer
        if self.triggertest_timer is not None:
            self.triggertest_timer.stop()

        # reset trigger test state variable
        self.triggertest_state = 0

        # start callback timer
        self.triggertest_timer = QTimer(self)
        self.triggertest_timer.timeout.connect(self.triggertest_callback)
        self.triggertest_timer.start(250)

        self.txt_triggertest.setText('Starting')

        self.update_gui_statuslabel_buttons()

    def triggertest_callback(self):
        print(datetime.datetime.now().isoformat() +
              ' → CLASGUI.triggertest_callback()  |  state = {:d}'.format(
                  self.triggertest_state))

        if self.triggertest_state > 15:
            self.triggertest_timer.stop()  # type:ignore
            self.triggertest_timer = None

            self.port.write((255).to_bytes(1, byteorder='little',
                                           signed=False))

            self.triggertest_reset()

        else:
            pw = self.triggertest_state if (self.triggertest_state < 8) else (
                15 - self.triggertest_state)
            self.triggertest_state += 1

            self.txt_triggertest.setText('Sent {:d}'.format(2**pw))
            self.port.write((2**pw).to_bytes(1,
                                             byteorder='little',
                                             signed=False))

    def triggertest_max(self):
        print(datetime.datetime.now().isoformat() +
              ' → CLASGUI.triggertest_max()')
        self.port.write((255).to_bytes(1, byteorder='little', signed=False))
        self.triggertest_state = -2
        self.txt_triggertest.setText('Sending 255...')
        QTimer.singleShot(100, self.triggertest_reset)

    def triggertest_reset(self):
        print(datetime.datetime.now().isoformat() +
              ' → CLASGUI.triggertest_reset()')
        self.triggertest_state = -1
        self.txt_triggertest.setText('Not running')
        self.update_gui_statuslabel_buttons()

    def cue_stim(self, delay: int, trig: int, muted: bool):
        if delay > 0:
            QTimer.singleShot(delay, self.get_defer_stim(trig, muted))
        else:
            self.deliver_stim(trig, muted)

        self.data_intent.write(
            int(self.eeg_data_time + (delay * 1e6)).to_bytes(
                8, byteorder='little', signed=False))
        self.data_intent.write(
            self.runner.ca.experiment_mode.to_bytes(1,
                                                    byteorder='little',
                                                    signed=False))
        self.data_intent.write(
            trig.to_bytes(1, byteorder='little', signed=False))

    def reset_pp(self):
        ''' Set parallel port output to zero. '''
        # this is no longer needed because the port automatically goes to zero
        pass

    def deliver_stim(self, trig: int, muted: bool = False):
        ''' Write trigger to parallel port and play sound if asked. '''
        self.port.write((trig).to_bytes(1, byteorder='little', signed=False))
        if not muted:
            QCLASAlgo.pink_noise.play()
        print('Stim {:d} delivered at {} (muted = {})'.format(
            trig,
            datetime.datetime.now().isoformat(), muted))

    def get_defer_stim(self, trig: int, muted: bool = False) -> Callable:
        ''' Return a callable with the trigger built in. '''

        def func():
            self.deliver_stim(trig, muted)

        return func

    def check_and_print_status(self):
        # print current mode
        now = datetime.datetime.now()
        if now.minute % 5 == 0:
            print('{} ? Current mode: {}'.format(
                now.isoformat(),
                str(self.runner.ca.experiment_mode)))

        # check if the mode needs to be changed
        # current mode should be the recent negative time
        past_modes = [
            mode['mode'] for mode in self.mode_info
            if (mode['sched_time'] - now).total_seconds() < 0
        ]

        if (len(past_modes) > 0) and (self.runner.ca.experiment_mode != past_modes[-1]):
            self.algo_set_mode(past_modes[-1])


# class TSPlot(FigureCanvasQTAgg):

#     def __init__(self, parent=None, dpi=60):
#         # initialize the figure
#         wsize = parent.size()
#         self.fig = matplotlib.figure.Figure(figsize=(wsize.width() / dpi,
#                                                      wsize.height() / dpi),
#                                             dpi=dpi)  #type:ignore
#         self.axes = self.fig.add_subplot(111)
#         self.axes.invert_yaxis()
#         self.axes.set_position((0.05, 0.18, 0.92, 0.77))

#         s = super(TSPlot, self)
#         s.__init__(self.fig)
#         self.setParent(parent)

#         s.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
#                         QtWidgets.QSizePolicy.Expanding)
#         s.updateGeometry()

#     def clear(self):
#         self.axes.cla()

#     def plot(self, time, data):
#         self.plt = self.axes.plot(time, data.T, lw=1, c='white', alpha=0.8)
#         self.axes.set_ylim(-500, 500)
#         self.draw()

#     def update_data(self, data):
#         # for kk, ln in enumerate(self.plt):
#         #     ln.set_ydata(data[kk, :])
#         self.plt[0].set_ydata(data)
#         self.draw()


# class HistoricalPlot(FigureCanvasQTAgg):

#     def __init__(self, parent=None, dpi=60):
#         # initialize the figure
#         wsize = parent.size()
#         self.fig = matplotlib.figure.Figure(figsize=(wsize.width() / dpi,
#                                                      wsize.height() / dpi),
#                                             dpi=dpi)  #type:ignore
#         self.axes = self.fig.add_subplot(111)
#         self.axes.invert_yaxis()
#         self.axes.set_position((0.05, 0.18, 0.92, 0.77))

#         s = super(HistoricalPlot, self)
#         s.__init__(self.fig)
#         self.setParent(parent)

#         s.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
#                         QtWidgets.QSizePolicy.Expanding)
#         s.updateGeometry()

#     def clear(self):
#         self.axes.cla()

#     def plot(self, time, data):
#         self.plt = self.axes.plot(time, data.T)
#         self.axes.set_ylim(-15, 0)
#         self.draw()

#     def update_data(self, data):
#         # for kk, ln in enumerate(self.plt):
#         #     ln.set_ydata(data[kk, :])
#         self.plt[0].set_ydata(data)
#         self.draw()

# class DummyPort():
#     def __init__(self):
#         pass

#     def write(self, val:int):
#         print('PORT WRITE {:d}'.format(val))


# def excepthook(exc_type, exc_value, exc_tb):
#     tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
#     print('################\n## escapehook ##\n################\n' + tb +
#           '################')
#     send_error(tb)



if __name__ == "__main__":
    sys.excepthook = excepthook
    App = QtWidgets.QApplication(sys.argv)
    window = CLASGUI()
    sys.exit(App.exec())
