import sys
import time
import os
import os.path
import struct
import json
import datetime
import io

from PyQt5.QtCore import Qt, QSize, QTimer, QThreadPool
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QPushButton

import numpy as np

import QCurryInterface, QCLASAlgo


class MainWindow(QMainWindow):
    BUFFER_LEN = 30
    pool = QThreadPool.globalInstance()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        button = QPushButton("Save streaming data")

        self.setFixedSize(QSize(200, 100))

        # Set the central widget of the Window.
        self.setCentralWidget(button)

        # files and stuff
        self.meta_path = ''
        self.meta_data = {}
        self.data_eeg = io.BytesIO()
        self.data_phase = io.BytesIO()
        self.data_intent = io.BytesIO()

        self.eeg = QCurryInterface.QCurryInterface()

        # bind close
        self.destroyed.connect(self.onclose)
        button.clicked.connect(self.start_streaming)

    def start_streaming(self):
        ''' Initiate streaming when the button is pressed '''

        # setup output files
        filepath = QFileDialog.getSaveFileName(self, caption="Destination save file", filter="JSON file (*.json)")[0]

        if filepath is None or len(filepath) == 0:
            print('No file selected')
            return

        # get base file root
        save_dir, save_basename = os.path.split(filepath)
        save_basename = os.path.splitext(os.path.splitext(save_basename)[0])[0]
        save_basename = os.path.join(save_dir, save_basename)

        # files
        self.meta_path = save_basename + '.json'
        self.meta_data = {'type': 'curry_debug_v1'}
        self.data_eeg = open(save_basename + '.eeg.bin', 'wb')
        self.data_phase = open(save_basename + '.phase.bin', 'wb')
        self.data_intent = open(save_basename + '.intent.bin', 'wb')

        # initialize CURRY
        self.eeg = QCurryInterface.QCurryInterface()
        self.eeg.dataReceived.connect(self.eeg_data_received)
        self.eeg.initialized.connect(self.eeg_connected)

        self.eeg.connectToHost()

    def stop_streaming(self):
        self.eeg.stop_streaming()

    def update_meta(self, **kwargs):
        for k in kwargs:
            self.meta_data[k] = kwargs[k]

        with open(self.meta_path, 'w') as f:
            json.dump(self.meta_data, f)

    def eeg_connected(self):
        ''' Callback for eegConnected signal.
        Display active channel information, initialize timeseries plots, initialize buffers.
        '''

        # print channel information to the textbox
        info_list_as_text = ['%d - %s: %d' % (x['id'], x['chanLabel'], x['deviceType']) for x in self.eeg.info_list]
        info_list_as_text = '\n'.join(info_list_as_text)
        print(info_list_as_text)

        # initialize circular buffer
        self.fsample = self.eeg.basic_info['sampleRate']
        self.update_meta(fsample=self.fsample, acq_start_time=datetime.datetime.now().isoformat())

        nsamples = self.fsample * (self.BUFFER_LEN)
        self.data_buffer = np.zeros((nsamples))
        self.time_values = np.arange(-1 * nsamples, 0) / self.fsample

        # initialize algorithm
        self.runner = QCLASAlgo.EEGProcessor(fsample=self.fsample,
                                             param_file=os.path.join(os.getcwd(), 'clas_params.json'))
        self.runner.signals.cue_stim.connect(self.cue_stim)
        self.runner.signals.datavals.connect(self.get_internals)

        # tell CURRY NetStreamer to start sending data
        self.eeg.start_streaming()

    def eeg_data_received(self, sample_start: int, data: np.ndarray):
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
        self.data_buffer[:nsamples_remain] = self.data_buffer[-nsamples_remain:]

        # add new data
        self.data_buffer[-nsamples_received:] = data[0, :]

        # run CLAS in separate thread
        self.runner.replace_data(self.latest_sample_num / self.fsample, self.data_buffer)
        self.pool.start(self.runner)

        # write stuff to file
        self.data_eeg.write("BLKBLKBLK".encode('utf8'))
        self.data_eeg.write(self.eeg_data_time.to_bytes(8, byteorder='little', signed=False))
        self.data_eeg.write(sample_start.to_bytes(8, byteorder='little', signed=False))

        npdata = data.astype('float32').tobytes()
        self.data_eeg.write(len(npdata).to_bytes(4, byteorder='little', signed=False))
        self.data_eeg.write(npdata)

    def cue_stim(self, delay: int, trig: int):
        self.data_intent.write((self.eeg_data_time + (delay * 1000)).to_bytes(8, byteorder='little', signed=False))
        self.data_intent.write(trig.to_bytes(1, byteorder='little', signed=False))

    def get_internals(self, internals: dict):
        self.data_phase.write("BLKBLKBLK".encode('utf8'))
        self.data_phase.write(self.eeg_data_time.to_bytes(8, byteorder='little', signed=False))
        self.data_phase.write(
            struct.pack('<fffff', internals['phase'], internals['freq'], internals['amp'], internals['meanamp'],
                        internals['quadrature']))

    def onclose(self):
        self.stop_streaming()

        self.data_eeg.close()
        self.data_phase.close()
        self.data_intent.close()


app = QApplication(sys.argv)
window = MainWindow()
window.show()

app.exec()
