'''
Simeon Wong
Ibrahim Lab
Hospital for Sick Children

Based on the CURRY MATLAB Interface.
'''

import struct
import array
import numpy as np
import re
import traceback
from enum import IntEnum
from typing import Optional

from PyQt5.QtCore import pyqtSignal, QObject, QTimer

# EEG connection
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations


class ConnectionState(IntEnum):
    DISCONNECTED = 1
    INIT_CONNECTING = 2
    INIT_BASIC = 3
    INIT_CHANNEL = 4
    READY = 5
    STREAMING = 6


class QEEGStreamer(QObject):
    #################################
    # Decode / Encode packet header #
    #################################
    # connection state
    state = ConnectionState.DISCONNECTED

    # packet header
    phdr = None

    # basic info
    basic_info = {'sampleRate': 0}
    info_list = []

    # Signals
    initialized = pyqtSignal()
    dataReceived = pyqtSignal(int, np.ndarray)
    eventReceived = pyqtSignal(object)

    # String cleaner
    strclean = re.compile('[^0-9a-zA-Z ]+')

    # setup board, but will be initialized when connecting to host
    board: BoardShim
    recv_timer: Optional[QTimer] = None  # type:ignore

    #####################
    # Functions n stuff #
    #####################
    def __init__(self,
                 debug: bool = False,
                 dump_data_path: Optional[str] = None):
        # initialize TCP connection
        super().__init__()

        self.debug = debug

        if dump_data_path is not None:
            self.dump_data_path = dump_data_path
            self.dump_data_file = open(self.dump_data_path, 'wb')
        else:
            self.dump_data_path = None
            self.dump_data_file = None

    def connectToHost(self, serial_port: str = 'COM3'):
        print('Connecting to socket...\n')
        bfip = BrainFlowInputParams()
        bfip.serial_port = serial_port
        self.board = BoardShim(BoardIds.CYTON_BOARD, bfip)
        self.board.prepare_session()
        self.board.config_board('!2345678')

        self.basic_info['sampleRate'] = self.board.get_sampling_rate(
            BoardIds.CYTON_BOARD)

        self.eeg_channels_idx = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD)

        self.initialized.emit()

    def data_handler(self):
        data = self.board.get_board_data()[self.eeg_channels_idx[0]]

        if (len(data) == 0):
            return

        # DataFilter.perform_bandpass(data, self.basic_info['sampleRate'], 15.5,
        #                             15, 2, FilterTypes.BESSEL, 0)
        data = np.array(data)[None, :]
        self.dataReceived.emit(0, data)

        print(data.shape)

    def connected_handler(self):
        pass

    def start_streaming(self):
        ''' Ask CURRY to start sending data '''
        if self.state == ConnectionState.STREAMING:
            return False
        self.board.start_stream()

        # poll for data
        if self.recv_timer is not None:
            self.recv_timer.stop()
        self.recv_timer = QTimer(self)
        self.recv_timer.timeout.connect(self.data_handler)
        self.recv_timer.setInterval(100)
        self.recv_timer.start()

        self.state = ConnectionState.STREAMING

    def stop_streaming(self):
        ''' Ask CURRY to stop sending data '''
        if self.state != ConnectionState.STREAMING:
            return False

        self.board.stop_stream()
        if self.recv_timer is not None:
            self.recv_timer.stop()

        self.state = ConnectionState.READY

    def close_connection(self):
        ''' Stop streaming, flush buffer, then close the connection. '''
        self.stop_streaming()
        self.board.release_session()
        self.state = ConnectionState.DISCONNECTED

    def process_eeg(self, sample_start, data):
        # emit the received signal
        self.dataReceived.emit(sample_start, data)
