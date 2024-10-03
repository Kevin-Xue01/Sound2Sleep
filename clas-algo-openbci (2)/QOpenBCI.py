'''
Simeon Wong
Ibrahim Lab
Hospital for Sick Children

Based on the CURRY MATLAB Interface.
'''

import numpy as np
import re
from enum import IntEnum
from typing import Union, Optional
import logging
import scipy.signal

from PyQt5.QtCore import pyqtSignal, QObject, QTimer

# EEG connection
import serial


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
    basic_info = {'sampleRate': 250}
    info_list = []

    # Signals
    initialized = pyqtSignal()
    dataReceived = pyqtSignal(int, np.ndarray, int)
    eventReceived = pyqtSignal(object)

    #####################
    # Functions n stuff #
    #####################
    def __init__(self,
                 debug: bool = False,
                 dump_data_path: Union[str, None] = None,
                 logger: Optional[logging.Logger] = None,
                 data_filters: Optional[list] = [{
                     'notch': [60]
                 }, {
                     'demean': 5
                 }]):
        # initialize TCP connection
        super().__init__()

        ##### TEMPORARILY USE ORIGINAL FILTERS #####
        # NOTE: These filters result in a group/phase delay!
        # data_filters = [{'bandpass': [0.3, 30]}, {'notch': [60]}]
        ############################################

        self.bcon = serial.Serial()
        self.buffer = bytearray()

        self.debug = debug

        # self.scale = -4.5 / 24 / ((2 ^ 23) - 1)  # based off OpenBCI documentation
        self.scale = -0.02235  # empirical

        self.recv_timer = QTimer(self)

        if logger is None:
            self.logger = logging.getLogger('QOpenBCI')
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

        if dump_data_path is not None:
            self.dump_data_path = dump_data_path
            self.dump_data_file = open(self.dump_data_path, 'wb')
        else:
            self.dump_data_path = None
            self.dump_data_file = None

        # initialize specified filters
        self.filters = []
        for f in data_filters:
            iirtype = list(
                filter(lambda x: x in f,
                       ['bandpass', 'highpass', 'lowpass', 'bandstop']))
            if iirtype:
                cfilt = scipy.signal.iirfilter(
                    4,
                    f[iirtype[0]],
                    btype=iirtype[0],
                    ftype='bessel',
                    output='sos',
                    fs=self.basic_info['sampleRate'])
                self.filters.append({'sos': cfilt})

            elif 'notch' in f:
                cfilt = scipy.signal.iirnotch(60,
                                              20,
                                              fs=self.basic_info['sampleRate'])
                self.filters.append({'b': cfilt[0], 'a': cfilt[1]})

            elif 'demean' in f:
                self.filters.append({
                    'demean_sp':
                    f['demean'] * self.basic_info['sampleRate']
                })  # convert seconds into samples


    def connectToHost(self, serial_port: Optional[str] = None):
        print('Connecting to socket...\n')

        if serial_port is None:
            # list serial ports, find one matching OpenBCI VID
            import serial.tools.list_ports
            comports = serial.tools.list_ports.comports()
            comports = [p for p in comports if p.vid == 1027]

            if len(comports) != 1:
                raise Exception(f'Cannot automatically identify OpenBCI port. {len(comports):d} matching ports detected!')
            
            serial_port = comports[0].device

        self.state = ConnectionState.INIT_CONNECTING
        self.bcon = serial.Serial(port=serial_port, baudrate=115200, timeout=0)
        self.bcon.write(b'v')  # this will reset Cyton

        # start a timer to read for ready prompt
        self.ready_wait_num = 0
        self.recv_timer.stop()
        self.recv_timer = QTimer(self)
        self.recv_timer.timeout.connect(self.wait_for_ready)
        self.recv_timer.setInterval(500)
        self.recv_timer.start()

        self.logger.info('EEG Scale = {:.6f}'.format(self.scale))

    def wait_for_ready(self):
        inp = self.bcon.read(size=500)
        self.buffer.extend(inp)

        self.ready_wait_num = self.ready_wait_num + 1

        print(inp)
        if (self.buffer[-3:] == b'$$$'):
            # ready!
            self.state = ConnectionState.READY
            self.recv_timer.stop()

            # clear buffer
            self.buffer = bytearray()
            self.bcon.write(b'ddd')  # reset default settings

            self.initialized.emit()  # let the calling app know we're ready

        elif self.ready_wait_num > 5:
            print('Trying to reset again...')
            self.bcon.write(b'v')   # try resetting again
            self.ready_wait_num = 0   # reset timeout





    def start_streaming(self):
        ''' Ask CURRY to start sending data '''
        if self.state != ConnectionState.READY:
            self.logger.warning(
                'QOpenBCI.start_streaming() requested while connection is not ready.'
            )
            return False

        # reinitialize filter initial conditions
        for f in self.filters:
            if 'b' in f:
                f['state'] = scipy.signal.lfilter_zi(f['b'], f['a'])
            elif 'sos' in f:
                f['state'] = scipy.signal.sosfilt_zi(f['sos'])
            elif 'demean_sp' in f:
                f['state'] = np.zeros(f['demean_sp'])

        # turn off all other channels except for Ch1
        self.bcon.write(b'!2345678bbbbbbbb')
        
        self.bcon.reset_input_buffer()

        # start a timer to check for incoming data
        self.recv_timer.stop()
        self.recv_timer = QTimer(self)
        self.recv_timer.timeout.connect(self.data_handler)
        self.recv_timer.setInterval(100)
        self.recv_timer.start()

        self.state = ConnectionState.STREAMING

    def data_handler(self):
        # read the connection
        while self.bcon.in_waiting >= 33:
            pkt = self.bcon.read(33)
            # print(pkt)

            if pkt[0] != 0xA0:
                # data error
                self.resync_data()

            sample_start = pkt[1]
            eeg_data = pkt[2:26]

            # pass the eeg data onwards for processing
            self.process_eeg(sample_start, eeg_data)

    def resync_data(self):
        # keep dumping bytes until we get to a stop byte
        self.logger.warning('Resyncing data frame...')

        n_bytes = 0
        nextbyte = 0
        while not ((nextbyte >= 192) and
                                       (nextbyte <= 198)):
            n_bytes += 1
            nextbyte = int.from_bytes(self.bcon.read(1),
                                      byteorder='little',
                                      signed=False)
            
            if nextbyte == 0xA0:
                self.logger.warn('Header byte found')
                
            if n_bytes > 20000:
                self.logger.warn('Stop byte not found...')
                self.bcon.reset_input_buffer()
                self.bcon.reset_output_buffer()
                self.bcon.write(b'bbb')
                break

        self.logger.warn(
            'Stop byte found after skipping {:d} bytes'.format(n_bytes))

    def stop_streaming(self):
        ''' Ask CURRY to stop sending data '''
        if self.state != ConnectionState.STREAMING:
            return False
        self.bcon.write(b'sssss')
        self.recv_timer.stop()
        self.bcon.reset_input_buffer()

        self.state = ConnectionState.READY

    def close_connection(self):
        ''' Stop streaming, flush buffer, then close the connection. '''
        self.stop_streaming()

        self.bcon.flush()
        self.bcon.close()
        self.state = ConnectionState.DISCONNECTED

    def process_eeg(self, sample_start, data):
        ''' Scale the data, then apply the requested filters. '''
        rawdata = int.from_bytes(data[0:3], byteorder='big', signed=True)
        processed = np.array([rawdata * self.scale])

        for f in self.filters:
            if 'b' in f:
                processed, f['state'] = scipy.signal.lfilter(f['b'],
                                                             f['a'],
                                                             processed,
                                                             zi=f['state'])
            elif 'sos' in f:
                processed, f['state'] = scipy.signal.sosfilt(sos=f['sos'],
                                                             x=processed,
                                                             zi=f['state'])
            elif 'demean_sp' in f:
                f['state'][:-1] = f['state'][1:]
                f['state'][-1] = processed
                processed = processed - np.mean(f['state'])

        # emit the received signal
        self.dataReceived.emit(sample_start, processed[:, None], rawdata)
