import logging
from threading import Thread
from time import sleep

import matplotlib
import numpy as np
import seaborn as sns
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import firwin, lfilter, lfilter_zi

MUSE_NB_EEG_CHANNELS = 5
MUSE_SAMPLING_EEG_RATE = 256
LSL_EEG_CHUNK = 12

MUSE_NB_PPG_CHANNELS = 3
MUSE_SAMPLING_PPG_RATE = 64
LSL_PPG_CHUNK = 6

MUSE_NB_ACC_CHANNELS = 3
MUSE_SAMPLING_ACC_RATE = 52
LSL_ACC_CHUNK = 1

MUSE_NB_GYRO_CHANNELS = 3
MUSE_SAMPLING_GYRO_RATE = 52
LSL_GYRO_CHUNK = 1

# 00001800-0000-1000-8000-00805f9b34fb Generic Access 0x05-0x0b
# 00001801-0000-1000-8000-00805f9b34fb Generic Attribute 0x01-0x04
MUSE_GATT_ATTR_SERVICECHANGED = '00002a05-0000-1000-8000-00805f9b34fb' # ble std 0x02-0x04
# 0000fe8d-0000-1000-8000-00805f9b34fb Interaxon Inc. 0x0c-0x42
MUSE_GATT_ATTR_STREAM_TOGGLE = '273e0001-4c4d-454d-96be-f03bac821358' # serial 0x0d-0x0f
MUSE_GATT_ATTR_LEFTAUX = '273e0002-4c4d-454d-96be-f03bac821358' # not implemented yet 0x1c-0x1e
MUSE_GATT_ATTR_TP9 = '273e0003-4c4d-454d-96be-f03bac821358' # 0x1f-0x21
MUSE_GATT_ATTR_AF7 = '273e0004-4c4d-454d-96be-f03bac821358' # fp1 0x22-0x24
MUSE_GATT_ATTR_AF8 = '273e0005-4c4d-454d-96be-f03bac821358' # fp2 0x25-0x27
MUSE_GATT_ATTR_TP10 = '273e0006-4c4d-454d-96be-f03bac821358' # 0x28-0x2a
MUSE_GATT_ATTR_RIGHTAUX = '273e0007-4c4d-454d-96be-f03bac821358' #0x2b-0x2d
MUSE_GATT_ATTR_REFDRL = '273e0008-4c4d-454d-96be-f03bac821358' # not implemented yet 0x10-0x12
MUSE_GATT_ATTR_GYRO = '273e0009-4c4d-454d-96be-f03bac821358' # 0x13-0x15
MUSE_GATT_ATTR_ACCELEROMETER = '273e000a-4c4d-454d-96be-f03bac821358' # 0x16-0x18
MUSE_GATT_ATTR_TELEMETRY = '273e000b-4c4d-454d-96be-f03bac821358' # 0x19-0x1b
#MUSE_GATT_ATTR_MAGNETOMETER = '273e000c-4c4d-454d-96be-f03bac821358' # 0x2e-0x30
#MUSE_GATT_ATTR_PRESSURE = '273e000d-4c4d-454d-96be-f03bac821358' # 0x31-0x33
#MUSE_GATT_ATTR_ULTRAVIOLET = '273e000e-4c4d-454d-96be-f03bac821358' # 0x34-0x36
MUSE_GATT_ATTR_PPG1 = "273e000f-4c4d-454d-96be-f03bac821358" # ambient 0x37-0x39
MUSE_GATT_ATTR_PPG2 = "273e0010-4c4d-454d-96be-f03bac821358" # infrared 0x3a-0x3c
MUSE_GATT_ATTR_PPG3 = "273e0011-4c4d-454d-96be-f03bac821358" # red 0x3d-0x3f
MUSE_GATT_ATTR_THERMISTOR = "273e0012-4c4d-454d-96be-f03bac821358" # muse S only, not implemented yet 0x40-0x42

MUSE_ACCELEROMETER_SCALE_FACTOR = 0.0000610352
MUSE_GYRO_SCALE_FACTOR = 0.0074768

# How long to wait while scanning for devices
LIST_SCAN_TIMEOUT = 10.5
# How long to wait after device stops sending data before ending the stream
AUTO_DISCONNECT_DELAY = 3
# How long to wait in between connection attempts
RETRY_SLEEP_TIMEOUT = 1

LSL_SCAN_TIMEOUT = 5
LSL_BUFFER = 360

VIEW_SUBSAMPLE = 2
VIEW_BUFFER = 12

LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}




def view(window, scale, refresh, figure, backend, version=1):
    matplotlib.use(backend)
    sns.set(style="whitegrid")

    figsize = np.int16(figure.split('x'))

    print("Looking for an EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=LSL_SCAN_TIMEOUT)

    if len(streams) == 0:
        raise(RuntimeError("Can't find EEG stream."))
    print("Start acquiring data.")

    fig, axes = matplotlib.pyplot.subplots(1, 1, figsize=figsize, sharex=True)
    lslv = LSLViewer(streams[0], fig, axes, window, scale)
    fig.canvas.mpl_connect('close_event', lslv.stop)

    help_str = """
                toggle filter : d
                toogle full screen : f
                zoom out : /
                zoom in : *
                increase time scale : -
                decrease time scale : +
               """
    print(help_str)
    lslv.start()
    matplotlib.pyplot.show()


class LSLViewer():
    def __init__(self, stream, fig, axes, window, scale, dejitter=True):
        """Init"""
        self.stream = stream
        self.window = window
        self.scale = scale
        self.dejitter = dejitter
        self.inlet = StreamInlet(stream, max_chunklen=LSL_EEG_CHUNK)
        self.filt = False
        self.subsample = VIEW_SUBSAMPLE

        info = self.inlet.info()
        description = info.desc()

        self.sfreq = info.nominal_srate()
        self.n_samples = int(self.sfreq * self.window)
        self.n_chan = info.channel_count()

        ch = description.child('channels').first_child()
        ch_names = [ch.child_value('label')]

        for i in range(self.n_chan):
            ch = ch.next_sibling()
            ch_names.append(ch.child_value('label'))

        self.ch_names = ch_names

        fig.canvas.mpl_connect('key_press_event', self.OnKeypress)
        fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.fig = fig
        self.axes = axes

        sns.despine(left=True)

        self.data = np.zeros((self.n_samples, self.n_chan))
        self.times = np.arange(-self.window, 0, 1. / self.sfreq)
        impedances = np.std(self.data, axis=0)
        lines = []

        for ii in range(self.n_chan):
            line, = axes.plot(self.times[::self.subsample],
                              self.data[::self.subsample, ii] - ii, lw=1)
            lines.append(line)
        self.lines = lines

        axes.set_ylim(-self.n_chan + 0.5, 0.5)
        ticks = np.arange(0, -self.n_chan, -1)

        axes.set_xlabel('Time (s)')
        axes.xaxis.grid(False)
        axes.set_yticks(ticks)

        ticks_labels = ['%s - %.1f' % (ch_names[ii], impedances[ii])
                        for ii in range(self.n_chan)]
        axes.set_yticklabels(ticks_labels)

        self.display_every = int(0.2 / (12 / self.sfreq))

        self.bf = firwin(32, np.array([1, 40]) / (self.sfreq / 2.), width=0.05,
                         pass_zero=False)
        self.af = [1.0]

        zi = lfilter_zi(self.bf, self.af)
        self.filt_state = np.tile(zi, (self.n_chan, 1)).transpose()
        self.data_f = np.zeros((self.n_samples, self.n_chan))

    def update_plot(self):
        k = 0
        try:
            while self.started:
                samples, timestamps = self.inlet.pull_chunk(timeout=1.0,
                                                            max_samples=LSL_EEG_CHUNK)

                if timestamps:
                    if self.dejitter:
                        timestamps = np.float64(np.arange(len(timestamps)))
                        timestamps /= self.sfreq
                        timestamps += self.times[-1] + 1. / self.sfreq
                    self.times = np.concatenate([self.times, timestamps])
                    self.n_samples = int(self.sfreq * self.window)
                    self.times = self.times[-self.n_samples:]
                    self.data = np.vstack([self.data, samples])
                    self.data = self.data[-self.n_samples:]
                    filt_samples, self.filt_state = lfilter(
                        self.bf, self.af,
                        samples,
                        axis=0, zi=self.filt_state)
                    self.data_f = np.vstack([self.data_f, filt_samples])
                    self.data_f = self.data_f[-self.n_samples:]
                    k += 1
                    if k == self.display_every:

                        if self.filt:
                            plot_data = self.data_f
                        elif not self.filt:
                            plot_data = self.data - self.data.mean(axis=0)
                        for ii in range(self.n_chan):
                            if ii == 0 or ii == 3: continue
                            self.lines[ii].set_xdata(self.times[::self.subsample] -
                                                     self.times[-1])
                            self.lines[ii].set_ydata(plot_data[::self.subsample, ii] /
                                                     self.scale - ii)
                            impedances = np.mean(plot_data, axis=0)

                        ticks_labels = ['%s - %.2f' % (self.ch_names[ii],
                                                       impedances[ii])
                                        for ii in range(self.n_chan)]
                        self.axes.set_yticklabels(ticks_labels)
                        self.axes.set_xlim(-self.window, 0)
                        self.fig.canvas.draw()
                        k = 0
                else:
                    sleep(0.2)
        except RuntimeError as e:
            raise

    def onclick(self, event):
        print((event.button, event.x, event.y, event.xdata, event.ydata))

    def OnKeypress(self, event):
        if event.key == '/':
            self.scale *= 1.2
        elif event.key == '*':
            self.scale /= 1.2
        elif event.key == '+':
            self.window += 1
        elif event.key == '-':
            if self.window > 1:
                self.window -= 1
        elif event.key == 'd':
            self.filt = not(self.filt)

    def start(self):
        self.started = True
        self.thread = Thread(target=self.update_plot)
        self.thread.daemon = True
        self.thread.start()

    def stop(self, close_event):
        self.started = False

view(window=5, scale=100, refresh=0.2, figure="15x6", version=1, backend='TkAgg')
