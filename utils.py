import subprocess
import traceback
import psutil
from datetime import datetime
from multiprocessing import Process
import sys
import subprocess
import traceback
import time
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QThreadPool, QRunnable, pyqtSignal, QObject, QTimer
from pylsl import StreamInlet, resolve_stream, resolve_byprop, StreamInfo
from multiprocessing import Process, Queue
from enum import Enum, auto
from datetime import datetime

class StreamType(Enum):
    EEG = 'EEG'
    Accelerometer = 'Accelerometer'
    PPG = 'PPG'

class BlueMuseSignal(QObject):
    update_data = pyqtSignal(np.ndarray, np.ndarray)  # Emit a tuple of (data, timestamp)
    
def screenoff():
    ''' Darken the screen by starting the blank screensaver '''
    try:
        subprocess.call(['C:\Windows\System32\scrnsave.scr', '/start'])
    except Exception as ex:
        # construct traceback
        tbstring = traceback.format_exception(type(ex), ex, ex.__traceback__)
        tbstring.insert(0, '=== ' + datetime.now().toISOString() + ' ===')

        # print to screen and error log file
        print('\n'.join(tbstring))
        # self.files['err'].writelines(tbstring)

def find_procs_by_name(name) -> list[Process]:
    ls = []
    for p in psutil.process_iter(['name']):
        if p.info['name'] == name:
            ls.append(p)
    return ls