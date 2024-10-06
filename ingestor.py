from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
import sys
from pylsl import StreamInlet, resolve_byprop
from datetime import datetime
import numpy as np


class Ingestor(QtWidgets.QMainWindow):  # Inherit from QMainWindow or QWidget
    datastreams = ['EEG', 'Accelerometer', 'PPG']

    no_data_count = 0
    reset_attempt_count = 0

    def __init__(self):
        super().__init__()  # Initialize the parent class
        self.datastreams = ['EEG', 'Accelerometer', 'PPG']
        self.lsl = dict()
        self.inlet = dict()
        self.lsl_timer = None

        # Start the LSL stream when initializing
        self.start_streaming()

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
            if d in self.inlet:
                chunk, times = self.inlet[d].pull_chunk()
                chunk = np.array(chunk)

                if len(times) > 0:
                    self.no_data_count = 0

                    print(f"Type: {d}, shape: {chunk.shape}")

    def start_streaming(self):
        if self.lsl_reload():
            self.inlet = dict()
            for k in self.datastreams:
                self.inlet[k] = StreamInlet(self.lsl[k])

            # Initialize the data stream timer
            self.lsl_timer = QTimer()
            self.lsl_timer.timeout.connect(self.lsl_timer_callback)
            self.lsl_timer.start(100)  # Timer to call the callback every 100 ms

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)  # Create the Qt application
    ingestor = Ingestor()  # Create an instance of the Ingestor
    ingestor.show()  # Show the main window
    sys.exit(app.exec_())  # Start the Qt event loop