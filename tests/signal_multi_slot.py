import statistics
import sys
import time

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget

# Constants
NUM_TESTS = 10  # Reduce tests for clarity
DATA_INTERVAL = 0.01  # Simulated delay
ARRAY_SIZE = 5  # NumPy array size

# Function to Simulate Data
def simulate_data():
    time.sleep(DATA_INTERVAL)
    return np.random.rand(ARRAY_SIZE)

# Worker Class (Generates Data in a Separate Thread)
class MultiSignalWorker(QObject):
    data_ready = pyqtSignal(object)  # One signal, multiple listeners

    def run(self):
        """Emits the signal multiple times with NumPy data"""
        for _ in range(NUM_TESTS):
            data = simulate_data()
            self.data_ready.emit((time.time(), data.copy()))

# UI Class to Display Logs
class SignalMultiHandlerTest(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal-Slot with Multiple Handlers")
        self.setGeometry(100, 100, 600, 400)

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Threaded worker
        self.thread = QThread()
        self.worker = MultiSignalWorker()
        self.worker.moveToThread(self.thread)

        # Multiple handlers for a single signal
        self.worker.data_ready.connect(self.handler_log)       # Logs the event
        self.worker.data_ready.connect(self.handler_avg)       # Calculates average
        self.worker.data_ready.connect(self.handler_threshold) # Checks if sum exceeds threshold

        self.latencies = []
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def handler_log(self, result):
        timestamp, data = result
        latency = time.time() - timestamp
        self.latencies.append(latency)
        self.log(f"Received Data: {data}, Latency: {latency:.6f} sec")

    def handler_avg(self, result):
        _, data = result
        avg_value = np.mean(data)
        self.log(f"Average Value: {avg_value:.6f}")

    def handler_threshold(self, result):
        _, data = result
        if np.sum(data) > 2.5:  # Arbitrary threshold
            self.log("Threshold exceeded! Sum = {:.2f}".format(np.sum(data)))

    def log(self, message):
        """Displays log messages in the UI"""
        self.text_edit.append(message)
        print(message)

# Run Application
def main():
    app = QApplication(sys.argv)
    window = SignalMultiHandlerTest()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()