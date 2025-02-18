import queue
import statistics
import sys
import time
from threading import Lock, Thread

import numpy as np
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget

# Constants
NUM_TESTS = 1000  # Number of latency tests per method
DATA_INTERVAL = 0.01  # Simulated data delay
ARRAY_SIZE = (12, 5)  # NumPy array size

# Standardized Data Function
def simulate_data():
    """Simulates external data and returns a NumPy array."""
    time.sleep(DATA_INTERVAL)
    return np.random.rand(*ARRAY_SIZE)

# Base UI for Displaying Results
class PerformanceTestUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thread Communication Performance Test")
        self.setGeometry(100, 100, 600, 400)

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def log(self, message):
        """Displays log messages in the UI."""
        self.text_edit.append(message)
        print(message)

# Method 1: Queue-Based Communication
class QueueWorker:
    def __init__(self, data_queue):
        self.data_queue = data_queue
        self.thread = Thread(target=self.run)
        self.thread.start()

    def run(self):
        for _ in range(NUM_TESTS):
            data = simulate_data()
            self.data_queue.put((time.time(), data.copy()))

class QueueTest(PerformanceTestUI):
    def __init__(self):
        super().__init__()
        self.data_queue = queue.Queue()
        self.worker = QueueWorker(self.data_queue)

        self.latencies = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_queue)
        self.timer.start(1)

    def process_queue(self):
        while not self.data_queue.empty():
            timestamp, _ = self.data_queue.get()
            latency = time.time() - timestamp
            self.latencies.append(latency)
            self.log(f"Queue Latency: {latency:.6f} sec")

        if len(self.latencies) >= NUM_TESTS:
            self.timer.stop()
            self.display_results("Queue", self.latencies)

    def display_results(self, method, latencies):
        avg_latency = statistics.mean(latencies[NUM_TESTS // 2:])
        self.log(f"\n{method} Average Latency: {avg_latency:.6f} sec\n")

# Method 2: Signal-Slot Communication
class SignalWorker(QObject):
    data_ready = pyqtSignal(object)

    def run(self):
        for _ in range(NUM_TESTS):
            data = simulate_data()
            self.data_ready.emit((time.time(), data.copy()))

class SignalTest(PerformanceTestUI):
    def __init__(self):
        super().__init__()
        self.thread = QThread()
        self.worker = SignalWorker()
        self.worker.moveToThread(self.thread)
        self.worker.data_ready.connect(self.process_data)
        self.latencies = []

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def process_data(self, result):
        timestamp, _ = result
        latency = time.time() - timestamp
        self.latencies.append(latency)
        self.log(f"Signal Latency: {latency:.6f} sec")

        if len(self.latencies) >= NUM_TESTS:
            self.thread.quit()
            self.display_results("Signal-Slot", self.latencies)

    def display_results(self, method, latencies):
        avg_latency = statistics.mean(latencies)
        self.log(f"\n{method} Average Latency: {avg_latency:.6f} sec\n")

# Method 3: Shared Variable with Lock
class LockWorker:
    def __init__(self):
        self.lock = Lock()
        self.timestamp = None
        self.data = None
        self.thread = Thread(target=self.run)
        self.thread.start()

    def run(self):
        for _ in range(NUM_TESTS):
            with self.lock:
                self.timestamp = time.time()
                self.data = simulate_data().copy()

    def get_data(self):
        with self.lock:
            return self.timestamp, self.data.copy() if self.data is not None else None

class LockTest(PerformanceTestUI):
    def __init__(self):
        super().__init__()
        self.worker = LockWorker()
        self.latencies = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_data)
        self.timer.start(1)

    def process_data(self):
        timestamp, _ = self.worker.get_data()
        if timestamp:
            latency = time.time() - timestamp
            self.latencies.append(latency)
            self.log(f"Lock Latency: {latency:.6f} sec")

        if len(self.latencies) >= NUM_TESTS:
            self.timer.stop()
            self.display_results("Lock", self.latencies)

    def display_results(self, method, latencies):
        avg_latency = statistics.mean(latencies)
        self.log(f"\n{method} Average Latency: {avg_latency:.6f} sec\n")

# Method 4: QTimer Polling
class TimerWorker:
    def __init__(self):
        self.timestamp = None
        self.data = None
        self.thread = Thread(target=self.run)
        self.thread.start()

    def run(self):
        for _ in range(NUM_TESTS):
            self.timestamp = time.time()
            self.data = simulate_data().copy()

    def get_data(self):
        return self.timestamp, self.data.copy() if self.data is not None else None

class TimerTest(PerformanceTestUI):
    def __init__(self):
        super().__init__()
        self.worker = TimerWorker()
        self.latencies = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_data)
        self.timer.start(10)  # Poll every 10ms

    def process_data(self):
        timestamp, _ = self.worker.get_data()
        if timestamp:
            latency = time.time() - timestamp
            self.latencies.append(latency)
            self.log(f"QTimer Latency: {latency:.6f} sec")

        if len(self.latencies) >= NUM_TESTS:
            self.timer.stop()
            self.display_results("QTimer", self.latencies)

    def display_results(self, method, latencies):
        avg_latency = statistics.mean(latencies)
        self.log(f"\n{method} Average Latency: {avg_latency:.6f} sec\n")

# Main function to run all tests
def main():
    app = QApplication(sys.argv)

    print("Running Queue Test...")
    queue_window = QueueTest()
    queue_window.show()
    app.exec_()

    print("Running Signal-Slot Test...")
    signal_window = SignalTest()
    signal_window.show()
    app.exec_()

    print("Running Lock Test...")
    lock_window = LockTest()
    lock_window.show()
    app.exec_()

    print("Running QTimer Test...")
    timer_window = TimerTest()
    timer_window.show()
    app.exec_()

if __name__ == "__main__":
    main()