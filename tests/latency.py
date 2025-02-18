import queue
import statistics
import sys
import time
from threading import Lock, Thread

from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget

# Constants
NUM_TESTS = 100  # Number of latency tests per method
DATA_INTERVAL = 0.01  # Simulated data arrival every 10ms

# Standardized Data Function
def simulate_data():
    """Simulates external data arrival with a fixed delay."""
    time.sleep(DATA_INTERVAL)
    return time.time()

# Base Class for UI & Result Display
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
        self.running = True
        self.thread = Thread(target=self.run)
        self.thread.start()

    def run(self):
        for _ in range(NUM_TESTS):
            timestamp = simulate_data()
            self.data_queue.put(timestamp)
        self.running = False

class QueueTest(PerformanceTestUI):
    def __init__(self):
        super().__init__()
        self.data_queue = queue.Queue()
        self.worker = QueueWorker(self.data_queue)

        self.latencies = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_queue)
        self.timer.start(1)  # Check every 1ms

    def process_queue(self):
        while not self.data_queue.empty():
            timestamp = self.data_queue.get()
            latency = time.time() - timestamp
            self.latencies.append(latency)
            self.log(f"Queue Latency: {latency:.6f} sec")

        if len(self.latencies) >= NUM_TESTS:
            self.timer.stop()
            self.display_results("Queue", self.latencies)

    def display_results(self, method, latencies):
        avg_latency = statistics.mean(latencies)
        self.log(f"\n{method} Average Latency: {avg_latency:.6f} sec\n")

# Method 2: Signal-Slot Communication
class SignalWorker(QObject):
    data_ready = pyqtSignal(float)

    def run(self):
        for _ in range(NUM_TESTS):
            timestamp = simulate_data()
            self.data_ready.emit(timestamp)

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

    def process_data(self, timestamp):
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
        self.running = True
        self.thread = Thread(target=self.run)
        self.thread.start()

    def run(self):
        for _ in range(NUM_TESTS):
            with self.lock:
                self.timestamp = simulate_data()

    def get_data(self):
        with self.lock:
            return self.timestamp

class LockTest(PerformanceTestUI):
    def __init__(self):
        super().__init__()
        self.worker = LockWorker()
        self.latencies = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_data)
        self.timer.start(1)

    def process_data(self):
        timestamp = self.worker.get_data()
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
        self.running = True
        self.thread = Thread(target=self.run)
        self.thread.start()

    def run(self):
        for _ in range(NUM_TESTS):
            self.timestamp = simulate_data()

    def get_data(self):
        return self.timestamp

class TimerTest(PerformanceTestUI):
    def __init__(self):
        super().__init__()
        self.worker = TimerWorker()
        self.latencies = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_data)
        self.timer.start(10)  # Poll every 10ms

    def process_data(self):
        timestamp = self.worker.get_data()
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