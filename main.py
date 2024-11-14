import sys
import time
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QThreadPool, QRunnable, pyqtSignal, QObject
from pylsl import StreamInlet, resolve_stream
from multiprocessing import Process, Queue

class DataSignal(QObject):
    update_data = pyqtSignal(np.ndarray, np.ndarray)  # Emit a tuple of (data, timestamp)
    update_fft_data = pyqtSignal(np.ndarray)

class FFTProcess(Process):
    def __init__(self, input_queue: Queue, output_queue: Queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        """Calculate FFT in a separate process."""
        while True:
            if not self.input_queue.empty():
                buffer_data = self.input_queue.get()
                if buffer_data is None:  # Check for termination signal
                    break
                # Perform FFT
                fft_result = np.fft.fft(buffer_data, axis=0)
                fft_magnitude = np.abs(fft_result)  # Calculate magnitude
                self.output_queue.put(fft_magnitude)  # Send results back to the main process

class EEGApp(QMainWindow):
    MAX_BUFFER_SIZE = 1000  # Define the maximum buffer size

    def __init__(self):
        super().__init__()

        self.label = QLabel('EEG Data', self)
        self.label.setGeometry(100, 100, 400, 200)

        self.fft_label = QLabel('FFT Data', self)
        self.fft_label.setGeometry(100, 300, 400, 200)

        # Set up the main window
        self.setWindowTitle('Real-time EEG Viewer')
        self.setGeometry(100, 100, 1200, 1200)

        # Create a central widget and set the layout for the plots
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create 4 time series plots using pyqtgraph
        self.plots = []
        self.curves = []
        self.plot_data = [[] for _ in range(4)]  # To hold data for each channel
        self.time_data = [[] for _ in range(4)]  # Time data for each channel

        for i in range(4):
            plot = pg.PlotWidget(title=f"EEG Channel {i + 1}")  # Create a new plot for each channel
            self.plots.append(plot)
            self.layout.addWidget(plot)  # Add plot to the vertical layout
            curve = plot.plot([], pen=pg.mkPen('w'))  # Initialize the plot line with an empty array
            self.curves.append(curve)

        # FFT Magnitude Plot
        self.fft_plots = []  # Stores FFT plot widgets
        self.fft_curves = []  # Stores FFT curve objects
        for i in range(4):
            fft_plot = pg.PlotWidget(title=f"FFT Magnitude Channel {i + 1}")
            self.fft_plots.append(fft_plot)
            self.layout.addWidget(fft_plot)  # Add FFT plot to the vertical layout
            fft_curve = fft_plot.plot([], pen=pg.mkPen('r'))  # Initialize the FFT plot line with an empty array
            self.fft_curves.append(fft_curve)

        # Create a data signal for GUI updates
        self.data_signal = DataSignal()
        self.data_signal.update_data.connect(self.update_gui)
        self.data_signal.update_fft_data.connect(self.update_fft_gui)

        # Create QQueues for inter-process communication
        self.input_queue = Queue()
        self.output_queue = Queue()

        # Start the FFT process
        self.fft_process = FFTProcess(self.input_queue, self.output_queue)
        self.fft_process.start()

        # Create a QThreadPool
        self.threadpool = QThreadPool()

        # Start the background task to ingest EEG data
        self.start_ingestion_task()

        self.show()

        # Initialize the buffer for FFT
        self.buffer = np.zeros((self.MAX_BUFFER_SIZE + 200, 4))  # Assuming 4 channels
        self.buffer_count = 0  # Current number of accumulated samples

    def start_ingestion_task(self):
        self.ingest_task = IngestEEGDataTask(self.data_signal)
        self.threadpool.start(self.ingest_task)

    def update_gui(self, data: np.ndarray, times: np.ndarray):
        for i in range(4):
            
            self.plot_data[i].extend(data[:, i].tolist())
            self.time_data[i].extend(times.tolist())
            
            # Check if we exceed the maximum buffer size
            if len(self.plot_data[i]) > self.MAX_BUFFER_SIZE:
                # Remove the oldest samples
                self.plot_data[i] = self.plot_data[i][-self.MAX_BUFFER_SIZE:]  # Keep only the most recent samples
                self.time_data[i] = self.time_data[i][-self.MAX_BUFFER_SIZE:]  # Corresponding timestamps

            # Update the plot curve using time data for x-axis
            self.curves[i].setData(self.time_data[i], self.plot_data[i])  # Update the plot curve

            # Accumulate data for FFT
            if self.buffer_count < self.MAX_BUFFER_SIZE:
                self.buffer[self.buffer_count:self.buffer_count + data.shape[0], i] = data[:, i]  # Store the data
                self.buffer_count += data.shape[0]
            
            # If buffer is full, send to FFT process
            if self.buffer_count >= self.MAX_BUFFER_SIZE:
                self.input_queue.put(self.buffer[:self.MAX_BUFFER_SIZE, :])  # Send data to FFT process
                self.buffer_count = 0  # Reset buffer count

                # Check for results from the FFT process
                if not self.output_queue.empty():
                    fft_result = self.output_queue.get()
                    self.update_fft_gui(fft_result)

    def update_fft_gui(self, fft_result):
        """Update the GUI with FFT magnitude results."""
        for i in range(fft_result.shape[1]):  # Assuming fft_result shape is (N, 4)
            self.fft_curves[i].setData(fft_result[:, i])  # Update the FFT plot curve

    def closeEvent(self, event):
        """Handle the window close event."""
        self.input_queue.put(None)  # Send termination signal to the FFT process
        self.fft_process.join()  # Wait for the process to finish
        self.ingest_task.stop()
        self.threadpool.waitForDone()
        event.accept()  # Accept the close event

def main():
    app = QApplication(sys.argv)
    eeg_app = EEGApp()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()