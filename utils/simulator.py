import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet
import threading
import random
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QHBoxLayout, QGroupBox, QPushButton, QDoubleSpinBox
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from typing import List
from enum import Enum
SRATE = 256  # [Hz]

class Band(Enum):
    DELTA = 'Delta'
    THETA = 'Theta'
    ALPHA = 'Alpha'
    BETA = 'Beta'
    GAMMA = 'Gamma'

class SignalParam(Enum):
    PERCENT = 'Percent'
    CENTER_FREQUENCY = 'Center Frequency'
    BANDWIDTH = 'Bandwidth'
    RANGE = 'Range'
    COLOR = 'Color'

INITIAL_CONFIG = {
    Band.DELTA: {SignalParam.PERCENT: 20, SignalParam.CENTER_FREQUENCY: 2, SignalParam.BANDWIDTH: 1, SignalParam.RANGE: (1, 4), SignalParam.COLOR: (255, 0, 0)},
    Band.THETA: {SignalParam.PERCENT: 20, SignalParam.CENTER_FREQUENCY: 6, SignalParam.BANDWIDTH: 1, SignalParam.RANGE: (4, 8), SignalParam.COLOR: (0, 255, 0)},
    Band.ALPHA: {SignalParam.PERCENT: 20, SignalParam.CENTER_FREQUENCY: 10, SignalParam.BANDWIDTH: 1, SignalParam.RANGE: (8, 12), SignalParam.COLOR: (0, 0, 255)},
    Band.BETA: {SignalParam.PERCENT: 20, SignalParam.CENTER_FREQUENCY: 20, SignalParam.BANDWIDTH: 1, SignalParam.RANGE: (12, 30), SignalParam.COLOR: (255, 165, 0)},
    Band.GAMMA: {SignalParam.PERCENT: 20, SignalParam.CENTER_FREQUENCY: 40, SignalParam.BANDWIDTH: 1, SignalParam.RANGE: (30, 80), SignalParam.COLOR: (128, 0, 128)},
}

class LSLSimulatorDataGenerator:
    def __init__(self, name, stream_type, num_channels, base_chunk_size, ics=True, output_signal_max_freq=80):

        self.name = name
        self.stream_type = stream_type
        self.num_channels = num_channels
        self.base_chunk_size = base_chunk_size
        self.ics = ics # irregular chunk size flag
        self.output_signal_max_freq = output_signal_max_freq
        self.streaming = True
        self.phases = np.random.uniform(0, 2 * np.pi, (self.output_signal_max_freq, self.num_channels))
        self.config = INITIAL_CONFIG
        self.noise_factor = 0.01
        self.gain_value = 1.0
        self.info = StreamInfo(name, stream_type, num_channels)
        self.outlet = StreamOutlet(self.info)

    def update_config(self, config, gain_value, noise_factor):
        self.config = config
        self.gain_value = gain_value
        self.noise_factor = noise_factor 

    def simulate_eeg_data(self):
        # Generate frequencies up to half the sampling rate
        frequencies = np.fft.fftfreq(2 * self.output_signal_max_freq, 1 / (2 * self.output_signal_max_freq))
        summed_signal = np.zeros_like(frequencies[:self.output_signal_max_freq])

        # Sum up the signals for each EEG band (Delta, Theta, Alpha, etc.)
        for band in Band:
            percent = self.config[band][SignalParam.PERCENT]
            center_frequency = self.config[band][SignalParam.CENTER_FREQUENCY]
            bandwidth = self.config[band][SignalParam.BANDWIDTH]

            # Gaussian distribution around center frequency
            summed_signal += (percent / 100) * np.exp(-0.5 * ((frequencies[:self.output_signal_max_freq] - center_frequency) / bandwidth) ** 2)

        summed_signal = summed_signal * self.gain_value
        summed_signal += np.random.uniform(-self.noise_factor*self.gain_value, self.noise_factor*self.gain_value, size=summed_signal.shape)
        # Update phases with small random perturbations for a continuous process
        phase_perturbation = np.random.uniform(-1.2, 1.2, self.phases.shape)  # Small Gaussian noise for gradual phase change
        self.phases += phase_perturbation  # Perform the random walk in phase

        # Generate the complex Fourier coefficients with the updated phases
        fft_complex_random_phase = summed_signal[:, np.newaxis] * np.exp(1j * self.phases)

        # Mirror the Fourier components to create a real-valued signal after the inverse FFT
        fft_complex_full_random_phase = np.concatenate([fft_complex_random_phase, np.conj(fft_complex_random_phase[-2:0:-1])])
        
        # Apply the inverse FFT to get the time-domain signal
        time_signal_random_phase = np.fft.ifft(fft_complex_full_random_phase, axis=0).real

        return time_signal_random_phase

    def determine_chunk_size(self):
        probabilities = [0.85, 0.1, 0.05]
        size_multiplier = random.choices([1, 2, 3], probabilities)[0]
        return self.base_chunk_size * size_multiplier

    # def push_data(self):
    #     generated_signal = self.simulate_eeg_data()
    #     while self.streaming:
    #         chunk_size = self.determine_chunk_size() if self.ics else self.base_chunk_size

    #         # If there isn't enough data left for the next chunk, regenerate signal
    #         if generated_signal.shape[0] < chunk_size:
    #             generated_signal = self.simulate_eeg_data()

    #         # Slice the generated signal into chunks
    #         chunk = generated_signal[:chunk_size, ...]

    #         # Push the chunk to the LSL stream
    #         self.outlet.push_chunk(chunk.tolist())
    #         # Remove the chunk from the signal so the next chunk can be pushed
    #         generated_signal = generated_signal[chunk_size:, ...]  
    #         time.sleep(chunk_size / SRATE)
    def push_data(self):
        generated_signal = self.simulate_eeg_data()
        while self.streaming:
            chunk_size = self.determine_chunk_size() if self.ics else self.base_chunk_size
            chunk_size = 2

            # If there isn't enough data left for the next chunk, regenerate signal
            if generated_signal.shape[0] < chunk_size:
                generated_signal = self.simulate_eeg_data()

            # Slice the generated signal into chunks
            chunk = generated_signal[0, ...]

            # Push the chunk to the LSL stream
            self.outlet.push_sample()
            # Remove the chunk from the signal so the next chunk can be pushed
            generated_signal = generated_signal[chunk_size:, ...]  
            time.sleep(chunk_size / SRATE)

    def stop(self):
        self.streaming = False

class LSLSimulatorGUI(QMainWindow):
    def __init__(self, streams: List[LSLSimulatorDataGenerator]):
        super().__init__()
        self.setWindowTitle("LSL Simulator GUI")
        # Get the screen size
        screen = QApplication.primaryScreen().availableGeometry()
        screen_width = screen.width()
        screen_height = screen.height() 

        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)

        self.setFixedSize(
            window_width,
            window_height
        )

        self.streams = streams
        self.config = INITIAL_CONFIG
        self.gain_value = 1.0  # Default gain value is 1.0
        self.noise_factor = 0.01

        # Controls for bands
        self.controls = {
            band: {
                SignalParam.PERCENT: {
                    'value': self.config[band][SignalParam.PERCENT],
                    'slider': QSlider(Qt.Orientation.Horizontal),
                    'label': QLabel(f"{SignalParam.PERCENT.value}: {self.config[band][SignalParam.PERCENT]}")
                },
                SignalParam.CENTER_FREQUENCY: {
                    'value': self.config[band][SignalParam.CENTER_FREQUENCY],
                    'slider': QSlider(Qt.Orientation.Horizontal),
                    'label': QLabel(f"{SignalParam.CENTER_FREQUENCY.value}: {self.config[band][SignalParam.CENTER_FREQUENCY]}")
                },
                SignalParam.BANDWIDTH: {
                    'value': self.config[band][SignalParam.BANDWIDTH],
                    'slider': QSlider(Qt.Orientation.Horizontal),
                    'label': QLabel(f"{SignalParam.BANDWIDTH.value}: {self.config[band][SignalParam.BANDWIDTH]}")
                }
            } for band in Band
        }

        # Main window layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)

        sliders_layout = QVBoxLayout()
        for band in Band:
            group_box = QGroupBox()
            group_box.setStyleSheet(f"background-color: rgba{INITIAL_CONFIG[band][SignalParam.COLOR] + (40,)};")
            group_layout = QVBoxLayout()

            min_center_freq, max_center_freq = self.config[band][SignalParam.RANGE]

            for curr_param in SignalParam:
                if curr_param != SignalParam.RANGE and curr_param != SignalParam.COLOR:
                    slider: QSlider = self.controls[band][curr_param]['slider']
                    slider.setValue(self.controls[band][curr_param]['value'])

                    if curr_param == SignalParam.PERCENT:
                        slider.setRange(0, 100)  # Percentage is from 0 to 100
                    elif curr_param == SignalParam.CENTER_FREQUENCY:
                        slider.setRange(min_center_freq, max_center_freq)  
                    else:
                        slider.setRange(1, 10)  
                    slider.valueChanged.connect(lambda value, b=band, s=curr_param: self.update_values(b, s, value))

                    group_layout.addWidget(self.controls[band][curr_param]['label'])
                    group_layout.addWidget(slider)

            group_box.setLayout(group_layout)
            sliders_layout.addWidget(group_box)

        right_layout = QVBoxLayout()

        self.gain_spinbox = QDoubleSpinBox()
        self.gain_spinbox.setRange(0.1, 5.0)  # Set range for gain (0.1 to 5.0)
        self.gain_spinbox.setSingleStep(0.1)  # Step size for gain adjustment
        self.gain_spinbox.setValue(1.0)  # Default gain is 1.0
        self.gain_spinbox.setDecimals(2)  # Allow two decimal places
        self.gain_spinbox.valueChanged.connect(self.update_gain)  # Connect value change event to update_gain

        self.noise_factor_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_factor_slider.setRange(1, 20)  # Set range from 1 to 100
        self.noise_factor_slider.setValue(int(self.noise_factor * 100))  # Convert 0.1 to 10 for slider
        self.noise_factor_slider.valueChanged.connect(self.update_noise_factor)

        # Create a horizontal layout for the gain control label and spinbox
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain Control: "))  # Add label to the layout
        gain_layout.addWidget(self.gain_spinbox)  # Add spinbox to the layout

        noise_factor_layout = QHBoxLayout()
        noise_factor_layout.addWidget(QLabel("Noise Control: "))  # Add label to the layout
        noise_factor_layout.addWidget(self.noise_factor_slider)  # Add spinbox to the layout

        # Add the gain layout to the right layout
        right_layout.addLayout(gain_layout)
        right_layout.addLayout(noise_factor_layout)

        self.normalize_button = QPushButton("Normalize Percentages")
        self.normalize_button.clicked.connect(self.normalize_percentages)
        right_layout.addWidget(self.normalize_button)

        # Primary plot for frequency bands
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.show()
        self.plot_widget.setYRange(0, 100)  # Primary Y-axis range for frequency bands
        self.plot_widget.setWindowTitle('Magnitude Plot')
        self.p1 = self.plot_widget.plotItem
        self.p1.setLabels(left='Percent (%)')  # Set label for the left Y-axis (percent)

        self.sum_viewbox = pg.ViewBox()
        self.sum_viewbox.setYRange(0, 500)
        self.p1.showAxis('right')  # Initial range for the summed signal
        self.p1.scene().addItem(self.sum_viewbox)
        self.p1.getAxis('right').linkToView(self.sum_viewbox)
        self.sum_viewbox.setXLink(self.p1)
        self.p1.getAxis('right').setLabel('Amplitude (μV)')
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.sum_viewbox.setMouseEnabled(x=False, y=False)
        self.p1.addLegend(size=(150, 150), labelTextSize='16pt')  # Adjust size as needed

        self.update_views()
        self.p1.vb.sigResized.connect(self.update_views)

        # Set up the legend with increased font size
        right_layout.addWidget(self.plot_widget)

        main_layout.addLayout(sliders_layout)
        main_layout.addLayout(right_layout)
        self.update_plot()
        self.update_config()

    def update_views(self):
        self.sum_viewbox.setGeometry(self.p1.vb.sceneBoundingRect())
        self.sum_viewbox.linkedViewChanged(self.p1.vb, self.sum_viewbox.XAxis)

    def update_gain(self, value):
        self.gain_value = value
        self.normalize_percentages()  
    
    def update_noise_factor(self, value):
        self.noise_factor = value / 100.0  # Convert slider value back to float (0.01 - 1.0)
        self.normalize_percentages()  

    def update_plot(self):
        self.p1.clear()
        self.sum_viewbox.clear()

        x = np.linspace(0, 120, 1000)
        summed_signal = np.zeros_like(x)

        for i, band in enumerate(Band):
            percent = self.controls[band][SignalParam.PERCENT]['value']
            center_frequency = self.controls[band][SignalParam.CENTER_FREQUENCY]['value']
            bandwidth = self.controls[band][SignalParam.BANDWIDTH]['value']

            y = percent * np.exp(-0.5 * ((x - center_frequency) / bandwidth) ** 2)

            pen = pg.mkPen(INITIAL_CONFIG[band][SignalParam.COLOR], width=2)
            self.plot_widget.plot(x, y, pen=pen, name=band.value)

            summed_signal += y

        summed_signal_microvolts = summed_signal * self.gain_value

        summed_pen = pg.mkPen((255, 255, 255), width=1, style=Qt.DashLine)
        self.sum_viewbox.addItem(pg.PlotCurveItem(x, summed_signal_microvolts, pen=summed_pen))


    def update_values(self, band, param, value):
        self.controls[band][param]['value'] = value
        self.controls[band][param]['label'].setText(f"{param.value}: {value}")
        self.update_plot()
        self.update_config()

    def update_config(self):
        new_config = {
            band: {
                SignalParam.PERCENT: self.controls[band][SignalParam.PERCENT]['value'],
                SignalParam.CENTER_FREQUENCY: self.controls[band][SignalParam.CENTER_FREQUENCY]['value'],
                SignalParam.BANDWIDTH: self.controls[band][SignalParam.BANDWIDTH]['value']
            }
            for band in Band
        }
        for curr_stream in self.streams:
            curr_stream.update_config(new_config, self.gain_value, self.noise_factor)

    def normalize_percentages(self):
        total_percent = sum(self.controls[band][SignalParam.PERCENT]['value'] for band in Band)
        if total_percent == 0:
            for i, band in enumerate(Band):
                self.controls[band][SignalParam.PERCENT]['value'] = 20
                self.controls[band][SignalParam.PERCENT]['slider'].setValue(20)  # Update slider position
                self.controls[band][SignalParam.PERCENT]['label'].setText(f"{SignalParam.PERCENT.value}: 20")
 
        else:
            normalized_values = [int(100 * self.controls[band][SignalParam.PERCENT]['value'] / total_percent) for band in Band]
            normalized_values[0] += 100 - sum(normalized_values)
            for i, band in enumerate(Band):
                self.controls[band][SignalParam.PERCENT]['value'] = normalized_values[i]
                self.controls[band][SignalParam.PERCENT]['slider'].setValue(int(normalized_values[i]))  # Update slider position
                self.controls[band][SignalParam.PERCENT]['label'].setText(f"{SignalParam.PERCENT.value}: {int(normalized_values[i])}")
        
        self.update_plot()
        self.update_config()

    def closeEvent(self, event):
        for simulator in self.streams:
            simulator.stop()
        event.accept()

if __name__ == "__main__":
    eeg_simulator = LSLSimulatorDataGenerator(name="EEG", stream_type="EEG", num_channels=4, base_chunk_size=12)
    acc_simulator = LSLSimulatorDataGenerator(name="Accelerometer", stream_type="ACC", num_channels=3, base_chunk_size=6)
    ppg_simulator = LSLSimulatorDataGenerator(name="PPG", stream_type="PPG", num_channels=3, base_chunk_size=6)

    streams = [eeg_simulator, acc_simulator, ppg_simulator]
    for stream in streams:
        thread = threading.Thread(target=stream.push_data)
        thread.daemon = True
        thread.start()
        break

    app = QApplication(sys.argv)
    window = LSLSimulatorGUI(streams)
    window.show()
    sys.exit(app.exec_())