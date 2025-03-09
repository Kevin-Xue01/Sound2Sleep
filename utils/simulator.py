import random
import sys
import threading
import time
from typing import List

import numpy as np
import pyqtgraph as pg
from audio import Audio
from constants import (
    CHUNK_SIZE,
    NUM_CHANNELS,
    SAMPLING_RATE,
    EEGSimulatorBand,
    EEGSimulatorSignalParam,
    MuseDataType,
)
from pylsl import StreamInfo, StreamOutlet
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

INITIAL_CONFIG = {
    EEGSimulatorBand.DELTA: {EEGSimulatorSignalParam.PERCENT: 20, EEGSimulatorSignalParam.CENTER_FREQUENCY: 2, EEGSimulatorSignalParam.BANDWIDTH: 1, EEGSimulatorSignalParam.RANGE: (1, 4), EEGSimulatorSignalParam.COLOR: (255, 0, 0)},
    EEGSimulatorBand.THETA: {EEGSimulatorSignalParam.PERCENT: 20, EEGSimulatorSignalParam.CENTER_FREQUENCY: 6, EEGSimulatorSignalParam.BANDWIDTH: 1, EEGSimulatorSignalParam.RANGE: (4, 8), EEGSimulatorSignalParam.COLOR: (0, 255, 0)},
    EEGSimulatorBand.ALPHA: {EEGSimulatorSignalParam.PERCENT: 20, EEGSimulatorSignalParam.CENTER_FREQUENCY: 10, EEGSimulatorSignalParam.BANDWIDTH: 1, EEGSimulatorSignalParam.RANGE: (8, 12), EEGSimulatorSignalParam.COLOR: (0, 0, 255)},
    EEGSimulatorBand.BETA: {EEGSimulatorSignalParam.PERCENT: 20, EEGSimulatorSignalParam.CENTER_FREQUENCY: 20, EEGSimulatorSignalParam.BANDWIDTH: 1, EEGSimulatorSignalParam.RANGE: (12, 30), EEGSimulatorSignalParam.COLOR: (255, 165, 0)},
    EEGSimulatorBand.GAMMA: {EEGSimulatorSignalParam.PERCENT: 20, EEGSimulatorSignalParam.CENTER_FREQUENCY: 40, EEGSimulatorSignalParam.BANDWIDTH: 1, EEGSimulatorSignalParam.RANGE: (30, 80), EEGSimulatorSignalParam.COLOR: (128, 0, 128)},
}

class LSLSimulatorDataGenerator:
    def __init__(self, muse_data_type: MuseDataType, output_signal_max_freq=80):
        self.muse_data_type = muse_data_type
        self.output_signal_max_freq = output_signal_max_freq
        self.streaming = True
        self.phases = np.random.uniform(0, 2 * np.pi, (self.output_signal_max_freq, NUM_CHANNELS[muse_data_type]))
        self.config = INITIAL_CONFIG
        self.noise_factor = 0.01
        self.gain_value = 1.0
        self.info = StreamInfo(muse_data_type.value, muse_data_type.name, NUM_CHANNELS[muse_data_type])
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
        for band in EEGSimulatorBand:
            percent = self.config[band][EEGSimulatorSignalParam.PERCENT]
            center_frequency = self.config[band][EEGSimulatorSignalParam.CENTER_FREQUENCY]
            bandwidth = self.config[band][EEGSimulatorSignalParam.BANDWIDTH]

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
        return CHUNK_SIZE[self.muse_data_type] * size_multiplier

    def push_data(self):
        generated_signal = self.simulate_eeg_data()
        while self.streaming:
            chunk_size = CHUNK_SIZE[self.muse_data_type]
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
            time.sleep(chunk_size / SAMPLING_RATE[self.muse_data_type])

    def stop(self):
        self.streaming = False

class LSLSimulatorGUI(QMainWindow):
    def __init__(self, streams: List[LSLSimulatorDataGenerator]):
        super().__init__()
        self.pool = QThreadPool.globalInstance()
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
                EEGSimulatorSignalParam.PERCENT: {
                    'value': self.config[band][EEGSimulatorSignalParam.PERCENT],
                    'slider': QSlider(Qt.Orientation.Horizontal),
                    'label': QLabel(f"{EEGSimulatorSignalParam.PERCENT.value}: {self.config[band][EEGSimulatorSignalParam.PERCENT]}")
                },
                EEGSimulatorSignalParam.CENTER_FREQUENCY: {
                    'value': self.config[band][EEGSimulatorSignalParam.CENTER_FREQUENCY],
                    'slider': QSlider(Qt.Orientation.Horizontal),
                    'label': QLabel(f"{EEGSimulatorSignalParam.CENTER_FREQUENCY.value}: {self.config[band][EEGSimulatorSignalParam.CENTER_FREQUENCY]}")
                },
                EEGSimulatorSignalParam.BANDWIDTH: {
                    'value': self.config[band][EEGSimulatorSignalParam.BANDWIDTH],
                    'slider': QSlider(Qt.Orientation.Horizontal),
                    'label': QLabel(f"{EEGSimulatorSignalParam.BANDWIDTH.value}: {self.config[band][EEGSimulatorSignalParam.BANDWIDTH]}")
                }
            } for band in EEGSimulatorBand
        }

        # Main window layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)

        sliders_layout = QVBoxLayout()
        for band in EEGSimulatorBand:
            group_box = QGroupBox()
            group_box.setStyleSheet(f"background-color: rgba{INITIAL_CONFIG[band][EEGSimulatorSignalParam.COLOR] + (40,)};")
            group_layout = QVBoxLayout()

            min_center_freq, max_center_freq = self.config[band][EEGSimulatorSignalParam.RANGE]

            for curr_param in EEGSimulatorSignalParam:
                if curr_param != EEGSimulatorSignalParam.RANGE and curr_param != EEGSimulatorSignalParam.COLOR:
                    slider: QSlider = self.controls[band][curr_param]['slider']
                    slider.setValue(self.controls[band][curr_param]['value'])

                    if curr_param == EEGSimulatorSignalParam.PERCENT:
                        slider.setRange(0, 100)  # Percentage is from 0 to 100
                    elif curr_param == EEGSimulatorSignalParam.CENTER_FREQUENCY:
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
        self.audio = Audio(3.0, 1.0)
        self.pool.start(self.audio)

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

        for i, band in enumerate(EEGSimulatorBand):
            percent = self.controls[band][EEGSimulatorSignalParam.PERCENT]['value']
            center_frequency = self.controls[band][EEGSimulatorSignalParam.CENTER_FREQUENCY]['value']
            bandwidth = self.controls[band][EEGSimulatorSignalParam.BANDWIDTH]['value']

            y = percent * np.exp(-0.5 * ((x - center_frequency) / bandwidth) ** 2)

            pen = pg.mkPen(INITIAL_CONFIG[band][EEGSimulatorSignalParam.COLOR], width=2)
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
                EEGSimulatorSignalParam.PERCENT: self.controls[band][EEGSimulatorSignalParam.PERCENT]['value'],
                EEGSimulatorSignalParam.CENTER_FREQUENCY: self.controls[band][EEGSimulatorSignalParam.CENTER_FREQUENCY]['value'],
                EEGSimulatorSignalParam.BANDWIDTH: self.controls[band][EEGSimulatorSignalParam.BANDWIDTH]['value']
            }
            for band in EEGSimulatorBand
        }
        for curr_stream in self.streams:
            curr_stream.update_config(new_config, self.gain_value, self.noise_factor)

    def normalize_percentages(self):
        total_percent = sum(self.controls[band][EEGSimulatorSignalParam.PERCENT]['value'] for band in EEGSimulatorBand)
        if total_percent == 0:
            for i, band in enumerate(EEGSimulatorBand):
                self.controls[band][EEGSimulatorSignalParam.PERCENT]['value'] = 20
                self.controls[band][EEGSimulatorSignalParam.PERCENT]['slider'].setValue(20)  # Update slider position
                self.controls[band][EEGSimulatorSignalParam.PERCENT]['label'].setText(f"{EEGSimulatorSignalParam.PERCENT.value}: 20")
 
        else:
            normalized_values = [int(100 * self.controls[band][EEGSimulatorSignalParam.PERCENT]['value'] / total_percent) for band in EEGSimulatorBand]
            normalized_values[0] += 100 - sum(normalized_values)
            for i, band in enumerate(EEGSimulatorBand):
                self.controls[band][EEGSimulatorSignalParam.PERCENT]['value'] = normalized_values[i]
                self.controls[band][EEGSimulatorSignalParam.PERCENT]['slider'].setValue(int(normalized_values[i]))  # Update slider position
                self.controls[band][EEGSimulatorSignalParam.PERCENT]['label'].setText(f"{EEGSimulatorSignalParam.PERCENT.value}: {int(normalized_values[i])}")
        
        self.update_plot()
        self.update_config()

    def closeEvent(self, event):
        for simulator in self.streams:
            simulator.stop()
        event.accept()

if __name__ == "__main__":
    eeg_simulator = LSLSimulatorDataGenerator(muse_data_type=MuseDataType.EEG)
    acc_simulator = LSLSimulatorDataGenerator(muse_data_type=MuseDataType.ACC)
    ppg_simulator = LSLSimulatorDataGenerator(muse_data_type=MuseDataType.PPG)

    streams = [eeg_simulator, acc_simulator, ppg_simulator]
    for stream in streams:
        thread = threading.Thread(target=stream.push_data)
        thread.daemon = True
        thread.start()
        # break

    app = QApplication(sys.argv)
    window = LSLSimulatorGUI(streams)
    window.show()
    sys.exit(app.exec_())