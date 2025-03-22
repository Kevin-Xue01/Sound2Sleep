import os
import random
import sys
import threading
import time
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pyqtgraph as pg
from pylsl import StreamInfo, StreamOutlet
from PyQt5.QtCore import QObject, Qt, QThread, QThreadPool, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from utils import CHUNK_SIZE, NUM_CHANNELS, SAMPLING_RATE, FileReader, MuseDataType


class SimulatorOutputType(Enum):
    PURE = 0
    PLAYBACK = 1

class LSLSimulatorDataGenerator(QObject):
    def __init__(self, parent_app: 'LSLSimulatorGUI', muse_data_type: MuseDataType = MuseDataType.EEG):
        super().__init__()
        self.muse_data_type = muse_data_type
        self.running = False
        self.parent_app = parent_app
        
        self.info = StreamInfo(muse_data_type.value, muse_data_type.name, NUM_CHANNELS[muse_data_type], nominal_srate=SAMPLING_RATE[muse_data_type])
        self.outlet = StreamOutlet(self.info)
        
        # Store the current time index to maintain phase continuity
        self.current_time_index = 0
        # Initial phase offsets for each channel
        self.channel_phase_offsets = [i * (np.pi / 4) for i in range(NUM_CHANNELS[self.muse_data_type])]

    def simulate_pure_sine(self, num_samples=None):
        """Generate a pure sine wave with configurable parameters and phase continuity"""
        if num_samples is None:
            num_samples = SAMPLING_RATE[self.muse_data_type]
        
        # Generate time vector starting from current time index
        t = np.linspace(
            self.current_time_index / SAMPLING_RATE[self.muse_data_type],
            (self.current_time_index + num_samples) / SAMPLING_RATE[self.muse_data_type],
            num_samples, 
            endpoint=False
        )
        print(t)
        # Update the time index for next call
        self.current_time_index += num_samples
        
        # Generate sine waves for each channel with consistent phase
        signals = []
        for i in range(NUM_CHANNELS[self.muse_data_type]):
            # Use stored phase offset
            phase_offset = self.channel_phase_offsets[i]
            signal = self.parent_app.pure_amp * np.sin(2 * np.pi * self.parent_app.pure_freq * t + phase_offset)
            
            # Add noise
            if self.parent_app.pure_noise > 0:
                signal += self.parent_app.pure_amp * np.random.normal(0, self.parent_app.pure_noise / 2, len(t))
            
            signals.append(signal)
        
        # Convert to shape [samples, channels]
        output = np.array(signals).T
        return output

    def simulate_eeg_data(self):
        """Generate EEG data based on the current simulator output type"""
        if self.parent_app.simulator_output_type == SimulatorOutputType.PURE:
            return self.simulate_pure_sine()
        else:
            if self.parent_app.file_reader is not None:
                # Get the current frame from the app's file reader
                frame_data = self.parent_app.file_reader.read_frame(self.parent_app.current_frame)
                eeg_data, timestamp_data = frame_data
                # Increment the current frame for next time
                self.parent_app.current_frame += 1
                
                # Check if we've reached the end of the file
                if self.parent_app.current_frame >= self.parent_app.total_frames:
                    # Loop back to the beginning
                    self.parent_app.current_frame = 0
                    
                # Return the frame data if it exists
                if eeg_data is not None:
                    return eeg_data
                    
            # Return zeros if no data is available
            return np.zeros((CHUNK_SIZE[self.muse_data_type], NUM_CHANNELS[self.muse_data_type]))

    def reset_phase(self):
        """Reset the time index and phase continuity"""
        self.current_time_index = 0

    def run(self):
        """Thread method to continuously push data to the LSL stream"""
        self.reset_phase()  # Reset phase when starting a new stream
        generated_signal = self.simulate_eeg_data()
        
        while self.parent_app.streaming:
            # Check if output type changed and regenerate signal if needed
            current_output_type = self.parent_app.simulator_output_type
            # If there isn't enough data left for the next chunk, or output type changed, regenerate signal
            if generated_signal.shape[0] < CHUNK_SIZE[self.muse_data_type] or current_output_type == SimulatorOutputType.PLAYBACK:
                # For playback mode, always get the latest frame
                if current_output_type == SimulatorOutputType.PLAYBACK:
                    generated_signal = self.simulate_eeg_data()
                else:  # For pure sine mode, generate more data
                    needed_samples = CHUNK_SIZE[self.muse_data_type] - generated_signal.shape[0] + CHUNK_SIZE[self.muse_data_type]
                    new_signal = self.simulate_pure_sine(num_samples=needed_samples)
                    generated_signal = np.vstack((generated_signal, new_signal))

            # Slice the generated signal into chunks
            chunk = generated_signal[:CHUNK_SIZE[self.muse_data_type], ...]

            # Push the chunk to the LSL stream
            self.outlet.push_chunk(chunk.tolist())
            
            # Remove the chunk from the signal so the next chunk can be pushed
            generated_signal = generated_signal[CHUNK_SIZE[self.muse_data_type]:, ...]  
            
            # Sleep for the duration of one chunk
            time.sleep(CHUNK_SIZE[self.muse_data_type] / SAMPLING_RATE[self.muse_data_type])

class LSLSimulatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.simulator_output_type = SimulatorOutputType.PURE

        self.streaming = False
        self.stream = LSLSimulatorDataGenerator(self)
        self.stream_thread = QThread()
        self.stream.moveToThread(self.stream_thread)
        self.stream_thread.started.connect(self.stream.run)
        
        self.file_reader: Union[FileReader, None] = None
        self.current_frame = 0
        self.total_frames = 0
        
        self.pure_freq = 1.0
        self.pure_amp = 1.0
        self.pure_noise = 0.0

        # Setup playback timer for updating UI
        self.playback_timer = QTimer()
        self.playback_timer.setInterval(100)  # Update every 100ms
        self.playback_timer.timeout.connect(self.update_ui)
        self.playback_timer.start()
        self.__init_ui__()

    def __init_ui__(self):
        self.setWindowTitle("LSL Simulator GUI")
        screen = QApplication.primaryScreen().availableGeometry()
        screen_width = screen.width()
        screen_height = screen.height() 
        window_width = int(screen_width * 0.5)
        window_height = int(screen_height * 0.6)
        self.setFixedSize(window_width, window_height)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        common_controls_layout = QHBoxLayout()
        
        self.stream_button = QPushButton("Start Streaming")
        self.stream_button.clicked.connect(self.toggle_streaming)
        common_controls_layout.addWidget(self.stream_button)
        
        common_controls_layout.addStretch()
        
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Simulator Mode:"))
        self.output_type_combo = QComboBox()
        self.output_type_combo.addItems([output_type.name for output_type in SimulatorOutputType])
        self.output_type_combo.setCurrentIndex(self.simulator_output_type.value)
        self.output_type_combo.currentIndexChanged.connect(self.change_output_type)
        selector_layout.addWidget(self.output_type_combo)
        common_controls_layout.addLayout(selector_layout)
        
        self.main_layout.addLayout(common_controls_layout)
        
        self.stacked_widget = QStackedWidget()
        
        self.pure_view = self.create_pure_view()
        self.playback_view = self.create_playback_view()
        
        self.stacked_widget.addWidget(self.pure_view)
        self.stacked_widget.addWidget(self.playback_view)
        
        self.stacked_widget.setCurrentIndex(self.simulator_output_type.value)  # PURE is the default
        
        self.main_layout.addWidget(self.stacked_widget)

    def update_pure_freq(self):
        curr_val = self.pure_freq_spin.value()
        if self.pure_freq != curr_val:
            self.pure_freq = curr_val

    def update_pure_amp(self):
        curr_val = self.pure_amp_spin.value()
        if self.pure_amp != curr_val:
            self.pure_amp = curr_val

    def update_pure_noise(self):
        curr_val = self.pure_noise_spin.value()
        if self.pure_noise != curr_val:
            self.pure_noise = curr_val

    def create_pure_view(self):
        view = QWidget()
        layout = QVBoxLayout(view)
        
        controls_group = QGroupBox()
        controls_layout = QVBoxLayout()
        
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Frequency (Hz):"))
        self.pure_freq_spin = QDoubleSpinBox()
        self.pure_freq_spin.setRange(0.1, 80.1)
        self.pure_freq_spin.setValue(self.pure_freq)
        self.pure_freq_spin.setSingleStep(0.5)
        self.pure_freq_spin.valueChanged.connect(self.update_pure_freq)
        freq_layout.addWidget(self.pure_freq_spin)
        controls_layout.addLayout(freq_layout)
        
        amp_layout = QHBoxLayout()
        amp_layout.addWidget(QLabel("Amplitude:"))
        self.pure_amp_spin = QDoubleSpinBox()
        self.pure_amp_spin.setRange(0.1, 100.0)
        self.pure_amp_spin.setValue(self.pure_amp)
        self.pure_amp_spin.setSingleStep(0.5)
        self.pure_amp_spin.valueChanged.connect(self.update_pure_amp)
        amp_layout.addWidget(self.pure_amp_spin)
        controls_layout.addLayout(amp_layout)
        
        noise_layout = QHBoxLayout()
        noise_layout.addWidget(QLabel("Noise Level:"))
        self.pure_noise_spin = QDoubleSpinBox()
        self.pure_noise_spin.setRange(0.0, 1.0)
        self.pure_noise_spin.setValue(self.pure_noise)
        self.pure_noise_spin.setSingleStep(0.1)
        self.pure_noise_spin.valueChanged.connect(self.update_pure_noise)
        noise_layout.addWidget(self.pure_noise_spin)
        controls_layout.addLayout(noise_layout)
        
        controls_layout.addStretch(1)
        controls_group.setLayout(controls_layout)
        
        layout.addWidget(controls_group)
        layout.addStretch(1)
        return view

    def create_playback_view(self):
        view = QWidget()
        layout = QVBoxLayout(view)
        
        session_group = QGroupBox()
        session_layout = QVBoxLayout()
        
        session_path_layout = QHBoxLayout()
        self.session_label = QLabel("No session selected")
        session_path_layout.addWidget(self.session_label)
        
        self.browse_button = QPushButton("Browse Session")
        self.browse_button.clicked.connect(self.browse_session)
        session_path_layout.addWidget(self.browse_button)
        
        session_layout.addLayout(session_path_layout)
        
        playback_layout = QHBoxLayout()
        self.position_label = QLabel("Progress: - %")
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 100)
        self.position_slider.setValue(0)
        self.position_slider.setEnabled(False)
        self.position_slider.sliderReleased.connect(self.position_slider_released)
        playback_layout.addWidget(self.position_label)
        playback_layout.addWidget(self.position_slider)
        session_layout.addLayout(playback_layout)
        session_group.setLayout(session_layout)
        layout.addWidget(session_group)
        layout.addStretch(1)
        
        return view

    def toggle_streaming(self):
        if self.streaming:
            self.streaming = False
            self.stream_button.setText("Start Streaming")
            self.output_type_combo.setEnabled(True)
            self.position_slider.setEnabled(True)
            if self.stream_thread.isRunning():
                self.stream_thread.quit()
                self.stream_thread.wait()
            self.output_type_combo.setDisabled(False)

        else:
            self.streaming = True
            self.stream_button.setText("Stop Streaming")
            self.output_type_combo.setEnabled(False)
            self.position_slider.setEnabled(False)
            if not self.stream_thread.isRunning():
                self.stream_thread.start()
            self.output_type_combo.setDisabled(True)
    
    def change_output_type(self, index):
        if self.streaming: 
            self.streaming = False
        self.simulator_output_type = list(SimulatorOutputType)[index]
        self.stacked_widget.setCurrentIndex(index)

        if self.simulator_output_type == SimulatorOutputType.PLAYBACK:
            self.stream_button.setEnabled(self.file_reader is not None)
        else:
            self.stream_button.setEnabled(True)

    def browse_session(self):
        """Open a dialog to select a session directory"""
        session_dir = QFileDialog.getExistingDirectory(
            self, "Select Session Directory", "data/"
        )
        
        if session_dir:
            session_key = os.path.basename(session_dir)
            self.file_reader = FileReader(session_key)
            self.total_frames = self.file_reader.get_total_frames()
            self.current_frame = 0
            self.session_label.setText(f"Session: {session_key}")
            
            has_frames = self.file_reader.get_total_frames() > 0
            self.position_slider.setEnabled(bool(has_frames))
            self.stream_button.setEnabled(has_frames)

            self.update_position_display()
    
    def position_slider_released(self):
        """Handle the position slider being released by the user"""
        if self.file_reader is not None and self.total_frames > 0:
            # Get the percentage from the slider
            percentage = self.position_slider.value()
            
            # Calculate the new frame position
            self.current_frame = int((percentage / 100) * self.total_frames)
            
            # Update the display
            self.update_position_display()
    
    def update_position_display(self):
        if self.file_reader is not None and self.total_frames > 0:
            # Calculate percentage progress
            progress = (self.current_frame / self.total_frames) * 100
            
            # Update the label
            self.position_label.setText(f"Progress: {progress:.1f}%")
            
            # Update the slider without triggering events
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(int(progress))
            self.position_slider.blockSignals(False)
    
    def update_ui(self):
        if self.streaming and self.simulator_output_type == SimulatorOutputType.PLAYBACK and self.file_reader is not None:
            self.update_position_display()

    def closeEvent(self, event):
        self.streaming = False
        if self.stream_thread.isRunning():
            self.stream_thread.quit()
            self.stream_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LSLSimulatorGUI()
    window.show()
    sys.exit(app.exec_())