# # import math
# # import sys
# # import time

# # import colorednoise
# # import numpy as np
# # import simpleaudio as sa
# # from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
# # from PyQt5.QtWidgets import (
# #     QApplication,
# #     QHBoxLayout,
# #     QLabel,
# #     QMainWindow,
# #     QPushButton,
# #     QSlider,
# #     QVBoxLayout,
# #     QWidget,
# # )


# # class AudioConfig:
# #     def __init__(self, total_s=1.0, ramp_s=0.1):
# #         self.total_s = total_s
# #         self.ramp_s = ramp_s

# # class AudioPlayer(QObject):
# #     playback_started = pyqtSignal()
# #     playback_finished = pyqtSignal()
    
# #     def __init__(self, config=None):
# #         super().__init__()
# #         self.config = config or AudioConfig()
# #         self.playing = False
# #         self.sound = None
# #         self.play_handle = None
# #         self.generate_audio()
        
# #     def generate_audio(self):
# #         fs = 44100
# #         noise_length = math.floor(self.config.total_s * fs)
# #         ramp_length = math.floor(self.config.ramp_s * fs)

# #         noisedata = colorednoise.powerlaw_psd_gaussian(1, noise_length)

# #         # Generate taper
# #         if ramp_length > 0:
# #             sineramp_x = np.linspace(0, np.pi / 2, np.round(self.config.ramp_s * fs).astype(int))
# #             sineramp = np.sin(sineramp_x)

# #             noisedata[:ramp_length] *= sineramp
# #             noisedata[-ramp_length:] *= np.flip(sineramp)

# #         # Normalize to int16
# #         noisedata -= np.mean(noisedata)
# #         noisedata /= np.max(np.abs(noisedata))
# #         noisedata *= 32766
# #         noisedata = noisedata.astype(np.int16)

# #         self.sound = sa.WaveObject(noisedata, 1, 2, fs)

# #     def play(self):
# #         if not self.playing:
# #             self.playing = True
# #             self.play_handle = self.sound.play()
# #             self.playback_started.emit()
# #             # Use a timer to check for completion
# #             check_timer = QTimer()
# #             check_timer.timeout.connect(lambda: self._check_playback_status(check_timer))
# #             check_timer.start(5)  # Check every 100ms
    
# #     def _check_playback_status(self, timer):
# #         if not self.play_handle.is_playing():
# #             timer.stop()
# #             self.playing = False
# #             self.playback_finished.emit()
    
# #     def stop(self):
# #         if self.playing and self.play_handle:
# #             self.play_handle.stop()
# #             self.playing = False
# #             self.playback_finished.emit()

# # class MainWindow(QMainWindow):
# #     def __init__(self):
# #         super().__init__()
# #         self.setWindowTitle("PyQt5 Audio Player")
# #         self.setGeometry(100, 100, 400, 300)
        
# #         # Create the audio player
# #         self.config = AudioConfig(total_s=2.0, ramp_s=0.1)
# #         self.audio_player = AudioPlayer(config=self.config)
        
# #         # Connect signals
# #         self.audio_player.playback_started.connect(self.on_playback_started)
# #         self.audio_player.playback_finished.connect(self.on_playback_finished)
        
# #         # Create periodic timer
# #         self.periodic_timer = QTimer()
# #         self.periodic_timer.timeout.connect(self.play_periodic_audio)
        
# #         # Setup UI
# #         self.setup_ui()
    
# #     def setup_ui(self):
# #         central_widget = QWidget()
# #         self.setCentralWidget(central_widget)
        
# #         layout = QVBoxLayout()
        
# #         # Play button
# #         self.play_button = QPushButton("Play Once")
# #         self.play_button.clicked.connect(self.play_audio)
# #         layout.addWidget(self.play_button)
        
# #         # Periodic controls
# #         periodic_layout = QHBoxLayout()
        
# #         self.periodic_button = QPushButton("Start Periodic")
# #         self.periodic_button.clicked.connect(self.toggle_periodic)
# #         periodic_layout.addWidget(self.periodic_button)
        
# #         interval_label = QLabel("Interval (s):")
# #         periodic_layout.addWidget(interval_label)
        
# #         self.interval_slider = QSlider(Qt.Horizontal)
# #         self.interval_slider.setMinimum(1)
# #         self.interval_slider.setMaximum(10)
# #         self.interval_slider.setValue(3)
# #         self.interval_slider.setTickPosition(QSlider.TicksBelow)
# #         self.interval_slider.setTickInterval(1)
# #         periodic_layout.addWidget(self.interval_slider)
        
# #         self.interval_value = QLabel("3")
# #         self.interval_slider.valueChanged.connect(
# #             lambda v: self.interval_value.setText(str(v))
# #         )
# #         periodic_layout.addWidget(self.interval_value)
        
# #         layout.addLayout(periodic_layout)
        
# #         # Status label
# #         self.status_label = QLabel("Ready")
# #         layout.addWidget(self.status_label)
        
# #         central_widget.setLayout(layout)
    
# #     def play_audio(self):
# #         if not self.audio_player.playing:
# #             self.audio_player.play()
    
# #     def play_periodic_audio(self):
# #         # This is called by the periodic timer
# #         self.play_audio()
    
# #     def toggle_periodic(self):
# #         if self.periodic_timer.isActive():
# #             self.periodic_timer.stop()
# #             self.periodic_button.setText("Start Periodic")
# #             self.status_label.setText("Periodic playback stopped")
# #         else:
# #             interval = self.interval_slider.value() * 1000  # Convert to ms
# #             self.periodic_timer.start(interval)
# #             self.periodic_button.setText("Stop Periodic")
# #             self.status_label.setText(f"Playing every {interval/1000} seconds")
    
# #     def on_playback_started(self):
# #         self.play_button.setEnabled(False)
# #         self.status_label.setText("Playing audio...")
    
# #     def on_playback_finished(self):
# #         self.play_button.setEnabled(True)
# #         if not self.periodic_timer.isActive():
# #             self.status_label.setText("Ready")

# import math

# # if __name__ == "__main__":
# #     app = QApplication(sys.argv)
# #     window = MainWindow()
# #     window.show()
# #     sys.exit(app.exec_())
# import sys
# import time

# import colorednoise
# import numpy as np
# import simpleaudio as sa
# from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
# from PyQt5.QtWidgets import (
#     QApplication,
#     QCheckBox,
#     QGroupBox,
#     QHBoxLayout,
#     QLabel,
#     QMainWindow,
#     QPushButton,
#     QSlider,
#     QSpinBox,
#     QVBoxLayout,
#     QWidget,
# )


# class AudioConfig:
#     def __init__(self, total_s=1.0, ramp_s=0.1):
#         self.total_s = total_s
#         self.ramp_s = ramp_s

# class AudioPlayer(QObject):
#     playback_started = pyqtSignal()
#     playback_finished = pyqtSignal()
#     delay_started = pyqtSignal(int)  # Signal emits remaining delay time
#     delay_finished = pyqtSignal()
    
#     def __init__(self, config=None):
#         super().__init__()
#         self.config = config or AudioConfig()
#         self.playing = False
#         self.sound = None
#         self.play_handle = None
#         self.delay_timer = QTimer()
#         self.delay_timer.setSingleShot(True)
#         self.delay_timer.timeout.connect(self._play_after_delay)
#         self.delay_countdown_timer = QTimer()
#         self.delay_countdown_timer.timeout.connect(self._update_delay_countdown)
#         self.remaining_delay = 0
#         self.generate_audio()
        
#     def generate_audio(self):
#         fs = 44100
#         noise_length = math.floor(self.config.total_s * fs)
#         ramp_length = math.floor(self.config.ramp_s * fs)

#         noisedata = colorednoise.powerlaw_psd_gaussian(1, noise_length)

#         # Generate taper
#         if ramp_length > 0:
#             sineramp_x = np.linspace(0, np.pi / 2, np.round(self.config.ramp_s * fs).astype(int))
#             sineramp = np.sin(sineramp_x)

#             noisedata[:ramp_length] *= sineramp
#             noisedata[-ramp_length:] *= np.flip(sineramp)

#         # Normalize to int16
#         noisedata -= np.mean(noisedata)
#         noisedata /= np.max(np.abs(noisedata))
#         noisedata *= 32766
#         noisedata = noisedata.astype(np.int16)

#         self.sound = sa.WaveObject(noisedata, 1, 2, fs)

#     def play(self, delay_ms=0):
#         if self.playing:
#             return
            
#         if delay_ms > 0:
#             self.start_delay(delay_ms)
#         else:
#             self._start_playback()
    
#     def start_delay(self, delay_ms):
#         self.remaining_delay = delay_ms
#         self.delay_timer.start(delay_ms)
#         self.delay_countdown_timer.start(100)  # Update every 100ms
#         self.delay_started.emit(delay_ms)
    
#     def _update_delay_countdown(self):
#         self.remaining_delay = max(0, self.remaining_delay - 100)
#         self.delay_started.emit(self.remaining_delay)
        
#         if self.remaining_delay <= 0:
#             self.delay_countdown_timer.stop()
    
#     def _play_after_delay(self):
#         self.delay_countdown_timer.stop()
#         self.delay_finished.emit()
#         self._start_playback()
    
#     def _start_playback(self):
#         if not self.playing:
#             self.playing = True
#             self.play_handle = self.sound.play()
#             self.playback_started.emit()
#             # Use a timer to check for completion
#             check_timer = QTimer()
#             check_timer.timeout.connect(lambda: self._check_playback_status(check_timer))
#             check_timer.start(100)  # Check every 100ms
    
#     def _check_playback_status(self, timer):
#         if not self.play_handle.is_playing():
#             timer.stop()
#             self.playing = False
#             self.playback_finished.emit()
    
#     def stop(self):
#         if self.delay_timer.isActive():
#             self.delay_timer.stop()
#             self.delay_countdown_timer.stop()
#             self.delay_finished.emit()
        
#         if self.playing and self.play_handle:
#             self.play_handle.stop()
#             self.playing = False
#             self.playback_finished.emit()

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("PyQt5 Audio Player")
#         self.setGeometry(100, 100, 500, 350)
        
#         # Create the audio player
#         self.config = AudioConfig(total_s=2.0, ramp_s=0.1)
#         self.audio_player = AudioPlayer(config=self.config)
        
#         # Connect signals
#         self.audio_player.playback_started.connect(self.on_playback_started)
#         self.audio_player.playback_finished.connect(self.on_playback_finished)
#         self.audio_player.delay_started.connect(self.on_delay_update)
#         self.audio_player.delay_finished.connect(self.on_delay_finished)
        
#         # Setup UI
#         self.setup_ui()
    
#     def setup_ui(self):
#         central_widget = QWidget()
#         self.setCentralWidget(central_widget)
        
#         main_layout = QVBoxLayout()
        
#         # Single Play Group
#         single_play_group = QGroupBox("Single Play")
#         single_play_layout = QVBoxLayout()
        
#         play_controls = QHBoxLayout()
        
#         self.play_button = QPushButton("Play Now")
#         self.play_button.clicked.connect(lambda: self.play_audio(0))
#         play_controls.addWidget(self.play_button)
        
#         self.play_delay_button = QPushButton("Play with Delay")
#         self.play_delay_button.clicked.connect(lambda: self.play_audio(self.delay_spinbox.value()))
#         play_controls.addWidget(self.play_delay_button)
        
#         delay_layout = QHBoxLayout()
#         delay_layout.addWidget(QLabel("Delay (ms):"))
        
#         self.delay_spinbox = QSpinBox()
#         self.delay_spinbox.setRange(100, 10000)
#         self.delay_spinbox.setSingleStep(100)
#         self.delay_spinbox.setValue(1000)
#         delay_layout.addWidget(self.delay_spinbox)
        
#         single_play_layout.addLayout(play_controls)
#         single_play_layout.addLayout(delay_layout)
#         single_play_group.setLayout(single_play_layout)
#         main_layout.addWidget(single_play_group)
        
#         # Periodic Group
#         periodic_group = QGroupBox("Periodic Playback")
#         periodic_layout = QVBoxLayout()
        
#         periodic_controls = QHBoxLayout()
#         self.periodic_button = QPushButton("Start Periodic")
#         self.periodic_button.clicked.connect(self.toggle_periodic)
#         periodic_controls.addWidget(self.periodic_button)
        
#         interval_layout = QHBoxLayout()
#         interval_layout.addWidget(QLabel("Interval (s):"))
        
#         self.interval_slider = QSlider(Qt.Horizontal)
#         self.interval_slider.setMinimum(1)
#         self.interval_slider.setMaximum(10)
#         self.interval_slider.setValue(3)
#         self.interval_slider.setTickPosition(QSlider.TicksBelow)
#         self.interval_slider.setTickInterval(1)
#         interval_layout.addWidget(self.interval_slider)
        
#         self.interval_value = QLabel("3")
#         self.interval_slider.valueChanged.connect(
#             lambda v: self.interval_value.setText(str(v))
#         )
#         interval_layout.addWidget(self.interval_value)
        
#         periodic_layout.addLayout(periodic_controls)
#         periodic_layout.addLayout(interval_layout)
        
#         # Add periodic delay checkbox
#         periodic_delay_layout = QHBoxLayout()
#         self.use_delay_checkbox = QCheckBox("Use delay in periodic playback")
#         periodic_delay_layout.addWidget(self.use_delay_checkbox)
#         periodic_layout.addLayout(periodic_delay_layout)
        
#         periodic_group.setLayout(periodic_layout)
#         main_layout.addWidget(periodic_group)
        
#         # Status label
#         self.status_label = QLabel("Ready")
#         self.status_label.setAlignment(Qt.AlignCenter)
#         main_layout.addWidget(self.status_label)
        
#         # Stop button
#         self.stop_button = QPushButton("Stop All")
#         self.stop_button.clicked.connect(self.stop_all)
#         self.stop_button.setEnabled(False)
#         main_layout.addWidget(self.stop_button)
        
#         central_widget.setLayout(main_layout)
    
#     def play_audio(self, delay_ms=0):
#         if not self.audio_player.playing and not self.audio_player.delay_timer.isActive():
#             self.audio_player.play(delay_ms)
#             self.stop_button.setEnabled(True)
    
#     def stop_all(self):
#         self.audio_player.stop()
#         self.status_label.setText("Stopped")
#         self.stop_button.setEnabled(False)
    
#     def on_playback_started(self):
#         self.play_button.setEnabled(False)
#         self.play_delay_button.setEnabled(False)
#         self.status_label.setText("Playing audio...")
    
#     def on_playback_finished(self):
#         self.play_button.setEnabled(True)
#         self.play_delay_button.setEnabled(True)
#         if not self.periodic_timer.isActive() and not self.audio_player.delay_timer.isActive():
#             self.status_label.setText("Ready")
#             self.stop_button.setEnabled(False)
    
#     def on_delay_update(self, remaining_ms):
#         self.play_button.setEnabled(False)
#         self.play_delay_button.setEnabled(False)
#         remaining_s = remaining_ms / 1000
#         self.status_label.setText(f"Waiting... {remaining_s:.1f}s before playback")
    
#     def on_delay_finished(self):
#         if not self.periodic_timer.isActive() and not self.audio_player.playing:
#             self.play_button.setEnabled(True)
#             self.play_delay_button.setEnabled(True)

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())

import math
import sys

import colorednoise
import numpy as np
import simpleaudio as sa
from PyQt5.QtCore import QObject, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from utils import Audio, AudioConfig


class AudioApp(QWidget):
    def __init__(self):
        super().__init__()
        self.audio = Audio(AudioConfig())
        
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel("Delay (ms):")
        layout.addWidget(self.label)
        
        self.delay_input = QSpinBox()
        self.delay_input.setRange(0, 5000)
        layout.addWidget(self.delay_input)
        
        self.play_button = QPushButton("Play Sound")
        self.play_button.clicked.connect(self.play_sound)
        layout.addWidget(self.play_button)
        
        self.setLayout(layout)
        self.setWindowTitle("Audio Player")
    
    def play_sound(self):
        delay = self.delay_input.value()
        self.audio.play(delay_ms=delay)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioApp()
    window.show()
    sys.exit(app.exec())
