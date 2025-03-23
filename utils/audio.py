import math

import colorednoise
import numpy as np
import simpleaudio as sa
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from .config import AudioConfig


class Audio(QObject):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.playing = False
        self.sound = None
        self.play_handle = None
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)
        self.delay_timer.timeout.connect(self._play_after_delay)
        self.remaining_delay = 0

        fs = 44100
        noise_length = math.floor(self.config.total_s * fs)
        ramp_length = math.floor(self.config.ramp_s * fs)

        noisedata = colorednoise.powerlaw_psd_gaussian(1, noise_length)

        # Generate taper
        if ramp_length > 0:
            sineramp_x = np.linspace(0, np.pi / 2, np.round(self.config.ramp_s * fs).astype(int))
            sineramp = np.sin(sineramp_x)

            noisedata[:ramp_length] *= sineramp
            noisedata[-ramp_length:] *= np.flip(sineramp)

        # Normalize to int16
        noisedata -= np.mean(noisedata)
        noisedata /= np.max(np.abs(noisedata))
        noisedata *= 32766
        noisedata *= self.config.volume
        noisedata = noisedata.astype(np.int16)

        self.sound = sa.WaveObject(noisedata, 1, 2, fs)
        self.play_handle = None

    def play(self, delay_s):
        if self.play_handle is not None and self.play_handle.is_playing():
            self.play_handle.stop()

        self._start_delay(delay_s)
    
    def _start_delay(self, delay_s):
        delay_ms = math.floor(delay_s * 1000)
        self.delay_timer.start(delay_ms)
    
    def _play_after_delay(self):
        self._start_playback()
    
    def _start_playback(self):
        self.play_handle = self.sound.play()