import math
import threading

import colorednoise
import numpy as np
import simpleaudio as sa
from PyQt5.QtCore import QRunnable

from .config import AudioConfig


class Audio(QRunnable):
    def __init__(self, config: AudioConfig):
        super().__init__()
        fs = 44100
        noise_length = math.floor(config.total_s * fs)
        ramp_length = math.floor(config.ramp_s * fs)

        noisedata = colorednoise.powerlaw_psd_gaussian(1, noise_length)

        # Generate taper
        if ramp_length > 0:
            sineramp_x = np.linspace(0, np.pi / 2, np.round(config.ramp_s * fs).astype(int))
            sineramp = np.sin(sineramp_x)

            noisedata[:ramp_length] *= sineramp
            noisedata[-ramp_length:] *= np.flip(sineramp)

        # Normalize to int16
        noisedata -= np.mean(noisedata)
        noisedata /= np.max(np.abs(noisedata))
        noisedata *= 32766
        noisedata = noisedata.astype(np.int16)

        self.sound = sa.WaveObject(noisedata, 1, 2, fs)

    def run(self):
        handle = self.sound.play()
        handle.wait_done()