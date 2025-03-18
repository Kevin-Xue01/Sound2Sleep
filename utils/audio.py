import asyncio
import math
import sys

import colorednoise
import numpy as np
import simpleaudio as sa
from PyQt5.QtCore import QRunnable, QThreadPool, QTimer
from PyQt5.QtWidgets import QApplication

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
        noisedata *= config.volume
        noisedata = noisedata.astype(np.int16)

        self.sound = sa.WaveObject(noisedata, 1, 2, fs)

    async def play(self):
        """Non-blocking audio playback using asyncio."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._blocking_play)

    def _blocking_play(self):
        """Plays the sound in a blocking way (but wrapped in an executor)."""
        handle = self.sound.play()
        handle.wait_done()