import math
import time
from multiprocessing import Process, Queue

import colorednoise
import numpy as np
import simpleaudio as sa
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from .config import AudioConfig


def audio_callback(queue):
    config = AudioConfig()
    
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

    sound = sa.WaveObject(noisedata, 1, 2, fs)
    """Runs in a separate process to ensure true parallel execution."""
    while True:
        delay = queue.get()  # Wait for an audio request
        if delay is None:
            break  # Stop process if None is received
        time.sleep(delay)  # Simulate delay
        sound.play()  # Replace this with actual audio playback logic


class Audio(QObject):
    """Manages audio playback using a separate process."""
    play_audio_signal = pyqtSignal(float)  # Signal to trigger audio playback

    def __init__(self):
        super().__init__()
        self.queue = Queue()  # Inter-process communication queue
        self.process = Process(target=audio_callback, args=(self.queue))
        self.process.start()

        # Connect the signal to the handler
        self.play_audio_signal.connect(self.handle_audio_request)

    def handle_audio_request(self, delay: float):
        """Sends audio playback request to the separate process."""
        self.queue.put(delay)

    def __del__(self):
        """Ensure proper cleanup of the process."""
        self.queue.put(None)  # Signal process to exit
        self.process.join()