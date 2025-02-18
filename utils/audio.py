import math
import threading

import colorednoise
import numpy as np
import simpleaudio as sa


class Audio:
    def __init__(self, length: float = 4.0, ramp: float = 1.0):
        """Initialize the noise player by generating the sound object."""
        fs = 44100
        noise_length = math.floor(length * fs)
        ramp_length = math.floor(ramp * fs)

        noisedata = colorednoise.powerlaw_psd_gaussian(1, noise_length)

        # Generate taper
        if ramp_length > 0:
            sineramp_x = np.linspace(0, np.pi / 2, np.round(ramp * fs).astype(int))
            sineramp = np.sin(sineramp_x)

            noisedata[:ramp_length] *= sineramp
            noisedata[-ramp_length:] *= np.flip(sineramp)

        # Normalize to int16
        noisedata -= np.mean(noisedata)
        noisedata /= np.max(np.abs(noisedata))
        noisedata *= 32766
        noisedata = noisedata.astype(np.int16)

        self.sound = sa.WaveObject(noisedata, 1, 2, fs)

    def play(self):
        """Play the sound asynchronously without overlapping previous playback."""
        def _play():
            # Wait for the previous sound to finish before playing a new one
            handle = self.sound.play()
            handle.wait_done()
        
        thread = threading.Thread(target=_play, daemon=True)
        thread.start()