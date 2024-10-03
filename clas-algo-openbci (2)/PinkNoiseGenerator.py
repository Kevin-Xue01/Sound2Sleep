import simpleaudio as sa
import numpy as np
import colorednoise
import math


def generate_noise(fsample: int = 48000,
                   lowpass: float = 80,
                   length: float = 0.05,
                   ramp: float = 0.005) -> sa.WaveObject:
    '''Generate a simpleaudio wave object containing pink noise bursts

    Noise bursts are 1-channel, 1/f pink noise, tapered with sinusoids at the beginning and end

    Parameters
    ----------
    fsample : int
        Sampling rate in Hz

    lowpass : float
        Low-pass cutoff frequency in Hz

    length : float
        Length in seconds of the burst

    ramp : float
        Ramp up / ramp down timing in seconds

    Returns
    -------
    simpleaudio WaveObject

    '''

    # compute timings in samples
    noise_length = math.floor(length * fsample)
    ramp_length = math.floor(ramp * fsample)

    # generate low-passed pink noise
    noisedata = colorednoise.powerlaw_psd_gaussian(1, noise_length,
                                                   lowpass / fsample * 2)

    # generate taper
    if (ramp_length > 0):
        sineramp_x = np.linspace(0, np.pi / 2,
                                 np.round(ramp * fsample).astype('int'))
        sineramp = np.sin(sineramp_x)

        noisedata[0:ramp_length] = noisedata[0:ramp_length] * sineramp
        noisedata[-1 *
                  ramp_length:] = noisedata[-1 *
                                            ramp_length:] * np.flip(sineramp)

    # normalize to int16
    noisedata = noisedata - np.mean(noisedata)
    noisedata = noisedata / (np.max(np.abs(noisedata)))
    noisedata = noisedata * 32766
    noisedata = noisedata.astype(np.int16)

    # create playable wave object
    sound = sa.WaveObject(noisedata, 1, 2, fsample)
    return sound