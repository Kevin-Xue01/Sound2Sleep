from enum import Enum, auto

import numpy as np
from muselsl.constants import (
    LSL_ACC_CHUNK,
    LSL_EEG_CHUNK,
    LSL_PPG_CHUNK,
    MUSE_NB_ACC_CHANNELS,
    MUSE_NB_EEG_CHANNELS,
    MUSE_NB_PPG_CHANNELS,
    MUSE_SAMPLING_ACC_RATE,
    MUSE_SAMPLING_EEG_RATE,
    MUSE_SAMPLING_PPG_RATE,
)


class MuseDataType(Enum):
    EEG = "EEG"
    ACC = "Accelerometer"
    PPG = "PPG"

SAMPLING_RATE = {
    MuseDataType.EEG: MUSE_SAMPLING_EEG_RATE,
    MuseDataType.ACC: MUSE_SAMPLING_ACC_RATE,
    MuseDataType.PPG: MUSE_SAMPLING_PPG_RATE,
}

NB_CHANNELS = {
    MuseDataType.EEG: MUSE_NB_EEG_CHANNELS - 1,
    MuseDataType.ACC: MUSE_NB_ACC_CHANNELS,
    MuseDataType.PPG: MUSE_NB_PPG_CHANNELS,
}

CHUNK_SIZE = {
    MuseDataType.EEG: LSL_EEG_CHUNK,
    MuseDataType.ACC: LSL_ACC_CHUNK,
    MuseDataType.PPG: LSL_PPG_CHUNK,
}

CHANNEL_NAMES = {
    MuseDataType.EEG: ['TP9', 'AF1', 'AF2', 'TP10'],
    MuseDataType.ACC: ['Acc_X', 'Acc_Y', 'Acc_Z'],
    MuseDataType.PPG: ['PPG'],
}

TIMESTAMPS = {
    MuseDataType.EEG: np.arange(CHUNK_SIZE[MuseDataType.EEG], dtype=np.float64) / np.float64(SAMPLING_RATE[MuseDataType.EEG]),
    MuseDataType.ACC: np.arange(CHUNK_SIZE[MuseDataType.ACC], dtype=np.float64) / np.float64(SAMPLING_RATE[MuseDataType.ACC]),
    MuseDataType.PPG: np.arange(CHUNK_SIZE[MuseDataType.PPG], dtype=np.float64) / np.float64(SAMPLING_RATE[MuseDataType.PPG])
}

DELAYS = {
    MuseDataType.EEG: CHUNK_SIZE[MuseDataType.EEG] / SAMPLING_RATE[MuseDataType.EEG],
    MuseDataType.ACC: CHUNK_SIZE[MuseDataType.ACC] / SAMPLING_RATE[MuseDataType.ACC],
    MuseDataType.PPG: CHUNK_SIZE[MuseDataType.PPG] / SAMPLING_RATE[MuseDataType.PPG]
}

class AppState(Enum):
    DISCONNECTED = "Disconnected"
    CONNECTING = "Connecting"
    CONNECTED = "Connected"
    RECORDING = "Recording"

class ExperimentMode(Enum):
    DISABLED = "Disabled" # disabled CLAS => NOTE: NULL HYPOTHESIS #1
    RANDOM_PHASE = "Random Phase" # CLAS with random target phase and audio off => NOTE: NULL HYPOTHESIS #2
    CONFIGURE_DELAY = "Configure Delay" # CLAS with specific target phase + configurable delay and audio off
    CLAS = "CLAS" # CLAS with specific target phase and audio on

# ## Running modes
# **CLAS**: (The full algorithm) Target specific phase and attempt to deliver stimulation exactly on target phase.  
# **SHAM Muted**: (The full algorith, no auditory stim) Target specific phase and attempt to deliver stimulation exactly on target phase, but do not actually output any audio. Parallel port markers are are still outputted.
# **SHAM Phase**: (previously "vary") Deliver stimulation on random phase. Rerandomize the target phase after each stimulation.
# **SHAM Delay**: (previuosly "sham") Target specific phase, when stim is triggered, wait a random delay interval then deliver auditory stimuli.  

class EEGProcessorOutput(Enum):
    NOT_RUNNING = auto()

    STIM = auto()
    STIM2 = auto()

    HL_RATIO = auto()
    AMPLITUDE = auto()
    BACKOFF = auto()
    BACKOFF2 = auto()
    FUTURE = auto()
    FUTURE2 = auto()