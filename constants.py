from enum import Enum

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


class DataStream(Enum):
    EEG = "EEG"
    ACCELEROMETER = "Accelerometer"
    PPG = "PPG"

SAMPLING_RATE = {
    DataStream.EEG: MUSE_SAMPLING_EEG_RATE,
    DataStream.ACCELEROMETER: MUSE_SAMPLING_ACC_RATE,
    DataStream.PPG: MUSE_SAMPLING_PPG_RATE,
}

NB_CHANNELS = {
    DataStream.EEG: MUSE_NB_EEG_CHANNELS - 1,
    DataStream.ACCELEROMETER: MUSE_NB_ACC_CHANNELS,
    DataStream.PPG: MUSE_NB_PPG_CHANNELS,
}

CHUNK_SIZE = {
    DataStream.EEG: LSL_EEG_CHUNK,
    DataStream.ACCELEROMETER: LSL_ACC_CHUNK,
    DataStream.PPG: LSL_PPG_CHUNK,
}

CHANNEL_NAMES = {
    DataStream.EEG: ['TP9', 'TP10', 'AF1', 'AF2'],
    DataStream.ACCELEROMETER: ['Acc_X', 'Acc_Y', 'Acc_Z'],
    DataStream.PPG: ['PPG'],
}

TIMESTAMPS = {
    DataStream.EEG: np.float64(np.arange(CHUNK_SIZE[DataStream.EEG])) / SAMPLING_RATE[DataStream.EEG],
    DataStream.ACCELEROMETER: np.float64(np.arange(CHUNK_SIZE[DataStream.ACCELEROMETER])) / SAMPLING_RATE[DataStream.ACCELEROMETER],
    DataStream.PPG: np.float64(np.arange(CHUNK_SIZE[DataStream.PPG])) / SAMPLING_RATE[DataStream.PPG]
}

DELAYS = {
    DataStream.EEG: CHUNK_SIZE[DataStream.EEG] / SAMPLING_RATE[DataStream.EEG],
    DataStream.ACCELEROMETER: CHUNK_SIZE[DataStream.ACCELEROMETER] / SAMPLING_RATE[DataStream.ACCELEROMETER],
    DataStream.PPG: CHUNK_SIZE[DataStream.PPG] / SAMPLING_RATE[DataStream.PPG]
}


class Config:
    class UI:
        window_s = 5
        refresh = 0.2
        dejitter = True
    class Connection:
        no_data_counter_max = 0

        reset_stream_step1_delay = 3 # [s]
        reset_stream_step2_delay = 3 # [s]
        reset_stream_step3_delay = 3 # [s]

        reset_attempt_count_max = 3
    
    class Processing:
        window_s = 2