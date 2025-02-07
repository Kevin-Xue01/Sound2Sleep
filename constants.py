from enum import Enum

from muselsl.constants import (  # MUSE_NB_GYRO_CHANNELS,; MUSE_SAMPLING_GYRO_RATE,; LSL_GYRO_CHUNK,
    LSL_ACC_CHUNK,
    LSL_EEG_CHUNK,
    LSL_PPG_CHUNK,
    LSL_SCAN_TIMEOUT,
    MUSE_NB_ACC_CHANNELS,
    MUSE_NB_EEG_CHANNELS,
    MUSE_NB_PPG_CHANNELS,
    MUSE_SAMPLING_ACC_RATE,
    MUSE_SAMPLING_EEG_RATE,
    MUSE_SAMPLING_PPG_RATE,
    VIEW_SUBSAMPLE,
)


class DataStream(Enum):
    EEG = "EEG"
    ACCELEROMETER = "Accelerometer"
    PPG = "PPG"

SAMPLING_RATE = {
    DataStream.EEG: MUSE_SAMPLING_EEG_RATE,
    DataStream.ACCELEROMETER: MUSE_SAMPLING_EEG_RATE,
    DataStream.PPG: MUSE_SAMPLING_EEG_RATE,
}

NB_CHANNELS = {
    DataStream.EEG: MUSE_NB_EEG_CHANNELS,
    DataStream.ACCELEROMETER: MUSE_NB_ACC_CHANNELS,
    DataStream.PPG: MUSE_NB_PPG_CHANNELS,
}

CHUNK_SIZE = {
    DataStream.EEG: LSL_EEG_CHUNK,
    DataStream.ACCELEROMETER: LSL_ACC_CHUNK,
    DataStream.PPG: LSL_PPG_CHUNK,
}

CHANNEL_NAMES = {
    DataStream.EEG: ['TP9', 'TP10', 'AF1', 'AF2', 'Right AUX'],
    DataStream.ACCELEROMETER: ['Acc_X', 'Acc_Y', 'Acc_Z'],
    DataStream.PPG: ['PPG'],
}

class Config:
    class UI:
        window_s = 5
        scale = 100
        refresh = 0.2
        figure = "15x6"
        backend = 'TkAgg'
        dejitter = True
        subsample = VIEW_SUBSAMPLE
    class Connection:
        no_data_counter_max = 0

        reset_stream_step1_delay = 3 # [s]
        reset_stream_step2_delay = 3 # [s]
        reset_stream_step3_delay = 3 # [s]

        reset_attempt_count_max = 3
    
    class Processing:
        window_s = 2
        # subsample = 1