import json
import os
import random
import string
from datetime import datetime
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING, _levelToName

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PrivateAttr

from .constants import ConnectionMode, ExperimentMode

load_dotenv(override=True)

class CLASAlgoConfig(BaseModel):
    processing_window_len_s: float = 2.0 # [seconds], duration of processing window

    w: int = 5
    t_wavelet_N: int = 20
    t_wavelet_freq_range: tuple = (0.6, 2.0)

    hl_ratio_buffer_len: int = 5
    hl_ratio_wavelet_freqs: list = [10, 20, 30]
    hl_ratio_latest_threshold: float = 0.15
    hl_ratio_buffer_threshold: float = 0.15

    amp_buffer_len: int = 10
    amp_threshold: float = 150
    amp_limit: float = 300.0

    target_phase: float = 3 * np.pi / 2 # radians
    backoff_time: float = 7.0

    quadrature_thresh: float = 0.8
    quadrature_len_s: float = 1.0

    stim2_start_delay: float = 0.6
    stim2_end_delay: float = 5.0

    stim1_prediction_limit_sec: float = 0.15
    stim2_prediction_limit_sec: float = 0.15

class AudioConfig(BaseModel):
    ramp_s: float = 0.005
    total_s: float = 0.05
    volume: float = 0.1

class SessionConfig(BaseModel):
    _session_key: str = PrivateAttr(default_factory=lambda: datetime.now().strftime("%m-%d_%H-%M-%S"))
    clas_algo: CLASAlgoConfig = Field(default_factory=CLASAlgoConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)

    _data_dir: str = PrivateAttr(default_factory=lambda: os.path.join("data", os.getenv('SUBJECT_NAME')))
    experiment_mode: ExperimentMode = ExperimentMode.CLAS_AUDIO_ON
    connection_mode: ConnectionMode = ConnectionMode.REALTIME

    switch_channel_period_s: float = 15.0
    time_to_target_offset: float = 0.003

    console_logging_level: int = WARNING
    file_logging_level: int = INFO
    filter_display_data: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        
        self._data_dir = os.path.join(self._data_dir, self._session_key)
        os.makedirs(self._data_dir, exist_ok=True)

        self._session_config_filename = os.path.join(self._data_dir, 'config.json')
        with open(self._session_config_filename, 'w') as file:
            json.dump(self.model_dump(), file, indent=4)
    
    def model_dump(self, **kwargs):
        base_dict = super().model_dump(**kwargs)
        base_dict["experiment_mode"] = self.experiment_mode.value  # Ensure .value is used
        base_dict["connection_mode"] = self.connection_mode.value  # Ensure .value is used
        base_dict["console_logging_level"] = _levelToName[self.console_logging_level]
        base_dict["file_logging_level"] = _levelToName[self.file_logging_level]
        return base_dict