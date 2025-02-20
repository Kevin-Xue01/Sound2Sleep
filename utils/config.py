import json
import random
import string
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr


def generate_random_key(length=6):
    """Generates a random alphanumeric key of the specified length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

class TruncatedWaveletConfig(BaseModel):
    n: int = 30 # [number of wavelets]
    w: int = 5
    low: float = 0.5
    high: float = 2.0

class ProcessingConfig(BaseModel):
    truncated_wavelet: TruncatedWaveletConfig = Field(default_factory=TruncatedWaveletConfig)
    window_len: float = 2.0 # [seconds]
    hl_ratio_buffer: int = 2
    hl_ratio_threshold: float = -2.0
    amp_buffer: int = 2
    amp_threshold: float = 4e-4
    target_phase_deg: float = 0.0
    backoff_max_time: float = 5.0
    queue_stim_max_delta_t_: float = 0.1

class AudioConfig(BaseModel):
    ramp_up_s: float = 1.0
    ramp_down_s: float = 1.0
    total_s: float = 3.0

class DisplayConfig(BaseModel):
    window_len: float = 5.0
    display_every: int = 5

class BlueMuseConfig(BaseModel):
    low_pass_cutoff: float = 40.0
    low_pass_transition_width: float = 0.05 # units of fs

class EEGSessionConfig(BaseModel):
    _key: str = PrivateAttr(default_factory=lambda: generate_random_key())
    _created_at: str = PrivateAttr(default_factory=lambda: datetime.now().isoformat())
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    connection: BlueMuseConfig = Field(default_factory=BlueMuseConfig)
    log_dir: str = "logs/"
    
    def __eq__(self, other):
        """Compare two EEGSessionConfig objects, ignoring private attributes."""
        if not isinstance(other, EEGSessionConfig):
            return NotImplemented
        return self.model_dump() == other.model_dump()

    # def update_processing_params(self, **kwargs):
    #     """Updates processing parameters with provided key-value pairs."""
    #     for key, value in kwargs.items():
    #         if hasattr(self.processing_params, key):
    #             setattr(self.processing_params, key, value)
    
    # def update_audio_params(self, **kwargs):
    #     """Updates audio parameters with provided key-value pairs."""
    #     self.audio_params.update(kwargs)
    
    # def save_to_file(self, file_path: str):
    #     """Saves the session configuration to a JSON file."""
    #     with open(file_path, 'w') as file:
    #         json.dump(self.dict(), file, indent=4)
    
    # @classmethod
    # def load_from_file(cls, file_path: str):
    #     """Loads session configuration from a JSON file."""
    #     with open(file_path, 'r') as file:
    #         data = json.load(file)
    #     return cls(**data)
    
    # def __repr__(self):
    #     return f"EEGSessionConfig({self.dict()})"