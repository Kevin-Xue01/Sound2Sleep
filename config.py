import json
import random
import string
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field


def generate_random_key(length=6):
    """Generates a random alphanumeric key of the specified length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

class WaveletParams(BaseModel):
    n: int = 30 # [number of wavelets]
    w: int = 5
    low: int = 0.5
    high: int = 2

class ProcessingParams(BaseModel):
    truncated_wavelet: WaveletParams = Field(default_factory=WaveletParams)
    window_len: int = 2 # [seconds]
    hl_ratio_threshold: int = -2
    amp_threshold: float = 4e-4
    target_phase: float = 2 * np.pi

class EEGSessionConfig(BaseModel):
    datetime: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M"))
    processing_params: ProcessingParams = Field(default_factory=ProcessingParams)
    audio_params: dict = Field(default_factory=dict)

    def update_processing_params(self, **kwargs):
        """Updates processing parameters with provided key-value pairs."""
        for key, value in kwargs.items():
            if hasattr(self.processing_params, key):
                setattr(self.processing_params, key, value)
    
    def update_audio_params(self, **kwargs):
        """Updates audio parameters with provided key-value pairs."""
        self.audio_params.update(kwargs)
    
    def save_to_file(self, file_path: str):
        """Saves the session configuration to a JSON file."""
        with open(file_path, 'w') as file:
            json.dump(self.dict(), file, indent=4)
    
    @classmethod
    def load_from_file(cls, file_path: str):
        """Loads session configuration from a JSON file."""
        with open(file_path, 'r') as file:
            data = json.load(file)
        return cls(**data)
    
    def __repr__(self):
        return f"EEGSessionConfig({self.dict()})"