import random
import string
from datetime import datetime

from pydantic import BaseModel, Field, PrivateAttr

from .constants import ExperimentMode


def generate_random_key(length=6):
    """Generates a random alphanumeric key of the specified length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

class TruncatedWaveletConfig(BaseModel):
    n: int = 30 # [number of wavelets]
    w: int = 5
    low: float = 0.5
    high: float = 2.0

class AudioConfig(BaseModel):
    ramp_s: float = 0.01
    total_s: float = 0.05
    volume: float = 0.001

class SessionConfig(BaseModel):
    _session_key: str = PrivateAttr(default_factory=lambda: datetime.now().strftime("%m-%d_%H-%M-%S"))
    _created_at: str = PrivateAttr(default_factory=lambda: datetime.now().isoformat())
    _audio: AudioConfig = PrivateAttr(default_factory=AudioConfig)
    experiment_mode: ExperimentMode = ExperimentMode.CLAS_AUDIO_ON
    truncated_wavelet: TruncatedWaveletConfig = Field(default_factory=TruncatedWaveletConfig)
    
    mean_subtraction_window_len_s: float = 15.0
    processing_window_len_s: float = 2.0 # [seconds], duration of processing window
    
    hl_ratio_buffer_len: int = 3
    hl_ratio_buffer_mean_threshold: float = -1.0
    hl_ratio_latest_threshold: float = -1.0

    amp_buffer_len: int = 3
    amp_buffer_mean_min: float = 75.0
    amp_buffer_mean_max: float = 400.0

    target_phase: float = 0.0 # radians
    
    backoff_time: float = 5.0
    stim2_start_delay: float = 0.5
    stim2_end_delay: float = 0.5

    low_bpf_cutoff: tuple = (0.5, 4.0)
    high_bpf_cutoff: tuple = (8.0, 12.0)
    bpf_order: int = 4

    switch_channel_period_s: float = 15.0
    stim1_prediction_limit_sec: float = 0.1
    stim2_prediction_limit_sec: float = 0.1

    time_to_target_offset: float = 0.002
    
    def __eq__(self, other):
        """Compare two SessionConfig objects, ignoring private attributes."""
        if not isinstance(other, SessionConfig):
            return NotImplemented
        return self.model_dump() == other.model_dump()
    
    def model_dump_json(self, *, indent: int = 4, **kwargs) -> str:
        """Override model_dump_json to exclude 'experiment_mode' from output."""
        return super().model_dump_json(indent=indent, exclude={'experiment_mode'}, **kwargs)
    
    def model_dump(self, **kwargs):
        """Override model_dump to exclude 'experiment_mode' from output."""
        kwargs.setdefault("exclude", {"experiment_mode"})
        return super().model_dump(**kwargs)

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
    #     return f"SessionConfig({self.dict()})"