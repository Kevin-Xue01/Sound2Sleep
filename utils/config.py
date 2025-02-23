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
    ramp_s: float = 1.0
    total_s: float = 3.0

class DisplayConfig(BaseModel):
    window_len: float = 5.0
    display_every: int = 2

class SessionConfig(BaseModel):
    _session_key: str = PrivateAttr(default_factory=lambda: datetime.now().strftime("%m-%d_%H-%M-%S"))
    _created_at: str = PrivateAttr(default_factory=lambda: datetime.now().isoformat())
    _audio: AudioConfig = PrivateAttr(default_factory=AudioConfig)
    _display: DisplayConfig = PrivateAttr(default_factory=DisplayConfig)
    experiment_mode: ExperimentMode = ExperimentMode.DISABLED
    truncated_wavelet: TruncatedWaveletConfig = Field(default_factory=TruncatedWaveletConfig)
    window_len_s: float = 2.0 # [seconds], duration of processing window
    hl_ratio_buffer_len: int = 2
    hl_ratio_buffer_mean_threshold: float = -2.0
    hl_ratio_latest_threshold: float = -2.0
    amp_buffer_len: int = 2
    amp_buffer_mean_threshold: float = 4e-4
    amp_latest_threshold: float = 4e-4
    target_phase_deg: float = 0.0
    backoff_time: float = 3.0
    stim2_start_delay: float = 2.0
    stim2_end_delay: float = 2.0
    low_bpf_cutoff: tuple = (1.0, 4.0)
    high_bpf_cutoff: tuple = (12.0, 80.0)
    bpf_order: int = 4
    
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