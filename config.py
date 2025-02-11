import json
import random
import string
from datetime import datetime

from pydantic import BaseModel, Field


def generate_random_key(length=6):
    """Generates a random alphanumeric key of the specified length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

class EEGSessionConfig(BaseModel):
    random_key: str = Field(default_factory=generate_random_key)
    datetime: str = Field(default_factory=lambda: datetime.now().isoformat())
    subject_name: str
    processing_params: dict = Field(default_factory=dict)
    audio_params: dict = Field(default_factory=dict)

    def update_processing_params(self, **kwargs):
        """Updates processing parameters with provided key-value pairs."""
        self.processing_params.update(kwargs)
    
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