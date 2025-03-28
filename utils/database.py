import os
from typing import Union

import numpy as np

from .config import SessionConfig
from .constants import CHANNEL_NAMES, CHUNK_SIZE, MuseDataType


class FileWriter:
    def __init__(self, config: SessionConfig, muse_data_type=MuseDataType.EEG):
        self.config = config
        self.muse_data_type = muse_data_type
        self.timestamp_file_path = os.path.join(config._data_dir, f"{self.muse_data_type.name}_timestamp.bin")
        self.data_file_path = os.path.join(config._data_dir, f"{self.muse_data_type.name}_data.bin")
        
        self.data_shape = (CHUNK_SIZE[MuseDataType.EEG], len(CHANNEL_NAMES[MuseDataType.EEG]))
        self.data_dtype = np.float32
        self.timestamp_shape = (CHUNK_SIZE[MuseDataType.EEG],)
        self.timestamp_dtype = np.float64
        
        self.data_frame_size = np.prod(self.data_shape) * np.dtype(self.data_dtype).itemsize # 12 * 4 * 4 bytes
        self.timestamp_frame_size = np.prod(self.timestamp_shape) * np.dtype(self.timestamp_dtype).itemsize # 12 * 8 bytes
        
        self.data_file = open(self.data_file_path, 'wb')
        self.timestamp_file = open(self.timestamp_file_path, 'wb')
        
    def write_chunk(self, data: np.ndarray, timestamp: np.ndarray):
        self.data_file.write(data.tobytes())
        self.timestamp_file.write(timestamp.tobytes())

        self.data_file.flush()
        self.timestamp_file.flush()

    def __del__(self):
        self.data_file.close()
        self.timestamp_file.close()


class FileReader:
    def __init__(self, config: Union[SessionConfig, str], muse_data_type=MuseDataType.EEG):
        self.muse_data_type = muse_data_type
        
        if isinstance(config, str):
            self.timestamp_file_path = os.path.join('data', *config.split(os.path.sep), f"{self.muse_data_type.name}_timestamp.bin")
            self.data_file_path = os.path.join('data', *config.split(os.path.sep), f"{self.muse_data_type.name}_data.bin")
        else:
            self.timestamp_file_path = os.path.join(config._data_dir, f"{self.muse_data_type.name}_timestamp.bin")
            self.data_file_path = os.path.join(config._data_dir, f"{self.muse_data_type.name}_data.bin")

        # Define expected shapes and dtypes
        self.data_shape = (CHUNK_SIZE[MuseDataType.EEG], len(CHANNEL_NAMES[MuseDataType.EEG]))
        self.data_dtype = np.float32
        self.timestamp_shape = (CHUNK_SIZE[MuseDataType.EEG],)
        self.timestamp_dtype = np.float64
        
        # Calculate sizes for verification and seeking
        self.data_frame_size = np.prod(self.data_shape) * np.dtype(self.data_dtype).itemsize # 12 * 4 * 4 bytes
        self.timestamp_frame_size = np.prod(self.timestamp_shape) * np.dtype(self.timestamp_dtype).itemsize # 12 * 8 bytes
        
        self.data_file = open(self.data_file_path, 'rb')
        self.timestamp_file = open(self.timestamp_file_path, 'rb')
        
        # Get file sizes for frame count calculation
        self.data_file.seek(0, 2)  # Seek to end
        data_file_size = self.data_file.tell()
        self.data_file.seek(0)  # Reset to beginning
        
        self.timestamp_file.seek(0, 2)  # Seek to end
        timestamp_file_size = self.timestamp_file.tell()
        self.timestamp_file.seek(0)  # Reset to beginning
        
        # Calculate total frames
        self.total_data_frames = data_file_size // self.data_frame_size
        self.total_timestamp_frames = timestamp_file_size // self.timestamp_frame_size
        
        # Use min of both for consistency
        self.total_frames = min(self.total_data_frames, self.total_timestamp_frames)
        
        # Verify consistency
        if self.total_data_frames != self.total_timestamp_frames:
            print(f"Warning: Mismatch between data frames ({self.total_data_frames}) "
                  f"and timestamp frames ({self.total_timestamp_frames}).")
            print(f"Using {self.total_frames} frames (minimum of both).")
    
    def read_frame(self, frame_index=None):
        # Seek to specific frame if requested
        if frame_index is not None:
            if frame_index < 0 or frame_index >= self.total_frames:
                raise IndexError(f"Frame index {frame_index} out of range (0-{self.total_frames-1})")
            
            self.data_file.seek(frame_index * self.data_frame_size)
            self.timestamp_file.seek(frame_index * self.timestamp_frame_size)
        
        # Read EEG data
        eeg_bytes = self.data_file.read(self.data_frame_size)
        if len(eeg_bytes) < self.data_frame_size:
            return None, None  # EOF reached
        
        # Read timestamp
        timestamp_bytes = self.timestamp_file.read(self.timestamp_frame_size)
        if len(timestamp_bytes) < self.timestamp_frame_size:
            return None, None  # EOF reached
        
        # Convert to numpy arrays
        eeg_data = np.frombuffer(eeg_bytes, dtype=self.data_dtype).reshape(self.data_shape)
        timestamp = np.frombuffer(timestamp_bytes, dtype=self.timestamp_dtype).reshape(self.timestamp_shape)
        
        return eeg_data, timestamp
    
    def read_all_frames(self, safe=True):
        """Reads all frames from the files.
        
        - If `safe=True`, yields frames one by one to avoid high memory usage.
        - If `safe=False`, reads all frames at once and returns numpy arrays.
        """
        self.data_file.seek(0)
        self.timestamp_file.seek(0)

        if safe:
            # Safe mode: Generator-based, memory-efficient
            for _ in range(self.total_frames):
                eeg_data, timestamp = self.read_frame()
                if eeg_data is None or timestamp is None:
                    break  # Stop if EOF is reached
                yield eeg_data, timestamp
        else:
            # Unsafe mode: Load all data at once into memory
            eeg_bytes = self.data_file.read(self.total_frames * self.data_frame_size)
            timestamp_bytes = self.timestamp_file.read(self.total_frames * self.timestamp_frame_size)

            if len(eeg_bytes) < self.total_frames * self.data_frame_size or \
            len(timestamp_bytes) < self.total_frames * self.timestamp_frame_size:
                raise ValueError("Unexpected EOF while reading files")

            eeg_data = np.frombuffer(eeg_bytes, dtype=self.data_dtype).reshape((self.total_frames,) + self.data_shape)
            timestamps = np.frombuffer(timestamp_bytes, dtype=self.timestamp_dtype).reshape((self.total_frames,) + self.timestamp_shape)
            print(eeg_data.shape, timestamps.shape)
            return eeg_data, timestamps  # Returns full dataset in unsafe mode
    
    def get_total_frames(self):
        return self.total_frames
    
    def __del__(self):
        self.data_file.close()
        self.timestamp_file.close()