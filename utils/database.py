import numpy as np

from .constants import CHUNK_SIZE, MuseDataType


class FileWriter:
    def __init__(self, log_dir):
        self.eeg_file = open(log_dir + '.eeg.bin', 'wb')
    
    def write_eeg_data(self, timestamp: np.ndarray, eeg_data: np.ndarray):
        self.eeg_file.write(timestamp.tobytes())
        self.eeg_file.write(eeg_data.tobytes())
    
    def close_files(self):
        """Closes the binary file explicitly."""
        if not self.eeg_file.closed:
            self.eeg_file.close()

    def __del__(self):
        """Destructor to ensure file closure when object is deleted."""
        self.close_files()

class FileReader:
    def __init__(self, log_dir):  # Default chunk size based on your use case
        self.log_dir = log_dir + '.eeg.bin'

    def stream_eeg_data(log_dir):
        """
        Generator that streams timestamp and EEG data from a binary file.
        
        - log_dir: Name of the EEG binary file (without .eeg.bin extension).
        - chunk_size: The number of rows per chunk (must match how data was written).
        
        Yields:
            (timestamps, eeg_data) tuple, where:
            - timestamps: NumPy array of shape (chunk_size,)
            - eeg_data: NumPy array of shape (chunk_size, 4)
        """
        timestamp_dtype = np.float64  # 8 bytes per timestamp
        eeg_dtype = np.float32        # 4 bytes per EEG value

        timestamp_size = CHUNK_SIZE[MuseDataType.EEG] * np.dtype(timestamp_dtype).itemsize  # 12 * 8 bytes
        eeg_size = CHUNK_SIZE[MuseDataType.EEG] * 4 * np.dtype(eeg_dtype).itemsize  # 12 * 4 * 4 bytes

        with open(log_dir + ".eeg.bin", "rb") as f:
            while True:
                # Read one full chunk of timestamps
                timestamp_bytes = f.read(timestamp_size)
                if len(timestamp_bytes) < timestamp_size:  # End of file
                    break
                timestamps = np.frombuffer(timestamp_bytes, dtype=timestamp_dtype)

                # Read corresponding EEG data chunk
                eeg_bytes = f.read(eeg_size)
                if len(eeg_bytes) < eeg_size:
                    break
                eeg_data = np.frombuffer(eeg_bytes, dtype=eeg_dtype).reshape(CHUNK_SIZE[MuseDataType.EEG], 4)

                yield timestamps, eeg_data  # Yield the complete chunk