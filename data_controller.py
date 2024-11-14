import h5py
import numpy as np

class DataWriter:
    def __init__(self, file_name: str, num_channels: int, nominal_chunk_size: int = 12):
        """
        Initialize the EEGDataWriter class.

        Parameters:
        - file_name (str): The name of the HDF5 file to store the data.
        - num_channels (int): Number of EEG channels.
        - nominal_chunk_size (int): Approximate size of each data chunk.
        """
        self.file_name = file_name
        self.num_channels = num_channels
        self.nominal_chunk_size = nominal_chunk_size
        
        # Open the HDF5 file and set up datasets if they don't exist
        with h5py.File(self.file_name, "a") as f:
            if "timestamps" not in f:
                f.create_dataset(
                    "timestamps", (0,), maxshape=(None,), dtype="f8",
                    chunks=(self.nominal_chunk_size,), compression="gzip"
                )
                f.create_dataset(
                    "data", (0, self.num_channels), maxshape=(None, self.num_channels),
                    dtype="f4", chunks=(self.nominal_chunk_size, self.num_channels), compression="gzip"
                )

    def write_data(self, timestamps_arr: np.ndarray, data_arr: np.ndarray):
        """
        Append a new chunk of timestamps and EEG data to the HDF5 file.

        Parameters:
        - timestamps_arr (np.ndarray): Array of timestamp data for the current chunk.
        - data_arr (np.ndarray): 2D array of EEG data for the current chunk.
        """
        with h5py.File(self.file_name, "a") as f:
            timestamps = f["timestamps"]
            data = f["data"]

            # Resize datasets to accommodate the new chunk
            timestamps.resize(timestamps.shape[0] + len(timestamps_arr), axis=0)
            data.resize(data.shape[0] + data_arr.shape[0], axis=0)

            # Append new data
            timestamps[-len(timestamps_arr):] = timestamps_arr
            data[-data_arr.shape[0]:] = data_arr