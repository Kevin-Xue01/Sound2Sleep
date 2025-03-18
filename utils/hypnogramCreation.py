import os
import numpy as np
import mne
import yasa


def hypnogramCreation(input_directory, chosen_channel="Ch2",num_channels=4, sfreq=256, epoch_length=30):
    """
    Processes EEG data, performs filtering, and predicts sleep stages.

    Parameters:
    - input_directory (str): Directory containing EEG binary files.
    - num_channels (int): Number of EEG channels.
    - sfreq (int): Sampling frequency in Hz.
    - epoch_length (int): Length of each epoch in seconds for sleep staging.

    Returns:
    - hypnogram_sec (np.ndarray): Predicted sleep stages in second-based format.
    """
    # Define file paths
    eeg_data_file = os.path.join(input_directory, "EEG_data.bin")
    timestamp_file = os.path.join(input_directory, "EEG_timestamp.bin")

    # Define data types
    data_dtype = np.float32  # EEG data stored as 32-bit floats
    timestamp_dtype = np.float64  # Timestamps stored as 64-bit floats

    # Read EEG Data
    def read_eeg_data(filename):
        with open(filename, "rb") as f:
            eeg_data = np.fromfile(f, dtype=data_dtype)
            eeg_data = eeg_data.reshape(-1, num_channels)  # Reshape based on channels
        return eeg_data.T  # Transpose for MNE (channels x time)

    # Read Timestamps
    def read_timestamps(filename):
        with open(filename, "rb") as f:
            timestamps = np.fromfile(f, dtype=timestamp_dtype)
        return timestamps

    # Load EEG data and timestamps
    eeg_data = read_eeg_data(eeg_data_file)
    timestamps = read_timestamps(timestamp_file)

    # Create MNE Info object
    ch_names = [f"Ch{i + 1}" for i in range(num_channels)]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")

    # Create MNE RawArray object
    raw = mne.io.RawArray(eeg_data, info)

    # Apply a bandpass filter (optional but recommended for sleep staging)
    raw.filter(0.3, 40., fir_design="firwin")

    # Predict sleep stages with YASA
    sls = yasa.SleepStaging(raw, eeg_name=chosen_channel)  # Assuming Ch1 is a reliable sleep channel
    hypnogram = sls.predict()

    # Convert hypnogram to YASA format (seconds-based)
    hypnogram_sec = yasa.hypno_str_to_int(hypnogram)

    return hypnogram_sec