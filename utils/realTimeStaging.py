import numpy as np
import mne
import yasa


def predict_sleep_stage_from_epoch(epoch_data, sfreq=256, eeg_name='Channel 2'):
    """
    Predicts the sleep stage for a 30-second epoch of EEG data.

    Parameters:
    - epoch_data: 1D array - The EEG data for a single 30-second epoch.
    - sfreq: The sampling frequency of the EEG data (default: 256 Hz).
    - eeg_name: The name of the EEG channel (default: 'Channel 2').

    Returns:
    - predicted_stage: The predicted sleep stage for the given epoch.
    """
    # Ensure the epoch is 30 seconds (if shorter, pad with zeros)
    target_length = sfreq * 30  # 30 seconds worth of samples
    if len(epoch_data) < target_length:
        epoch_data = np.pad(epoch_data, (0, target_length - len(epoch_data)), 'constant')

    # Convert the epoch data into an MNE Raw object
    info = mne.create_info([eeg_name], sfreq=sfreq, ch_types=['eeg'])
    raw_epoch = mne.io.RawArray(epoch_data[np.newaxis, :], info)

    # Perform sleep staging using yasa
    ls = yasa.SleepStaging(raw_epoch, eeg_name=eeg_name)
    hypno_pred = ls.predict()

    return hypno_pred[0]  # Return the predicted stage for this epoch
