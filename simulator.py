import os
import numpy as np
from typing import Callable

# phase estimator imports
import scipy.signal 
import scipy.stats
from math import pi

# data generation
from collections import deque

# visualization 
import matplotlib.pyplot as plt

from tqdm import tqdm # progress bar

def phase_estimator(signal: np.ndarray, wavelet_freqs: list, trunc_wavelets: list, fs: int, analysis_len: int): 
    # dot product of truncated wavelet
    conv_vals = [np.dot(signal, w) for w in trunc_wavelets]
    max_idx = np.argmax(np.abs(conv_vals))
    # phase = np.angle(conv_vals[max_idx]) % (2 * pi)
    freq = wavelet_freqs[max_idx]
    phase = np.angle(conv_vals[max_idx]) % (2 * pi)
    
    return phase, freq, conv_vals

def identify_sw(filtered_signal_lp: np.ndarray, filtered_signal_hp: np.ndarray, fs: int):
    filtered_signal_lp = filtered_signal_lp[1 * fs:]
    filtered_signal_hp = filtered_signal_hp[1 * fs:]
    # compute the lp envelope of the signal
    envelope_lp = np.abs(scipy.signal.hilbert(filtered_signal_lp))
    power_lf = envelope_lp**2
    # compute the hf envelope of the signal
    envelope_hf = np.abs(scipy.signal.hilbert(filtered_signal_hp))
    power_hf = envelope_hf**2
    # compute ratio and store
    hl_ratio = np.mean(power_hf) / np.mean(power_lf)
    hl_ratio = np.log10(hl_ratio)
    return hl_ratio
    
def butter_filter(cut, type, fs, order = 4):
    # nyquist = fs / 2
    # cut = [x / nyquist for x in cut]
    sos = scipy.signal.butter(order, cut, btype = type, output = 'sos', fs = fs)
    return sos

def bootstrap_confidence_interval(data, num_bootstrap_samples=1000, confidence_level=0.95):
    bootstrap_means = []
    n = len(data)
    
    for _ in range(num_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile
    lower_bound = np.percentile(bootstrap_means, lower_percentile * 100)
    upper_bound = np.percentile(bootstrap_means, upper_percentile * 100)
    
    return np.mean(data), (lower_bound, upper_bound)


class Simulator():
    def __init__(self, phase_estimator: Callable, identify_sw: Callable, butter_filter: Callable, fs: int = 256, window_size: int = 2):
        self.phase_estimator = phase_estimator
        self.identify_sw = identify_sw
        self.butter_filter = butter_filter
        self.fs = fs
        self.window_size = window_size
        self.window_len = int(window_size * fs)
        self.signal_buffer = deque(maxlen = self.window_len)

    @staticmethod
    def generate_eeg_signal(fs: int, duration: int, noise_level=0.2, mode = 'stationary', eeg_path = None):
        if mode == 'stationary':
            f = 1.25
            test_signal = np.sin(2*np.pi*t / f)
            t = np.linspace(0, duration, fs * duration, endpoint=False) 
        elif mode == 'time-varying':
            w_n = 1 + 0.5*np.sin((2*np.pi*t) / duration)
            phi = 2 * np.pi * np.cumsum(w_n) / fs
            test_signal = np.sin(phi)
            t = np.linspace(0, duration, fs * duration, endpoint=False)
        elif mode == 'eeg':
            if eeg_path is None:
                raise ValueError("Please provide an EEG path")
            if duration == None:
                raw = np.load(eeg_path)
                duration = len(raw) // fs   # duration in seconds
                test_signal = raw[:fs * duration]
                t = np.linspace(0, duration, fs * duration, endpoint=False)
            else:
                val = 9625600 # from visual inspection
                test_signal = raw[val:val + fs * duration]
                test_signal = raw[:fs * duration]
                t = np.linspace(0, duration, fs * duration, endpoint=False)
        return test_signal, t
    
    @staticmethod
    def true_phase(signal: np.ndarray, fs: int):
        lowcut = 0.5
        highcut = 2
        # nyquist = fs / 2
        # lowcut = lowcut / nyquist
        # highcut = highcut / nyquist
        b, a = scipy.signal.butter(4, (lowcut, highcut), btype = 'bandpass', fs = fs)
        filtered = scipy.signal.filtfilt(b, a, signal)
        analytic_signal = scipy.signal.hilbert(filtered)
        true_phase = np.unwrap(np.angle(analytic_signal)) % (2 * pi) 

        return true_phase

    def simulate(self, signal: np.ndarray, real_time_step: float = 0.1):
        # backoff period
        backoff = False
        stim_count = 0  
        last_stim_time = -np.inf
        
        step_len = int(real_time_step * self.fs)
        phase_list =   []
        time_indices = []     
        stim_time = [] # time of stimulation
        stim_freqs = []

        # filter parameters
        lowband = [1, 4]
        highband = [12, 80] 
        sos_low = butter_filter(lowband, 'bandpass', fs)
        sos_high = butter_filter(highband, 'bandpass', fs)

        # initialize filters
        zi_low = scipy.signal.sosfilt_zi(sos_low)
        zi_high = scipy.signal.sosfilt_zi(sos_high)

        # initialize wavelets
        w = 5
        wavelet_width = lambda f: w*fs / (2*f*np.pi)
        wavelet_freqs = np.linspace(0.5, 2, 30)

        analysis_len = self.window_size
        trunc_wavelet_len = analysis_len * fs * 2 # double the length of the signal
        trunc_wavelets = [scipy.signal.morlet2(trunc_wavelet_len, wavelet_width(f), w = 5)[:trunc_wavelet_len // 2] for f in wavelet_freqs]

        for i in tqdm(range(0, len(signal), step_len)):
            self.signal_buffer.extend(signal[i:i+step_len])
                
            lp_signal, zi_low = scipy.signal.sosfilt(sos_low, self.signal_buffer, zi = zi_low)
            hp_signal, zi_high = scipy.signal.sosfilt(sos_high, self.signal_buffer, zi = zi_high)

            if len(self.signal_buffer) == self.window_len:
                window = np.array(self.signal_buffer)
                phase, freq, conv_vals = self.phase_estimator(window, wavelet_freqs, trunc_wavelets, self.fs, self.window_size)
                phase_list.append(phase)
                time_indices.append(i/self.fs)

                # check high-low frequency ratio
                hl_ratio = self.identify_sw(np.array(lp_signal), np.array(hp_signal), self.fs)
                max_amp = np.nanmax(window[fs:])

                if hl_ratio < - 2 and max_amp >  80e-6:
                    target_phase = 2 * pi
                    delta_t = (target_phase - phase) % (2 * pi) / (2 * pi * freq)
                    if delta_t < 0.1 and not backoff:
                    # queue stim
                        stim_time.append(i / self.fs + delta_t)
                        stim_freqs.append(freq)
                        stim_count += 1

                        if stim_count == 2:
                            backoff = True
                            last_stim_time = i / self.fs + delta_t
                            stim_count = 0 

                if backoff and (i / self.fs - last_stim_time) >= 5:
                    backoff = False
        
        # unwrap phase
        phase_list = -np.unwrap(phase_list) % (2 * pi)
        return phase_list, time_indices, stim_time, stim_freqs
    

# Test accuracy and precision of phase estimation
def phase_hist(signal, stim_trigs, outpath, stim_freqs, fs = 256, lowcut = 0.5, highcut = 2, window_size = 2):
    stim_idx = (np.array(stim_trigs) * fs).astype(int)
    # bandpass filter from 0.5 to 2 Hz
    # nyquist = fs / 2
    # lowcut = lowcut / nyquist
    # highcut = highcut / nyquist
    b, a = scipy.signal.butter(4, [lowcut, highcut], 'bandpass', fs = fs)
    filtered = scipy.signal.filtfilt(b, a, signal)
    # compute the analytic signal
    analytic_signal = scipy.signal.hilbert(filtered)
    phase = np.angle(analytic_signal) % (2 * pi)
    # phase of signal at stim_trigs
    stim_phases = phase[stim_idx]
    mean_phase = scipy.stats.circmean(stim_phases)
    std_phase = scipy.stats.circstd(stim_phases)   

    # plot histogram of phase
    plt.figure(figsize=(8, 10))
    ax = plt.subplot(211, polar=True)
    ax.hist(stim_phases, bins=30, color='skyblue')
    ax.set_xticks([ax.get_xticks()[-1]])
    ax.grid(True, alpha = 0.2)
    ax.set_xlabel('Phase')
    sub = outpath.split('_raw')[0]
    ax.set_title(f'TWave - {sub}\nMean: {mean_phase:.2f}, Std: {std_phase:.2f}')
    ax = plt.subplot(212, polar=False)
    ax.hist(stim_freqs, bins=30, color = 'skyblue')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Count')
    # removespine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(os.path.join('twave', outpath + '.png'))
    return stim_phases

    
# modes = ['stationary', 'time-varying', 'eeg']
modes = ['eeg']

datadir = '/d/gmi/1/vickili/clas/data/processed_data'
# musedir
# datadir = '/d/gmi/1/vickili/clas/muse_data'

data_paths = [os.path.join(datadir, f) for f in os.listdir(datadir) if f.endswith('.npy')]

all_stim_phases = []
all_stim_freqs = []
for mode in modes:
    for data_path in data_paths:
        print(data_path)
        fs = 256
        duration = None
        window_size = 2
        real_time_step = 0.1

        sim = Simulator(phase_estimator, identify_sw, butter_filter, fs, window_size)
        test_signal, t_signal = sim.generate_eeg_signal(fs, duration, mode=mode, eeg_path = data_path)

        true_phase = sim.true_phase(test_signal, fs)
        estimated_phases, phase_times, stim_time, stim_freqs = sim.simulate(test_signal, real_time_step)
        stim_phases = phase_hist(test_signal, stim_time, os.path.basename(data_path).split('.')[0], stim_freqs)
        all_stim_phases.append(stim_phases)
        all_stim_freqs.append(stim_freqs)

# calcualte mean and standard deviation of all stim phases
all_stim_phases = np.concatenate(all_stim_phases)
mean_phase = scipy.stats.circmean(all_stim_phases)
std_phase = scipy.stats.circstd(all_stim_phases)


# plot histogram of phase
plt.figure(figsize=(8, 10))
ax = plt.subplot(211, polar=True)
ax.hist(all_stim_phases, bins=30, color='skyblue')
plt.title(f'TWave - multi-subject\nMean: {mean_phase:.2f}, Std: {std_phase:.2f}')
ax.set_xlabel('Phase')
ax.grid(True, alpha = 0.2)
ax = plt.subplot(212, polar=False)
ax.hist(np.concatenate(all_stim_freqs), bins=30, color = 'skyblue')
ax.set_xlabel('Frequency')
ax.set_ylabel('Count')
# removespine
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join('twave', 'all.png'))


#         # Interpolate the true phase to match the estimated phase times
#         true_phase_interp = np.interp(phase_times, t_signal, true_phase)

#         # plot unwrapped signals
#         plt.figure(figsize=(14, 7))
#         plt.plot(phase_times, np.unwrap(true_phase_interp), label='True Phase (Interpolated)')
#         plt.plot(phase_times, np.unwrap(estimated_phases), label='Estimated Phase')
#         plt.savefig(f'unwrapped_signals_{mode}.png')

#         errors = np.abs(np.unwrap(true_phase_interp) - np.unwrap(estimated_phases))  # Phase errors
    

#         # Visualization
#         plt.figure(figsize=(14, 7))

#         # Plot EEG signal
#         plt.subplot(2, 1, 1)
#         plt.plot(t_signal, test_signal, label="EEG Signal", alpha=0.7)
#         for stim in stim_time:
#             plt.axvline(stim, color='r', linestyle='--', alpha=0.5)
#         plt.xlabel("Time (s)")
#         plt.ylabel("Amplitude")
#         plt.title("Synthetic EEG Signal")

#         # Plot true phase vs. estimated phase
#         plt.subplot(2, 1, 2)
#         plt.plot(t_signal, true_phase, label="True Phase", alpha=0.4)
#         # plt.step(phase_times, estimated_phases, label="Estimated Phase (Real-Time)", where="post", alpha=0.4)
#         plt.plot(phase_times, estimated_phases, label="Estimated Phase", alpha=0.4)
#         plt.xlabel("Time (s)")
#         plt.ylabel("Phase (radians)")
#         plt.title("True Phase vs. Estimated Phase")
#         plt.legend()

#         plt.tight_layout()
#         plt.savefig(f'simulated_phase_estimation_{mode}_{wavelet_method}.png')