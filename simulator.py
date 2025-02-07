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

# yasa for sleep staging
# import yasa

def phase_estimator(signal: np.ndarray, fs: int, analysis_len: int, method = 'truncated_wavelet'):
    # wavelet params
    w = 5
    wavelet_width = lambda f: w*fs / (2*f*np.pi)
    wavelet_freqs = np.linspace(0.5, 2, 30)

    trunc_wavelet_len = analysis_len * fs * 2 # double the length of the signal
    trunc_wavelets = [scipy.signal.morlet2(trunc_wavelet_len, wavelet_width(f), w = 5)[:trunc_wavelet_len // 2] for f in wavelet_freqs]

    # dot product of truncated wavelet
    if method == 'truncated_wavelet':
        conv_vals = [np.dot(signal, w) for w in trunc_wavelets]
        max_idx = np.argmax(np.abs(conv_vals))
        # phase = np.angle(conv_vals[max_idx]) % (2 * pi)
        freq = wavelet_freqs[max_idx]
        phase = np.angle(conv_vals[max_idx])

    # dot product of full wavelet
    if method == 'full_wavelet':
        wavelet_len = analysis_len * fs # length of the signal
        wavelets = [scipy.signal.morlet2(wavelet_len, wavelet_width(f), w) for f in wavelet_freqs]
        dot_prod_vals = [np.dot(signal, w) for w in wavelets]
        max_idx = np.argmax(np.abs(dot_prod_vals))
        freq = wavelet_freqs[max_idx]
        conv_vals = np.dot(signal, trunc_wavelets[max_idx])
        phase = np.angle(conv_vals) % (2 * pi)
    
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
    nyquist = fs / 2
    cut = cut / nyquist
    sos = scipy.signal.butter(order, cut, btype = type, output = 'sos')
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
    def __init__(self, phase_estimator: Callable, identify_sw: Callable, butter_filter: Callable, fs: int = 256, window_size: int = 2, wavelet_method = 'truncated_wavelet'):
        self.phase_estimator = phase_estimator
        self.identify_sw = identify_sw
        self.butter_filter = butter_filter
        self.fs = fs
        self.window_size = window_size
        self.window_len = int(window_size * fs)
        self.signal_buffer = deque(maxlen = self.window_len)
        self.wavelet_method = wavelet_method
    @staticmethod
    def generate_eeg_signal(fs: int, duration: int, noise_level=0.2, mode = 'stationary', eeg_path = None):
        t = np.linspace(0, duration, fs * duration, endpoint=False)
        if mode == 'stationary':
            f = 1.25
            test_signal = np.sin(2*np.pi*t / f) 
        elif mode == 'time-varying':
            w_n = 1 + 0.5*np.sin((2*np.pi*t) / duration)
            phi = 2 * np.pi * np.cumsum(w_n) / fs
            # test_signal = np.sin(2*np.pi*t / w_n)
            test_signal = np.sin(phi)
        elif mode == 'eeg':
            if eeg_path is None:
                raise ValueError("Please provide an EEG path")
            raw = np.load(eeg_path)
            duration = len(raw) // fs   # duration in seconds
            test_signal = raw[:fs * duration]
            t = np.linspace(0, duration, fs * duration, endpoint=False)

            # val = 9625600 # from visual inspection
            # test_signal = raw[val:val + fs * duration]
            # test_signal = raw[:fs * duration]
        return test_signal, t
    
    @staticmethod
    def true_phase(signal: np.ndarray, fs: int):

        b, a = scipy.signal.butter(4, 2, 'low', fs=fs)
        filtered = scipy.signal.filtfilt(b, a, signal)
        analytic_signal = scipy.signal.hilbert(filtered)
        true_phase = np.unwrap(np.angle(analytic_signal)) % (2 * pi) 

        return true_phase
    
    def parametric_ci(stim_phases, ci=95):
        mean_phase = np.angle(np.mean(np.exp(1j * stim_phases)))
        R = np.abs(np.mean(np.exp(1j * stim_phases)))  # Mean resultant length
        n = len(stim_phases)

        # Compute Circular Standard Error (CSE)
        CSE = 1 / np.sqrt(n * R)

        # Compute confidence interval bounds
        z = stats.norm.ppf(1 - (1 - ci/100) / 2)  # Critical value for 95% CI
        ci_lower = (mean_phase - z * CSE) % (2 * np.pi)
        ci_upper = (mean_phase + z * CSE) % (2 * np.pi)

        return ci_lower, ci_upper



    def simulate(self, signal: np.ndarray, real_time_step: float = 0.1):
        # backoff period
        backoff = False
        stim_count = 0  
        last_stim_time = -np.inf
        
        step_len = int(real_time_step * self.fs)
        phase_list =   []
        time_indices = []     
        stim_time = [] # time of stimulation

        # filter parameters
        highcut = 2
        lowcut = 8
        sos_low = butter_filter(highcut, 'lowpass', fs)
        sos_high = butter_filter(lowcut, 'highpass', fs)

        # initialize filters
        zi_low = scipy.signal.sosfilt_zi(sos_low)
        zi_high = scipy.signal.sosfilt_zi(sos_high)

        for i in tqdm(range(0, len(signal), step_len)):
            self.signal_buffer.extend(signal[i:i+step_len])
                
            lp_signal, zi_low = scipy.signal.sosfilt(sos_low, self.signal_buffer, zi = zi_low)
            hp_signal, zi_high = scipy.signal.sosfilt(sos_high, self.signal_buffer, zi = zi_high)

            if len(self.signal_buffer) == self.window_len:
                window = np.array(self.signal_buffer)
                phase, freq, conv_vals = self.phase_estimator(window, self.fs, self.window_size, method=self.wavelet_method)
                phase_list.append(phase)
                time_indices.append(i/self.fs)

                # check high-low frequency ratio
                hl_ratio = self.identify_sw(np.array(lp_signal), np.array(hp_signal), self.fs)
                max_amp = np.nanmax(np.abs(window[fs:]))

                if hl_ratio < - 2 and max_amp >  75e-6:
                    target_phase = 2 * pi
                    # delta_t = (target_phase - phase) % (2 * pi) / (2 * pi * freq)
                    delta_t = (target_phase - phase) % (2 * pi) / (2 * pi * freq)
                    if delta_t < 0.1 and not backoff:
                    # queue stim
                        stim_time.append(i / self.fs + delta_t)
                        stim_count += 1

                    if stim_count == 2:
                        backoff = True
                        stim_time.append(i / self.fs + delta_t)
                        last_stim_time = i / self.fs + delta_t
                        stim_count = 0 

                if backoff and (i / self.fs - last_stim_time) >= 5:
                    backoff = False
        
        # unwrap phase
        phase_list = -np.unwrap(phase_list) % (2 * pi)
        return phase_list, time_indices, stim_time
    

# Test accuracy and precision of phase estimation
def phase_hist(signal, stim_trigs, outpath, fs = 256, lowcut = 0.5, highcut = 2, window_size = 2):
    stim_idx = (np.array(stim_trigs) * fs).astype(int)
    # bandpass filter from 0.5 to 2 Hz
    b, a = scipy.signal.butter(4, [lowcut, highcut], 'band', fs=fs)
    filtered = scipy.signal.filtfilt(b, a, signal)
    # compute the analytic signal
    analytic_signal = scipy.signal.hilbert(filtered)
    phase = np.angle(analytic_signal) % (2 * pi)
    # phase of signal at stim_trigs
    stim_phases = phase[stim_idx]
    mean_phase = scipy.stats.circmean(stim_phases)
    std_phase = scipy.stats.circstd(stim_phases)   

    # mean_phase = np.angle(np.mean(np.exp(1j * stim_phases)))
    # R = np.abs(np.mean(np.exp(1j * stim_phases)))  # Mean resultant length
    # n = len(stim_phases)

    # # Compute Circular Standard Error (CSE)
    # CSE = 1 / np.sqrt(n * R)

    # # Compute confidence interval bounds
    # z = scipy.stats.norm.ppf(1 - (1 - 95/100) / 2)  # Critical value for 95% CI
    # ci_lower = (mean_phase - z * CSE) % (2 * np.pi)
    # ci_upper = (mean_phase + z * CSE) % (2 * np.pi)

    _, ci = bootstrap_confidence_interval(stim_phases, num_bootstrap_samples=1000, confidence_level=0.95)
    ci_lower, ci_upper = ci
    ci_lower = mean_phase - ci_lower
    ci_upper = mean_phase + ci_upper
    # plot histogram of phase
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    ax.hist(stim_phases, bins=30)
    ax.set_xlabel('Phase')
    ax.set_title(f'Mean: {mean_phase:.2f}, Std: {std_phase:.2f}\nCI: [{ci_lower:.2f}, {ci_upper:.2f}]')
    plt.savefig(outpath + '.png')
    return 0


    
# modes = ['stationary', 'time-varying', 'eeg']
modes = ['eeg']
wavelet_methods = ['truncated_wavelet']

datadir = '/d/gmi/1/vickili/clas/data/processed_data'

data_paths = [os.path.join(datadir, f) for f in os.listdir(datadir) if f.endswith('.npy')]
data_paths = data_paths[1:]

for wavelet_method in wavelet_methods:
    for mode in modes:
        for data_path in data_paths:
            print(data_path)
            fs = 256
            duration = 20000
            window_size = 2
            real_time_step = 0.1

            sim = Simulator(phase_estimator, identify_sw, butter_filter, fs, window_size, wavelet_method = wavelet_method)
            test_signal, t_signal = sim.generate_eeg_signal(fs, duration, mode=mode, eeg_path = data_path)

            true_phase = sim.true_phase(test_signal, fs)
            estimated_phases, phase_times, stim_time = sim.simulate(test_signal, real_time_step)
            phase_hist(test_signal, stim_time, os.path.basename(data_path).split('.')[0])
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