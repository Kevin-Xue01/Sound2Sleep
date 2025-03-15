import os
import numpy as np
from typing import Callable

# phase estimator imports
import scipy.signal 
import scipy.stats
from math import pi

# data generation
from collections import deque
from math import pi
from typing import Callable

# visualization 
import matplotlib.pyplot as plt
import numpy as np

# phase estimator imports
import scipy.signal
import scipy.stats
from tqdm import tqdm  # progress bar

class Simulator():
    wavelet_freqs:list
    trunc_wavelets:list

    def __init__(self, fs: int = 256, window_size: int = 2):
        self.fs = fs
        self.window_size = window_size
        self.analysis_len = window_size
        self.window_len = int(window_size * fs)
        self.signal_buffer = deque(maxlen = self.window_len)

    def phase_estimator(self, signal: np.ndarray): 
        # dot product of truncated wavelet
        conv_vals = [np.dot(signal, w) for w in self.trunc_wavelets]
        max_idx = np.argmax(np.abs(conv_vals))
        freq = self.wavelet_freqs[max_idx]
        phase = np.angle(conv_vals[max_idx]) % (2 * pi)
        return phase, freq, conv_vals

    @staticmethod
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
        
    @staticmethod
    def butter_filter(cut, type, fs, order = 4):
        sos = scipy.signal.butter(order, cut, btype = type, output = 'sos', fs = fs)
        return sos

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
        target_phase = 6 * np.pi / 4
        
        step_len = int(real_time_step * self.fs)
        phase_list =   []
        time_indices = []     
        stim_time = [] # time of stimulation
        stim_freqs = []

        # filter parameters
        lowband = [1, 4]
        highband = [12, 80] 
        sos_low = self.butter_filter(lowband, 'bandpass', self.fs)
        sos_high = self.butter_filter(highband, 'bandpass', self.fs)

        # initialize filters
        zi_low = scipy.signal.sosfilt_zi(sos_low)
        zi_high = scipy.signal.sosfilt_zi(sos_high)

        # initialize wavelets
        w = 5
        wavelet_width = lambda f: w*self.fs / (2*f*np.pi)
        self.wavelet_freqs = np.linspace(0.4, 2.1, 34)

        trunc_wavelet_len = self.analysis_len * self.fs * 2 # double the length of the signal
        self.trunc_wavelets = [scipy.signal.morlet2(trunc_wavelet_len, wavelet_width(f), w = 5)[:trunc_wavelet_len // 2] for f in self.wavelet_freqs]

        for i in tqdm(range(0, len(signal), step_len)):
            self.signal_buffer.extend(signal[i:i+step_len])
                
            lp_signal, zi_low = scipy.signal.sosfilt(sos_low, self.signal_buffer, zi = zi_low)
            hp_signal, zi_high = scipy.signal.sosfilt(sos_high, self.signal_buffer, zi = zi_high)

            if len(self.signal_buffer) == self.window_len:
                window = np.array(self.signal_buffer)

                # check high-low frequency ratio
                hl_ratio = self.identify_sw(np.array(lp_signal), np.array(hp_signal), self.fs)
                if hl_ratio > -2:
                    continue
                max_amp = np.nanmax(np.abs(window[self.fs:]))
                if max_amp < 80e-6:
                    continue

                phase, freq, conv_vals = self.phase_estimator(window)
                phase_list.append(phase)
                time_indices.append(i/self.fs)

                if freq < 0.5 or freq > 2:
                    continue

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
    ax.hist(stim_phases, bins=30, color='slateblue')
    ax.set_xticks([ax.get_xticks()[-1]])
    ax.grid(True, alpha = 0.2)
    ax.set_xlabel('Phase')
    sub = outpath.split('_raw')[0]
    ax.set_title(f'TWave - {sub}\nMean: {mean_phase:.2f}, Std: {std_phase:.2f}')
    ax = plt.subplot(212, polar=False)
    ax.hist(stim_freqs, bins=30, color = 'slateblue')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Count')
    # removespine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(os.path.join('twave', outpath + '.png'))

    # plot stim phases broken up by phase bin
    bins = np.linspace(7*np.pi/4 - np.pi/2, 7 * np.pi / 4 + np.pi/2, 10)
    for bin in bins:
        indices = np.array([idx for idx in stim_idx if (phase[idx] >= bin - np.pi/10) and (phase[idx] < bin + np.pi/10)])
        # drop indices that are within a second of the end of the signal
        indices = indices[indices < len(signal) - fs]
        indices = indices[indices > fs]

        if len(indices) > 0:
            start_indices = np.array(indices - fs)
            end_indices = np.array(indices + fs)
            phase_aligned_signal = np.array([signal[start:end] for start, end in zip(start_indices, end_indices)])
        else:
            continue
        lower_quartile = np.percentile(phase_aligned_signal, 25, axis = 0)
        upper_quartile = np.percentile(phase_aligned_signal, 75, axis = 0)
        mean_signal = np.nanmean(phase_aligned_signal, axis = 0)

        plt.figure(figsize=(6, 4))
        plt.fill_between(np.arange(len(mean_signal)) / fs - 1, lower_quartile, upper_quartile, alpha = 0.2, color = 'slateblue')
        plt.plot(np.arange(len(mean_signal)) / fs - 1, mean_signal * 1e6, linewidth = 2, color = 'slateblue')
        plt.axvline(linestyle = '--', alpha = 0.1)
        plt.ylim(-150, 150)
        plt.xlabel('Time')
        plt.ylabel('Amplitude (μV)')
        plt.title(f'Phase: {bin:.2f} \n {len(indices)} stimulations')
        # revmoe spine
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        output_dir = os.path.join('twave', outpath)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'phase_{bin:.2f}.png'))
        plt.close()

    return stim_phases, mean_signal

    

# modes = ['stationary', 'time-varying', 'eeg']
modes = ['eeg']

datadir = '/d/gmi/1/vickili/clas/data/processed_data'
# musedir
# datadir = '/d/gmi/1/vickili/clas/muse_data'

data_paths = [os.path.join(datadir, f) for f in os.listdir(datadir) if f.endswith('.npy')]

all_stim_phases = []
all_stim_freqs = []

mean_signals = []
for mode in modes:
    for data_path in data_paths:
        print(data_path)
        fs = 256
        duration = None
        window_size = 2
        real_time_step = 0.1

        sim = Simulator(fs, window_size)
        test_signal, t_signal = sim.generate_eeg_signal(fs, duration, mode=mode, eeg_path = data_path)

        true_phase = sim.true_phase(test_signal, fs)
        estimated_phases, phase_times, stim_time, stim_freqs = sim.simulate(test_signal, real_time_step)
        stim_phases, mean_signal = phase_hist(test_signal, stim_time, os.path.basename(data_path).split('.')[0], stim_freqs)
        all_stim_phases.append(stim_phases)
        all_stim_freqs.append(stim_freqs)
        mean_signals.append(mean_signal)

# calcualte mean and standard deviation of all stim phases
all_stim_phases = np.concatenate(all_stim_phases)
mean_phase = scipy.stats.circmean(all_stim_phases)
std_phase = scipy.stats.circstd(all_stim_phases)

# plot histogram of frequencies
# plt.figure(figsize=(8, 8))
# ax = plt.subplot(111, polar=False)
# ax.hist(np.concatenate(all_stim_freqs), bins=30, color = 'slateblue')
# ax.set_xlabel('Frequency')
# ax.set_ylabel('Count')
# # removespine
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.savefig(os.path.join('twave', 'all_frequency_histogram.png'))

# save all stim phases
np.save(os.path.join('simulation_results', 'twave_stim_phases.npy'), all_stim_phases)
np.save(os.path.join('simulation_results', 'twave_stim_freqs.npy'), all_stim_freqs)

# plot histogram of phase
plt.figure(figsize=(3, 3))
ax = plt.subplot(111, polar=True)
ax.hist(all_stim_phases, bins=30, color='slateblue', edgecolor='black', alpha=0.75)
ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
ax.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$'], fontsize=10)
ax.set_title(f'Mean: {mean_phase:.2f}, Std: {std_phase:.2f}', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)
# ax.set_rticks([-1])  # Fewer radial ticks
ax.set_rticks([ax.get_yticks()[-1]])
ax.grid(True, linestyle='--', alpha=0.6)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.savefig(os.path.join('simulation_results', 'twave.pdf'), dpi=300, transparent=True)
plt.close()



# plot mean signal
mean_signals = np.concatenate(mean_signals)
lower_percentile = np.percentile(mean_signals, 25, axis=0)
upper_percentile = np.percentile(mean_signals, 75, axis=0)
plt.figure(figsize=(10, 10))
plt.fill_between(np.arange(len(mean_signals)) / fs - 1.5, lower_percentile, upper_percentile, alpha = 0.2, color = 'slateblue')
plt.plot(np.arange(len(mean_signals)) / fs - 1.5, mean_signals * 1e6, linewidth = 2, color = 'slateblue')
plt.axvline(linestyle = '--', alpha = 0.1)
plt.ylim(-150, 150)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (μV)')
plt.title('Average Signal')
plt.tight_layout()
plt.savefig(os.path.join('simulation_results', 'twave_mean_signal.pdf'), dpi=300, transparent=True)
plt.close()