from dataclasses import dataclass
from math import isnan, nan, pi

import numpy as np

from .config import CLASAlgoConfig
from .constants import CHUNK_SIZE, SAMPLING_RATE, CLASAlgoResultType, MuseDataType


@dataclass
class CLASAlgoResult:
    type: CLASAlgoResultType
    time_to_target: float
    stim_time: float
    phase: float
    freq: float
    quadrature: float

    current_amp: float
    mean_amp: float
    current_hl_ratio: float
    mean_hl_ratio: float

class CLASAlgo:
    def __init__(self, config: CLASAlgoConfig):
        self.config = config
        self.last_stim = 0.0
        self.processor_elapsed_time = 0.0

        self.target_phase = self.config.target_phase
        
        self.second_stim_start = nan
        self.second_stim_end = nan

        self.processing_window_len_n = int(SAMPLING_RATE[MuseDataType.EEG] * self.config.processing_window_len_s)
        self.quadrature_window_len_n = int(SAMPLING_RATE[MuseDataType.EEG] * self.config.quadrature_len_s)
        
        self.amp_buffer = np.zeros(self.config.amp_buffer_len)
        self.hl_ratio_buffer = np.zeros(self.config.hl_ratio_buffer_len)

        self.t_wavelet_freqs = np.linspace(self.config.t_wavelet_freq_range[0], self.config.t_wavelet_freq_range[1], self.config.t_wavelet_N)
        self.t_wavelet_arr = [self.gen_tmorlet2(f) for f in self.t_wavelet_freqs]
        
        self.hl_wavelet_arr = [self.gen_tmorlet2(f) for f in self.config.hl_ratio_wavelet_freqs]

        self.selected_channel_ind = 1 # AF7

    def gen_tmorlet2(self, f: float):
        s = self.config.w * SAMPLING_RATE[MuseDataType.EEG] / (2 * pi * f)
        x = np.arange(0, self.processing_window_len_n * 2) - (self.processing_window_len_n * 2 - 1.0) / 2
        x /= s
        wavelet = np.exp(1j * self.config.w * x) * np.exp(-0.5 * x**2) * np.pi**(-0.25)
        output = np.sqrt(1 / s) * wavelet

        return output[:self.processing_window_len_n]

    def update(self, processor_elapsed_time, data):
        self.processor_elapsed_time = processor_elapsed_time

        if self.second_stim_end < self.processor_elapsed_time:
            # go back to normal functioning
            self.second_stim_start = nan
            self.second_stim_end = nan

        ### estimate phase ###
        phase, freq, current_amp, quadrature, current_hl_ratio = self.estimate(data)

        # roll amplitude buffer
        self.amp_buffer[:-1] = self.amp_buffer[1:]
        self.amp_buffer[-1] = current_amp
        mean_amp = self.amp_buffer.mean()

        self.hl_ratio_buffer[:-1] = self.hl_ratio_buffer[1:]
        self.hl_ratio_buffer[-1] = current_hl_ratio
        mean_hl_ratio = self.hl_ratio_buffer.mean()

        # check if we're waiting for the 2nd stim
        # if NOT, run normal checks
        if isnan(self.second_stim_start):
            ### check backoff criteria ###
            if ((self.last_stim + self.config.backoff_time) > (self.processor_elapsed_time + self.config.stim1_prediction_limit_sec)):
                return CLASAlgoResult(CLASAlgoResultType.BACKOFF, 0, 0, phase, freq, current_amp, mean_amp, current_hl_ratio, mean_hl_ratio, quadrature)

            ### check amplitude criteria ###
            if (mean_amp < self.config.amp_threshold) or (mean_amp > self.config.amp_limit):
                return CLASAlgoResult(CLASAlgoResultType.AMPLITUDE, 0, 0, phase, freq, current_amp, mean_amp, current_hl_ratio, mean_hl_ratio, quadrature)

            ### check quadrature ###
            if (quadrature is not None) and (quadrature < self.config.quadrature_thresh):
                return CLASAlgoResult(CLASAlgoResultType.QUADRATURE, 0, 0, phase, freq, current_amp, mean_amp, current_hl_ratio, mean_hl_ratio, quadrature)


            if (mean_hl_ratio > self.config.hl_ratio_buffer_threshold) or (current_hl_ratio > self.config.hl_ratio_latest_threshold):
                return CLASAlgoResult(CLASAlgoResultType.HL_RATIO, 0, 0, phase, freq, current_amp, mean_amp, current_hl_ratio, mean_hl_ratio, quadrature)

        # if we are waiting for 2nd stim, but before the backoff window, only use phase targeting
        if self.processor_elapsed_time < self.second_stim_start:
            return CLASAlgoResult(CLASAlgoResultType.BACKOFF2, 0, 0, phase, freq, current_amp, mean_amp, current_hl_ratio, mean_hl_ratio, quadrature)

        ### perform forward prediction ###
        delta_t = ((self.target_phase - phase) % (2 * pi)) / (freq * 2 * pi)

        # cue a stim for the next target phase
        if isnan(self.second_stim_start):
            if delta_t > self.config.stim1_prediction_limit_sec:
                return CLASAlgoResult(CLASAlgoResultType.FUTURE, delta_t, 0, phase, freq, current_amp, mean_amp, current_hl_ratio, mean_hl_ratio, quadrature)

            self.last_stim = self.processor_elapsed_time + delta_t  # update stim time to compute backoff
            self.second_stim_start = self.last_stim + self.config.stim2_start_delay  # update
            self.second_stim_end = self.last_stim + self.config.stim2_end_delay

            return CLASAlgoResult(CLASAlgoResultType.STIM, delta_t, self.last_stim, phase, freq, current_amp, mean_amp, current_hl_ratio, mean_hl_ratio, quadrature)

        else:
            if delta_t > self.config.stim2_prediction_limit_sec:
                return CLASAlgoResult(CLASAlgoResultType.FUTURE2, delta_t, 0, phase, freq, current_amp, mean_amp, current_hl_ratio, mean_hl_ratio, quadrature)

            self.second_stim_start = nan
            self.second_stim_end = nan

            return CLASAlgoResult(CLASAlgoResultType.STIM2, delta_t, self.processor_elapsed_time + delta_t, phase, freq, current_amp, mean_amp, current_hl_ratio, mean_hl_ratio, quadrature) 


    def estimate(self, data):
        # convolve the list of wavelets
        conv_vals = [np.dot(data, w) for w in self.t_wavelet_arr]

        # choose the one with highest amp/phase
        amp_conv_vals = np.abs(conv_vals)
        amp_max = np.argmax(amp_conv_vals)

        # create outputs
        amp = amp_conv_vals[amp_max] / 2
        freq = self.t_wavelet_freqs[amp_max]
        phase = np.angle(conv_vals[amp_max])

        ### high low ratio ###
        conv_vals_hl = [np.dot(data, w) for w in self.hl_wavelet_arr]
        hf_amp = np.mean(np.abs(conv_vals_hl))
        hl_ratio = hf_amp / np.mean(np.abs(conv_vals))

        ### determine if we're locked on ###
        est_phase = (np.arange(self.quadrature_window_len_n) / SAMPLING_RATE[MuseDataType.EEG]) * freq * 2 * pi
        est_phase = est_phase - est_phase[-1] + phase
        est_sig = np.cos(est_phase)
        est_sig = est_sig / np.trapz(np.abs(est_sig)) * self.quadrature_window_len_n

        # normalize the signal
        normsig = data[-self.quadrature_window_len_n:] / np.trapz(
            np.abs(data[-self.quadrature_window_len_n:])) * self.processing_window_len_n
        quadrature = np.trapz(normsig * est_sig) / self.processing_window_len_n

        return phase, freq, amp, quadrature, hl_ratio
