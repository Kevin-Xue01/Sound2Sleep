from math import ceil, floor, isnan, nan, pi
from typing import Tuple, Union

# pip
import numpy as np
import scipy
import scipy.signal
from PyQt5.QtCore import QMutex, QObject, QTimer, pyqtSignal

from utils import (
    CHANNEL_NAMES,
    SAMPLING_RATE,
    ExperimentMode,
    Logger,
    MuseDataType,
    SessionConfig,
)

# class CLASResult(IntEnum):
#     NOT_RUNNING = 700

#     STIM = 701
#     STIM2 = 702

#     BACKOFF = 710
#     AMPLITUDE = 711
#     QUADRATURE = 712
#     FUTURE = 713

#     BACKOFF2 = 714
#     FUTURE2 = 715

#     HL_RATIO = 716


# class ExperimentMode(IntEnum):
#     DISABLED = 0
#     SHAM_DELAY = 1
#     CLAS = 2
#     SHAM_PHASE = 3
#     SHAM_MUTED = 4


# class PhaseTrackerMode(Enum):
#     fft = EnumAuto()
#     fftwin = EnumAuto()
#     wavelet = EnumAuto()

class Processor(QObject):
    results_ready = pyqtSignal(object)  # Signal that will emit results once processing is done

    def __init__(self, muse_data_type: MuseDataType):
        super().__init__()
        self.muse_data_type = muse_data_type
        self.running = False

    def run(self):
        """Starts the process in a separate thread."""
        self.running = True

    def stop(self):
        """Stops the process."""
        self.running = False

    def process(self):
        """To be overridden by subclasses."""
        raise NotImplementedError("Subclasses should implement the 'process' method.")


class EEGProcessor(Processor):
    stim = pyqtSignal()
    _last_stim = 0.0  # float: units of time elapsed
    _time_elapsed = 0.0  # float: in seconds, relative to end of last block/start of current block
    
    def __init__(self, config: SessionConfig):
        super().__init__(MuseDataType.EEG)
        self.config = config
        self.logger = Logger(self.config._session_key, self.__class__.__name__)
        self.amp_buffer = np.zeros(self.config.amp_buffer_len)
        self.hl_ratio_buffer = np.zeros(self.config.hl_ratio_buffer_len)

        self.window_len_n = SAMPLING_RATE[self.muse_data_type] * self.config.window_len_s
        self.target_phase_deg = self.config.target_phase_deg
        self.times = np.zeros(self.window_len_n)
        self.data = np.zeros((self.window_len_n, len(CHANNEL_NAMES[self.muse_data_type])))

        self.second_stim_start = nan
        self.second_stim_end = nan

        self.sos_low = scipy.signal.butter(self.config.bpf_order, self.config.low_bpf_cutoff, btype = 'bandpass', output = 'sos', fs = SAMPLING_RATE[self.muse_data_type])
        self.sos_high = scipy.signal.butter(self.config.bpf_order, self.config.high_bpf_cutoff, btype = 'bandpass', output = 'sos', fs = SAMPLING_RATE[self.muse_data_type])

        self.zi_low = scipy.signal.sosfilt_zi(self.sos_low)
        self.zi_high = scipy.signal.sosfilt_zi(self.sos_high)

        self.wavelet_freqs = np.linspace(self.config.truncated_wavelet.low, self.config.truncated_wavelet.high, self.config.truncated_wavelet.n)

        trunc_wavelet_len = self.window_len_n * 2 # double the length of the signal
        self.trunc_wavelets = [scipy.signal.morlet2(trunc_wavelet_len, self.config.truncated_wavelet.w * SAMPLING_RATE[self.muse_data_type] / (2 * f * np.pi), w = self.config.truncated_wavelet.w)[:trunc_wavelet_len // 2] for f in self.wavelet_freqs]

        self.selected_channel_ind = 0  # Default to channel 0
        self.selected_channel_ind_mutex = QMutex()
        self.channel_switch_timer = QTimer()
        self.channel_switch_timer.timeout.connect(self.switch_channel)
        self.channel_switch_timer.start(3000)  # 3 seconds

    def get_hl_ratio(self, selected_channel_data):
        lp_signal, self.zi_low = scipy.signal.sosfilt(self.sos_low, selected_channel_data, zi = self.zi_low)
        hp_signal, self.zi_high = scipy.signal.sosfilt(self.sos_high, selected_channel_data, zi = self.zi_high)

        # compute the lp envelope of the signal
        envelope_lp = np.abs(scipy.signal.hilbert(lp_signal[SAMPLING_RATE:]))
        power_lf = envelope_lp**2

        # compute the hf envelope of the signal
        envelope_hf = np.abs(scipy.signal.hilbert(hp_signal[SAMPLING_RATE:]))
        power_hf = envelope_hf**2
        # compute ratio and store
        hl_ratio = np.mean(power_hf) / np.mean(power_lf)
        hl_ratio = np.log10(hl_ratio)
        return hl_ratio
    
    def estimate_phase(self, selected_channel): 
        conv_vals = [np.dot(selected_channel, w) for w in self.trunc_wavelets]
        max_idx = np.argmax(np.abs(conv_vals))
        freq = self.wavelet_freqs[max_idx]
        phase = np.angle(conv_vals[max_idx]) % (2 * pi)
        
        return phase, freq, conv_vals

    def process_data(self, times: np.ndarray, data: np.ndarray):
        self.times = np.concatenate([self.times, times])
        self.times = self.times[-self.window_len_n:]
        self.data = np.vstack([self.data, data])
        self.data = self.data[-self.window_len_n:]
        self.selected_channel_ind_mutex.lock()
        try:
            curr_selected_channel = self.selected_channel_ind
        finally:
            self.selected_channel_ind_mutex.unlock()

        phase, freq, _ = self.estimate_phase(self.data[:, curr_selected_channel])
        hl_ratio = self.get_hl_ratio(self.data[:, curr_selected_channel])
        print(phase, freq, hl_ratio)

    def switch_channel(self):
        self.selected_channel_ind_mutex.lock()
        try:
            self.selected_channel_ind = np.argmin(np.sqrt(np.mean(self.data**2, axis=0)))
        finally:
            self.selected_channel_ind_mutex.unlock()  # Release lock after computing RMS
    
    # def process_block(self, currsig: Union[np.ndarray, None] = None) -> Tuple[CLASResult, float, dict]:
    #     ''' 
    #     Process a block of data through phase tracker and return stimulation status.

    #     Parameters
    #     ----------
    #     currsig
    #         Signal in a 1D numpy array

    #     Returns
    #     -------
    #     CLASResult
    #     delta_t (in seconds)
    #     internals
    #     '''

    #     # if no new data is given in parameters, run estimate on existing internal data
    #     # this requires that data is updated asynchronously using new/replace data functions
    #     if currsig is not None:
    #         blocksize = currsig.shape[0]
    #         blocksize_sec = blocksize / self.fs  # length in seconds

    #         ### adjust time_elapsed accumulator ###
    #         block_start_time = self.time_elapsed
    #         block_end_time = block_start_time + blocksize_sec
    #         self.time_elapsed = block_end_time

    #     if self.second_stim_end < self.time_elapsed:
    #         # go back to normal functioning
    #         self.second_stim_start = nan
    #         self.second_stim_end = nan

    #     ### estimate phase ###
    #     phase, cfreq, camp, quadrature, hl_ratio = self.estimate(currsig)

    #     # roll amplitude buffer
    #     self.ampbuffer[:-1] = self.ampbuffer[1:]
    #     self.ampbuffer[-1] = camp
    #     meanamp = self.ampbuffer.mean()

    #     if self.high_low_analysis:
    #         self.high_low_data[:-1] = self.high_low_data[1:]
    #         self.high_low_data[-1] = hl_ratio
    #         mean_hl_ratio = self.high_low_data.mean()
    #     else:
    #         mean_hl_ratio = nan

    #     internals = {
    #         'phase': phase,
    #         'freq': cfreq,
    #         'amp': camp,
    #         'meanamp': meanamp,
    #         'quadrature': quadrature,
    #         'hl_ratio': hl_ratio
    #     }

    #     if self.experiment_mode == ExperimentMode.DISABLED:
    #         return CLASResult.NOT_RUNNING, 0, internals

    #     # check if we're waiting for the 2nd stim
    #     # if NOT, run normal checks
    #     if isnan(self.second_stim_start):
    #         ### check backoff criteria ###
    #         if ((self.last_stim + self.backoff_time) > (self.time_elapsed + self.prediction_limit_sec)):
    #             return CLASResult.BACKOFF, 0, internals

    #         ### check amplitude criteria ###
    #         if (meanamp < self.amp_threshold) or (meanamp > self.amp_limit):
    #             return CLASResult.AMPLITUDE, 0, internals

    #         ### check quadrature ###
    #         if (quadrature is not None) and (quadrature < self.quadrature_thresh):
    #             return CLASResult.QUADRATURE, 0, internals

    #         if self.high_low_analysis and ((mean_hl_ratio > self.high_low_freq_lookback_ratio) or
    #                                        (hl_ratio > self.high_low_freq_ratio)):
    #             return CLASResult.HL_RATIO, 0, internals

    #     # if we are waiting for 2nd stim, but before the backoff window, only use phase targeting
    #     if self.time_elapsed < self.second_stim_start:
    #         return CLASResult.BACKOFF2, 0, internals

    #     ### perform forward prediction ###
    #     delta_t = ((self.target_phase - phase) % (2 * pi)) / (cfreq * 2 * pi)

    #     # cue a stim for the next target phase
    #     if isnan(self.second_stim_start):
    #         if delta_t > self.prediction_limit_sec:
    #             return CLASResult.FUTURE, delta_t, internals

    #         self.last_stim = self.time_elapsed + delta_t  # update stim time to compute backoff
    #         self.second_stim_start = self.last_stim + self.stim2_start_delay  # update
    #         self.second_stim_end = self.last_stim + self.stim2_end_delay

    #         return CLASResult.STIM, delta_t, internals

    #     else:
    #         if delta_t > self.stim2_prediction_limit_sec:
    #             return CLASResult.FUTURE2, delta_t, internals

    #         self.second_stim_start = nan
    #         self.second_stim_end = nan

    #         if self.experiment_mode == ExperimentMode.SHAM_PHASE:
    #             self.vary_regen()

    #         return CLASResult.STIM2, delta_t, internals

    # def estimate(self, block: Union[np.ndarray, None] = None) -> Tuple[float, float, float, float, float]:
    #     '''
    #     Return estimated phase / amplitude / frequency / quadrature at most recent point of the internal buffer using the wavelet transform.

    #     If block is provided, this function also rolls the internal buffer and appends the data from block.

    #     Parameters
    #     ----------
    #     block : np.ndarray (Default: None)
    #         New data

    #     Returns
    #     -------
    #     phase : float
    #         The current estimated phase

    #     freq : float
    #         The current estimated frequency

    #     amp : float
    #         The current estimated amplitude

    #     quadrature : float

    #     '''
    #     hl_ratio = np.nan

    #     if block is not None:
    #         self.new_data(block)

    #     # the data that we're analyzing
    #     cdata = self.data[-1 * self.analysis_sp:]


    #     # convolve the list of wavelets
    #     conv_vals = [np.dot(cdata, w) for w in self.wavelet]

    #     # choose the one with highest amp/phase
    #     amp_conv_vals = np.abs(conv_vals)
    #     amp_max = np.argmax(amp_conv_vals)

    #     # create outputs
    #     amp = amp_conv_vals[amp_max] / 2
    #     freq = self.wavelet_freqs[amp_max]
    #     phase = np.angle(conv_vals[amp_max])

    #     ### high low ratio ###
    #     # if high-low analysis is enabled, estimate high-low frequency ratio
    #     if self.high_low_analysis:
    #         # convolve the list of wavelets
    #         conv_vals_hl = [np.dot(cdata, w) for w in self.high_low_wavelets]

    #         # get average amplitude
    #         hf_amp = np.mean(np.abs(conv_vals_hl))

    #         # compute ratio and store
    #         hl_ratio = hf_amp / np.mean(np.abs(conv_vals))


    #     ### determine if we're locked on ###
    #     est_phase = (np.arange(self.quadrature_sp) / self.fs) * freq * 2 * pi
    #     est_phase = est_phase - est_phase[-1] + phase
    #     est_sig = np.cos(est_phase)
    #     est_sig = est_sig / np.trapz(np.abs(est_sig)) * est_sig.size

    #     # normalize the signal
    #     #normsig = cdata / np.sqrt(np.mean(np.square(cdata)))
    #     normsig = cdata[-self.quadrature_sp:] / np.trapz(np.abs(cdata[-self.quadrature_sp:])) * cdata.size
    #     quadrature = np.trapz(normsig * est_sig) / cdata.size

    #     return phase, freq, amp, quadrature, hl_ratio
    
    # def new_data(self, block: np.ndarray):
    #     '''
    #     Roll the internal buffer and append new data block.
    #     Parameters
    #     ----------
    #     block : np.ndarray
    #         1D array with data to append
    #         Must be smaller than self.data_len
    #     '''

    #     blocksize = block.shape[0]
    #     blocksize_sec = blocksize / self.fs  # length in seconds

    #     ### adjust time_elapsed accumulator ###
    #     block_start_time = self.time_elapsed
    #     block_end_time = block_start_time + blocksize_sec
    #     self.time_elapsed = block_end_time

    #     # append latest block to internal buffer
    #     n_new_samp = block.size
    #     self.data[:-1 * n_new_samp] = self.data[n_new_samp:]  # rotate buffer
    #     self.data[-1 * n_new_samp:] = block

    # def set_experiment_mode(self, mode: ExperimentMode) -> None:
    #     self.experiment_mode = mode

    #     if (mode == ExperimentMode.CLAS) or (mode == ExperimentMode.SHAM_MUTED):
    #         self.target_phase = self.default_target_phase
    #         self.target_phase_trig = self.phase_to_trigger(self.target_phase)

    #     elif mode == ExperimentMode.SHAM_PHASE:
    #         self.vary_regen()

    #     elif mode == ExperimentMode.SHAM_DELAY:
    #         self.target_phase = self.default_target_phase
    #         self.target_phase_trig = 0

    # def vary_regen(self):
    #     phase_idx = np.random.randint(0, 63) << 2  # encode target phase into the 6 MSB
    #     self.target_phase = self.trigger_to_phase(phase_idx)  # scale idx to 2pi
    #     self.target_phase_trig = phase_idx