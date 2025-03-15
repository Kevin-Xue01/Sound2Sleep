# native
import json
import logging
from enum import Enum, IntEnum
from enum import auto as EnumAuto
from math import ceil, floor, isnan, nan, pi
from typing import Tuple, Union

# pip
import numpy as np
import scipy
import scipy.signal


class CLASResult(IntEnum):
    NOT_RUNNING = 700

    STIM = 701
    STIM2 = 702

    BACKOFF = 710
    AMPLITUDE = 711
    QUADRATURE = 712
    FUTURE = 713

    BACKOFF2 = 714
    FUTURE2 = 715

    HL_RATIO = 716


class ExperimentMode(IntEnum):
    DISABLED = 0
    SHAM_DELAY = 1
    CLAS = 2
    SHAM_PHASE = 3
    SHAM_MUTED = 4


class PhaseTrackerMode(Enum):
    fft = EnumAuto()
    fftwin = EnumAuto()
    wavelet = EnumAuto()


class CLASAlgo():

    def __init__(self, fs: float, param_file: str, **kwargs):
        logging.info('CLASAlgo 2022-08-26')

        # load parameters from file
        with open(param_file, 'r') as f:
            params = json.load(f)

        self.params = params

        ############################################################
        # signal parameters
        self.ampbuffer = np.zeros(params['amp_lookback_nblocks'])
        self.fs = fs

        # internal parameters
        self.last_stim = 0.0  # float: units of time elapsed
        self.time_elapsed = 0.0  # float: in seconds, relative to end of last block/start of current block

        # store CLAS parameters
        self.amp_threshold = params['amp_threshold']
        # self.amp_limit = params['amp_limit']
        self.prediction_limit_sec = params['prediction_limit_sec']
        self.backoff_time = params['backoff_time']
        # self.quadrature_thresh = params['quadrature_thresh'] or nan
        # self.quadrature_len = params['quadrature_len']
        self.sham_mindelay = params['sham_mindelay']
        self.sham_maxdelay = params['sham_maxdelay']
        self.freq_limits = params['freq_limits']

        self.stim2_start_delay = params['stim2_start_delay']
        self.stim2_end_delay = params['stim2_end_delay']
        self.stim2_prediction_limit_sec = params['stim2_prediction_limit_sec']

        if 'high_freq_vals' in params:
            self.high_low_analysis = True
            self.high_freq_vals = params['high_freq_vals']
            self.high_low_freq_ratio = params['high_low_freq_ratio']
            self.high_low_freq_lookback_ratio = params['high_low_freq_lookback_ratio']
            self.high_low_lookback_nblocks = params['high_low_lookback_nblocks']
        else:
            self.high_low_analysis = False

        # phase target parameters
        self.target_phase = 0
        self.target_phase_trig = 0
        self.default_target_phase = params['target_phase'] % (2 * pi)

        # second stim parameters in msec
        self.second_stim_start = nan
        self.second_stim_end = nan

        # buffer
        self.analysis_len = params['analysis_len']

        # overrides
        for k in kwargs:
            if kwargs[k] is not None:
                print('PhaseTracker: Overriding {:s}:{:s} with {:s}.'.format(k, self.__dict__[k], kwargs[k]))
                self.__dict__[k] = kwargs[k]

        ############################################################
        # initialize phase tracker
        self.data_len = int(fs * 2 * self.analysis_len)
        self.data = np.zeros(self.data_len)
        self.mode = PhaseTrackerMode[params['mode']]
        self.analysis_sp = int(fs * self.analysis_len)
        self.quadrature_sp = int(fs * self.quadrature_len)  # sp is short for sample

        if self.mode == PhaseTrackerMode.wavelet:
            # construct the wavelet
            M = int(self.analysis_sp * 2)
            w = 5
            s = lambda f: w * fs / (2 * pi * f)

            # set of frequencies, to identify primary freq
            self.wavelet_freqs = np.linspace(self.freq_limits[0], self.freq_limits[1], 20)

            # create wavelet for each frequency, truncated at the middle
            self.wavelet = [scipy.signal.morlet2(M, s(f), w)[:self.analysis_sp] for f in self.wavelet_freqs]

            # if requested, initialize the high-low frequency ratio
            if self.high_low_analysis:
                self.high_low_data = np.zeros(self.high_low_lookback_nblocks)
                self.high_low_wavelets = [
                    scipy.signal.morlet2(M, s(f), w)[:self.analysis_sp] for f in self.high_freq_vals
                ]

        elif self.mode == PhaseTrackerMode.fftwin:
            self.fft_window = np.hanning(self.analysis_sp * 2)[:self.analysis_sp]

            if self.high_low_analysis:
                raise (NotImplementedError('high-low analysis not implemented for fftwin'))

        ############################################################
        # other initialization

        # initialize experiment mode
        self.set_experiment_mode(ExperimentMode.CLAS)  # default is disabled

    def replace_data(self, time_elapsed: float, data: np.ndarray):
        '''
        Replace the internal buffer with the provided array.
        Parameters
        ----------
        data : np.ndarray
            1D array with new data buffer
        '''
        self.data = data
        self.data_len = data.shape[0]
        self.time_elapsed = time_elapsed

    def new_data(self, block: np.ndarray):
        '''
        Roll the internal buffer and append new data block.
        Parameters
        ----------
        block : np.ndarray
            1D array with data to append
            Must be smaller than self.data_len
        '''

        blocksize = block.shape[0]
        blocksize_sec = blocksize / self.fs  # length in seconds

        ### adjust time_elapsed accumulator ###
        block_start_time = self.time_elapsed
        block_end_time = block_start_time + blocksize_sec
        self.time_elapsed = block_end_time

        # append latest block to internal buffer
        n_new_samp = block.size
        self.data[:-1 * n_new_samp] = self.data[n_new_samp:]  # rotate buffer
        self.data[-1 * n_new_samp:] = block

    @staticmethod
    def phase_to_trigger(phase: float) -> int:
        return int((phase % (2 * pi)) / (2 * pi) * 64) << 2

    @staticmethod
    def trigger_to_phase(trig: int) -> float:
        return (trig >> 2) / 64 * 2 * pi

    def set_experiment_mode(self, mode: ExperimentMode) -> None:
        self.experiment_mode = mode

        if (mode == ExperimentMode.CLAS) or (mode == ExperimentMode.SHAM_MUTED):
            self.target_phase = self.default_target_phase
            self.target_phase_trig = self.phase_to_trigger(self.target_phase)

        elif mode == ExperimentMode.SHAM_PHASE:
            self.vary_regen()

        elif mode == ExperimentMode.SHAM_DELAY:
            self.target_phase = self.default_target_phase
            self.target_phase_trig = 0

    def vary_regen(self):
        phase_idx = np.random.randint(0, 63) << 2  # encode target phase into the 6 MSB
        self.target_phase = self.trigger_to_phase(phase_idx)  # scale idx to 2pi
        self.target_phase_trig = phase_idx

    def set(self, **kwargs):
        for k in kwargs:
            if k in [
                    'amp_threshold', 'amp_limit', 'prediction_limit', 'target_phase', 'backoff_time', 'quadrature_thresh',
                    'sham_mindelay', 'sham_maxdelay'
            ]:
                self.__dict__[k] = kwargs[k]
            elif k == 'experiment_mode':
                self.set_experiment_mode(mode=kwargs[k])
            else:
                raise (ValueError('Invalid key'))

    def process_block(self, currsig: Union[np.ndarray, None] = None) -> Tuple[CLASResult, float, dict]:
        ''' 
        Process a block of data through phase tracker and return stimulation status.

        Parameters
        ----------
        currsig
            Signal in a 1D numpy array

        Returns
        -------
        CLASResult
        delta_t (in seconds)
        internals
        '''

        # if no new data is given in parameters, run estimate on existing internal data
        # this requires that data is updated asynchronously using new/replace data functions
        if currsig is not None:
            blocksize = currsig.shape[0]
            blocksize_sec = blocksize / self.fs  # length in seconds

            ### adjust time_elapsed accumulator ###
            block_start_time = self.time_elapsed
            block_end_time = block_start_time + blocksize_sec
            self.time_elapsed = block_end_time

        if self.second_stim_end < self.time_elapsed:
            # go back to normal functioning
            self.second_stim_start = nan
            self.second_stim_end = nan

        ### estimate phase ###
        phase, cfreq, camp, quadrature, hl_ratio = self.estimate(currsig)

        # roll amplitude buffer
        self.ampbuffer[:-1] = self.ampbuffer[1:]
        self.ampbuffer[-1] = camp
        meanamp = self.ampbuffer.mean()

        if self.high_low_analysis:
            self.high_low_data[:-1] = self.high_low_data[1:]
            self.high_low_data[-1] = hl_ratio
            mean_hl_ratio = self.high_low_data.mean()
        else:
            mean_hl_ratio = nan

        internals = {
            'phase': phase,
            'freq': cfreq,
            'amp': camp,
            'meanamp': meanamp,
            'quadrature': quadrature,
            'hl_ratio': hl_ratio
        }

        if self.experiment_mode == ExperimentMode.DISABLED:
            return CLASResult.NOT_RUNNING, 0, internals

        # check if we're waiting for the 2nd stim
        # if NOT, run normal checks
        if isnan(self.second_stim_start):
            ### check backoff criteria ###
            if ((self.last_stim + self.backoff_time) > (self.time_elapsed + self.prediction_limit_sec)):
                return CLASResult.BACKOFF, 0, internals

            ### check amplitude criteria ###
            if (meanamp < self.amp_threshold) or (meanamp > self.amp_limit):
                return CLASResult.AMPLITUDE, 0, internals

            ### check quadrature ###
            if (quadrature is not None) and (quadrature < self.quadrature_thresh):
                return CLASResult.QUADRATURE, 0, internals

            if self.high_low_analysis and ((mean_hl_ratio > self.high_low_freq_lookback_ratio) or
                                           (hl_ratio > self.high_low_freq_ratio)):
                return CLASResult.HL_RATIO, 0, internals

        # if we are waiting for 2nd stim, but before the backoff window, only use phase targeting
        if self.time_elapsed < self.second_stim_start:
            return CLASResult.BACKOFF2, 0, internals

        ### perform forward prediction ###
        delta_t = ((self.target_phase - phase) % (2 * pi)) / (cfreq * 2 * pi)

        # cue a stim for the next target phase
        if isnan(self.second_stim_start):
            if delta_t > self.prediction_limit_sec:
                return CLASResult.FUTURE, delta_t, internals

            self.last_stim = self.time_elapsed + delta_t  # update stim time to compute backoff
            self.second_stim_start = self.last_stim + self.stim2_start_delay  # update
            self.second_stim_end = self.last_stim + self.stim2_end_delay

            return CLASResult.STIM, delta_t, internals

        else:
            if delta_t > self.stim2_prediction_limit_sec:
                return CLASResult.FUTURE2, delta_t, internals

            self.second_stim_start = nan
            self.second_stim_end = nan

            if self.experiment_mode == ExperimentMode.SHAM_PHASE:
                self.vary_regen()

            return CLASResult.STIM2, delta_t, internals

    def estimate(self,
                 block: Union[np.ndarray, None] = None,
                 nfft: int = 4096) -> Tuple[float, float, float, float, float]:
        '''
        Return estimated phase / amplitude / frequency / quadrature at most recent point of the internal buffer using the wavelet transform.

        If block is provided, this function also rolls the internal buffer and appends the data from block.

        Parameters
        ----------
        block : np.ndarray (Default: None)
            New data

        Returns
        -------
        phase : float
            The current estimated phase

        freq : float
            The current estimated frequency

        amp : float
            The current estimated amplitude

        quadrature : float

        '''
        hl_ratio = np.nan

        if block is not None:
            self.new_data(block)

        # the data that we're analyzing
        cdata = self.data[-1 * self.analysis_sp:]

        # apply window if requested
        if self.mode == PhaseTrackerMode.fftwin:
            cdata = cdata * self.fft_window

        if (self.mode == PhaseTrackerMode.fft) or (self.mode == PhaseTrackerMode.fftwin):
            # run FFT on analysis segment
            freqdat = scipy.fft.fft(cdata, n=nfft, workers=-2)

            # identify frequency peak
            freq_limit_idx = np.array(self.freq_limits) / self.fs * nfft
            freq_limit_idx[0] = floor(freq_limit_idx[0])
            freq_limit_idx[1] = ceil(freq_limit_idx[1])

            spectralamp = np.abs(freqdat[freq_limit_idx[0]:freq_limit_idx[1]])  # only data within limits
            max_idx = np.argmax(spectralamp) + freq_limit_idx[0]

            # get phase
            phase_start = np.angle(freqdat[max_idx])
            amp = spectralamp[max_idx] / nfft * 2 * 10

            # get freq
            freq = max_idx / nfft * self.fs

            # estimate sine
            time = np.arange(int(self.analysis_sp / 2)) / self.fs
            phase = ((phase_start + (time * freq * 2 * pi)) % (2 * pi))[-1]

            # compute a forward looking function
            # pred = lambda t: phase[-1] + (t * freq * 2 * np.pi) % (2 * np.pi)

        elif self.mode == PhaseTrackerMode.wavelet:
            # convolve the list of wavelets
            conv_vals = [np.dot(cdata, w) for w in self.wavelet]

            # choose the one with highest amp/phase
            amp_conv_vals = np.abs(conv_vals)
            amp_max = np.argmax(amp_conv_vals)

            # create outputs
            amp = amp_conv_vals[amp_max] / 2
            freq = self.wavelet_freqs[amp_max]
            phase = np.angle(conv_vals[amp_max])

            ### high low ratio ###
            # if high-low analysis is enabled, estimate high-low frequency ratio
            if self.high_low_analysis:
                # convolve the list of wavelets
                conv_vals_hl = [np.dot(cdata, w) for w in self.high_low_wavelets]

                # get average amplitude
                hf_amp = np.mean(np.abs(conv_vals_hl))

                # compute ratio and store
                hl_ratio = hf_amp / np.mean(np.abs(conv_vals))

        else:
            raise (NotImplementedError('Unknown mode'))

        ### determine if we're locked on ###
        est_phase = (np.arange(self.quadrature_sp) / self.fs) * freq * 2 * pi # create time vector based on the number of samples and the sampling frequency (angular)
        est_phase = est_phase - est_phase[-1] + phase # shift time vector to match the phase
        est_sig = np.cos(est_phase) # cosine signal with the estimated phase
        est_sig = est_sig / np.trapz(np.abs(est_sig)) * est_sig.size # normalize estimated signal

        # normalize the signal
        #normsig = cdata / np.sqrt(np.mean(np.square(cdata)))
        normsig = cdata[-self.quadrature_sp:] / np.trapz(np.abs(cdata[-self.quadrature_sp:])) * cdata.size # normalized measured signal
        quadrature = np.trapz(normsig * est_sig) / cdata.size # compute quadrature

        return phase, freq, amp, quadrature, hl_ratio
