# native
from enum import IntEnum
from math import pi, nan, isnan
from typing import Tuple, Union
import json

# pip
import numpy as np

# custom
try:
    from .CLASAlgo import CLASResult, ExperimentMode
except:
    from CLASAlgo import CLASResult, ExperimentMode

class CLASAlgo():

    def __init__(self, fs: float, param_file: str, **kwargs):

        # load parameters from file
        with open(param_file, 'r') as f:
            params = json.load(f)

        self.params = params

        # signal parameters
        self.ampbuffer = np.zeros(params['amp_lookback_nblocks'])
        self.fs = fs

        # internal parameters
        self.last_stim = 0.0  # float: units of time elapsed
        self.time_elapsed = 0.0  # float: in seconds, relative to end of last block/start of current block

        # store CLAS parameters
        self.amp_threshold = params['amp_threshold']
        self.backoff_time = params['backoff_time']
        self.sham_mindelay = params['sham_mindelay']
        self.sham_maxdelay = params['sham_maxdelay']
        self.freq_limits = params['freq_limits']

        self.stim2_delay = params['stim2_delay']

        self.current_thresh = nan

        # overrides
        for k in kwargs:
            if kwargs[k] is not None:
                print('PhaseTracker: Overriding {:s}:{:s} with {:s}.'.format(k, self.__dict__[k], kwargs[k]))
                self.__dict__[k] = kwargs[k]

        # initialize experiment mode
        self.set_experiment_mode(ExperimentMode.CLAS)  # default is CLAS


    def replace_data(self, time_elapsed:float, data:np.ndarray):
        self.time_elapsed = time_elapsed
        self.pt.replace_data(data)

    def new_data(self, block:np.ndarray):
        blocksize = block.shape[0]
        blocksize_sec = blocksize / self.fs  # length in seconds

        ### adjust time_elapsed accumulator ###
        block_start_time = self.time_elapsed
        block_end_time = block_start_time + blocksize_sec
        self.time_elapsed = block_end_time


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
        raise (NotImplementedError('SHAM_PHASE not supported with amplitude threshold algorithm.'))

    def set(self, **kwargs):
        for k in kwargs:
            if k in [
                    'amp_threshold', 'prediction_limit', 'target_phase', 'backoff_time', 'quadrature_thresh',
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
        phase, cfreq, camp, quadrature = self.pt.estimate(currsig)

        # roll amplitude
        self.ampbuffer[:-1] = self.ampbuffer[1:]
        self.ampbuffer[-1] = camp
        meanamp = self.ampbuffer.mean()
        internals = {'phase': phase, 'freq': cfreq, 'amp': camp, 'meanamp': meanamp, 'quadrature': quadrature}

        if self.experiment_mode == ExperimentMode.DISABLED:
            return CLASResult.NOT_RUNNING, 0, internals

        # check if we're waiting for the 2nd stim
        # if NOT, run normal checks
        if isnan(self.second_stim_start):
            ### check backoff criteria ###
            if ((self.last_stim + self.backoff_time) > (self.time_elapsed + self.prediction_limit_sec)):
                return CLASResult.BACKOFF, 0, internals

            ### check amplitude criteria ###
            if meanamp < self.amp_threshold:
                return CLASResult.AMPLITUDE, 0, internals

            ### check quadrature ###
            if quadrature < self.quadrature_thresh:
                return CLASResult.QUADRATURE, 0, internals

        # if we are waiting for 2nd stim, but before the backoff window
        if  self.time_elapsed < self.second_stim_start:
            return CLASResult.BACKOFF2, 0, internals

        # if we're waiting for 2nd stim, but within the 2nd stim window, try to deliver a stim with only phase targeting.

        ### perform forward prediction ###
        delta_t = ((self.target_phase - phase) % (2 * pi)) / (cfreq * 2 * pi)

        # cue a stim for the next target phase
        if isnan(self.second_stim_start):
            if delta_t > self.prediction_limit_sec:
                return CLASResult.FUTURE, 0, internals

            self.last_stim = self.time_elapsed + delta_t  # update stim time to compute backoff
            self.second_stim_start = self.last_stim + self.stim2_start_delay  # update
            self.second_stim_end = self.last_stim + self.stim2_end_delay

            return CLASResult.STIM, delta_t, internals

        else:
            if delta_t > self.stim2_prediction_limit_sec:
                return CLASResult.FUTURE2, 0, internals

            self.second_stim_start = nan
            self.second_stim_end = nan

            if self.experiment_mode == ExperimentMode.SHAM_PHASE:
                self.vary_regen()

            return CLASResult.STIM2, delta_t, internals
