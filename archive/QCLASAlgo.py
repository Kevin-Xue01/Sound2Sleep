# pyright: reportGeneralTypeIssues=false

# Qt framework
from PyQt5.QtCore import pyqtSignal, QRunnable, QObject

# built in
import datetime
from enum import IntEnum
from math import pi
import json

# pip
import numpy as np

# custom
import PinkNoiseGenerator
import CLASAlgo
from CLASAlgo import ExperimentMode, CLASResult

##########
# STATIC #
##########
pink_noise = PinkNoiseGenerator.generate_noise()


###########
# CLASSES #
###########
class PortCodes(IntEnum):
    STIM1_DELIVERED = 1
    STIM2_DELIVERED = 2


class EEGProcessorSignal(QObject):
    ''' QObject helper so MLRunner can send Qt signals '''
    result = pyqtSignal(int)
    cue_stim = pyqtSignal(int, int, bool)
    datavals = pyqtSignal(dict)


class EEGProcessor(QRunnable):
    def __init__(self, fsample: float, param_file: str):
        ''' 
        Initialize the runnable.

        Parameters
        ----------
        fsample : float
            EEG data sampling rate
        '''
        super(EEGProcessor, self).__init__()
        self.setAutoDelete(False)

        # load parameters
        with open(param_file, 'r') as f:
            self.params = json.load(f)

        # setup signals to return data
        self.signals = EEGProcessorSignal()

        # initialize algorithm
        self.ca = CLASAlgo.CLASAlgo(fs=fsample, param_file=param_file)
        self.new_data = self.ca.new_data
        self.replace_data = self.ca.replace_data

    def set_amp_threshold(self, amp_threshold: float):
        self.ca.set(amp_threshold=amp_threshold)

    def set_experiment_mode(self, experiment_mode: CLASAlgo.ExperimentMode):
        self.ca.set_experiment_mode(mode=experiment_mode)

    def run(self):
        ### run CLAS algorithm ###
        result, time_to_target, internals = self.ca.process_block()

        ### initial results handling ###
        # return signal-derived values to GUI to update
        self.signals.datavals.emit(internals)

        # if we're not stimming, return results
        if (result != CLASResult.STIM) and (result != CLASResult.STIM2):
            self.signals.result.emit(result)
            return

        # if we are stimming, continue...

        ### get ready to stim ###
        # update last stim time
        if result == CLASResult.STIM:
            self.last_stim = datetime.datetime.now() + datetime.timedelta(
                seconds=time_to_target)

        # compute 2nd pulse delay interval based on dominant frequency

        # queue stim
        if (self.ca.experiment_mode == ExperimentMode.CLAS) or (
                self.ca.experiment_mode == ExperimentMode.SHAM_PHASE) or (
                    self.ca.experiment_mode == ExperimentMode.SHAM_MUTED):
            ## prepare to deliver timelocked stim

            if result == CLASResult.STIM:
                self.signals.cue_stim.emit(
                    time_to_target,
                    PortCodes.STIM1_DELIVERED + self.ca.target_phase_trig,
                    (self.ca.experiment_mode == ExperimentMode.SHAM_MUTED))
                    
            elif result == CLASResult.STIM2:
                self.signals.cue_stim.emit(
                    time_to_target,
                    PortCodes.STIM2_DELIVERED,
                    (self.ca.experiment_mode == ExperimentMode.SHAM_MUTED))


        elif self.ca.experiment_mode == ExperimentMode.SHAM_DELAY:
            ## prepare to deliver delayed stim

            # use current 
            stim2_delay = time_to_target + int(1 / internals['freq'] * 1000) 

            # compute random delay to introduce phase jitter
            rand_delay = np.random.randint(int(self.SHAM_MINDELAY * 1000),
                                           int(self.SHAM_MAXDELAY * 1000))

            # deliver randomly shifted sham stim
            self.signals.cue_stim.emit(rand_delay + time_to_target,
                                       PortCodes.STIM1_DELIVERED, False)
            self.signals.cue_stim.emit(rand_delay + stim2_delay,
                                       PortCodes.STIM2_DELIVERED, False)
