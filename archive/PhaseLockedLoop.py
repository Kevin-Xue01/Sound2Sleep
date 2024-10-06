import scipy
import scipy.signal
import numpy as np


class PhaseLockedLoop(object):
    """The phase locked loop class.

    Internally records the current PLL and filter states and updates as blocks of data become available.
    """

    def __init__(self, fs):
        """

        Parameters
        ----------
        fs : int
        Sampling frequency
        """
        self.fs = fs

        self.pll_vco_fq = 0.85 
        self.pll_loopgain = 0.5
        self.filter_gain = 2
        self.vco_sens = 0.3
        self.vco_gain = 1.1
        self.filter_cutoff = 3

        # self.sig_b, self.sig_a = scipy.signal.butter(2, np.divide([0.1, 2], (fs / 2)), btype='bandpass')
        self.loopf_b, self.loopf_a = scipy.signal.cheby1(4, 1, self.filter_cutoff / (fs / 2), btype='lowpass')
        self.lockf_b, self.lockf_a = scipy.signal.cheby1(4, 1, 3 / (fs / 2), btype='lowpass')

        self.loopf_zi = scipy.signal.lfiltic(self.loopf_b, self.loopf_a, 0, 0)
        self.lockf_zi = scipy.signal.lfilter_zi(self.lockf_b, self.lockf_a)

        self.sig_ref = np.zeros(1)
        self.pll_integral = np.zeros(1)

        self.ctime = np.zeros(1)

    def process_block(self, data):
        """Run a block of EEG data through the PLL.

        Process a block of EEG data and return the PLL internal phase, reference sinusoid, and lock detector output. This function 
        assumes subsequent EEG blocks contain continuous data. Will reuse and update the internal filter and PLL state stored in
        the current instance.

        Parameters
        ----------
        data : numpy-like vector
        A vector of EEG timeseries data

        Returns
        -------
        ts_phase : numpy vector
        A timeseries of the internal PLL phase.
        Unwrapped, has values from 0 until +Inf
        ts_ref : numpy vector
        The PLL reference sinusoid timeseries from the VCO
        Has values -vco_gain to +vco_gain
        ts_lockbin : numpy vector
        A binary timeseries indicating PLL locked or not
        Has a value of 0 or 1 per sample
        """
        block_size = data.size

        # variables to be returned
        ts_phase = np.zeros(block_size)
        ts_ref = np.zeros(block_size)
        ts_lockbin = np.zeros(block_size)

        for n in range(block_size):
            self.ctime = self.ctime + (1 / self.fs) # update current time

            # do pll stuff
            control = data[n] * self.sig_ref * self.pll_loopgain  # phase detector
            control, self.loopf_zi = scipy.signal.lfilter(self.loopf_b,
                                                          self.loopf_a,
                                                          control,
                                                          axis=-1,
                                                          zi=self.loopf_zi)  # loop filter
            control = control * self.filter_gain
            control = min(max(control, -100), 100)  # hard limit control signal (just in case)

            self.pll_integral += control / self.fs  # cumulative sum, error signal

            # compute PLL oscillator output
            current_phase = 2 * np.pi * (self.pll_vco_fq * self.ctime + self.vco_sens * self.pll_integral)
            self.sig_ref = self.vco_gain * np.cos(current_phase)  # reference signal

            # compute PLL lock detector
            sig_quad = np.sin(current_phase)  # quadrature signal
            sig_lock, self.lockf_zi = scipy.signal.lfilter(self.lockf_b,
                                                           self.lockf_a,
                                                           -sig_quad * data[n],
                                                           axis=-1,
                                                           zi=self.lockf_zi)
            sig_lockbin = abs(sig_lock) > 0.8

            # variables that need to be returned
            ts_phase[n] = current_phase
            ts_ref[n] = self.sig_ref
            ts_lockbin[n] = sig_lockbin

        return (ts_phase, ts_ref, ts_lockbin)
