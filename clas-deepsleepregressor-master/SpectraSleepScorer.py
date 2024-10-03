# -*- coding: utf-8 -*-

import fooof
import numpy as np
import scipy.signal
import pickle

class SpectraSleepScorer():
  def __init__(self):
    # load SVC classifier
    self.clf = pickle.load('n3classifier.pickle')

  def compute_spectra(self, fsample, data):
    f, psd = scipy.signal.welch(data, fsample, scaling='density', nperseg=1024)
    return f, psd

  def compute_power_ratio(self, fsample, data, f = None):
    # compute power spectral density
    if f is None:
      _, psd = self.compute_spectra(fsample, data)
    else:
      psd = data
      
    f_to_idx = lambda x: int(round(x / (fsample / 2) * psd.size))

    # get average over frequency bands of interest
    delta_alpha = np.mean(psd[f_to_idx(0.5):f_to_idx(4)]) / np.mean(psd[f_to_idx(8):f_to_idx(12)])
    delta_gamma = np.mean(psd[f_to_idx(0.5):f_to_idx(4)]) / np.mean(psd[f_to_idx(30):f_to_idx(50)])

    return (delta_alpha, delta_gamma)

  def compute_fooof_aperiodic(self, fsample, data, f = None):
    if f is None:
      f, psd = self.compute_spectra(fsample, data)
    else:
      psd = data
    
    fm = fooof.FOOOF(max_n_peaks=6)
    freq_range = [0.5, 50]
    
    fm.fit(f, psd, freq_range)
    dat = fm.get_results()
    
    bg = dat.background_params
    pp = dat.peak_params
    
    # find all peaks between 0 and 2 Hz
    pp = pp[pp[:,0] < 2,:]
    
    #deltapeaks = np.mean(np.multiply(pp[:,1], pp[:,2]))
    deltapeaks = np.mean(pp[:,1])
    
    return (bg[0], deltapeaks)

  def score_power_ratio(self, fsample, data):
    return None

  def score(self, fsample, data):
    # compute the spectra once
    f,psd = self.compute_spectra(fsample, data)

    delta_alpha, delta_gamma = self.compute_power_ratio(fsample, psd, f = f)
    fooof_bg, _ = self.compute_fooof_aperiodic(fsample, psd, f = f)

    isN3 = self.clf.predict([delta_alpha, delta_gamma, fooof_bg[0]])

    return 3 if isN3 else -1


