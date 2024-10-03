# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:03:57 2019

@author: simeo
"""

import SpectraSleepScorer
import pyedflib
import os.path
import numpy as np
import math
import pandas as pd
import scipy.signal


EPOCH_LENGTH = 30 # epoch length in seconds

#%% Load and preproc data
filelist = os.listdir(os.path.abspath(os.path.join('..', 'sleepedfdata')))
allsubjects = list(filter(lambda x: x.endswith('-PSG.edf'), filelist))
allsubjects = list(map(lambda x: x[0:6], allsubjects))

#allsubjects.reverse()   # on 2nd run, go backwards so we can be more efficient

for subjid in allsubjects:
  print("Running %s" % subjid)
  hypname = list(filter(lambda x: x.startswith(subjid) and x.endswith('-Hypnogram.edf'), filelist))[0]
  
  eeg = pyedflib.EdfReader(os.path.abspath(os.path.join("..", "sleepedfdata", "%sE0-PSG.edf" % subjid)))
  hyp = pyedflib.EdfReader(os.path.abspath(os.path.join("..", "sleepedfdata", hypname)))
  
  # read first signal (Fpz-Cz)
  sigidx = eeg.getSignalLabels().index('EEG Fpz-Cz')
  
  fsample = eeg.getSampleFrequencies()[sigidx]
  n_samples = eeg.getNSamples()[sigidx]
  
  sigbufs = np.zeros(n_samples)
  sigbufs[:] = eeg.readSignal(sigidx)
  
  # split into 30 second epochs
  epoch_samples = fsample * EPOCH_LENGTH
  n_epochs = math.floor(n_samples / epoch_samples)
  
  sigbufs = sigbufs[0:n_epochs*epoch_samples]
  sigbufs = np.reshape(sigbufs, (n_epochs, epoch_samples), order='C')
  
  # get labels for each epoch
  sleepstages = np.zeros(n_epochs)
  
  
  hyp_dat = hyp.readAnnotations()
  
  # get sleep stage for each epoch
  for kk1 in range(0, n_epochs):
    # find the start/end indices of sleep stage
    idx1 = 0
    idx2 = 0
    
    e_srt = kk1*epoch_samples
    
    # find the most recent annotation
    for kk2 in range(hyp_dat[0].size-1, 0, -1):
      if (hyp_dat[0][kk2]*fsample) <= e_srt:
        break
      
    stage = hyp_dat[2][kk2][-1:]
    
    if stage == 'W':
      stage = -1
    elif stage == 'R':
      stage = 0
    elif stage == '1':
      stage = 1
    elif stage == '2':
      stage = 2
    elif stage == '3':
      stage = 3
    elif stage == '4':
      stage = 4
    else:
      stage = -1
      
    if stage >= 3:
      stage = 3
    
    sleepstages[kk1] = stage
    
  #%% Compute stuff
  delta_alpha = np.zeros(n_epochs)
  delta_gamma = np.zeros(n_epochs)
  
  aper_slope = np.zeros(n_epochs)
  aper_delta = np.zeros(n_epochs)
  
  # run power spectra calculator
  for kk in range(n_epochs):
    print("Computing %d of %d" % (kk, n_epochs))
    #f, psd = scipy.signal.welch(sigbufs[kk,:], fsample, scaling='density', nperseg=1024)
    f, psd = scipy.signal.periodogram(sigbufs[kk,:], fsample, scaling='density')
    (delta_alpha[kk], delta_gamma[kk]) = SpectraSleepScorer.SpectraSleepScorer.compute_power_ratio(fsample, psd, f=f)
    (aper_slope[kk], aper_delta[kk]) = SpectraSleepScorer.SpectraSleepScorer.compute_fooof_aperiodic(fsample, psd, f=f)
  
  #%% Plot
  df = pd.DataFrame({'Stage': sleepstages, 'DeltaAlpha': delta_alpha, 'DeltaGamma': delta_gamma, 'FOOOFslope': aper_slope, 'FOOOFdelta': aper_delta})
  df.to_csv(os.path.abspath(os.path.join("..", "sleepedfdata", "%s_eegmetrics.csv" % subjid)))






