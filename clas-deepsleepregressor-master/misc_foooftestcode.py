# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:34:54 2019

@author: simeo
"""


import scipy.signal
import fooof

plt.close()
f, psd = scipy.signal.welch(sigbufs[100,:], fsample, scaling='density', nperseg=1024)

fm = fooof.FOOOF()
freq_range = [0.5, 50]

fm.fit(f, psd, freq_range)
fm.print_results()
fm.plot()
    