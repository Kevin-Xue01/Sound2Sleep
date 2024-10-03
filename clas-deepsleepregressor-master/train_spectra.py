# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:59:09 2019

@author: simeo
"""

import os.path
import numpy as np
import ptitprince as pt  # raincloud plots
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm
from sklearn.model_selection import GroupKFold

#%% Load all subjects
filelist = os.listdir(os.path.abspath(os.path.join('..', 'sleepedfdata')))
allsubjects = list(filter(lambda x: x.endswith('-PSG.edf'), filelist))
allsubjects = list(map(lambda x: x[0:6], allsubjects))

#allsubjects.reverse()   # on 2nd run, go backwards so we can be more efficient

df = None

for subjid in allsubjects:
  df_subj = pd.read_csv(os.path.abspath(os.path.join("..", "sleepedfdata", "%s_eegmetrics.csv" % subjid)))
  
  df_subj['subj'] = [subjid] * df_subj.shape[0]
  
  if df is None:
    df = df_subj
  else:
    df = df.append(df_subj)
    

#%% Plot
fig, axs = plt.subplots(2, 2)
pt.RainCloud(x = 'Stage', y = 'DeltaAlpha', data = df, ax = axs[0][0])
axs[0][0].set_title('Delta Alpha ratios')

pt.RainCloud(x = 'Stage', y = 'DeltaGamma', data = df, ax = axs[0][1])
axs[0][1].set_title('Delta Gamma ratios')


pt.RainCloud(x = 'Stage', y = 'FOOOFslope', data = df, ax = axs[1][0])
axs[1][0].set_title('FOOOF: Aperiodic spectral slope')

pt.RainCloud(x = 'Stage', y = 'FOOOFdelta', data = df, ax = axs[1][1])
axs[1][1].set_title('FOOOF: Mean delta peak amplitude')

plt.show()


#%% Plot binarized
isN3 = np.array(df['Stage'] == 3)

fig, axs = plt.subplots(2, 2)
pt.RainCloud(x = isN3, y = df['DeltaAlpha'], ax = axs[0][0])
axs[0][0].set_title('Delta Alpha ratios')

pt.RainCloud(x = isN3, y = df['DeltaGamma'], ax = axs[0][1])
axs[0][1].set_title('Delta Gamma ratios')


pt.RainCloud(x = isN3, y = df['FOOOFslope'], ax = axs[1][0])
axs[1][0].set_title('FOOOF: Aperiodic spectral slope')

pt.RainCloud(x = isN3, y = df['FOOOFdelta'], ax = axs[1][1])
axs[1][1].set_title('FOOOF: Mean delta peak amplitude')



fig, axs = plt.subplots(1, 3)
fig.set_size_inches([20, 5.5])
axs[0].scatter(x = df['DeltaAlpha'], y = df['DeltaGamma'], s = 0.7, c = isN3, alpha = 0.3, cmap = 'Dark2')
axs[0].set_xlabel("Delta-Alpha ratio")
axs[0].set_ylabel("Delta-Gamma ratio")

axs[1].scatter(x = df['DeltaGamma'], y = df['FOOOFslope'], s = 0.7, c = isN3, alpha = 0.3, cmap = 'Dark2')
axs[1].set_xlabel("Delta-Gamma ratio")
axs[1].set_ylabel("FOOOF aperiodic slope")

axs[2].scatter(x = df['DeltaAlpha'], y = df['FOOOFslope'], s = 0.7, c = isN3, alpha = 0.3, cmap = 'Dark2')
axs[2].set_xlabel("Delta-Alpha ratio")
axs[2].set_ylabel("FOOOF aperiodic slope")

  
#%% Stats
clf = svm.LinearSVC(max_iter = 10000)

clf.fit(df.values[:,2:5], isN3)
noncv = clf.score(df.values[:,2:5], isN3)

print("Non cross-validated score: %f" % noncv)


# perform cross-validation on SVM model design
n_splits = 20
gkf = GroupKFold(n_splits=n_splits)

cvscores = np.zeros(0)

for i_train, i_test in gkf.split(X=df.values[:,2:5], y=isN3, groups=df['subj']):
  cvclf = svm.LinearSVC()
  cvclf.fit(df.values[i_train,2:5], isN3[i_train])
  score = cvclf.score(df.values[i_test,2:5], isN3[i_test])
  cvscores = np.append(cvscores, score)
  
  

print("Mean score: %f±%f" % (np.mean(cvscores), np.std(cvscores)))
