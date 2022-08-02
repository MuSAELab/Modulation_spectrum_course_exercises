# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:14:13 2022

@author: Yi.Zhu
"""

"""
Modulation spectrogram toolbox - Exercise 1

1.Generate modulation spectrogram from single audio file;
2.Visualize 2D modulation spectrogram;
3.Compute F-ratio topographical plots to visualize group difference.
"""

import msr_func # customized function script for the excercise
import librosa # speech related function library
import librosa.display # to display mel-spectrogram
import numpy as np
import matplotlib.pyplot as plt
 

if __name__ == "__main__": 
    
    
    """ Step 1: load a single audio file """
    ex_speech = msr_func.load_audio('./data/speech/noncovid.flac', fs=16000) # modify it to your own path if necessary
    
    
    """ Step 2: generate modulation spectrogram using SRMR toolbox """
    # parameters to generate modulation spectrogram
    fs = 16000 # sampling frequency
    n_cochlear_filters = 23 # number of acoustic frequency bands
    low_freq = 125 # lowest frequency cutoff
    min_cf = 2 # lower bound of modulation frequency  
    max_cf= 32 # upper bound of modulation frequency
   
    _, ex_srmr = msr_func.single_srmr(ex_speech,
                                      fs=fs,
                                      n_cochlear_filters=n_cochlear_filters,
                                      low_freq=low_freq,
                                      min_cf=min_cf,
                                      max_cf=max_cf)
    
    
    """ Step 3: plot modulation spectrogram (SRMR) """
    # average the SRMR tensor over time axis
    ave_srmr = np.mean(ex_srmr,axis=2)
    
    # plot temporally averaged SRMR plot
    msr_func.srmr_plot(ave_srmr)
    
    
    """ Step 4: plot mel-spectrogram for comparison """
    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=ex_speech, 
                                       sr=fs, 
                                       n_mels=128, 
                                       fmax=8000)
    fig, ax = plt.subplots(dpi=300)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, 
                                   x_axis='time',
                                   y_axis='mel', 
                                   sr=fs,
                                   fmax=8000, 
                                   ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    
    
    """ Step 5: Try the COVID speech recording or your own samples """
    
    
