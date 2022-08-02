# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 17:41:05 2022

@author: Yi.Zhu
"""

"""
Functions to compute modulation spectrograms and to generate corresponding plots.
"""

from srmrpy import srmr
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pickle as pkl


def load_audio(ad,fs:int=16000):
    data, fs = librosa.load(ad,sr=fs)
    return data

def single_srmr(ad,  
                fs:int=16000,
                n_cochlear_filters:int=23,
                low_freq:int=125,
                min_cf:int=2,
                max_cf:int=32,
                norm=True):
    
    ratio, s = srmr(ad, fs=fs, n_cochlear_filters=n_cochlear_filters, low_freq=low_freq, min_cf=min_cf, max_cf=max_cf, fast=True, norm=norm)
    
    return ratio, s

# temporally averaged SRMR
def group_srmr(ad_list,
             fs:int=16000,
             n_cochlear_filters:int=23,
             low_freq:int=125,
             min_cf:int=2,
             max_cf:int=32,
             norm=True
             ):
    # s = np.empty((len(ad_list),8*23))*np.nan
    s=np.empty((len(ad_list),23,8))*np.nan
    r=np.empty((len(ad_list),1))*np.nan
    for i,j in enumerate(ad_list):
        ratio , res = srmr(j, fs=fs, n_cochlear_filters=n_cochlear_filters, low_freq=low_freq, min_cf=min_cf, max_cf=max_cf, fast=True, norm=norm)
        s[i,:,:] = np.mean(res,axis=2) # average across frames then flatten
        r[i] = ratio
    
    return r, s

# plot average SRMR image
def srmr_plot(srmr_img,
              require_log=True):
    
    assert srmr_img.ndim == 2, "srmr input shape needs to be 2D"
    
    num_y = srmr_img.shape[0]
    num_x = srmr_img.shape[1]
    
    plt.figure(dpi=300)
    
    if require_log is True:
        plt.pcolormesh(10*np.log10(srmr_img))
        
    elif require_log is False:
        plt.pcolormesh(srmr_img)
        
    plt.colorbar()
    plt.xticks(np.arange(0,num_x+1,1),np.arange(0,num_x+1,1))
    plt.yticks(np.arange(0,num_y+1,1),np.arange(0,num_y+1,1))
    plt.xlabel('Modulation frequency channel')
    plt.ylabel('Acoustic frequency channel')
    
    print('plot finished!')
    
    return 0

# calculate F-ratio between two groups
def f_ratio (g1,g2):
    # g1 and g2 are sample values from two groups
    n1 = g1.size
    n2 = g2.size
    s1 = np.sum(g1)
    s2 = np.sum(g2)
    
    SS_b = np.square(s1)/n1 + np.square(s2/n2) - np.square(s1+s2)/(n1+n2)
    
    SS_total = (np.sum(np.square(g1))+np.sum(np.square(g2)))-np.square(s1+s2)/(n1+n2)
    
    SS_w = SS_total - SS_b
    
    MS_b = SS_b
    MS_w = SS_w/(n1+n2-2)
    
    return MS_b/MS_w

def load_saved_fea(fea_name:str):
    with open("%s.pkl"%(fea_name), "rb") as infile:
        saved_fea = pkl.load(infile)
    return saved_fea


def load_saved_lab(file_name:str):
    with open("%s.pkl"%(file_name), "rb") as infile:
        saved_lab = pkl.load(infile)
        saved_lab = saved_lab.astype(np.int)
    return saved_lab

