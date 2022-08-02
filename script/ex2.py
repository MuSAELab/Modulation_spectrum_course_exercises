# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:33:12 2022

@author: Yi.Zhu
"""

import msr_func # customized function script for the excercise
import msr_ml # customized machine learning script for the exercise
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__": 
    
    """ Step 1: import preprocessed SRMR data """
    covid_fea = msr_func.load_saved_fea('./preprocessed/covid_srmr') # preprocessed features
    covid_fea = covid_fea.iloc[:,1:].to_numpy() # dataframe to numpy
    covid_fea = 10*np.log10(covid_fea) # raw energy to dB
    covid_lab = msr_func.load_saved_lab('./preprocessed/covid_lab') # labels: COVID-postive/COVID-negative
    
    
    """ Step 2: calculate F-ratio between COVID and non-COVID samples """
    # split features into two groups
    pos_ind = np.where(covid_lab == 1)[0]
    neg_ind = np.where(covid_lab == 0)[0]
    msr_pos = covid_fea[pos_ind]
    msr_neg = covid_fea[neg_ind]
    
    num_fea = covid_fea.shape[1]
    fr = np.empty((num_fea))*np.nan
    
    # calculate F-ratio    
    for i in range(num_fea):
        fr[i] = msr_func.f_ratio(msr_pos[:,i], msr_neg[:,i])
    
    
    """ Step 3: make F-ratio plots"""
    num_freq = 23
    num_mod = 8
    
    assert num_freq*num_mod == num_fea, "number of frequency/modulation frequency bands \
        are inconsistent with saved file"
        
    fr = fr.reshape((num_freq,num_mod))
    msr_func.srmr_plot(fr,require_log=False)
    
    
    """ Step 4: COVID prediction """
    # data preprocessing: train-test-split and normalization
    X_train, y_train, X_test, y_test = msr_ml.data_preprocess(covid_fea,covid_lab)
    
    # train a svm
    # here we use a 3-fold cross-validation to determine the optimal hyper-paramters
    trained_model = msr_ml.train_svm(X_train, y_train)
    
    # model evaluation
    yfit,score = msr_ml.model_evaluate(X_test, y_test, trained_model)
    
    # draw confusion martrix
    msr_ml.get_confmat(y_test,yfit)
