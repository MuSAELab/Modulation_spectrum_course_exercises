# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 22:13:05 2022

@author: Yi.Zhu
"""

import numpy as np
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def data_preprocess(x,y):
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)
    
    return X_train, y_train, X_test, y_test


def train_svm(X_train,y_train):
    
    params = {'C':[0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.007,0.01,0.05,0.1,0.5,1]}
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=26)
    model = GridSearchCV(SVC(kernel = 'linear',probability=True), param_grid=params,cv=cv,scoring='roc_auc')
    model.fit(X_train,y_train)
    print (model.best_params_)

    return model


def model_evaluate(X_test,y_test,trained_model):
    
    yfit = trained_model.predict(X_test)
    print(classification_report(y_test,yfit))
    y_prob = trained_model.predict_proba(X_test)
    score = sklearn.metrics.roc_auc_score(y_test,y_prob[:,1])
    print('Test AUC-ROC score' + str(score))
    
    return yfit,score


def get_confmat(label,prediction):
    cf_matrix = confusion_matrix(label,prediction)

    group_names = ['True neg','False pos','False neg','True pos']
    group_percentages1 = ["{0:.2%}".format(value) for value in
                          cf_matrix[0]/np.sum(cf_matrix[0])]
    group_percentages2 = ["{0:.2%}".format(value) for value in
                          cf_matrix[1]/np.sum(cf_matrix[1])]

    group_percentages = group_percentages1+group_percentages2
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2,v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    yticklabels = ['negative','positive']
    xticklabels = ['negative','positive']
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues',yticklabels=yticklabels,
                xticklabels=xticklabels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

