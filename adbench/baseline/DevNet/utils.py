#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guansong Pang
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019. 
Deep Anomaly Detection with Deviation Networks. 
In The 25th ACM SIGKDDConference on Knowledge Discovery and Data Mining (KDD ’19),
August4–8, 2019, Anchorage, AK, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3292500.3330871
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import auc,roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
# from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file

# mem = Memory("./dataset/svm_data")

# @mem.cache
# def get_data_from_svmlight_file(path):
#     data = load_svmlight_file(path)
#     return data[0], data[1]

def dataLoading(input_path:str, dataset, RIA, seed):
    # loading data
    data_ul = pd.read_csv(os.path.join(input_path, dataset + '_ul_' + str(RIA) + '_' + str(seed) + '.csv'))
    data_ia = pd.read_csv(os.path.join(input_path, dataset + '_ia_' + str(RIA) + '_' + str(seed) + '.csv'))
    data_test = pd.read_csv(os.path.join(input_path, dataset + '_test' + '_' + str(seed) + '.csv'))

    data_train = pd.concat([data_ul,data_ia], axis=0)
    #rename column
    data_train.rename(columns={'Unnamed: 0': 'index'}, inplace = True)
    #sort data by the orginal index
    data_train = data_train.sort_values('index', ascending = True)
    data_train.reset_index(drop = True, inplace=True)

    # training set
    X_train = np.array(data_train.drop(['index', 'y','y_gt'], axis=1))
    y_train = np.array(data_train['y']).astype('float64')
    y_train_gt = np.array(data_train['y_gt']).astype('float64')

    # testing set
    X_test = np.array(data_test.drop(['Unnamed: 0', 'y','y_gt'], axis=1))
    y_test = np.array(data_test['y']).astype('float64')
    
    return X_train,y_train,y_train_gt,X_test,y_test


def aucPerformance(mse, labels):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;


