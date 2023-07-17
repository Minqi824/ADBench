#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
import csv
mem = Memory("./dataset/")

@mem.cache
def get_data_from_svmlight_file(path):
    data = load_svmlight_file(path)
    return data[0], data[1]

def dataLoading(path, byte_num):
    # loading data
    x=[]
    labels=[]
    
    with (open(path,'r')) as data_from:
        csv_reader=csv.reader(data_from)
        for i in csv_reader:
            x.append(i[0:byte_num])
            labels.append(i[byte_num])

    for i in range(len(x)):
        for j in range(byte_num):
            x[i][j] = float(x[i][j])
    for i in range(len(labels)):
        labels[i] = float(labels[i])
    x = np.array(x)
    labels = np.array(labels)

    return x, labels;


def aucPerformance(mse, labels):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;

def writeResults(name, n_samples_trn,  n_outliers, n_samples_test,test_outliers ,test_inliers, avg_AUC_ROC, avg_AUC_PR, std_AUC_ROC,std_AUC_PR, path):    
    csv_file = open(path, 'a') 
    row = name + ","  + n_samples_trn + ','+n_outliers  + ','+n_samples_test+','+test_outliers+','+test_inliers+','+avg_AUC_ROC+','+avg_AUC_PR+','+std_AUC_ROC+','+std_AUC_PR + "\n"
    csv_file.write(row)

