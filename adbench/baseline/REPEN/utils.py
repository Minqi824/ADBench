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

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.datasets import load_svmlight_file
import numpy as np


def get_data_from_svmlight_file(path):
    data = load_svmlight_file(path)
    return data[0], data[1]


def dataLoading(path):
    # loading data
    df = pd.read_csv(path)

    labels = df['class']

    x_df = df.drop(['class'], axis=1)

    x = x_df.values
    print("Data shape: (%d, %d)" % x.shape)

    return x, labels;


def aucPerformance(mse, labels):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;


def writeResults(name, n_samples_trn, n_outliers_trn, n_outliers, ap, std_ap, cr, path="./results/auc_performance.csv"):
    csv_file = open(path, 'a')
    row = name + "," + str(n_samples_trn) + ',' + str(n_outliers_trn) + ',' + str(n_outliers) + ',' + str(
        ap) + "," + str(std_ap) + "," + str(cr) + "," + "\n"
    csv_file.write(row)


def cutoff_unsorted(values, th=1.7321):
    #    print(values)
    v_mean = np.mean(values)
    v_std = np.std(values)
    th = v_mean + th * v_std  # 1.7321
    if th >= np.max(values):  # return the top-10 outlier scores
        temp = np.sort(values)
        th = temp[-11]

    outlier_ind = np.where(values > th)[0]
    inlier_ind = np.where(values <= th)[0]
    return inlier_ind, outlier_ind;