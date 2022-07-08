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
import gc

import numpy as np
import tensorflow as tf
from keras import regularizers
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard

try:
    from keras.optimizers import RMSprop # old tf version
except:
    from tensorflow.keras.optimizers import RMSprop

import argparse
import numpy as np
import pandas as pd
from scipy.special import comb
import matplotlib.pyplot as plt
import sys
import os
from scipy.sparse import vstack, csc_matrix
# from utils import dataLoading, aucPerformance, writeResults, get_data_from_svmlight_file
from baseline.DevNet.utils import dataLoading, aucPerformance
from sklearn.model_selection import train_test_split
from myutils import Utils
import time

class DevNet():
    def __init__(self, seed, model_name='DevNet', save_suffix='test'):
        self.utils = Utils()
        self.device = self.utils.get_device()  # get device
        self.seed = seed
        self.MAX_INT = np.iinfo(np.int32).max

        # self.sess = tf.Session() #for old version tf
        self.sess = tf.compat.v1.Session()

        parser = argparse.ArgumentParser()
        parser.add_argument("--network_depth", choices=['1', '2', '4'], default='2',
                            help="the depth of the network architecture")
        parser.add_argument("--batch_size", type=int, default=512, help="batch size used in SGD")
        parser.add_argument("--nb_batch", type=int, default=20, help="the number of batches per epoch")
        parser.add_argument("--epochs", type=int, default=50, help="the number of epochs")
        parser.add_argument("--runs", type=int, default=10,
                            help="how many times we repeat the experiments to obtain the average performance")
        parser.add_argument("--cont_rate", type=float, default=0.02,
                            help="the outlier contamination rate in the training data")
        parser.add_argument("--input_path", type=str, default='./dataset/', help="the path of the data sets")
        parser.add_argument("--data_set", type=str, default='annthyroid', help="a list of data set names")
        parser.add_argument("--output", type=str,
                            default='./results/devnet_auc_performance_30outliers_0.02contrate_2depth_10runs.csv',
                            help="the output file path")
        parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
        # self.args = parser.parse_args()
        self.args, unknown = parser.parse_known_args()

        # network depth
        self.network_depth = int(self.args.network_depth)
        # random_seed = args.ramdn_seed

        self.save_suffix = save_suffix
        if not os.path.exists('baseline/DevNet/model'):
            os.makedirs('baseline/DevNet/model')
        self.ref = None # normal distribution reference, created for reusing across subsequent function calls

    def dev_network_d(self,input_shape):
        '''
        deeper network architecture with three hidden layers
        '''
        x_input = Input(shape=input_shape)
        intermediate = Dense(1000, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
        intermediate = Dense(250, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01), name = 'hl2')(intermediate)
        intermediate = Dense(20, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01), name = 'hl3')(intermediate)
        intermediate = Dense(1, activation='linear', name = 'score')(intermediate)
        return Model(x_input, intermediate)

    def dev_network_s(self,input_shape):
        '''
        network architecture with one hidden layer
        '''
        x_input = Input(shape=input_shape)
        intermediate = Dense(20, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
        intermediate = Dense(1, activation='linear',  name = 'score')(intermediate)
        return Model(x_input, intermediate)

    def dev_network_linear(self,input_shape):
        '''
        network architecture with no hidden layer, equivalent to linear mapping from
        raw inputs to anomaly scores
        '''
        x_input = Input(shape=input_shape)
        intermediate = Dense(1, activation='linear',  name = 'score')(x_input)
        return Model(x_input, intermediate)

    def deviation_loss(self, y_true, y_pred):
        '''
        z-score-based deviation loss
        '''

        confidence_margin = 5.
        ## size=5000 is the setting of l in algorithm 1 in the paper
        if self.ref is None:
            self.ref = K.variable(np.random.normal(loc = 0., scale= 1.0, size = 5000), dtype='float32')
        dev = (y_pred - K.mean(self.ref)) / K.std(self.ref)
        inlier_loss = K.abs(dev)
        outlier_loss = K.abs(K.maximum(confidence_margin - dev, 0.))

        return K.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)

    def deviation_network(self, input_shape, network_depth):
        '''
        construct the deviation network-based detection model
        '''
        if network_depth == 4:
            model = self.dev_network_d(input_shape)
        elif network_depth == 2:
            model = self.dev_network_s(input_shape)
        elif network_depth == 1:
            model = self.dev_network_linear(input_shape)
        else:
            sys.exit("The network depth is not set properly")
        rms = RMSprop(clipnorm=1.)
        model.compile(loss=self.deviation_loss, optimizer=rms)
        return model

    def batch_generator_sup(self, x, outlier_indices, inlier_indices, batch_size, nb_batch, rng):
        """batch generator
        """
        rng = np.random.RandomState(rng.randint(self.MAX_INT, size = 1))
        counter = 0
        while 1:
            ref, training_labels = self.input_batch_generation_sup(x, outlier_indices, inlier_indices, batch_size, rng)
            counter += 1
            yield(ref, training_labels)
            if (counter > nb_batch):
                counter = 0

    def input_batch_generation_sup(self, X_train, outlier_indices, inlier_indices, batch_size, rng):
        '''
        batchs of samples. This is for csv data.
        Alternates between positive and negative pairs.
        '''
        dim = X_train.shape[1]
        ref = np.empty((batch_size, dim))
        training_labels = []
        n_inliers = len(inlier_indices)
        n_outliers = len(outlier_indices)
        for i in range(batch_size):
            if(i % 2 == 0):
                sid = rng.choice(n_inliers, 1)
                ref[i] = X_train[inlier_indices[sid]]
                training_labels += [0]
            else:
                sid = rng.choice(n_outliers, 1)
                ref[i] = X_train[outlier_indices[sid]]
                training_labels += [1]
        return np.array(ref), np.array(training_labels, dtype=float)

    def input_batch_generation_sup_sparse(self, X_train, outlier_indices, inlier_indices, batch_size, rng):
        '''
        batchs of samples. This is for libsvm stored sparse data.
        Alternates between positive and negative pairs.
        '''
        ref = np.empty((batch_size))
        training_labels = []
        n_inliers = len(inlier_indices)
        n_outliers = len(outlier_indices)
        for i in range(batch_size):
            if(i % 2 == 0):
                sid = rng.choice(n_inliers, 1)
                ref[i] = inlier_indices[sid]
                training_labels += [0]
            else:
                sid = rng.choice(n_outliers, 1)
                ref[i] = outlier_indices[sid]
                training_labels += [1]
        ref = X_train[ref, :].toarray()
        return ref, np.array(training_labels)

    def load_model_weight_predict(self, model_name, input_shape, network_depth, X_test):
        '''
        load the saved weights to make predictions
        '''
        model = self.deviation_network(input_shape, network_depth)
        model.load_weights(model_name)
        scoring_network = Model(inputs=model.input, outputs=model.output)

        scores = scoring_network.predict(X_test)

        return scores

    def fit(self, X_train, y_train, ratio=None):
        #index
        outlier_indices = np.where(y_train == 1)[0]
        inlier_indices = np.where(y_train == 0)[0]
        n_outliers = len(outlier_indices)
        print("Training size: %d, No. outliers: %d" % (X_train.shape[0], n_outliers))

        #set seed using myutils
        self.utils.set_seed(self.seed)
        #rng = np.random.RandomState(random_seed)
        rng = np.random.RandomState(self.seed)

        #start time
        self.input_shape = X_train.shape[1:]
        epochs = self.args.epochs
        batch_size = self.args.batch_size
        nb_batch = self.args.nb_batch
        self.model = self.deviation_network(self.input_shape, self.network_depth)
        self.model_name = os.path.join(os.getcwd(),'baseline','DevNet','model','devnet_'+self.save_suffix+'.h5')
        checkpointer = ModelCheckpoint(self.model_name, monitor='loss', verbose=0,
                                       save_best_only = True, save_weights_only = True)

        self.model.fit_generator(self.batch_generator_sup(X_train, outlier_indices, inlier_indices, batch_size, nb_batch, rng),
                                  steps_per_epoch = nb_batch,
                                  epochs = epochs,
                                  callbacks=[checkpointer])

        return self

    def predict_score(self, X):
        score = self.load_model_weight_predict(self.model_name, self.input_shape, self.network_depth, X)
        # score = self.model.predict(X)

        return score