# -*- coding: utf-8 -*- 
"""
@author：Xucheng Song
The algorithm was implemented using Python 3.6.12, Keras 2.3.1 and TensorFlow 1.13.1 based on the code (https://github.com/GuansongPang/deviation-network).
The major contributions are summarized as follows.
This code adds a feature encoder to encode the input data and utilizes three factors, hidden representation, reconstruction residual vector,
and reconstruction error, as the new representation for the input data. The representation is then fed into an MLP based anomaly score generator,
similar to the code (https://github.com/GuansongPang/deviation-network), but with a twist, i.e., the reconstruction error is fed to each layer
of the MLP in the anomaly score generator. A different loss function in the anomaly score generator is also included. Additionally,
the pre-training procedure is adopted in the training process. More details can be found in our TNNLS paper as follows.
Yingjie Zhou, Xucheng Song, Yanru Zhang, Fanxing Liu, Ce Zhu and Lingqiao Liu,
Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection,
in IEEE Transactions on Neural Networks and Learning Systems, 2021, 12 pages,
which can be found in IEEE Xplore or arxiv (https://arxiv.org/abs/2105.10500).
"""
import argparse
import numpy as np
import os
import sys
from scipy.sparse import vstack, csc_matrix
from myutils import Utils
import gc

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Subtract,concatenate,Lambda,Reshape
from keras.callbacks import ModelCheckpoint
from keras.losses import mean_squared_error

try:
    from keras.optimizers import Adam, RMSprop
except:
    from tensorflow.keras.optimizers import Adam, RMSprop

# Disable TF eager execution mode for avoid the errors caused by the custom loss function
# the disable_eager_execution may occur error with DeepSVDD in pyod (2022.08.05)
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

class FEAWAD():
    def __init__(self, seed, model_name='FEAWAD', save_suffix='test'):
        self.utils = Utils()
        self.device = self.utils.get_device()  # get device
        self.seed = seed
        self.MAX_INT = np.iinfo(np.int32).max

        # self.sess = tf.Session() #for old version tf
        self.sess = tf.compat.v1.Session()

        parser = argparse.ArgumentParser()
        parser.add_argument("--network_depth", choices=['1', '2', '4'], default='4',
                            help="the depth of the network architecture")
        parser.add_argument("--batch_size", type=int, default=512, help="batch size used in SGD")
        parser.add_argument("--nb_batch", type=int, default=20, help="the number of batches per epoch")
        parser.add_argument("--epochs", type=int, default=30, help="the number of epochs")
        parser.add_argument("--runs", type=int, default=10,
                            help="how many times we repeat the experiments to obtain the average performance")
        parser.add_argument("--known_outliers", type=int, default=30,
                            help="the number of labeled outliers available at hand")
        parser.add_argument("--cont_rate", type=float, default=0.02,
                            help="the outlier contamination rate in the training data")
        parser.add_argument("--input_path", type=str, default='./dataset/', help="the path of the data sets")
        parser.add_argument("--data_set", type=str, default='nslkdd_normalization', help="a list of data set names")
        parser.add_argument("--data_format", choices=['0', '1'], default='0',
                            help="specify whether the input data is a csv (0) or libsvm (1) data format")
        parser.add_argument("--data_dim", type=int, default=122, help="the number of dims in each data sample")
        parser.add_argument("--output", type=str, default='./proposed_devnet_auc_performance.csv',
                            help="the output file path")
        parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
        # self.args = parser.parse_args()
        self.args, unknown = parser.parse_known_args()
        self.data_format = 0

        self.save_suffix = save_suffix
        if not os.path.exists('baseline/FEAWAD/model'):
            os.makedirs('baseline/FEAWAD/model')

    def auto_encoder(self, input_shape):
        x_input = Input(shape=input_shape)
        length = K.int_shape(x_input)[1]

        input_vector = Dense(length, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'ain')(x_input)
        en1 = Dense(128, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'ae1')(input_vector)
        en2 = Dense(64,kernel_initializer='glorot_normal', use_bias=True,activation='relu',name = 'ae2')(en1)
        de1 = Dense(128, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'ad1')(en2)
        de2 = Dense(length, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'ad2')(de1)

        model =  Model(x_input, de2)
        adm = Adam(lr=0.0001)
        model.compile(loss=mean_squared_error, optimizer=adm)

        return model

    def dev_network_d(self, input_shape, modelname, testflag):
        '''
        deeper network architecture with three hidden layers
        '''
        x_input = Input(shape=input_shape)
        length = K.int_shape(x_input)[1]

        input_vector = Dense(length, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'ain')(x_input)
        en1 = Dense(128, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'ae1')(input_vector)
        en2 = Dense(64,kernel_initializer='glorot_normal', use_bias=True,activation='relu',name = 'ae2')(en1)
        de1 = Dense(128, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'ad1')(en2)
        de2 = Dense(length, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'ad2')(de1)

        if testflag==0:
            AEmodel = Model(x_input,de2)
            AEmodel.load_weights(modelname)
            print('load autoencoder model')

            sub_result = Subtract()([x_input, de2]) # reconstruction residual error
            cal_norm2 = Lambda(lambda x: tf.norm(x,ord = 2,axis=1))
            sub_norm2 = cal_norm2(sub_result)
            sub_norm2 = Reshape((1,))(sub_norm2)
            division = Lambda(lambda x:tf.divide(x[0],x[1]))
            sub_result = division([sub_result,sub_norm2]) # normalized reconstruction residual error
            conca_tensor = concatenate([sub_result,en2],axis=1) # [hidden representation, normalized reconstruction residual error]

            conca_tensor = concatenate([conca_tensor,sub_norm2],axis=1) # [hidden representation, normalized reconstruction residual error, residual error]
        else:
            sub_result = Subtract()([x_input, de2])
            cal_norm2 = Lambda(lambda x: tf.norm(x,ord = 2,axis=1))
            sub_norm2 = cal_norm2(sub_result)
            sub_norm2 = Reshape((1,))(sub_norm2)
            division = Lambda(lambda x:tf.divide(x[0],x[1]))
            sub_result = division([sub_result,sub_norm2])
            conca_tensor = concatenate([sub_result,en2],axis=1)

            conca_tensor = concatenate([conca_tensor,sub_norm2],axis=1)

        intermediate = Dense(256, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'hl2')(conca_tensor)
        intermediate = concatenate([intermediate,sub_norm2],axis=1) # concat the intermediate vector with the residual error
        intermediate = Dense(32, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'hl3')(intermediate)
        intermediate = concatenate([intermediate,sub_norm2],axis=1) # again, concat the intermediate vector with the residual error
        output_pre = Dense(1, kernel_initializer='glorot_normal',use_bias=True,activation='linear', name = 'score')(intermediate)
        dev_model = Model(x_input, output_pre)

        def multi_loss(y_true,y_pred):
            confidence_margin = 5.

            dev = y_pred
            inlier_loss = K.abs(dev)
            outlier_loss = K.abs(K.maximum(confidence_margin - dev, 0.))

            sub_nor = tf.norm(sub_result,ord = 2,axis=1)
            outlier_sub_loss = K.abs(K.maximum(confidence_margin - sub_nor, 0.))
            loss1 =  (1 - y_true) * (inlier_loss+sub_nor) + y_true * (outlier_loss+outlier_sub_loss)

            return loss1

        adm = Adam(lr=0.0001)
        dev_model.compile(loss=multi_loss, optimizer=adm)

        return dev_model

    def deviation_network(self, input_shape, network_depth, modelname, testflag):
        '''
        construct the deviation network-based detection model
        '''
        if network_depth == 4:
            model = self.dev_network_d(input_shape, modelname, testflag)
        elif network_depth == 2:
            model = self.auto_encoder(input_shape)

        else:
            sys.exit("The network depth is not set properly")
        return model

    def auto_encoder_batch_generator_sup(self, x, inlier_indices, batch_size, nb_batch, rng):
        """auto encoder batch generator
        """
        self.utils.set_seed(self.seed)
        # rng = np.random.RandomState(rng.randint(self.MAX_INT, size = 1))
        rng = np.random.RandomState(np.random.randint(self.MAX_INT, size=1))
        counter = 0
        while 1:
            if self.data_format == 0:
                ref, training_labels = self.AE_input_batch_generation_sup(x, inlier_indices, batch_size, rng)
            else:
                ref, training_labels = self.input_batch_generation_sup_sparse(x, inlier_indices, batch_size, rng)
            counter += 1
            yield(ref, training_labels)
            if (counter > nb_batch):
                counter = 0

    def AE_input_batch_generation_sup(self, train_x, inlier_indices, batch_size, rng):
        '''
        batchs of samples. This is for csv data.
        Alternates between positive and negative pairs.
        '''
        rng = np.random.RandomState(self.seed)

        dim = train_x.shape[1]
        ref = np.empty((batch_size, dim))
        training_labels = np.empty((batch_size, dim))
        n_inliers = len(inlier_indices)
        for i in range(batch_size):
            sid = rng.choice(n_inliers, 1)
            ref[i] = train_x[inlier_indices[sid]]
            training_labels[i] = train_x[inlier_indices[sid]]
        return np.array(ref), np.array(training_labels, dtype=float)

    def batch_generator_sup(self, x, outlier_indices, inlier_indices, batch_size, nb_batch, rng):
        """batch generator
        """
        self.utils.set_seed(self.seed)
        # rng = np.random.RandomState(rng.randint(self.MAX_INT, size = 1))
        rng = np.random.RandomState(np.random.randint(self.MAX_INT, size=1))
        counter = 0
        while 1:
            if self.data_format == 0:
                ref, training_labels = self.input_batch_generation_sup(x, outlier_indices, inlier_indices, batch_size, rng)
            else:
                ref, training_labels = self.input_batch_generation_sup_sparse(x, outlier_indices, inlier_indices, batch_size, rng)
            counter += 1
            yield(ref, training_labels)
            if (counter > nb_batch):
                counter = 0

    def input_batch_generation_sup(self, train_x, outlier_indices, inlier_indices, batch_size, rng):
        '''
        batchs of samples. This is for csv data.
        Alternates between positive and negative pairs.
        '''
        rng = np.random.RandomState(self.seed)
        dim = train_x.shape[1]
        ref = np.empty((batch_size, dim))
        training_labels = []
        n_inliers = len(inlier_indices)
        n_outliers = len(outlier_indices)
        for i in range(batch_size):
            if(i % 2 == 0):
                sid = rng.choice(n_inliers, 1)
                ref[i] = train_x[inlier_indices[sid]]
                training_labels += [0]
            else:
                sid = rng.choice(n_outliers, 1)
                ref[i] = train_x[outlier_indices[sid]]
                training_labels += [1]
        return np.array(ref), np.array(training_labels, dtype=float)

    def input_batch_generation_sup_sparse(self, train_x, outlier_indices, inlier_indices, batch_size, rng):
        '''
        batchs of samples. This is for libsvm stored sparse data.
        Alternates between positive and negative pairs.
        '''
        rng = np.random.RandomState(self.seed)
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
        ref = train_x[ref, :].toarray()
        return ref, np.array(training_labels, dtype=float)

    def load_model_weight_predict(self, model_name, input_shape, network_depth, test_x):
        '''
        load the saved weights to make predictions
        '''
        model = self.deviation_network(input_shape, network_depth,model_name,1)
        model.load_weights(model_name)
        scoring_network = Model(inputs=model.input, outputs=model.output)

        if self.data_format == 0:
            scores = scoring_network.predict(test_x)
        else:
            data_size = test_x.shape[0]
            scores = np.zeros([data_size, 1])
            count = 512
            i = 0
            while i < data_size:
                subset = test_x[i:count].toarray()
                scores[i:count] = scoring_network.predict(subset)
                if i % 1024 == 0:
                    print(i)
                i = count
                count += 512
                if count > data_size:
                    count = data_size
            assert count == data_size
        return scores

    def inject_noise_sparse(self, seed, n_out, random_seed):
        '''
        add anomalies to training data to replicate anomaly contaminated data sets.
        we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
        This is for sparse data.
        '''
        rng = np.random.RandomState(random_seed)
        n_sample, dim = seed.shape
        swap_ratio = 0.05
        n_swap_feat = int(swap_ratio * dim)
        seed = seed.tocsc()
        noise = csc_matrix((n_out, dim))
        print(noise.shape)
        for i in np.arange(n_out):
            outlier_idx = rng.choice(n_sample, 2, replace = False)
            o1 = seed[outlier_idx[0]]
            o2 = seed[outlier_idx[1]]
            swap_feats = rng.choice(dim, n_swap_feat, replace = False)
            noise[i] = o1.copy()
            noise[i, swap_feats] = o2[0, swap_feats]
        return noise.tocsr()

    def inject_noise(self, seed, n_out, random_seed):
        '''
        add anomalies to training data to replicate anomaly contaminated data sets.
        we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
        this is for dense data
        '''
        rng = np.random.RandomState(random_seed)
        n_sample, dim = seed.shape
        swap_ratio = 0.05
        n_swap_feat = int(swap_ratio * dim)
        noise = np.empty((n_out, dim))
        for i in np.arange(n_out):
            outlier_idx = rng.choice(n_sample, 2, replace = False)
            o1 = seed[outlier_idx[0]]
            o2 = seed[outlier_idx[1]]
            swap_feats = rng.choice(dim, n_swap_feat, replace = False)
            noise[i] = o1.copy()
            noise[i, swap_feats] = o2[swap_feats]
        return noise

    def fit(self, X_train, y_train, ratio=None):
        # network_depth = int(self.args.network_depth)
        self.utils.set_seed(self.seed)
        rng = np.random.RandomState(self.seed)

        # index
        outlier_indices = np.where(y_train == 1)[0]
        inlier_indices = np.where(y_train == 0)[0]

        # X_train_inlier = np.delete(X_train, outlier_indices, axis=0)
        self.input_shape = X_train.shape[1:]

        # pre-trained autoencoder
        self.utils.set_seed(self.seed)
        AEmodel = self.deviation_network(self.input_shape, 2, None, 0)  # pretrain auto-encoder model
        print('autoencoder pre-training start....')
        AEmodel_name = os.path.join(os.getcwd(), 'baseline', 'FEAWAD', 'model', 'pretrained_autoencoder_'+self.save_suffix+'.h5')
        ae_checkpointer = ModelCheckpoint(AEmodel_name, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True)
        AEmodel.fit_generator(self.auto_encoder_batch_generator_sup(X_train, inlier_indices, self.args.batch_size, self.args.nb_batch, rng),
                                         steps_per_epoch=self.args.nb_batch, epochs=100, callbacks=[ae_checkpointer])


        #end-to-end devnet model
        print('load pretrained autoencoder model....')
        self.utils.set_seed(self.seed)
        self.dev_model = self.deviation_network(self.input_shape, 4, AEmodel_name, 0)
        print('end-to-end training start....')
        self.dev_model_name = os.path.join(os.getcwd(), 'baseline', 'FEAWAD', 'model', 'devnet_'+self.save_suffix+'.h5')
        checkpointer = ModelCheckpoint(self.dev_model_name, monitor='loss', verbose=0,
                                       save_best_only=True, save_weights_only=True)
        self.dev_model.fit_generator(self.batch_generator_sup(X_train, outlier_indices, inlier_indices, self.args.batch_size, self.args.nb_batch, rng),
                                      steps_per_epoch=self.args.nb_batch,
                                      epochs=self.args.epochs,
                                      callbacks=[checkpointer])

        return self

    def predict_score(self, X):
        score = self.load_model_weight_predict(self.dev_model_name, self.input_shape, 4, X)
        return score