from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras.optimizers import SGD
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import keras
import math
import os
from scipy.special import comb
import argparse

#local additive packages
from tqdm import tqdm
from itertools import product
import random
import tensorflow as tf
from keras.models import load_model
import time
from myutils import Utils

from sklearn.metrics import roc_curve,auc,average_precision_score
import warnings
warnings.filterwarnings("ignore")

class MOGAAL():
    def __init__(self, seed, model_name='MOGAAL'):
        self.seed = seed
        self.utils = Utils()

        parser = argparse.ArgumentParser(description="Run MO-GAAL.")
        parser.add_argument('--k', type=int, default=10,
                            help='Number of sub_generator.')
        parser.add_argument('--stop_epochs', type=int, default=25,
                            help='Stop training generator after stop_epochs.')
        parser.add_argument('--lr_d', type=float, default=0.01,
                            help='Learning rate of discriminator.')
        parser.add_argument('--lr_g', type=float, default=0.0001,
                            help='Learning rate of generator.')
        parser.add_argument('--decay', type=float, default=1e-6,
                            help='Decay.')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='Momentum.')

        # self.args = parser.parse_args()
        self.args, unknown = parser.parse_known_args()

    # Generator
    def create_generator(self, latent_size):
        gen = Sequential()
        gen.add(Dense(latent_size, input_dim=latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
        gen.add(Dense(latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
        latent = Input(shape=(latent_size,))
        fake_data = gen(latent)
        return Model(latent, fake_data)

    # Discriminator
    def create_discriminator(self, data_size, latent_size):
        dis = Sequential()
        dis.add(Dense(math.ceil(math.sqrt(data_size)), input_dim=latent_size, activation='relu', kernel_initializer= keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
        dis.add(Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
        data = Input(shape=(latent_size,))
        fake = dis(data)
        return Model(data, fake)

    def fit(self, X_train, y_train, print_log=False):
        # set seed
        self.utils.set_seed(self.seed)

        # initialize dataset
        data_x, data_y = X_train, y_train
        data_size = data_x.shape[0]
        latent_size = data_x.shape[1]

        verbose = 0
        if print_log:
            print("The dimension of the training data :{}*{}".format(data_size, latent_size))
            verbose = 1

        train_history = defaultdict(list)
        names = locals()
        epochs = self.args.stop_epochs * 3
        stop = 0
        k = self.args.k

        # Create discriminator
        discriminator = self.create_discriminator(data_size = data_size,latent_size = latent_size)
        discriminator.compile(optimizer=SGD(lr=self.args.lr_d, decay=self.args.decay, momentum=self.args.momentum),
                              loss='binary_crossentropy')

        # Create k combine models
        for i in range(k):
            names['sub_generator' + str(i)] = self.create_generator(latent_size)
            latent = Input(shape=(latent_size,))
            names['fake' + str(i)] = names['sub_generator' + str(i)](latent)
            discriminator.trainable = False
            names['fake' + str(i)] = discriminator(names['fake' + str(i)])
            names['combine_model' + str(i)] = Model(latent, names['fake' + str(i)])
            names['combine_model' + str(i)].compile(optimizer=SGD(lr=self.args.lr_g, decay=self.args.decay, momentum=self.args.momentum),
                                                    loss='binary_crossentropy')

        # Start iteration
        for epoch in range(epochs):
            if print_log:
                print('Epoch {} of {}'.format(epoch + 1, epochs))
            batch_size = min(500, data_size)
            num_batches = int(data_size / batch_size)

            for index in range(num_batches):
                if print_log:
                    print('\nTesting for epoch {} index {}:'.format(epoch + 1, index + 1))

                # Generate noise
                noise_size = batch_size
                noise = np.random.uniform(0, 1, (int(noise_size), latent_size))

                # Get training data
                data_batch = data_x[index * batch_size: (index + 1) * batch_size]

                # Generate potential outliers
                block = ((1 + k) * k) // 2
                for i in range(k):
                    if i != (k - 1):
                        noise_start = int((((k + (k - i + 1)) * i) / 2) * (noise_size // block))
                        noise_end = int((((k + (k - i)) * (i + 1)) / 2) * (noise_size // block))
                        names['noise' + str(i)] = noise[noise_start: noise_end]
                        names['generated_data' + str(i)] = names['sub_generator' + str(i)].predict(names['noise' + str(i)],
                                                                                                   verbose=verbose)
                    else:
                        noise_start = int((((k + (k - i + 1)) * i) / 2) * (noise_size // block))
                        names['noise' + str(i)] = noise[noise_start: noise_size]
                        names['generated_data' + str(i)] = names['sub_generator' + str(i)].predict(names['noise' + str(i)],
                                                                                                   verbose=verbose)

                # Concatenate real data to generated data
                for i in range(k):
                    if i == 0:
                        X = np.concatenate((data_batch, names['generated_data' + str(i)]))
                    else:
                        X = np.concatenate((X, names['generated_data' + str(i)]))
                Y = np.array([1] * batch_size + [0] * int(noise_size))

                # Train discriminator
                discriminator_loss = discriminator.train_on_batch(X, Y)
                train_history['discriminator_loss'].append(discriminator_loss)

                # Get the target value of sub-generator
                p_value = discriminator.predict(data_x)
                p_value = pd.DataFrame(p_value)
                for i in range(k):
                    names['T' + str(i)] = p_value.quantile(i / k)
                    names['trick' + str(i)] = np.array([float(names['T' + str(i)])] * noise_size)

                # Train generator
                noise = np.random.uniform(0, 1, (int(noise_size), latent_size))
                if stop == 0:
                    for i in range(k):
                        names['sub_generator' + str(i) + '_loss'] = names['combine_model' + str(i)].train_on_batch(noise,
                                                                                                                   names[
                                                                                                                       'trick' + str(
                                                                                                                           i)])
                        train_history['sub_generator{}_loss'.format(i)].append(names['sub_generator' + str(i) + '_loss'])
                else:
                    for i in range(k):
                        names['sub_generator' + str(i) + '_loss'] = names['combine_model' + str(i)].evaluate(noise, names[
                            'trick' + str(i)],verbose = verbose)
                        train_history['sub_generator{}_loss'.format(i)].append(names['sub_generator' + str(i) + '_loss'])

                generator_loss = 0
                for i in range(k):
                    generator_loss = generator_loss + names['sub_generator' + str(i) + '_loss']
                generator_loss = generator_loss / k
                train_history['generator_loss'].append(generator_loss)

                # Stop training generator
                if epoch + 1 > self.args.stop_epochs:
                    stop = 1

                    # saving model
                if (epoch + 1) == epochs:
                    # discriminator weights
                    discriminator.save(os.path.join('baseline','MOGAAL','MOGAAL_D.h5'))

            # Detection result
            data_y = pd.DataFrame(data_y)
            result = np.concatenate((p_value, data_y), axis=1)
            result = pd.DataFrame(result, columns=['p', 'y'])
            result = result.sort_values('p', ascending=True)

        return self

    # evaluate
    def predict_score(self, X):
        data_x = X

        discriminator_all = load_model(os.path.join('baseline', 'MOGAAL', 'MOGAAL_D.h5'))
        p_value = discriminator_all.predict(data_x)
        p_value = pd.DataFrame(p_value)
        result = pd.DataFrame(p_value, columns=['p'])
        result = result.sort_values('p', ascending=True)
        score = np.array(1 - result['p'])

        return score