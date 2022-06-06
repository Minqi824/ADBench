from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras.optimizers import SGD
import numpy as np
import pandas as pd
from collections import defaultdict
import keras
import math
import argparse
from keras.models import load_model
import os
from sklearn.metrics import roc_curve, auc, average_precision_score
from myutils import Utils
import warnings
warnings.filterwarnings("ignore")

class SOGAAL():
    def __init__(self, seed, model_name='SOGAAL'):
        self.seed = seed
        self.utils = Utils()

        parser = argparse.ArgumentParser(description="Run SO-GAAL.")
        parser.add_argument('--k', type=int, default=10,
                            help='Number of sub_generator.')
        # default = 25
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

        data_x, data_y = X_train, y_train
        verbose = 0
        if print_log:
            print("The dimension of the training data :{}*{}".format(data_x.shape[0], data_x.shape[1]))
            verbose = 1
        latent_size = data_x.shape[1]
        data_size = data_x.shape[0]
        stop = 0
        epochs = self.args.stop_epochs * 3

        train_history = defaultdict(list)

        # Create discriminator
        discriminator = self.create_discriminator(data_size = data_size,latent_size = latent_size)
        discriminator.compile(optimizer=SGD(lr=self.args.lr_d, decay=self.args.decay, momentum=self.args.momentum),
                              loss='binary_crossentropy')

        # Create combine model
        generator = self.create_generator(latent_size)
        latent = Input(shape=(latent_size,))
        fake = generator(latent)
        discriminator.trainable = False
        fake = discriminator(fake)
        combine_model = Model(latent, fake)
        combine_model.compile(optimizer=SGD(lr=self.args.lr_g, decay=self.args.decay, momentum=self.args.momentum),
                              loss='binary_crossentropy')

        # Start iteration
        print('Training...')
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
                generated_data = generator.predict(noise, verbose=verbose)

                # Concatenate real data to generated data
                X = np.concatenate((data_batch, generated_data))
                Y = np.array([1] * batch_size + [0] * int(noise_size))

                # Train discriminator
                discriminator_loss = discriminator.train_on_batch(X, Y)
                train_history['discriminator_loss'].append(discriminator_loss)

                # Train generator
                if stop == 0:
                    trick = np.array([1] * noise_size)
                    generator_loss = combine_model.train_on_batch(noise, trick)
                    train_history['generator_loss'].append(generator_loss)
                else:
                    trick = np.array([1] * noise_size)
                    generator_loss = combine_model.evaluate(noise, trick,verbose=verbose)
                    train_history['generator_loss'].append(generator_loss)

            # Stop training generator
            if epoch + 1 > self.args.stop_epochs:
                stop = 1

            # saving model
            if (epoch + 1) == epochs:
                # discriminator weights
                discriminator.save(os.path.join('baseline','SOGAAL','SOGAAL_D.h5'))

            # Detection result
            # p_value = discriminator.predict(data_x)
            # p_value = pd.DataFrame(p_value)
            # data_y = pd.DataFrame(data_y)
            # result = np.concatenate((p_value, data_y), axis=1)
            # result = pd.DataFrame(result, columns=['p', 'y'])
            # result = result.sort_values('p', ascending=True)

        return self

    # evaluate
    def predict_score(self, X, phase=None):
        data_x = X
        discriminator_all = load_model(os.path.join('baseline', 'SOGAAL', 'SOGAAL_D.h5'))
        p_value = discriminator_all.predict(data_x)
        result = pd.DataFrame(p_value, columns=['p'])
        result = result.sort_values('p', ascending=True)
        score = np.array(1 - result['p'])

        return score