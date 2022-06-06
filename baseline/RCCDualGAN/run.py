from keras.layers import Input, Dense, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import keras
import math
from scipy.special import comb
import argparse
from keras.models import load_model
from baseline.RCCDualGAN import RCC

#local addtive packages
from tqdm import tqdm
from itertools import product
import random
import tensorflow as tf
import time
import os
from myutils import Utils
from sklearn.metrics import roc_curve,auc,average_precision_score,precision_score,recall_score,f1_score
import warnings
warnings.filterwarnings("ignore")

class RccDualGAN():
    def __init__(self, seed, model_name='RCCDualGAN'):
        self.seed = seed
        self.utils = Utils()

        parser = argparse.ArgumentParser(description="Run RCC-Dual-GAN.")
        # default = 1000,考虑到计算成本这里设置为100
        parser.add_argument('--max_iter', type=int, default=100,
                            help='The maximum number of iterations.')
        parser.add_argument('--lr_d', type=float, default=0.01,
                            help='Learning rate of discriminator.')
        parser.add_argument('--lr_g_unl', type=float, default=0.0001,
                            help='Learning rate of generator.')
        parser.add_argument('--lr_g_out', type=float, default=0.0001,
                            help='Learning rate of generator.')
        parser.add_argument('--decay', type=float, default=1e-6,
                            help='Decay.')
        parser.add_argument('--batch_size', type=int, default=1000,
                            help='batch_size.')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='Momentum.')
        parser.add_argument('--nash_thr_1', type=float, default=0.5,
                            help='Threshold 1.')
        parser.add_argument('--nash_thr_2', type=float, default=0.4,
                            help='Threshold 2.')

        # self.args = parser.parse_args()
        self.args, unknown = parser.parse_known_args()

    # Sub-Generator
    def create_generator(self, latent_size):
        gen = Sequential()
        gen.add(Dense(latent_size, input_dim=latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
        gen.add(Dense(latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
        latent = Input(shape=(latent_size,))
        fake_data = gen(latent)
        return Model(latent, fake_data)

    # Sub-Discriminator
    def create_sub_discriminator(self, size,latent_size):
        dis = Sequential()
        dis.add(Dense(size, input_dim=latent_size, activation='relu', kernel_initializer= keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
        dis.add(Dense(10, activation='relu', kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
        dis.add(Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal',seed=None)))
        data = Input(shape=(latent_size,))
        fake = dis(data)
        return Model(data, fake)

    # Discriminator
    def create_discriminator(self, size,latent_size):
        dis = Sequential()
        dis.add(Dense(size, input_dim=latent_size, activation='relu', kernel_initializer= keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
        dis.add(Dense(10, activation='relu', kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
        dis.add(Dropout(0.2))
        dis.add(Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal',seed=None)))
        data = Input(shape=(latent_size,))
        fake = dis(data)
        return Model(data, fake)

    # Euclidean distance
    def distEclud(self, x, weights):
        dist = []
        for w in weights:
            d = np.linalg.norm(x-w)
            dist.append(d)
        return np.array(dist)

    def fit(self, X_train, y_train, print_log=False):
        # set seed
        self.utils.set_seed(self.seed)

        data_out_x, data_unl_x, data_out_y, data_unl_y = X_train[y_train==1], X_train[y_train==0], y_train[y_train==1], y_train[y_train==0]

        if print_log:
            print(data_unl_x)
            print(data_unl_x.shape)
        data_x = np.concatenate((data_out_x, data_unl_x), axis=0)
        data_y = np.concatenate((data_out_y, data_unl_y), axis=0)
        data_out_size = data_out_x.shape[0]
        data_unl_size = data_unl_x.shape[0]
        data_size = data_out_size + data_unl_size
        latent_size = data_x.shape[1]
        batch_size = min(self.args.batch_size, data_size)
        batch_out_size = min(100, data_out_size)
        batch_unl_size = min(batch_size - batch_out_size, data_unl_size)
        mul = math.ceil(batch_unl_size / batch_out_size)
        if print_log:
            print("The dimensions of the outliers:{}*{}".format(data_out_size, latent_size))
            print("The dimensions of the unlabeled data:{}*{}".format(data_unl_size, latent_size))

        names = locals()
        eva_list = []
        eva_save = 1

        # RCC
        clusterer = RCC.RccCluster(measure='cosine')
        if data_out_size > 2:
            clu_out, k_out = clusterer.fit(data_out_x)
            clu_out = pd.DataFrame(clu_out)
            data_out_x = pd.DataFrame(data_out_x)
            data_out_x_clu = np.concatenate((data_out_x, clu_out), axis=1)
        elif data_out_size == 2:
            clu_out = np.array([0] * (1) + [1] * (1))
            clu_out = pd.DataFrame(clu_out)
            data_out_x_clu = np.concatenate((data_out_x, clu_out), axis=1)
            k_out = 2
        else:
            clu_out = np.array([0] * (1))
            clu_out = pd.DataFrame(clu_out)
            data_out_x_clu = np.concatenate((data_out_x, clu_out), axis=1)
            k_out = 1

        clu_unl, k_unl = clusterer.fit(data_unl_x)
        clu_unl = pd.DataFrame(clu_unl)
        data_unl_x = pd.DataFrame(data_unl_x)
        data_unl_x_clu = np.concatenate((data_unl_x, clu_unl), axis=1)

        # Divide data into different data subsets
        for i in range(k_out):
            names['data_out_x_' + str(i)] = []
        for idx in range(data_out_size):
            for i in range(k_out):
                if data_out_x_clu[idx, -1] == i:
                    names['data_out_x_clu_' + str(idx)] = data_out_x_clu[idx, :-1].reshape(1, latent_size)
                    if names['data_out_x_' + str(i)] == []:
                        names['data_out_x_' + str(i)] = names['data_out_x_clu_' + str(idx)]
                    else:
                        names['data_out_x_' + str(i)] = np.concatenate(
                            (names['data_out_x_' + str(i)], names['data_out_x_clu_' + str(idx)]), axis=0)
        for i in range(k_unl):
            names['data_unl_x_' + str(i)] = []
        for idx in range(data_unl_size):
            for i in range(k_unl):
                if data_unl_x_clu[idx, -1] == i:
                    names['data_unl_x_clu_' + str(idx)] = data_unl_x_clu[idx, :-1].reshape(1, latent_size)
                    if names['data_unl_x_' + str(i)] == []:
                        names['data_unl_x_' + str(i)] = names['data_unl_x_clu_' + str(idx)]
                    else:
                        names['data_unl_x_' + str(i)] = np.concatenate(
                            (names['data_unl_x_' + str(i)], names['data_unl_x_clu_' + str(idx)]), axis=0)
        for i in range(k_out):
            if print_log:
                print(names['data_out_x_' + str(i)].shape)
        for i in range(k_unl):
            if print_log:
                print(names['data_unl_x_' + str(i)].shape)

        for i in range(k_out):
            names['stop_out_' + str(i)] = 0
        for i in range(k_unl):
            names['stop_unl_' + str(i)] = 0

        # Create sub-discriminator
        for i in range(k_out):
            names['discriminator_out_' + str(i)] = self.create_sub_discriminator(size = min(data_size, 1000), latent_size = latent_size)
            names['discriminator_out_' + str(i)].compile(
                optimizer=SGD(lr=self.args.lr_d, decay=self.args.decay, momentum=self.args.momentum), loss='binary_crossentropy')
        for i in range(k_unl):
            names['discriminator_unl_' + str(i)] = self.create_sub_discriminator(size = min(data_size, 1000), latent_size = latent_size)
            names['discriminator_unl_' + str(i)].compile(
                optimizer=SGD(lr=self.args.lr_d, decay=self.args.decay, momentum=self.args.momentum), loss='binary_crossentropy')

        # Create discriminator
        discriminator_all = self.create_discriminator(size = min(data_size, 1000), latent_size = latent_size)
        discriminator_all.compile(optimizer=SGD(lr=self.args.lr_d, decay=self.args.decay, momentum=self.args.momentum),
                                  loss='binary_crossentropy', metrics=['accuracy'])

        # Create sub-generator and combine_model
        for i in range(k_out):
            names['generator_out_' + str(i)] = self.create_generator(latent_size)
            latent = Input(shape=(latent_size,))
            names['fake_out_' + str(i)] = names['generator_out_' + str(i)](latent)
            names['discriminator_out_' + str(i)].trainable = False
            names['fake_out_' + str(i)] = names['discriminator_out_' + str(i)](names['fake_out_' + str(i)])
            names['combine_model_out_' + str(i)] = Model(latent, names['fake_out_' + str(i)])
            names['combine_model_out_' + str(i)].compile(
                optimizer=SGD(lr=self.args.lr_g_out, decay=self.args.decay, momentum=self.args.momentum), loss='binary_crossentropy')
        for i in range(k_unl):
            names['generator_unl_' + str(i)] = self.create_generator(latent_size)
            latent = Input(shape=(latent_size,))
            names['fake_unl_' + str(i)] = names['generator_unl_' + str(i)](latent)
            names['discriminator_unl_' + str(i)].trainable = False
            names['fake_unl_' + str(i)] = names['discriminator_unl_' + str(i)](names['fake_unl_' + str(i)])
            names['combine_model_unl_' + str(i)] = Model(latent, names['fake_unl_' + str(i)])
            names['combine_model_unl_' + str(i)].compile(
                optimizer=SGD(lr=self.args.lr_g_unl, decay=self.args.decay, momentum=self.args.momentum), loss='binary_crossentropy')

        # Initialize the stop node
        epochs = self.args.max_iter
        stop_iter = epochs
        stop_iter_all = np.array([stop_iter] * (int(k_out + k_unl)))

        # Start iteration
        if print_log:
            print('Training...')
        for epoch in tqdm(range(epochs)):
            if print_log:
                print('Epoch {} of {}'.format(epoch + 1, epochs))

            # Sample mini-batch data
            for i in range(k_out):
                names['data_out_x_' + str(i)] = pd.DataFrame(names['data_out_x_' + str(i)])
                names['data_out_batch_x_' + str(i)] = names['data_out_x_' + str(i)].sample(
                    n=math.ceil((names['data_out_x_' + str(i)].shape[0] * batch_out_size) / data_out_size), replace=False,
                    random_state=None, axis=0)
            for i in range(k_unl):
                names['data_unl_x_' + str(i)] = pd.DataFrame(names['data_unl_x_' + str(i)])
                names['data_unl_batch_x_' + str(i)] = names['data_unl_x_' + str(i)].sample(
                    n=math.ceil((names['data_unl_x_' + str(i)].shape[0] * batch_unl_size) / data_unl_size), replace=False,
                    random_state=None, axis=0)

            # Train sub-generators and sub-discriminators
            for i in range(k_out):
                names['data_out_batch_x_' + str(i)] = pd.DataFrame(names['data_out_batch_x_' + str(i)])
                # Train sub-discriminators
                noise = np.random.uniform(0, 1, (int(names['data_out_batch_x_' + str(i)].shape[0]), latent_size))
                names['generated_data_out_' + str(i)] = names['generator_out_' + str(i)].predict(noise, verbose=0)
                names['x_out_' + str(i)] = np.concatenate(
                    (names['data_out_batch_x_' + str(i)], names['generated_data_out_' + str(i)]), axis=0)
                names['y_out_' + str(i)] = np.array([1] * (int(names['data_out_batch_x_' + str(i)].shape[0])) + [0] * (
                    int(names['data_out_batch_x_' + str(i)].shape[0])))
                names['discriminator_out' + str(i)] = names['discriminator_out_' + str(i)].train_on_batch(
                    names['x_out_' + str(i)], names['y_out_' + str(i)])

                # Train sub-generators
                if names['stop_out_' + str(i)] == 0:
                    trick_out = np.array([1] * (names['data_out_batch_x_' + str(i)].shape[0]))
                    names['generator_out' + str(i)] = names['combine_model_out_' + str(i)].train_on_batch(noise, trick_out)

                    # The evaluation of Nash equilibrium
                    if stop_iter_all[i] == epochs:
                        names['generated_data_out_' + str(i)] = pd.DataFrame(names['generated_data_out_' + str(i)])
                        sample_num = min(10, names['data_out_batch_x_' + str(i)].shape[0])
                        names['eva_nash_data_out_' + str(i)] = names['generated_data_out_' + str(i)].sample(sample_num,
                                                                                                            replace=False,
                                                                                                            random_state=None,
                                                                                                            axis=0)
                        names['eva_nash_data_out_' + str(i)] = names['eva_nash_data_out_' + str(i)].values
                        names['nash_out_' + str(i)] = 0
                        for idx in range(sample_num):
                            real = 0
                            dists_out = self.distEclud(names['eva_nash_data_out_' + str(i)][idx,], names['x_out_' + str(i)])
                            dists_out = pd.DataFrame(dists_out)
                            names['y_out_' + str(i)] = pd.DataFrame(names['y_out_' + str(i)])
                            dists_out = np.concatenate((dists_out, names['y_out_' + str(i)]), axis=1)
                            dists_out = pd.DataFrame(dists_out, columns=['d', 'y'])
                            dists_out = dists_out.sort_values('d', ascending=True)
                            dists_out = dists_out.values
                            for index in range(max(sample_num, 2)):
                                if dists_out[index, 1] == 1:
                                    real = real + 1
                            gen_real = real / max(sample_num, 2)
                            if gen_real >= self.args.nash_thr_1:
                                names['nash_out_' + str(i)] = names['nash_out_' + str(i)] + 1
                        names['nash_out_' + str(i)] = names['nash_out_' + str(i)] / max(sample_num, 2)
                        if names['nash_out_' + str(i)] >= self.args.nash_thr_2:
                            stop_iter_all[i] = epoch + 1
                        # print("The {}th subset contains {} samples, the evaluation of Nash equilibrium is {}".format(i, sample_num, names['nash_out_' + str(i)]))

            for i in range(k_unl):
                names['data_unl_batch_x_' + str(i)] = pd.DataFrame(names['data_unl_batch_x_' + str(i)])

                # Train sub-discriminators
                noise = np.random.uniform(0, 1, (int(names['data_unl_batch_x_' + str(i)].shape[0]), latent_size))
                names['generated_data_unl_' + str(i)] = names['generator_unl_' + str(i)].predict(noise, verbose=0)
                names['x_unl_' + str(i)] = np.concatenate(
                    (names['data_unl_batch_x_' + str(i)], names['generated_data_unl_' + str(i)]), axis=0)
                names['y_unl_' + str(i)] = np.array([1] * (int(names['data_unl_batch_x_' + str(i)].shape[0])) + [0] * (
                    int(names['data_unl_batch_x_' + str(i)].shape[0])))
                names['discriminator_unl' + str(i)] = names['discriminator_unl_' + str(i)].train_on_batch(
                    names['x_unl_' + str(i)], names['y_unl_' + str(i)])

                # Train sub-generators
                if names['stop_unl_' + str(i)] == 0:
                    trick_unl = np.array([1] * (int(names['data_unl_batch_x_' + str(i)].shape[0])))
                    names['generator_unl' + str(i)] = names['combine_model_unl_' + str(i)].train_on_batch(noise, trick_unl)

                    # The evaluation of Nash equilibrium
                    if stop_iter_all[i + k_out] == epochs:
                        names['generated_data_unl_' + str(i)] = pd.DataFrame(names['generated_data_unl_' + str(i)])
                        sample_num = min(10, names['data_unl_batch_x_' + str(i)].shape[0])
                        names['eva_nash_data_unl_' + str(i)] = names['generated_data_unl_' + str(i)].sample(sample_num,
                                                                                                            replace=False,
                                                                                                            random_state=None,
                                                                                                            axis=0)
                        names['eva_nash_data_unl_' + str(i)] = names['eva_nash_data_unl_' + str(i)].values
                        names['nash_unl_' + str(i)] = 0
                        for idx in range(sample_num):
                            real = 0
                            dists_unl = self.distEclud(names['eva_nash_data_unl_' + str(i)][idx,], names['x_unl_' + str(i)])
                            dists_unl = pd.DataFrame(dists_unl)
                            names['y_unl_' + str(i)] = pd.DataFrame(names['y_unl_' + str(i)])
                            dists_unl = np.concatenate((dists_unl, names['y_unl_' + str(i)]), axis=1)
                            dists_unl = pd.DataFrame(dists_unl, columns=['d', 'y'])
                            dists_unl = dists_unl.sort_values('d', ascending=True)
                            dists_unl = dists_unl.values
                            for index in range(max(sample_num, 2)):
                                if dists_unl[index, 1] == 1:
                                    real = real + 1
                            gen_real = real / max(sample_num, 2)
                            if gen_real >= self.args.nash_thr_1:
                                names['nash_unl_' + str(i)] = names['nash_unl_' + str(i)] + 1
                        names['nash_unl_' + str(i)] = names['nash_unl_' + str(i)] / max(sample_num, 2)
                        if sample_num >= 2 and names['nash_unl_' + str(i)] >= self.args.nash_thr_2:
                            names['stop_unl_' + str(i)] = 1
                            stop_iter_all[i + k_out] = epoch + 1
                        elif names['nash_unl_' + str(i)] >= self.args.nash_thr_2:
                            stop_iter_all[i + k_out] = epoch + 1
                        # print("The {}th subset contains {} samples, the evaluation of Nash equilibrium is {}".format(i, sample_num, names['nash_unl_' + str(i)]))

            if stop_iter == epochs:
                stop_iter = max(stop_iter_all)
            else:
                for i in range(k_unl):
                    names['stop_unl_' + str(i)] = 1

            # Train discriminators
            for i in range(k_out):
                if i == 0:
                    data_out_batch = names['data_out_batch_x_' + str(i)]
                else:
                    data_out_batch = np.concatenate((data_out_batch, names['data_out_batch_x_' + str(i)]), axis=0)
            for i in range(k_unl):
                if i == 0:
                    data_unl_batch = names['data_unl_batch_x_' + str(i)]
                else:
                    data_unl_batch = np.concatenate((data_unl_batch, names['data_unl_batch_x_' + str(i)]), axis=0)
            for i in range(k_out):
                noise_all = np.random.uniform(0, 1, (int(mul * names['data_out_batch_x_' + str(i)].shape[0]), latent_size))
                names['generated_data_out_all_' + str(i)] = names['generator_out_' + str(i)].predict(noise_all, verbose=0)
                if i == 0:
                    x_out_all = np.concatenate((data_out_batch, names['generated_data_out_all_' + str(i)]), axis=0)
                else:
                    x_out_all = np.concatenate((x_out_all, names['generated_data_out_all_' + str(i)]), axis=0)
            for i in range(k_unl):
                noise_all = np.random.uniform(0, 1, (math.ceil(batch_unl_size / k_unl), latent_size))
                names['generated_data_unl_all_' + str(i)] = names['generator_unl_' + str(i)].predict(noise_all, verbose=0)
                if i == 0:
                    x_unl_all = np.concatenate((data_unl_batch, names['generated_data_unl_all_' + str(i)]), axis=0)
                else:
                    x_unl_all = np.concatenate((x_unl_all, names['generated_data_unl_all_' + str(i)]), axis=0)

            x_all = np.concatenate((x_unl_all, x_out_all), axis=0)
            y_all = np.array([1] * (int(data_unl_batch.shape[0])) + [0] * (x_all.shape[0] - data_unl_batch.shape[0]))
            discriminator_all_loss = discriminator_all.train_on_batch(x_all, y_all)

            # The selection of optimal model
            eva_x = discriminator_all.predict(data_x)
            eva_x = pd.DataFrame(eva_x)
            eva_y = np.array([0] * (int(data_out_x.shape[0])) + [1] * (data_unl_x.shape[0]))
            eva_y = pd.DataFrame(eva_y)
            eva_xy = np.concatenate((eva_x, eva_y), axis=1)
            eva_xy = pd.DataFrame(eva_xy, columns=['x', 'y'])
            eva_xy = eva_xy.sort_values('x', ascending=True)
            eva_xy = eva_xy.values
            eva = 0
            for i in range(eva_xy.shape[0]):
                if eva_xy[i, 1] == 0:
                    eva = eva + i + 1
            eva = eva / (data_x.shape[0] * data_out_x.shape[0])
            eva_list.append(eva)
            if eva_save >= eva:
                eva_save = eva
                discriminator_all.save(os.path.join('baseline','RCCDualGAN','RCCDualGAN_D.h5'))

            if stop_iter == epochs:
                stop_iter = max(stop_iter_all)
            else:
                for i in range(k_unl):
                    names['stop_unl_' + str(i)] = 1

            # 评估并保存检测结果
            if epoch % 100 == 0:
                p_value = discriminator_all.predict(data_x)
                p_value = pd.DataFrame(p_value)
                data_y = pd.DataFrame(data_y)
                result = np.concatenate((p_value, data_y), axis=1)
                result = pd.DataFrame(result, columns=['p', 'y'])
                result = result.sort_values('p', ascending=True)
                # print(result)

                fpr, tpr, _ = roc_curve(y_true=result['y'], y_score=1 - result['p'], pos_label=1)
                aucroc = auc(fpr, tpr)
                aucpr = average_precision_score(y_true=result['y'], y_score=1 - result['p'], pos_label=1)
                if print_log:
                    print(f'AUCROC value in training set :{round(aucroc * 100, 2)}')
                    print(f'AUCPR value in training set :{round(aucpr * 100, 2)}')

        return self

    def predict_score(self, X, phase=None):
        data_x = X
        discriminator_all = load_model(os.path.join('baseline', 'RCCDualGAN', 'RCCDualGAN_D.h5'))
        p_value = discriminator_all.predict(data_x)
        result = pd.DataFrame(p_value, columns=['p'])
        result = result.sort_values('p', ascending=True)
        score = np.array(1 - result['p'])

        return score