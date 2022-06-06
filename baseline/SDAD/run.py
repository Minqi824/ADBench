import pandas as pd
import numpy as np
import random
import os
import sys
from itertools import product
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Subset, DataLoader, TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F

# import warnings
# warnings.filterwarnings("ignore")

from myutils import Utils
from baseline.SDAD.model import SDAD, SDAD_pro
from baseline.SDAD.fit import fit

class sdad():
    def __init__(self, seed:int, model_name='SDAD',
                 epochs:int=200, batch_size:int=256, act_fun=nn.ReLU(),
                 lr:float=1e-2, mom=0.7, weight_decay:float=1e-2,
                 resample=False, noise=False, pseudo=True,
                 select_bw=False, select_epoch=False, early_stopping=False,
                 bw_u:int=10, bw_a:int=10, analysis=False):
        '''
        noise: whether to add Gaussian noise for the output score of anomalies
        select_bw: whether to use the validation set for selecting the best bandwidth. In practice, we observe that
        the bandwidth would significantly affect the model performance
        select_epoch: whether to use the validation set for selecting the best epoch to prevent overfitting problem
        early stopping: whether to use the validation set for early stopping
        '''

        self.seed = seed
        self.utils = Utils()
        self.device = self.utils.get_device()

        #hyper-parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.act_fun = act_fun
        self.lr = lr
        self.mom = mom
        self.weight_decay = weight_decay

        self.resample = resample
        self.noise = noise
        self.pseudo = pseudo

        self.select_bw = select_bw
        self.select_epoch = select_epoch
        self.early_stopping = early_stopping

        #change the current hyper-parameter
        # self.epochs = 200
        # self.lr = 1e-3
        # self.select_bw = False
        # self.select_epoch = True
        # self.early_stopping = False

        self.epochs = 20
        self.lr = 1e-3
        self.select_bw = False
        self.select_epoch = False
        self.early_stopping = False

        if self.select_bw:
            self.bw_pool = [0.01, 0.1, 1.0, 10.0, 100]
        else:
            # self.bw_u = bw_u
            # self.bw_a = bw_a

            self.bw_u = 1.0
            self.bw_a = 1.0

        self.model_init = SDAD_pro
        self.analysis = analysis

    def fit2test(self, data):
        #data
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        input_size = X_train.shape[1] #input size
        X_test_tensor = torch.from_numpy(X_test).float() # testing set

        # using the training set as validation set
        X_val = X_train.copy()
        y_val = y_train.copy()

        X_val_tensor = torch.from_numpy(X_val).float() # validation set

        # resampling
        X_train, y_train = self.utils.sampler(X_train, y_train, self.batch_size)
        # X_train, y_train = self.utils.sampler_2(X_train, y_train, step=20)

        X_train_tensor = torch.from_numpy(X_train).float()
        train_tensor = TensorDataset(torch.from_numpy(X_train).float(), torch.tensor(y_train))
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=False, drop_last=True)

        # grid search for the best bandwidth based on the training data
        # the choice of bandwidth would significantly affect the performance of model
        # we recommend to use the k-fold method for selecting the best bandwidth

        if self.select_bw:
            bw_val_select = []
            epoch_val_select = []

            bw_combination = list(product(self.bw_pool, self.bw_pool))
            for bw_u, bw_a in bw_combination:
                try:
                    self.utils.set_seed(self.seed)
                    model = self.model_init(input_size=input_size, act_fun=self.act_fun) #model initialization
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.mom,
                                                weight_decay=self.weight_decay)  # optimizer

                    # fit
                    if self.select_epoch: # using the validation set for selecting the best epoch
                        score_val_epoch = fit(train_loader=train_loader, model=model, optimizer=optimizer,
                                              epochs=self.epochs,
                                              resample=self.resample, noise=self.noise, pseudo=self.pseudo,
                                              bw_u=bw_u, bw_a=bw_a,
                                              device=self.device, X_val_tensor=X_val_tensor)

                        aucpr_val_epoch = []
                        for i in range(score_val_epoch.shape[1]):
                            result_val = self.utils.metric(y_true=y_val, y_score=score_val_epoch[:, i])
                            aucpr_val_epoch.append(result_val['aucpr'])

                        bw_val_select.append(np.max(aucpr_val_epoch))
                        epoch_val_select.append(np.argmax(aucpr_val_epoch) + 1)

                        print(f'Grid search for: {bw_u, bw_a},'
                              f'The best aucpr in validation set: {np.max(aucpr_val_epoch)}')

                    elif self.early_stopping: # using the validation set for early stopping
                        _ = fit(train_loader=train_loader, model=model, optimizer=optimizer,
                                epochs=self.epochs,
                                resample=self.resample, noise=self.noise, pseudo=self.pseudo,
                                bw_u=bw_u, bw_a=bw_a,
                                device=self.device, X_val_tensor=X_val_tensor, y_val=y_val,
                                early_stopping=True)

                        # load the best model
                        model = torch.load(os.path.join(os.getcwd(),'baseline','SDAD','model','SDAD.pt'))
                        model.eval()

                        with torch.no_grad():
                            score_val = model(X_val_tensor)

                        result_val = self.utils.metric(y_true=y_val, y_score=score_val)

                        bw_val_select.append(result_val['aucpr'])
                        print(f'Grid search for: {bw_u, bw_a},'
                              f"The best aucpr in validation set: {result_val['aucpr']}")

                    else: # do nothing
                        _ = fit(train_loader=train_loader, model=model, optimizer=optimizer,
                                epochs=self.epochs,
                                resample=self.resample, noise=self.noise, pseudo=self.pseudo,
                                bw_u=bw_u, bw_a=bw_a,
                                device=self.device, X_val_tensor=X_val_tensor)

                        with torch.no_grad():
                            _, score_val = model(X_val_tensor)
                            score_val = score_val.cpu().numpy()
                            result_val = self.utils.metric(y_true=y_val, y_score=score_val)

                            bw_val_select.append(result_val['aucpr'])

                        print(f'Grid search for: {bw_u, bw_a},'
                              f"The aucpr in validation set: {result_val['aucpr']}")

                except Exception as e:
                    bw_val_select.append(0.0)
                    epoch_val_select.append(self.epochs)
                    print(f'error for bw: {(bw_u, bw_a)}, the error message: {e}')
                    pass

            bw_best = bw_combination[np.argmax(bw_val_select)]

            if self.select_epoch:
                epoch_best = epoch_val_select[np.argmax(bw_val_select)]
            elif self.early_stopping:
                epoch_best = None
            else:
                epoch_best = self.epochs

            print(f'The best bandwidth: {bw_best}, the best epoch: {epoch_best}')

        else:
            if self.select_epoch:
                self.utils.set_seed(self.seed)
                model = self.model_init(input_size=input_size, act_fun=self.act_fun)
                optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.mom,
                                            weight_decay=self.weight_decay)  # optimizer

                # fit
                score_val_epoch = fit(train_loader=train_loader, model=model, optimizer=optimizer,
                                      epochs=self.epochs,
                                      resample=self.resample, noise=self.noise, pseudo=self.pseudo,
                                      bw_u=self.bw_u, bw_a=self.bw_a, device=self.device,
                                      X_val_tensor=X_val_tensor)

                aucpr_val_epoch = []
                for i in range(score_val_epoch.shape[1]):
                    result_val = self.utils.metric(y_true=y_val, y_score=score_val_epoch[:, i])
                    aucpr_val_epoch.append(result_val['aucpr'])

                bw_best = (self.bw_u, self.bw_a)
                epoch_best = np.argmax(aucpr_val_epoch) + 1

                print(f'Using the default bandwidth..., the best epoch: {epoch_best}')

            elif self.early_stopping:
                self.utils.set_seed(self.seed)
                model = self.model_init(input_size=input_size, act_fun=self.act_fun)
                optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.mom,
                                            weight_decay=self.weight_decay)  # optimizer

                # fit
                _ = fit(train_loader=train_loader, model=model, optimizer=optimizer,
                        epochs=self.epochs,
                        resample=self.resample, noise=self.noise, pseudo=self.pseudo,
                        bw_u=self.bw_u, bw_a=self.bw_a, device=self.device,
                        X_val_tensor=X_val_tensor, y_val=y_val, early_stopping=True)

                bw_best = (self.bw_u, self.bw_a)
                epoch_best = None

                print(f'Using the default bandwidth..., the best model has been saved...')

            else:
                bw_best = (self.bw_u, self.bw_a)
                epoch_best = self.epochs

        # refit
        if self.early_stopping:
            model = torch.load(os.path.join(os.getcwd(),'baseline','SDAD','model','SDAD.pt'))

        else:
            X_train = data['X_train']
            y_train = data['y_train']
            X_train, y_train = self.utils.sampler(X_train, y_train, self.batch_size)
            # X_train, y_train = self.utils.sampler_2(X_train, y_train, step=20)
            train_tensor = TensorDataset(torch.from_numpy(X_train).float(), torch.tensor(y_train))
            train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=False, drop_last=True)

            self.utils.set_seed(self.seed)
            model = self.model_init(input_size=input_size, act_fun=self.act_fun)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.mom,
                                        weight_decay=self.weight_decay)  # optimizer

            # training
            _ = fit(train_loader=train_loader, model=model, optimizer=optimizer, epochs=epoch_best,
                    resample=self.resample, noise=self.noise, pseudo=self.pseudo,
                    bw_u=bw_best[0], bw_a=bw_best[1], device=self.device)

        #evaluating in the testing set
        model.eval()
        with torch.no_grad():
            _, score_test = model(X_test_tensor)
            score_test = score_test.cpu().numpy()

        result = self.utils.metric(y_true=y_test, y_score=score_test)

        if self.analysis:
            return model, result
        else:
            return result

'''
below is the original code for selecting the bandwidth
'''
# #resampling
# X_train, y_train = self.utils.sampler(X_train, y_train, self.batch_size)
#
# X_train_tensor = torch.from_numpy(X_train).float()
# train_tensor = TensorDataset(torch.from_numpy(X_train).float(), torch.tensor(y_train))
# train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=False, drop_last=True)
#
# #grid search for the best bandwidth based on the training data
# #the choice of bandwidth would significantly affect the performance of model
# #we recommend to use the k-fold method for selecting the best bandwidth
#
# bw_aucpr = []
# for bw in self.bw_pool:
#     try:
#         self.utils.set_seed(self.seed)
#         model = self.model_init(input_size=input_size, act_fun=self.act_fun)
#         optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.mom,
#                                     weight_decay=self.weight_decay)  # optimizer
#
#         # fit
#         fit(train_loader=train_loader, model=model, optimizer=optimizer, epochs=self.epochs,
#             noise=self.noise, bw=bw, device=self.device)
#
#         model.eval()
#         with torch.no_grad():
#             _, score_train = model(X_train_tensor)
#             score_train = score_train.squeeze().numpy()
#             result_train = self.utils.metric(y_true=y_train, y_score=score_train)
#             aucpr_train = result_train['aucpr']
#
#         bw_aucpr.append(aucpr_train)
#     except:
#         pass
#         bw_aucpr.append(0.0)
#         print(f'error for bw: {bw}')
#
# print(f'The best bandwidth: {self.bw_pool[np.argmax(bw_aucpr)]}')
# bw_best = self.bw_pool[np.argmax(bw_aucpr)]
#
# #refit
# self.utils.set_seed(self.seed)
# model = self.model_init(input_size=input_size, act_fun=self.act_fun)
# optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.mom, weight_decay=self.weight_decay) #optimizer
#
# # training
# fit(train_loader=train_loader, model=model, optimizer=optimizer, epochs=self.epochs,
#     noise=self.noise, bw=bw_best, device=self.device)


'''
below is the code for the k-folds selection for bandwidth
'''
# kf = KFold(n_splits=5)
# bw_aucpr = np.zeros((5, 5))
#
# for i, (train_idx, val_idx) in enumerate(kf.split(y_train)):
#     X_train_kf, X_val_kf = X_train[train_idx], X_train[val_idx]
#     y_train_kf, y_val_kf = y_train[train_idx], y_train[val_idx]
#
#     #resampling
#     X_train_kf, y_train_kf = self.utils.sampler(X_train_kf, y_train_kf, self.batch_size)
#     X_val_tensor = torch.from_numpy(X_val_kf).float()
#
#     train_tensor = TensorDataset(torch.from_numpy(X_train_kf).float(), torch.tensor(y_train_kf))
#     train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=False, drop_last=True)
#
#     for j, bw in enumerate(self.bw_pool):
#         try:
#             self.utils.set_seed(self.seed)
#             model = self.model_init(input_size=input_size, act_fun=self.act_fun)
#             optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.mom,
#                                         weight_decay=self.weight_decay)  # optimizer
#
#             # fit
#             fit(train_loader=train_loader, model=model, optimizer=optimizer, epochs=self.epochs,
#                 noise=self.noise, bw=bw, device=self.device)
#
#             model.eval()
#             with torch.no_grad():
#                 _, score_val = model(X_val_tensor)
#                 score_val = score_val.squeeze().numpy()
#                 result_val = self.utils.metric(y_true=y_val_kf, y_score=score_val)
#                 aucpr_val = result_val['aucpr']
#
#             bw_aucpr[i, j] = aucpr_val
#         except:
#             pass
#             bw_aucpr[i, j] = 0.0
#             print(f'error for fold: {i}; bw: {bw}')
#
# bw_aucpr = np.mean(bw_aucpr, axis=0) #average performance among k-folds
# print(f'The best bandwidth: {self.bw_pool[np.argmax(bw_aucpr)]}')
# bw_best = self.bw_pool[np.argmax(bw_aucpr)]
#
# # refit
# X_train, y_train = self.utils.sampler(X_train, y_train, self.batch_size)
# train_tensor = TensorDataset(torch.from_numpy(X_train).float(), torch.tensor(y_train))
# train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=False, drop_last=True)
#
# self.utils.set_seed(self.seed)
# model = self.model_init(input_size=input_size, act_fun=self.act_fun)
# optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.mom,
#                             weight_decay=self.weight_decay)  # optimizer
#
# # training
# fit(train_loader=train_loader, model=model, optimizer=optimizer, epochs=self.epochs,
#     noise=self.noise, bw=bw_best, device=self.device)