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
from baseline.SDAD_GMM.fit import fit

class sdad_gmm():
    def __init__(self, seed:int, model_name='SDAD_GMM',
                 epochs:int=200, batch_size:int=256, act_fun=nn.ReLU(),
                 lr:float=1e-2, mom=0.7, weight_decay:float=1e-2,
                 resample=False, noise=False, pseudo=True,
                 select_epoch=False, early_stopping=False):
        '''
        noise: whether to add Gaussian noise for the output score of anomalies
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

        self.select_epoch = select_epoch
        self.early_stopping = early_stopping

        #change the current hyper-parameter
        self.epochs = 200
        self.lr = 1e-3
        self.select_epoch = True
        self.early_stopping = False

        self.model_init = SDAD_pro

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

        X_train_tensor = torch.from_numpy(X_train).float()
        train_tensor = TensorDataset(torch.from_numpy(X_train).float(), torch.tensor(y_train))
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=False, drop_last=True)

        if self.select_epoch:
            self.utils.set_seed(self.seed)
            model = self.model_init(input_size=input_size, act_fun=self.act_fun)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.mom,
                                        weight_decay=self.weight_decay)  # optimizer

            # fit
            score_val_epoch = fit(train_loader=train_loader, model=model, optimizer=optimizer,
                                  epochs=self.epochs,
                                  resample=self.resample, noise=self.noise, pseudo=self.pseudo,
                                  device=self.device,
                                  X_val_tensor=X_val_tensor)

            aucpr_val_epoch = []
            for i in range(score_val_epoch.shape[1]):
                result_val = self.utils.metric(y_true=y_val, y_score=score_val_epoch[:, i])
                aucpr_val_epoch.append(result_val['aucpr'])

            epoch_best = np.argmax(aucpr_val_epoch) + 1

            print(f'Selecting the epoch... the best epoch: {epoch_best}')

        elif self.early_stopping:
            self.utils.set_seed(self.seed)
            model = self.model_init(input_size=input_size, act_fun=self.act_fun)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.mom,
                                        weight_decay=self.weight_decay)  # optimizer

            # fit
            _ = fit(train_loader=train_loader, model=model, optimizer=optimizer,
                    epochs=self.epochs,
                    resample=self.resample, noise=self.noise, pseudo=self.pseudo,
                    device=self.device,
                    X_val_tensor=X_val_tensor, y_val=y_val, early_stopping=True)

            epoch_best = None

            print(f'Earlystopping... the best model has been saved...')

        else:
            epoch_best = self.epochs

        # refit
        if self.early_stopping:
            model = torch.load(os.path.join(os.getcwd(),'baseline','SDAD','model','SDAD_GMM.pt'))

        else:
            X_train = data['X_train']
            y_train = data['y_train']
            X_train, y_train = self.utils.sampler(X_train, y_train, self.batch_size)
            train_tensor = TensorDataset(torch.from_numpy(X_train).float(), torch.tensor(y_train))
            train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=False, drop_last=True)

            self.utils.set_seed(self.seed)
            model = self.model_init(input_size=input_size, act_fun=self.act_fun)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.mom,
                                        weight_decay=self.weight_decay)  # optimizer

            # training
            _ = fit(train_loader=train_loader, model=model, optimizer=optimizer, epochs=epoch_best,
                    resample=self.resample, noise=self.noise, pseudo=self.pseudo,
                    device=self.device)

        #evaluating in the testing set
        model.eval()
        with torch.no_grad():
            _, score_test = model(X_test_tensor)
            score_test = score_test.cpu().numpy()

        result = self.utils.metric(y_true=y_test, y_score=score_test)

        return result