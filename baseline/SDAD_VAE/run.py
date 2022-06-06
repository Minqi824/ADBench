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
from baseline.SDAD_VAE.model import SDAD_VAE, SDAD_pro_VAE
from baseline.SDAD_VAE.fit import fit

class sdad_vae():
    def __init__(self, seed:int, model_name='SDAD_VAE',
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

        # self.model_init = SDAD_VAE
        self.model_init = SDAD_pro_VAE
        self.analysis = analysis

    def fit2test(self, data):
        #data
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        input_size = X_train.shape[1] #input size
        X_test_tensor = torch.from_numpy(X_test).float() # testing set

        # resampling
        X_train, y_train = self.utils.sampler(X_train, y_train, self.batch_size)

        X_train_tensor = torch.from_numpy(X_train).float()
        train_tensor = TensorDataset(torch.from_numpy(X_train).float(), torch.tensor(y_train))
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=False, drop_last=True)

        self.utils.set_seed(self.seed)
        model = self.model_init(input_size=input_size, act_fun=self.act_fun)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=self.lr, momentum=self.mom,
                                        weight_decay=self.weight_decay)  # optimizer

        # training
        fit(train_loader=train_loader, model=model, optimizer=optimizer, epochs=self.epochs,
            resample=self.resample, noise=self.noise, pseudo=self.pseudo,
            bw_u=self.bw_u, bw_a=self.bw_a, device=self.device)

        #evaluating in the testing set
        model.eval()
        with torch.no_grad():
            _, _, _, score_test = model(X_test_tensor)
            score_test = score_test.cpu().numpy()

        result = self.utils.metric(y_true=y_test, y_score=score_test)

        if self.analysis:
            return model, result
        else:
            return result