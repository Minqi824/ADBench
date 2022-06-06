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
from baseline.FS.model import FS
from baseline.FS.fit import fit

class fs():
    def __init__(self, seed:int, model_name=None,
                 epochs:int=200, batch_size:int=256, act_fun=nn.ReLU(),
                 lr:float=1e-3, weight_decay:float=1e-6):
        self.seed = seed
        self.utils = Utils()
        self.device = self.utils.get_device()

        #hyper-parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.act_fun = act_fun
        self.lr = lr
        self.weight_decay = weight_decay

    def fit2test(self, data):
        #data
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        input_size = X_train.shape[1] #input size
        X_test_tensor = torch.from_numpy(X_test).float() # testing set
        X_train_tensor = torch.from_numpy(X_train).float()
        train_tensor = TensorDataset(torch.from_numpy(X_train).float(), torch.tensor(y_train).float())
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=False, drop_last=True)

        self.utils.set_seed(self.seed)
        model = FS(input_size=input_size, act_fun=self.act_fun)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)  # optimizer

        # training
        fit(train_loader=train_loader, model=model, optimizer=optimizer, epochs=self.epochs, device=self.device)

        #evaluating in the testing set
        model.eval()
        with torch.no_grad():
            _, score_test = model(X_test_tensor)
            score_test = score_test.cpu().numpy()

        result = self.utils.metric(y_true=y_test, y_score=score_test)

        return result