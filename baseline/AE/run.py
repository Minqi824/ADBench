import os
import sys

from myutils import Utils
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader, TensorDataset

from baseline.AE.model import network
from baseline.AE.fit import fit

class AE():
    def __init__(self, model_name:str, seed:int, epochs:int=50, batch_size:int=64, act_fun=nn.Tanh(),
                 lr:float=1e-2, mom:float=0.7):

        self.utils = Utils()
        self.device = self.utils.get_device() #get device
        self.seed = seed

        #hyper-parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.act_fun = act_fun
        self.lr = lr
        self.mom = mom

    def fit(self, X_train, y_train, ratio=None):
        #only use the normal data
        X_train = X_train[y_train == 0]
        y_train = y_train[y_train == 0]

        train_tensor = TensorDataset(torch.from_numpy(X_train).float(), torch.tensor(y_train).float())
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=False, drop_last=True)

        input_size = X_train.shape[1]
        if input_size < 8:
            hidden_size = input_size // 2
        else:
            hidden_size = input_size // 4

        # model initialization, there exists randomness because of weight initialization***
        self.utils.set_seed(self.seed)
        self.model = network(input_size=input_size,  act_fun=self.act_fun)
        self.model = self.model.to(self.device)

        #optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.mom)
        # fitting
        fit(train_loader=train_loader, model=self.model, optimizer=optimizer, epochs=self.epochs, print_loss=False, device=self.device)

        return self

    #calculate the anomaly score based on the reconstruction loss
    def predict_score(self, X):
        L1_criterion = nn.L1Loss(reduction='none')
        self.model.eval()

        if torch.is_tensor(X):
            pass
        else:
            X = torch.from_numpy(X)

        X = X.float()
        X = X.to(self.device)

        with torch.no_grad():
            _, X_hat = self.model(X)
            score = L1_criterion(X, X_hat)
            score = torch.mean(score, dim=1).cpu().detach().numpy()

        return score