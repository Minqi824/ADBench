import sys
import os
# sys.path.append(os.path.dirname(__file__))

import torch
from torch import nn
from torch.utils.data import Subset,DataLoader,TensorDataset
from myutils import Utils

from baseline.WSGAN.model import generator
from baseline.WSGAN.model import discriminator
from baseline.WSGAN.fit import fit

class WSGAN():
    def __init__(self, seed, model_name='WSGAN',
                 epochs:int=50, batch_size:int=64, act_fun=nn.ReLU(), lr:float=1e-2, mom:float=0.7, eta:float=0.5):
        self.utils = Utils()
        self.device = self.utils.get_device()  # get device
        self.seed = seed

        self.epochs = epochs
        self.batch_size = batch_size
        self.act_fun = act_fun
        self.lr = lr
        self.mom = mom
        self.eta = eta

    def evaluation(self, data_tensor, model):
        data_tensor = data_tensor.to(self.device)
        L1_criterion = nn.L1Loss(reduction='none')

        z, _, z_hat = model(data_tensor)
        score = L1_criterion(z, z_hat)
        score = torch.sum(score, dim=1).cpu().detach().numpy()

        return score

    def fit2test(self, data):
        # data
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        X_train, y_train = self.utils.sampler(X_train, y_train, self.batch_size)

        train_tensor = TensorDataset(torch.from_numpy(X_train).float(), torch.tensor(y_train).float())
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=False, drop_last=True)

        # testing set
        X_test_tensor = torch.from_numpy(X_test).float()

        input_size = X_test_tensor.size(1)
        if input_size < 8:
            hidden_size = input_size // 2
        else:
            hidden_size = input_size // 4

        # model initialization, there exists randomness because of weight initialization***
        self.utils.set_seed(self.seed)
        net_generator = generator(input_size=input_size, hidden_size=hidden_size, act_fun=self.act_fun)
        net_discriminator = discriminator(input_size=input_size, act_fun=self.act_fun)

        net_generator = net_generator.to(self.device)
        net_discriminator = net_discriminator.to(self.device)

        optimizer_G = torch.optim.SGD(net_generator.parameters(), lr=self.lr, momentum=self.mom)
        optimizer_D = torch.optim.SGD(net_discriminator.parameters(), lr=self.lr, momentum=self.mom)

        # fitting
        fit(dataloader=train_loader, net_generator=net_generator, net_discriminator=net_discriminator,
            optimizer_G=optimizer_G, optimizer_D=optimizer_D,
            eta=self.eta, epochs=self.epochs, seed=self.seed, batch_size=self.batch_size,
            input_size=input_size, act_fun=self.act_fun,
            device=self.device, print_loss=False)

        # evaluation
        score_test = self.evaluation(data_tensor=X_test_tensor, model=net_generator)
        result = self.utils.metric(y_true=y_test, y_score=score_test)

        return result