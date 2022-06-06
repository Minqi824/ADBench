import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from myutils import Utils
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.distribution import Distribution

def fit(train_loader, model, optimizer, epochs, device=None):
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for i, data in enumerate(train_loader):

            X, y = data
            X = X.to(device); y = y.to(device)
            X = Variable(X); y = Variable(y)

            # clear gradient
            model.zero_grad()

            # loss forward
            # 注意cv中由于batchnorm的存在要一起计算score
            _, prob = model(X)
            loss = criterion(prob, y)

            # loss backward
            loss.backward()
            # parameter update
            optimizer.step()