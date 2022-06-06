import os
import numpy as np
from tqdm import tqdm

from myutils import Utils
import matplotlib.pyplot as plt


from other_utils.gmm.gmm import GaussianMixture

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.distribution import Distribution
from torch.autograd import Variable

#实例化utils
utils = Utils()

def my_gmm(x, n_components:int=3, num_iter:int=100, lr:float=0.001, momentum:float=0.9):
    '''
    refer: https://stackoverflow.com/questions/65755730/estimating-mixture-of-gaussian-models-in-pytorch
    refer: https://pytorch.org/docs/stable/distributions.html#categorical
    '''
    # parameters initialization
    x = x.detach()

    logits = torch.ones(n_components, requires_grad=True)
    means = torch.randn((n_components, x.size(1)), requires_grad=True)
    log_vars = torch.randn((n_components, x.size(1)), requires_grad=True)

    # model parameters
    parameters = [logits, means, log_vars]
    optimizer = optim.SGD(parameters, lr=lr, momentum=momentum)

    for i in range(num_iter):
        mix = D.Categorical(F.softmax(logits, dim=0))
        comp = D.Independent(D.Normal(means, torch.exp(log_vars / 2)), 1) # mean and std
        gmm = D.MixtureSameFamily(mix, comp)

        # clear gradient
        optimizer.zero_grad()
        # loss forward
        loss = -gmm.log_prob(x.detach()).mean()
        # loss backward
        loss.backward(retain_graph=True)
        # update
        optimizer.step()

    # refit
    mix = D.Categorical(F.softmax(logits, dim=0))
    comp = D.Independent(D.Normal(means, torch.exp(log_vars / 2)), 1)  # mean and std
    gmm = D.MixtureSameFamily(mix, comp)

    return gmm

def loss_overlap(s_u, s_a, seed, x_num=1000, resample=False, pseudo=True, plot=False):
    if not resample:
        # we remove the duplicated anomalies, since they may not be helpful for estimating the overall distribution
        unique, inverse = torch.unique(s_a, sorted=True, return_inverse=True, dim=0)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
        s_a = s_a[perm]
    else:
        assert len(s_u) == len(s_a)

    # set seed
    utils.set_seed(seed)

    # reshape
    s_u = s_u.reshape(-1, 1)
    s_a = s_a.reshape(-1, 1)

    # estimate the GMM model
    gmm_u = my_gmm(s_u)
    gmm_a = my_gmm(s_a)

    if pseudo:
        # using the fitted GMM to generate pseudo anomaly scores
        s_a = gmm_a.sample(torch.tensor([s_u.size(0)]))

    xmin = torch.min(torch.min(s_u), torch.min(s_a))
    xmax = torch.max(torch.max(s_u), torch.max(s_a))

    dx = 0.2 * (xmax - xmin)
    xmin -= dx
    xmax += dx

    x = torch.linspace(xmin.detach(), xmax.detach(), x_num)
    pdf_u_x = torch.exp(gmm_u.log_prob(x.reshape(-1, 1)))
    pdf_a_x = torch.exp(gmm_a.log_prob(x.reshape(-1, 1)))

    if plot:
        plt.plot(x, pdf_u_x, color='blue')
        plt.plot(x, pdf_a_x, color='red')

    inters_x = torch.min(pdf_u_x, pdf_a_x)
    area = torch.trapz(inters_x, x)

    return area

def fit(train_loader, model, optimizer, epochs, print_loss=False, device=None,
        resample=False, noise=False, pseudo=True,
        X_val_tensor=None, y_val=None, early_stopping=False, tol=5):
    '''
    noise: whether to add Gaussian noise of the output score of labeled anomalies
    early_stopping: whether to use early stopping based on the performance in validation set
    tol: the tolerance for early stopping
    '''

    # margin loss for keeping the order of score between normal samples and anomalies
    ranking_loss = torch.nn.MarginRankingLoss()
    if X_val_tensor is not None:
        score_val_epoch = np.empty([X_val_tensor.size(0), epochs])
    else:
        score_val_epoch = None

    best_metric_val = 0.0
    tol_count = 0

    for epoch in tqdm(range(epochs)):
        model.train()
        for i, data in enumerate(train_loader):



            X, y = data
            X = X.to(device); y = y.to(device)
            X = Variable(X); y = Variable(y)

            # clear gradient
            model.zero_grad()

            # loss forward
            # 注意由于batchnorm的存在要一起计算score
            _, score = model(X)

            idx_u = torch.where(y == 0)[0]
            idx_a = torch.where(y == 1)[0]

            score_u = score[idx_u]
            score_a = score[idx_a]

            if noise: #additionally inject Gaussian noise for improving robustness
                score_a = score_a + torch.zeros_like(score_a).normal_(0.0, 1.0)

            # loss forward
            loss_1 = loss_overlap(s_u=score_u, s_a=score_a, seed=utils.unique(epoch, i),
                                  resample=resample, pseudo=pseudo)

            loss_2 = ranking_loss(score_a, score_u, torch.ones_like(score_a))
            # combine the loss
            # loss = loss_1 + loss_2
            loss = loss_2

            # loss backward
            loss.backward()
            # parameter update
            optimizer.step()

            # if (i % 50 == 0) & print_loss:
            #     print('[%d/%d] [%d/%d] Loss: %.4f' % (epoch + 1, epochs, i, len(train_loader), loss))

        # storing the network output score in validation set
        if X_val_tensor is not None:
            model.eval()
            with torch.no_grad():
                _, score_val = model(X_val_tensor)
                score_val_epoch[:, epoch] = score_val.detach().numpy()


            # using the validation set for early stopping
            if early_stopping:
                # the metric in validation set
                metric_val = utils.metric(y_true=y_val, y_score=score_val)['aucpr']

                if best_metric_val < metric_val:
                    best_metric_val = metric_val
                    tol_count = 0

                    # save model
                    torch.save(model, os.path.join(os.getcwd(),'baseline','SDAD','model','SDAD_GMM.pt'))
                else:
                    tol_count += 1

                if tol_count >= tol:
                    print(f'Early stopping in epoch: {epoch}')
                    break

    return score_val_epoch