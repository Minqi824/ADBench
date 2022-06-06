import os
import sys
from myutils import Utils

from baseline.WSGAN.model import generator
from baseline.WSGAN.model import discriminator

import torch
from torch import nn
from torch.autograd import Variable


def fit(dataloader, net_generator, net_discriminator, optimizer_G, optimizer_D,
        eta, epochs, seed,  batch_size, input_size, act_fun,
        device, org_loss=True, print_loss=False):
    '''
    :param dataloader:
    :param net_generator:
    :param net_discriminator:
    :param optimizer_G:
    :param optimizer_D:
    :param eta: weight for the combination of loss function
    :param epochs:
    :param seed:
    :param batch_size:
    :param input_size:
    :param act_fun:
    :param device:
    :param org_loss: whether to use the original loss function in WSGAN
    :param print_loss:
    :return:
    '''
    L1_criterion = nn.L1Loss(reduction='none')
    L2_criterion = nn.MSELoss(reduction='none')
    BCE_criterion = nn.BCELoss(reduction='mean')

    # my utils
    utils = Utils()

    for epoch in range(epochs):
        for i, data in enumerate(dataloader):

            # y_aclabel means the acquired label information (which may be contaminated)
            X, y_aclabel = data
            y_real = torch.FloatTensor(batch_size).fill_(0)  # real label=0,size=batch_size
            y_fake = torch.FloatTensor(batch_size).fill_(1)  # fake label=1,size=batch_size

            # to cuda
            X = X.to(device)
            y_aclabel = y_aclabel.to(device)
            y_real = y_real.to(device)
            y_fake = y_fake.to(device)

            X = Variable(X)
            y_aclabel = Variable(y_aclabel)
            y_real = Variable(y_real)
            y_fake = Variable(y_fake)

            # zero grad for discriminator
            net_discriminator.zero_grad()

            # training the discriminator with real sample
            _, output = net_discriminator(X)
            loss_D_real = BCE_criterion(output.view(-1), y_real)

            # training the discriminator with fake sample
            _, X_hat, _ = net_generator(X)
            _, output = net_discriminator(X_hat)
            loss_D_fake = BCE_criterion(output.view(-1), y_fake)

            # entire loss in discriminator
            loss_D = (loss_D_real + loss_D_fake) / 2

            # backward
            loss_D.backward()
            optimizer_D.step()

            # reinitialization
            if loss_D < 1e-1:
                print('Reinitialization of discriminator...')
                utils.set_seed(seed)
                net_discriminator = discriminator(input_size=input_size, act_fun=act_fun)
                net_discriminator.to(device)

            # training the generator based on the result from the discriminator
            net_generator.zero_grad()

            z, X_hat, z_hat = net_generator(X)
            feature_real, _ = net_discriminator(X)
            feature_fake, _ = net_discriminator(X_hat)

            # 2021.5.17:注意此处应该是sum而非mean(之前代码为torch.mean)
            loss_G_contextual = torch.mean(L1_criterion(X, X_hat), 1)  # contextual loss
            loss_G_encoder = torch.mean(L1_criterion(z, z_hat), 1)  # encdoer loss
            loss_G_latent = torch.mean(L2_criterion(feature_fake, feature_real), 1)  # latent loss

            loss_G = (loss_G_contextual + loss_G_encoder + loss_G_latent) / 3

            if org_loss:
                loss_G_u = torch.mean(loss_G[y_aclabel==0])
                loss_G_a = torch.mean(torch.pow(loss_G[y_aclabel==1],-1))
                if loss_G_a.size(0) > 0:
                    loss_G = (1 - eta) * loss_G_u + eta * loss_G_a
                else:
                    loss_G = loss_G_u
            else:
                loss_G_u = torch.mean(loss_G[y_aclabel==0])
                loss_G_a = torch.mean(loss_G[y_aclabel==1])

                loss_G = (1 - eta) * loss_G_u - eta * loss_G_a


            loss_G.backward()
            optimizer_G.step()

            if (i % 50 == 0) & print_loss:
                print('[%d/%d] [%d/%d] Loss D: %.4f / Loss G: %.4f' % (
                epoch + 1, epochs, i, len(dataloader), loss_D, loss_G))
