import os
import sys
from myutils import Utils

from baseline.GANomaly.model import generator
from baseline.GANomaly.model import discriminator

import torch
from torch import nn
from torch.autograd import Variable

def fit(train_loader, net_generator, net_discriminator, optimizer_G, optimizer_D,
        epochs, batch_size, print_loss, device,
        seed, input_size, hidden_size, act_fun):

    #my utils
    utils = Utils()

    # loss criterion
    L1_criterion = nn.L1Loss(reduction='mean')
    L2_criterion = nn.MSELoss(reduction='mean')
    BCE_criterion = nn.BCELoss(reduction='mean')

    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            X, _ = data
            y_real = torch.FloatTensor(batch_size).fill_(0)  # real label=0,size=batch_size
            y_fake = torch.FloatTensor(batch_size).fill_(1)  # fake label=1,size=batch_size

            X = X.to(device)
            y_real = y_real.to(device)
            y_fake = y_fake.to(device)

            X = Variable(X)
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

            loss_D.backward()
            optimizer_D.step()

            # if loss_D < 1e-1:
            #     print('Reinitialization of discriminator...')
            #     utils.set_seed(seed)
            #     net_discriminator = discriminator(input_size=input_size, act_fun=act_fun)
            #     net_discriminator.to(device)

            # training the generator based on the result from the discriminator
            net_generator.zero_grad()

            z, X_hat, z_hat = net_generator(X)

            # latent loss
            feature_real, _ = net_discriminator(X)
            feature_fake, _ = net_discriminator(X_hat)

            loss_G_latent = L2_criterion(feature_fake, feature_real)

            # contexutal loss
            loss_G_contextual = L1_criterion(X, X_hat)
            # entire loss in generator

            # encoder loss
            loss_G_encoder = L1_criterion(z, z_hat)

            loss_G = (loss_G_latent + loss_G_contextual + loss_G_encoder) / 3

            loss_G.backward()
            optimizer_G.step()

            if (i % 50 == 0) & print_loss:
                print('[%d/%d] [%d/%d] Loss D: %.4f / Loss G: %.4f' % (
                epoch + 1, epochs, i, len(train_loader), loss_D, loss_G))