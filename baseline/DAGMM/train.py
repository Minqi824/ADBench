import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar

from baseline.DAGMM.model import DAGMM
from baseline.DAGMM.forward_step import ComputeLoss
from baseline.DAGMM.utils.utils import weights_init_normal

from torch.utils.data import DataLoader, TensorDataset

class TrainerDAGMM:
    """Trainer class for DAGMM."""
    def __init__(self, args, data, device):
        self.args = args
        self.device = device

        # input data
        X_train = data
        self.input_size = X_train.shape[1]

        # dataloader
        self.train_loader = DataLoader(torch.from_numpy(X_train).float(),
                                       batch_size=self.args.batch_size, shuffle=False, drop_last=True)

    def train(self):
        """Training the DAGMM model"""
        self.model = DAGMM(self.input_size, self.args.n_gmm, self.args.latent_dim).to(self.device)
        self.model.apply(weights_init_normal)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov, 
                                   self.device, self.args.n_gmm)
        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x in self.train_loader:
                x = x.float().to(self.device)
                optimizer.zero_grad()
                
                _, x_hat, z, gamma = self.model(x)

                loss = self.compute.forward(x, x_hat, z, gamma)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                total_loss += loss.item()

            # print('Training DAGMM... Epoch: {}, Loss: {:.3f}'.format(
            #        epoch, total_loss/len(self.train_loader)))
                

        

