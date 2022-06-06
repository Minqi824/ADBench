import torch.nn as nn
import torch.nn.functional as F

from baseline.DeepSAD.src.base.base_net import BaseNet


class MLP(BaseNet):

    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [x_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        self.code = nn.Linear(h_dims[-1], rep_dim, bias=bias)

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        return self.code(x)


class MLP_Decoder(BaseNet):

    def __init__(self, x_dim, h_dims=[64, 128], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [rep_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        self.reconstruction = nn.Linear(h_dims[-1], x_dim, bias=bias)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        x = self.reconstruction(x)
        return self.output_activation(x)


class MLP_Autoencoder(BaseNet):

    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = MLP(x_dim, h_dims, rep_dim, bias)
        self.decoder = MLP_Decoder(x_dim, list(reversed(h_dims)), rep_dim, bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Linear_BN_leakyReLU(nn.Module):
    """
    A nn.Module that consists of a Linear layer followed by BatchNorm1d and a leaky ReLu activation
    """

    def __init__(self, in_features, out_features, bias=False, eps=1e-04):
        super(Linear_BN_leakyReLU, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features, eps=eps, affine=bias)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.linear(x)))
