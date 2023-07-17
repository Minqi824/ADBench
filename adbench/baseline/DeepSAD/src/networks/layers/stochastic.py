import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


# Acknowledgements: https://github.com/wohlert/semi-supervised-pytorch
class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the reparametrization trick (Kingma and Welling, 2013) to draw a sample from a
    distribution parametrized by mu and log_var.
    """

    def __init__(self):
        super(Stochastic, self).__init__()

    def reparametrize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        if mu.is_cuda:
            epsilon = epsilon.to(mu.device)

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul(std, epsilon)

        return z

    def forward(self, x):
        raise NotImplementedError


class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a Gaussian distribution.
    """

    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))
        return self.reparametrize(mu, log_var), mu, log_var
