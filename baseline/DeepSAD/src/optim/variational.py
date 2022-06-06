import torch
import torch.nn.functional as F

from torch import nn
from itertools import repeat
from baseline.DeepSAD.src.utils import enumerate_discrete, log_sum_exp
from baseline.DeepSAD.src.networks import log_standard_categorical


# Acknowledgements: https://github.com/wohlert/semi-supervised-pytorch
class ImportanceWeightedSampler(object):
    """
    Importance weighted sampler (Burda et al., 2015) to be used together with SVI.

    :param mc: number of Monte Carlo samples
    :param iw: number of Importance Weighted samples
    """

    def __init__(self, mc=1, iw=1):
        self.mc = mc
        self.iw = iw

    def resample(self, x):
        return x.repeat(self.mc * self.iw, 1)

    def __call__(self, elbo):
        elbo = elbo.view(self.mc, self.iw, -1)
        elbo = torch.mean(log_sum_exp(elbo, dim=1, sum_op=torch.mean), dim=0)
        return elbo.view(-1)


class SVI(nn.Module):
    """
    Stochastic variational inference (SVI) optimizer for semi-supervised learning.

    :param model: semi-supervised model to evaluate
    :param likelihood: p(x|y,z) for example BCE or MSE
    :param beta: warm-up/scaling of KL-term
    :param sampler: sampler for x and y, e.g. for Monte Carlo
    """

    base_sampler = ImportanceWeightedSampler(mc=1, iw=1)

    def __init__(self, model, likelihood=F.binary_cross_entropy, beta=repeat(1), sampler=base_sampler):
        super(SVI, self).__init__()
        self.model = model
        self.likelihood = likelihood
        self.sampler = sampler
        self.beta = beta

    def forward(self, x, y=None):
        is_labeled = False if y is None else True

        # Prepare for sampling
        xs, ys = (x, y)

        # Enumerate choices of label
        if not is_labeled:
            ys = enumerate_discrete(xs, self.model.y_dim)
            xs = xs.repeat(self.model.y_dim, 1)

        # Increase sampling dimension
        xs = self.sampler.resample(xs)
        ys = self.sampler.resample(ys)

        reconstruction = self.model(xs, ys)

        # p(x|y,z)
        likelihood = -self.likelihood(reconstruction, xs)

        # p(y)
        prior = -log_standard_categorical(ys)

        # Equivalent to -L(x, y)
        elbo = likelihood + prior - next(self.beta) * self.model.kl_divergence
        L = self.sampler(elbo)

        if is_labeled:
            return torch.mean(L)

        logits = self.model.classify(x)

        L = L.view_as(logits.t()).t()

        # Calculate entropy H(q(y|x)) and sum over all labels
        eps = 1e-8
        H = -torch.sum(torch.mul(logits, torch.log(logits + eps)), dim=-1)
        L = torch.sum(torch.mul(logits, L), dim=-1)

        # Equivalent to -U(x)
        U = L + H

        return torch.mean(U)
