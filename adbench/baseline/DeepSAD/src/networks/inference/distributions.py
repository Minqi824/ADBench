import math
import torch
import torch.nn.functional as F


# Acknowledgements: https://github.com/wohlert/semi-supervised-pytorch
def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x.

    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)


def log_gaussian(x, mu, log_var):
    """
    Evaluates the log pdf of a normal distribution parametrized by mu and log_var at x.

    :param x: point to evaluate
    :param mu: mean
    :param log_var: log variance
    :return: log N(x|µ,σI)
    """
    log_pdf = -0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))
    return torch.sum(log_pdf, dim=-1)


def log_standard_categorical(p):
    """
    Computes the cross-entropy between a (one-hot) categorical vector and a standard (uniform) categorical distribution.
    :param p: one-hot categorical distribution
    :return: H(p,u)
    """
    eps = 1e-8
    prior = F.softmax(torch.ones_like(p), dim=1)  # Uniform prior over y
    prior.requires_grad = False
    cross_entropy = -torch.sum(p * torch.log(prior + eps), dim=1)

    return cross_entropy
