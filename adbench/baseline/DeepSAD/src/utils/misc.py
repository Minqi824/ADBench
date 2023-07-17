import torch

from torch.autograd import Variable


# Acknowledgements: https://github.com/wohlert/semi-supervised-pytorch
def enumerate_discrete(x, y_dim):
    """
    Generates a 'torch.Tensor' of size batch_size x n_labels of the given label.

    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """

    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.to(x.device)

    return Variable(generated.float())


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.

    :param tensor: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=True)
    return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max


def binary_cross_entropy(x, y):
    eps = 1e-8
    return -torch.sum(y * torch.log(x + eps) + (1 - y) * torch.log(1 - x + eps), dim=-1)
