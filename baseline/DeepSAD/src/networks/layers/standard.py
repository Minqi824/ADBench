import torch

from torch.nn import Module
from torch.nn import init
from torch.nn.parameter import Parameter


# Acknowledgements: https://github.com/wohlert/semi-supervised-pytorch
class Standardize(Module):
    """
    Applies (element-wise) standardization with trainable translation parameter μ and scale parameter σ, i.e. computes
    (x - μ) / σ where '/' is applied element-wise.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn a translation parameter μ.
            Default: ``True``

    Attributes:
        mu: the learnable translation parameter μ.
        std: the learnable scale parameter σ.
    """
    __constants__ = ['mu']

    def __init__(self, in_features, bias=True, eps=1e-6):
        super(Standardize, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.eps = eps
        self.std = Parameter(torch.Tensor(in_features))
        if bias:
            self.mu = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('mu', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.std, 1)
        if self.mu is not None:
            init.constant_(self.mu, 0)

    def forward(self, x):
        if self.mu is not None:
            x -= self.mu
        x = torch.div(x, self.std + self.eps)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.mu is not None
        )
