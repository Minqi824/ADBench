import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .layers.stochastic import GaussianSample
from .inference.distributions import log_standard_gaussian, log_gaussian


# Acknowledgements: https://github.com/wohlert/semi-supervised-pytorch
class Encoder(nn.Module):
    """
    Encoder, i.e. the inference network.

    Attempts to infer the latent probability distribution p(z|x) from the data x by fitting a
    variational distribution q_φ(z|x). Returns the two parameters of the distribution (µ, log σ²).

    :param dims: dimensions of the network given by [input_dim, [hidden_dims], latent_dim].
    """

    def __init__(self, dims, sample_layer=GaussianSample):
        super(Encoder, self).__init__()

        [x_dim, h_dim, z_dim] = dims
        neurons = [x_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(linear_layers)
        self.sample = sample_layer(h_dim[-1], z_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.sample(x)


class Decoder(nn.Module):
    """
    Decoder, i.e. the generative network.

    Generates samples from an approximation p_θ(x|z) of the original distribution p(x)
    by transforming a latent representation z.

    :param dims: dimensions of the network given by [latent_dim, [hidden_dims], input_dim].
    """

    def __init__(self, dims):
        super(Decoder, self).__init__()

        [z_dim, h_dim, x_dim] = dims
        neurons = [z_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(linear_layers)
        self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.output_activation(self.reconstruction(x))


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) (Kingma and Welling, 2013) model consisting of an encoder-decoder pair for which
    a variational distribution is fitted to the encoder.
    Also known as the M1 model in (Kingma et al., 2014)

    :param  dims: dimensions of the networks given by [input_dim, latent_dim, [hidden_dims]]. Encoder and decoder
    are build symmetrically.
    """

    def __init__(self, dims):
        super(VariationalAutoencoder, self).__init__()

        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim
        self.flow = None

        self.encoder = Encoder([x_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim, list(reversed(h_dim)), x_dim])
        self.kl_divergence = 0

        # Init linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, q_param, p_param=None):
        """
        Computes the KL-divergence of some latent variable z.

        KL(q||p) = - ∫ q(z) log [ p(z) / q(z) ] = - E_q[ log p(z) - log q(z) ]

        :param z: sample from q-distribuion
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        (mu, log_var) = q_param

        if self.flow is not None:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        else:
            qz = log_gaussian(z, mu, log_var)

        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)

        kl = qz - pz

        return kl

    def add_flow(self, flow):
        self.flow = flow

    def forward(self, x, y=None):
        """
        Runs a forward pass on a data point through the VAE model to provide its reconstruction and the parameters of
        the variational approximate distribution q.

        :param x: input data
        :return: reconstructed input
        """
        z, q_mu, q_log_var = self.encoder(x)
        self.kl_divergence = self._kld(z, (q_mu, q_log_var))
        rec = self.decoder(z)

        return rec

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from the learned distribution based on p_θ(x|z).

        :param z: (torch.autograd.Variable) latent normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)
