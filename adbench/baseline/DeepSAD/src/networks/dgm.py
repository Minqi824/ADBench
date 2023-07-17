import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from .vae import VariationalAutoencoder, Encoder, Decoder


# Acknowledgements: https://github.com/wohlert/semi-supervised-pytorch
class Classifier(nn.Module):
    """
    Classifier network, i.e. q(y|x), for two classes (0: normal, 1: outlier)

    :param net: neural network class to use (as parameter to use the same network over different shallow_ssad)
    """

    def __init__(self, net, dims=None):
        super(Classifier, self).__init__()
        self.dims = dims
        if dims is None:
            self.net = net()
            self.logits = nn.Linear(self.net.rep_dim, 2)
        else:
            [x_dim, h_dim, y_dim] = dims
            self.dense = nn.Linear(x_dim, h_dim)
            self.logits = nn.Linear(h_dim, y_dim)

    def forward(self, x):
        if self.dims is None:
            x = self.net(x)
        else:
            x = F.relu(self.dense(x))
        x = F.softmax(self.logits(x), dim=-1)
        return x


class DeepGenerativeModel(VariationalAutoencoder):
    """
    M2 model from the paper 'Semi-Supervised Learning with Deep Generative Models' (Kingma et al., 2014).

    The 'Generative semi-supervised model' (M2) is a probabilistic model that incorporates label information in both
    inference and generation.

    :param dims: dimensions of the model given by [input_dim, label_dim, latent_dim, [hidden_dims]].
    :param classifier_net: classifier network class to use.
    """

    def __init__(self, dims, classifier_net=None):
        [x_dim, self.y_dim, z_dim, h_dim] = dims
        super(DeepGenerativeModel, self).__init__([x_dim, z_dim, h_dim])

        self.encoder = Encoder([x_dim + self.y_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim + self.y_dim, list(reversed(h_dim)), x_dim])
        if classifier_net is None:
            self.classifier = Classifier(net=None, dims=[x_dim, h_dim[0], self.y_dim])
        else:
            self.classifier = Classifier(classifier_net)

        # Init linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        z, q_mu, q_log_var = self.encoder(torch.cat((x, y), dim=1))
        self.kl_divergence = self._kld(z, (q_mu, q_log_var))
        rec = self.decoder(torch.cat((z, y), dim=1))

        return rec

    def classify(self, x):
        logits = self.classifier(x)
        return logits

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.

        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        y = y.float()
        x = self.decoder(torch.cat((z, y), dim=1))
        return x


class StackedDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims, features):
        """
        M1+M2 model as described in (Kingma et al., 2014).

        :param dims: dimensions of the model given by [input_dim, label_dim, latent_dim, [hidden_dims]].
        :param classifier_net: classifier network class to use.
        :param features: a pre-trained M1 model of class 'VariationalAutoencoder' trained on the same dataset.
        """
        [x_dim, y_dim, z_dim, h_dim] = dims
        super(StackedDeepGenerativeModel, self).__init__([features.z_dim, y_dim, z_dim, h_dim])

        # Be sure to reconstruct with the same dimensions
        in_features = self.decoder.reconstruction.in_features
        self.decoder.reconstruction = nn.Linear(in_features, x_dim)

        # Make vae feature model untrainable by freezing parameters
        self.features = features
        self.features.train(False)

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Sample a new latent x from the M1 model
        x_sample, _, _ = self.features.encoder(x)

        # Use the sample as new input to M2
        return super(StackedDeepGenerativeModel, self).forward(x_sample, y)

    def classify(self, x):
        _, x, _ = self.features.encoder(x)
        logits = self.classifier(x)
        return logits
