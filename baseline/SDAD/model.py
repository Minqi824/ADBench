import torch
from torch import nn


class SDAD(nn.Module):
    def __init__(self, input_size, act_fun):
        super(SDAD, self).__init__()

        # self.feature = nn.Sequential(
        #     nn.Linear(input_size, 20),
        #     act_fun,
        # )

        self.feature = nn.Sequential(
            nn.Linear(input_size, 100),
            act_fun,
            nn.Linear(100, 20),
            act_fun
        )

        self.reg = nn.Sequential(
            nn.Linear(20, 1),
            nn.BatchNorm1d(num_features=1)
        )

    def forward(self, X):
        feature = self.feature(X)
        score = self.reg(feature)

        return feature, score.squeeze()

# improving the score distribution based anomaly detection (SDAD) by the backbone in:
# "Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection"
class SDAD_pro(nn.Module):
    def __init__(self, input_size, act_fun):
        super(SDAD_pro, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            act_fun,
            nn.Linear(128, 64),
            act_fun
        )

        # if we add relu layer in the encoder, how to represent the direction for non-negative value?

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            act_fun,
            nn.Linear(128, input_size),
            act_fun
        )

        self.reg_1 = nn.Sequential(
            nn.Linear(input_size+64+1, 256),
            act_fun
        )

        self.reg_2 = nn.Sequential(
            nn.Linear(256+1, 32),
            act_fun
        )

        self.reg_3 = nn.Sequential(
            nn.Linear(32+1, 1),
            nn.BatchNorm1d(num_features=1)
        )

    def forward(self, X):
        # hidden representation
        h = self.encoder(X)

        # reconstructed input vector
        X_hat = self.decoder(h)

        # reconstruction residual vector
        r = torch.sub(X_hat, X)

        # reconstruction error
        e = r.norm(dim=1).reshape(-1, 1)

        # normalized reconstruction residual vector
        r = torch.div(r, e) #div by broadcast

        # regression
        feature = self.reg_1(torch.cat((h, r, e), dim=1))
        feature = self.reg_2(torch.cat((feature, e), dim=1))
        score = self.reg_3(torch.cat((feature, e), dim=1))

        return feature, score.squeeze()