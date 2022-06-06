import torch
from torch import nn

class FS(nn.Module):
    def __init__(self, input_size, act_fun):
        super(FS, self).__init__()

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
            nn.Sigmoid()
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
        prob = self.reg_3(torch.cat((feature, e), dim=1))

        return feature, prob.squeeze()