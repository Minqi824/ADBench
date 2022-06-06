import torch
from torch import nn


class SDAD_VAE(nn.Module):
    def __init__(self, input_size, act_fun):
        super(SDAD_VAE, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_size, 100),
            act_fun,
            nn.Linear(100, 20),
            act_fun
        )

        self.mu = nn.Sequential(
            nn.Linear(20, 1)
        )

        self.log_var = nn.Sequential(
            nn.Linear(20, 1)
        )

    def sample(self, mu, std):
        # sampling from normal distribution (could be vector)
        # 这边用到了Reparameterization Trick使得梯度可以传播
        z = mu + torch.randn_like(std) * std
        return z

    def forward(self, X):
        feature = self.feature(X)
        # mu and std
        mu, log_var = self.mu(feature), self.log_var(feature)
        mu = mu.squeeze()
        log_var = log_var.squeeze()
        std = torch.exp(log_var / 2)

        # sampling
        score = self.sample(mu, std)

        return feature, mu, std, score.squeeze()

class SDAD_pro_VAE(nn.Module):
    def __init__(self, input_size, act_fun):
        super(SDAD_pro_VAE, self).__init__()

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

        self.mu = nn.Sequential(
            nn.Linear(32+1, 1)
        )

        self.log_var = nn.Sequential(
            nn.Linear(32+1, 1)
        )

    def sample(self, mu, std):
        # sampling from normal distribution (could be vector)
        # 这边用到了Reparameterization Trick使得梯度可以传播
        z = mu + torch.randn_like(std) * std
        return z

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

        # mu and std
        mu, log_var = self.mu(torch.cat((feature, e), dim=1)), self.log_var(torch.cat((feature, e), dim=1))
        mu = mu.squeeze()
        log_var = log_var.squeeze()
        std = torch.exp(log_var / 2)

        # sampling
        score = self.sample(mu, std)
        return feature, mu, std, score.squeeze()