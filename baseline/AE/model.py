from torch import nn

class network(nn.Module):
    def __init__(self, input_size:int, hidden_size:int=20, act_fun=None):
        super(network, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            act_fun,
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, input):
        z = self.encoder(input)
        X_hat = self.decoder(z)

        return z, X_hat