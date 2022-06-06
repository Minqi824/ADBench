from torch import nn

class generator(nn.Module):
    def __init__(self, input_size, hidden_size, act_fun):
        super(generator, self).__init__()

        self.encoder_1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            act_fun,
        )

        self.decoder_1 = nn.Sequential(
            nn.Linear(hidden_size, input_size),
        )

        self.encoder_2 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            act_fun,
        )

    def forward(self, input):
        z = self.encoder_1(input)
        X_hat = self.decoder_1(z)
        z_hat = self.encoder_2(X_hat)

        return z, X_hat, z_hat

class discriminator(nn.Module):
    def __init__(self, input_size, act_fun):
        super(discriminator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 100),
            act_fun,
            nn.Linear(100, 20),
            act_fun
        )

        self.classifier = nn.Sequential(
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        latent_vector = self.encoder(input)
        output = self.classifier(latent_vector)

        return latent_vector, output
