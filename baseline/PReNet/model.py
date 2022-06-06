import torch
from torch import nn

class prenet(nn.Module):
    def __init__(self, input_size, act_fun):
        super(prenet, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_size, 20),
            act_fun
        )

        self.reg = nn.Linear(40, 1)

    #the input vector of prenet should be a pair
    def forward(self, X_left, X_right):
        feature_left = self.feature(X_left)
        feature_right = self.feature(X_right)

        # concat feature
        feature = torch.cat((feature_left, feature_right), dim=1)
        # generate score based on the concat feature
        score = self.reg(feature)

        return score.squeeze()