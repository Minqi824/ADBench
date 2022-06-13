import torch
from torch import nn
import numpy as np

from myutils import Utils
from baseline.PReNet.model import prenet
from baseline.PReNet.fit import fit

'''
The unofficial implement (with PyTorch) of the PReNet model in the paper "Deep Weakly-supervised Anomaly Detection"
The default hyper-parameter is the same as in the original paper
'''

class PReNet():
    def __init__(self, seed:int, model_name='PReNet', epochs:int=50, batch_num:int=20, batch_size:int=512,
                act_fun=nn.ReLU(), lr:float=1e-3, weight_decay:float=1e-2,
                s_a_a=8, s_a_u=4, s_u_u=0):

        self.seed = seed
        self.utils = Utils()
        self.device = self.utils.get_device()

        # hyper-parameters
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.act_fun = act_fun
        self.lr = lr
        self.weight_decay = weight_decay

        self.s_a_a = s_a_a
        self.s_a_u = s_a_u
        self.s_u_u = s_u_u

    def fit(self, X_train, y_train, ratio=None):

        input_size = X_train.shape[1]  # input size
        self.X_train_tensor = torch.from_numpy(X_train).float()  # testing set
        self.y_train = y_train

        self.utils.set_seed(self.seed)
        self.model = prenet(input_size=input_size, act_fun=self.act_fun)
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)  # optimizer

        # training
        fit(X_train_tensor=self.X_train_tensor, y_train=y_train, model=self.model, optimizer=optimizer,
            epochs=self.epochs, batch_num=self.batch_num, batch_size=self.batch_size,
            s_a_a=self.s_a_a, s_a_u=self.s_a_u, s_u_u=self.s_u_u, device=self.device)

        return self


    def predict_score(self, X, num=30):
        self.model = self.model.eval()

        if torch.is_tensor(X):
            pass
        else:
            X = torch.from_numpy(X)

        X = X.float()
        X = X.to(self.device)

        score = []
        for i in range(X.size(0)):
            index_a = np.random.choice(np.where(self.y_train==1)[0], num, replace=True) #postive sample in training set
            index_u = np.random.choice(np.where(self.y_train==0)[0], num, replace=True) #negative sample in training set

            X_train_a_tensor = self.X_train_tensor[index_a]
            X_train_u_tensor = self.X_train_tensor[index_u]

            with torch.no_grad():
                score_a_x = self.model(X_train_a_tensor, torch.cat(num * [X[i].view(1, -1)]))
                score_x_u = self.model(torch.cat(num * [X[i].view(1, -1)]), X_train_u_tensor)

            score_sub = torch.mean(score_a_x + score_x_u)
            score_sub = score_sub.numpy()[()]

            # entire score
            score.append(score_sub)

        return np.array(score)






