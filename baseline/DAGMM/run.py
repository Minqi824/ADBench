from myutils import Utils
from baseline.DAGMM.train import TrainerDAGMM
from baseline.DAGMM.test import eval

import numpy as np

class DAGMM():
    '''
    PyTorch implementation of DAGMM from "https://github.com/mperezcarrasco/PyTorch-DAGMM"
    '''
    def __init__(self, seed, model_name='DAGMM', tune=False,
                 num_epochs=200, patience=50, lr=1e-4, lr_milestones=[50], batch_size=256,
                 latent_dim=1, n_gmm=4, lambda_energy=0.1, lambda_cov=0.005):
        '''
        The default batch_size is 1024
        The default latent_dim is 1
        The default lambda_cov is 0.005
        '''
        self.utils = Utils()
        self.device = self.utils.get_device()  # get device
        self.seed = seed
        self.tune = tune

        # hyper-parameter
        class Args:
            pass

        self.args = Args()
        self.args.num_epochs = num_epochs
        self.args.patience = patience
        self.args.lr = lr
        self.args.lr_milestones = lr_milestones
        self.args.batch_size = batch_size
        self.args.latent_dim = latent_dim
        self.args.n_gmm = n_gmm
        self.args.lambda_energy = lambda_energy
        self.args.lambda_cov = lambda_cov

    def grid_search(self, X_train, y_train, ratio):
        '''
        implement the grid search for unsupervised models and return the best hyper-parameters
        the ratio could be the ground truth anomaly ratio of input dataset
        '''

        # set seed
        self.utils.set_seed(self.seed)
        # get the hyper-parameter grid (n_gmm, default=4)
        param_grid = [4, 6, 8, 10]

        # index of normal ana abnormal samples
        idx_a = np.where(y_train==1)[0]
        idx_n = np.where(y_train==0)[0]
        idx_n = np.random.choice(idx_n, int((len(idx_a) * (1-ratio)) / ratio), replace=True)

        idx = np.append(idx_n, idx_a) #combine
        np.random.shuffle(idx) #shuffle

        # valiation set (and the same anomaly ratio as in the original dataset)
        X_val = X_train[idx]
        y_val = y_train[idx]

        # fitting
        metric_list = []
        for param in param_grid:
            try:
                self.args.n_gmm = param
                model = TrainerDAGMM(self.args, X_train, self.device)
                model.train()

            except:
                metric_list.append(0.0)
                continue

            try:
                # model performance on the validation set
                data = {'X_train': X_train, 'X_test':X_val}

                score_val = eval(model.model, data, self.device, self.args.n_gmm, self.args.batch_size)
                metric = self.utils.metric(y_true=y_val, y_score=score_val, pos_label=1)
                metric_list.append(metric['aucpr'])

            except:
                metric_list.append(0.0)
                continue

        self.args.n_gmm = param_grid[np.argmax(metric_list)]

        print(f'The candidate hyper-parameter: {param_grid},',
              f' corresponding metric: {metric_list}',
              f' the best candidate: {self.args.n_gmm}')

        return self

    def fit(self, X_train, y_train=None, ratio=None):
        # set seed using myutils
        self.utils.set_seed(self.seed)

        if sum(y_train) > 0 and self.tune:
            self.grid_search(X_train, y_train, ratio)
        else:
            pass

        print(f'using the params: {self.args.n_gmm}')

        # initialization
        self.model = TrainerDAGMM(self.args, X_train, self.device)
        # fitting
        self.model.train()

        return self

    def predict_score(self, X_train, X_test):
        data = {'X_train': X_train, 'X_test': X_test}

        # predicting
        score = eval(self.model.model, data, self.device, self.args.n_gmm, self.args.batch_size)
        return score

# X_train=np.random.randn(5000, 16)
# X_test=np.random.randn(2000, 16)
#
# model = DAGMM(seed=42)
# model.fit(X_train=X_train, y_train=np.repeat(0, X_train.shape[0]))
# score_test = model.predict_score(X_train, X_test)
# print(len(score_test))