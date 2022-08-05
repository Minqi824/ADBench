from typing import Any, Dict

import numpy as np
import rtdl
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
# import zero
import delu # the new version of zero package


from myutils import Utils

class FTTransformer():
    '''
    The original code: https://yura52.github.io/rtdl/stable/index.html
    The original paper: "Revisiting Deep Learning Models for Tabular Data", NIPS 2019
    '''
    def __init__(self, seed:int, model_name:str, n_epochs=100, batch_size=64):

        self.seed = seed
        self.model_name = model_name
        self.utils = Utils()

        # device
        if model_name == 'FTTransformer':
            self.device = self.utils.get_device(gpu_specific=True)
        else:
            self.device = self.utils.get_device(gpu_specific=False)

        # Docs: https://yura52.github.io/zero/0.0.4/reference/api/zero.improve_reproducibility.html
        # zero.improve_reproducibility(seed=self.seed)
        delu.improve_reproducibility(base_seed=int(self.seed))

        # hyper-parameter
        self.n_epochs = n_epochs # default is 1000
        self.batch_size = batch_size # default is 256

    def apply_model(self, x_num, x_cat=None):
        if isinstance(self.model, rtdl.FTTransformer):
            return self.model(x_num, x_cat)
        elif isinstance(self.model, (rtdl.MLP, rtdl.ResNet)):
            assert x_cat is None
            return self.model(x_num)
        else:
            raise NotImplementedError(
                f'Looks like you are using a custom model: {type(self.model)}.'
                ' Then you have to implement this branch first.'
            )

    @torch.no_grad()
    def evaluate(self, X, y=None):
        self.model.eval()
        score = []
        # for batch in delu.iter_batches(X[part], 1024):
        for batch in delu.iter_batches(X, self.batch_size):
            score.append(self.apply_model(batch))
        score = torch.cat(score).squeeze(1).cpu().numpy()
        score = scipy.special.expit(score)

        # calculate the metric
        if y is not None:
            target = y.cpu().numpy()
            metric = self.utils.metric(y_true=target, y_score=score)
        else:
            metric = {'aucroc': None, 'aucpr': None}

        return score, metric['aucpr']

    def fit(self, X_train, y_train, ratio=None):
        # set seed
        self.utils.set_seed(self.seed)

        # training set is used as the validation set in the anomaly detection task
        X = {'train': torch.from_numpy(X_train).float().to(self.device),
             'val': torch.from_numpy(X_train).float().to(self.device)}

        y = {'train': torch.from_numpy(y_train).float().to(self.device),
             'val': torch.from_numpy(y_train).float().to(self.device)}

        task_type = 'binclass'
        n_classes = None
        d_out = n_classes or 1

        if self.model_name == 'ResNet':
            self.model = rtdl.ResNet.make_baseline(
                d_in=X_train.shape[1],
                d_main=128,
                d_hidden=256,
                dropout_first=0.2,
                dropout_second=0.0,
                n_blocks=2,
                d_out=d_out,
            )
            lr = 0.001
            weight_decay = 0.0

        elif self.model_name == 'FTTransformer':
            self.model = rtdl.FTTransformer.make_default(
                n_num_features=X_train.shape[1],
                cat_cardinalities=None,
                last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
                d_out=d_out,
            )

        else:
            raise NotImplementedError

        self.model.to(self.device)
        optimizer = (
            self.model.make_default_optimizer()
            if isinstance(self.model, rtdl.FTTransformer)
            else torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        )
        loss_fn = (
            F.binary_cross_entropy_with_logits
            if task_type == 'binclass'
            else F.cross_entropy
            if task_type == 'multiclass'
            else F.mse_loss
        )

        # Create a dataloader for batches of indices
        # Docs: https://yura52.github.io/zero/reference/api/zero.data.IndexLoader.html
        train_loader = delu.data.IndexLoader(len(X['train']), self.batch_size, device=self.device)

        # Create a progress tracker for early stopping
        # Docs: https://yura52.github.io/zero/reference/api/zero.ProgressTracker.html
        progress = delu.ProgressTracker(patience=100)

        # training
        # report_frequency = len(X['train']) // self.batch_size // 5

        for epoch in range(1, self.n_epochs + 1):
            for iteration, batch_idx in enumerate(train_loader):
                self.model.train()
                optimizer.zero_grad()
                x_batch = X['train'][batch_idx]
                y_batch = y['train'][batch_idx]
                loss = loss_fn(self.apply_model(x_batch).squeeze(1), y_batch)
                loss.backward()
                optimizer.step()
                # if iteration % report_frequency == 0:
                #     print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')

            _, val_metric = self.evaluate(X=X['val'], y=y['val'])
            print(f'Epoch {epoch:03d} | Validation metric: {val_metric:.4f}', end='')
            progress.update((-1 if task_type == 'regression' else 1) * val_metric)
            if progress.success:
                print(' <<< BEST VALIDATION EPOCH', end='')
            print()
            if progress.fail:
                break

        return self

    def predict_score(self, X):
        X = torch.from_numpy(X).float().to(self.device)
        score, _ = self.evaluate(X=X, y=None)
        return score



