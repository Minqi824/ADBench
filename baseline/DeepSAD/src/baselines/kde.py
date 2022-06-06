import json
import logging
import time
import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import GridSearchCV
from base.base_dataset import BaseADDataset
from networks.main import build_autoencoder


class KDE(object):
    """A class for Kernel Density Estimation models."""

    def __init__(self, hybrid=False, kernel='gaussian', n_jobs=-1, seed=None, **kwargs):
        """Init Kernel Density Estimation instance."""
        self.kernel = kernel
        self.n_jobs = n_jobs
        self.seed = seed

        self.model = KernelDensity(kernel=kernel, **kwargs)
        self.bandwidth = self.model.bandwidth

        self.hybrid = hybrid
        self.ae_net = None  # autoencoder network for the case of a hybrid model

        self.results = {
            'train_time': None,
            'test_time': None,
            'test_auc': None,
            'test_scores': None
        }

    def train(self, dataset: BaseADDataset, device: str = 'cpu', n_jobs_dataloader: int = 0,
              bandwidth_GridSearchCV: bool = True):
        """Trains the Kernel Density Estimation model on the training data."""
        logger = logging.getLogger()

        # do not drop last batch for non-SGD optimization shallow_ssad
        train_loader = DataLoader(dataset=dataset.train_set, batch_size=128, shuffle=True,
                                  num_workers=n_jobs_dataloader, drop_last=False)

        # Get data from loader
        X = ()
        for data in train_loader:
            inputs, _, _, _ = data
            inputs = inputs.to(device)
            if self.hybrid:
                inputs = self.ae_net.encoder(inputs)  # in hybrid approach, take code representation of AE as features
            X_batch = inputs.view(inputs.size(0), -1)  # X_batch.shape = (batch_size, n_channels * height * width)
            X += (X_batch.cpu().data.numpy(),)
        X = np.concatenate(X)

        # Training
        logger.info('Starting training...')
        start_time = time.time()

        if bandwidth_GridSearchCV:
            # use grid search cross-validation to select bandwidth
            logger.info('Using GridSearchCV for bandwidth selection...')
            params = {'bandwidth': np.logspace(0.5, 5, num=10, base=2)}
            hyper_kde = GridSearchCV(KernelDensity(kernel=self.kernel), params, n_jobs=self.n_jobs, cv=5, verbose=0)
            hyper_kde.fit(X)
            self.bandwidth = hyper_kde.best_estimator_.bandwidth
            logger.info('Best bandwidth: {:.8f}'.format(self.bandwidth))
            self.model = hyper_kde.best_estimator_
        else:
            # if exponential kernel, re-initialize kde with bandwidth minimizing the numerical error
            if self.kernel == 'exponential':
                self.bandwidth = np.max(pairwise_distances(X)) ** 2
                self.model = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)

            self.model.fit(X)

        train_time = time.time() - start_time
        self.results['train_time'] = train_time

        logger.info('Training Time: {:.3f}s'.format(self.results['train_time']))
        logger.info('Finished training.')

    def test(self, dataset: BaseADDataset, device: str = 'cpu', n_jobs_dataloader: int = 0):
        """Tests the Kernel Density Estimation model on the test data."""
        logger = logging.getLogger()

        _, test_loader = dataset.loaders(batch_size=128, num_workers=n_jobs_dataloader)

        # Get data from loader
        idx_label_score = []
        X = ()
        idxs = []
        labels = []
        for data in test_loader:
            inputs, label_batch, _, idx = data
            inputs, label_batch, idx = inputs.to(device), label_batch.to(device), idx.to(device)
            if self.hybrid:
                inputs = self.ae_net.encoder(inputs)  # in hybrid approach, take code representation of AE as features
            X_batch = inputs.view(inputs.size(0), -1)  # X_batch.shape = (batch_size, n_channels * height * width)
            X += (X_batch.cpu().data.numpy(),)
            idxs += idx.cpu().data.numpy().astype(np.int64).tolist()
            labels += label_batch.cpu().data.numpy().astype(np.int64).tolist()
        X = np.concatenate(X)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        scores = (-1.0) * self.model.score_samples(X)
        self.results['test_time'] = time.time() - start_time
        scores = scores.flatten()

        # Save triples of (idx, label, score) in a list
        idx_label_score += list(zip(idxs, labels, scores.tolist()))
        self.results['test_scores'] = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.results['test_auc'] = roc_auc_score(labels, scores)

        # Log results
        logger.info('Test AUC: {:.2f}%'.format(100. * self.results['test_auc']))
        logger.info('Test Time: {:.3f}s'.format(self.results['test_time']))
        logger.info('Finished testing.')

    def load_ae(self, dataset_name, model_path):
        """Load pretrained autoencoder from model_path for feature extraction in a hybrid KDE model."""

        model_dict = torch.load(model_path, map_location='cpu')
        ae_net_dict = model_dict['ae_net_dict']
        if dataset_name in ['mnist', 'fmnist', 'cifar10']:
            net_name = dataset_name + '_LeNet'
        else:
            net_name = dataset_name + '_mlp'

        if self.ae_net is None:
            self.ae_net = build_autoencoder(net_name)

        # update keys (since there was a change in network definition)
        ae_keys = list(self.ae_net.state_dict().keys())
        for i in range(len(ae_net_dict)):
            k, v = ae_net_dict.popitem(False)
            new_key = ae_keys[i]
            ae_net_dict[new_key] = v
            i += 1

        self.ae_net.load_state_dict(ae_net_dict)
        self.ae_net.eval()

    def save_model(self, export_path):
        """Save KDE model to export_path."""
        pass

    def load_model(self, import_path, device: str = 'cpu'):
        """Load KDE model from import_path."""
        pass

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
