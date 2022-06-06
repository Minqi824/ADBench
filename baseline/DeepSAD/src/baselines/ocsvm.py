import json
import logging
import time
import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from base.base_dataset import BaseADDataset
from networks.main import build_autoencoder


class OCSVM(object):
    """A class for One-Class SVM models."""

    def __init__(self, kernel='rbf', nu=0.1, hybrid=False):
        """Init OCSVM instance."""
        self.kernel = kernel
        self.nu = nu
        self.rho = None
        self.gamma = None

        self.model = OneClassSVM(kernel=kernel, nu=nu)

        self.hybrid = hybrid
        self.ae_net = None  # autoencoder network for the case of a hybrid model
        self.linear_model = None  # also init a model with linear kernel if hybrid approach

        self.results = {
            'train_time': None,
            'test_time': None,
            'test_auc': None,
            'test_scores': None,
            'train_time_linear': None,
            'test_time_linear': None,
            'test_auc_linear': None
        }

    def train(self, dataset: BaseADDataset, device: str = 'cpu', n_jobs_dataloader: int = 0):
        """Trains the OC-SVM model on the training data."""
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

        # Select model via hold-out test set of 1000 samples
        gammas = np.logspace(-7, 2, num=10, base=2)
        best_auc = 0.0

        # Sample hold-out set from test set
        _, test_loader = dataset.loaders(batch_size=128, num_workers=n_jobs_dataloader)

        X_test = ()
        labels = []
        for data in test_loader:
            inputs, label_batch, _, _ = data
            inputs, label_batch = inputs.to(device), label_batch.to(device)
            if self.hybrid:
                inputs = self.ae_net.encoder(inputs)  # in hybrid approach, take code representation of AE as features
            X_batch = inputs.view(inputs.size(0), -1)  # X_batch.shape = (batch_size, n_channels * height * width)
            X_test += (X_batch.cpu().data.numpy(),)
            labels += label_batch.cpu().data.numpy().astype(np.int64).tolist()
        X_test, labels = np.concatenate(X_test), np.array(labels)
        n_test, n_normal, n_outlier = len(X_test), np.sum(labels == 0), np.sum(labels == 1)
        n_val = int(0.1 * n_test)
        n_val_normal, n_val_outlier = int(n_val * (n_normal/n_test)), int(n_val * (n_outlier/n_test))
        perm = np.random.permutation(n_test)
        X_val = np.concatenate((X_test[perm][labels[perm] == 0][:n_val_normal],
                                X_test[perm][labels[perm] == 1][:n_val_outlier]))
        labels = np.array([0] * n_val_normal + [1] * n_val_outlier)

        i = 1
        for gamma in gammas:

            # Model candidate
            model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=gamma)

            # Train
            start_time = time.time()
            model.fit(X)
            train_time = time.time() - start_time

            # Test on small hold-out set from test set
            scores = (-1.0) * model.decision_function(X_val)
            scores = scores.flatten()

            # Compute AUC
            auc = roc_auc_score(labels, scores)

            logger.info(f'  | Model {i:02}/{len(gammas):02} | Gamma: {gamma:.8f} | Train Time: {train_time:.3f}s '
                        f'| Val AUC: {100. * auc:.2f} |')

            if auc > best_auc:
                best_auc = auc
                self.model = model
                self.gamma = gamma
                self.results['train_time'] = train_time

            i += 1

        # If hybrid, also train a model with linear kernel
        if self.hybrid:
            self.linear_model = OneClassSVM(kernel='linear', nu=self.nu)
            start_time = time.time()
            self.linear_model.fit(X)
            train_time = time.time() - start_time
            self.results['train_time_linear'] = train_time

        logger.info(f'Best Model: | Gamma: {self.gamma:.8f} | AUC: {100. * best_auc:.2f}')
        logger.info('Training Time: {:.3f}s'.format(self.results['train_time']))
        logger.info('Finished training.')

    def test(self, dataset: BaseADDataset, device: str = 'cpu', n_jobs_dataloader: int = 0):
        """Tests the OC-SVM model on the test data."""
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

        scores = (-1.0) * self.model.decision_function(X)

        self.results['test_time'] = time.time() - start_time
        scores = scores.flatten()
        self.rho = -self.model.intercept_[0]

        # Save triples of (idx, label, score) in a list
        idx_label_score += list(zip(idxs, labels, scores.tolist()))
        self.results['test_scores'] = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.results['test_auc'] = roc_auc_score(labels, scores)

        # If hybrid, also test model with linear kernel
        if self.hybrid:
            start_time = time.time()
            scores_linear = (-1.0) * self.linear_model.decision_function(X)
            self.results['test_time_linear'] = time.time() - start_time
            scores_linear = scores_linear.flatten()
            self.results['test_auc_linear'] = roc_auc_score(labels, scores_linear)
            logger.info('Test AUC linear model: {:.2f}%'.format(100. * self.results['test_auc_linear']))
            logger.info('Test Time linear model: {:.3f}s'.format(self.results['test_time_linear']))

        # Log results
        logger.info('Test AUC: {:.2f}%'.format(100. * self.results['test_auc']))
        logger.info('Test Time: {:.3f}s'.format(self.results['test_time']))
        logger.info('Finished testing.')

    def load_ae(self, dataset_name, model_path):
        """Load pretrained autoencoder from model_path for feature extraction in a hybrid OC-SVM model."""

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
        """Save OC-SVM model to export_path."""
        pass

    def load_model(self, import_path, device: str = 'cpu'):
        """Load OC-SVM model from import_path."""
        pass

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
