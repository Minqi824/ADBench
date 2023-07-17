import json
import torch

from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder
from optim import SemiDeepGenerativeTrainer, VAETrainer


class SemiDeepGenerativeModel(object):
    """A class for the Semi-Supervised Deep Generative model (M1+M2 model).

    Paper: Kingma et al. (2014). Semi-supervised learning with deep generative models. In NIPS (pp. 3581-3589).
    Link: https://papers.nips.cc/paper/5352-semi-supervised-learning-with-deep-generative-models.pdf

    Attributes:
        net_name: A string indicating the name of the neural network to use.
        net: The neural network.
        trainer: SemiDeepGenerativeTrainer to train a Semi-Supervised Deep Generative model.
        optimizer_name: A string indicating the optimizer to use for training.
        results: A dictionary to save the results.
    """

    def __init__(self, alpha: float = 0.1):
        """Inits SemiDeepGenerativeModel."""

        self.alpha = alpha

        self.net_name = None
        self.net = None

        self.trainer = None
        self.optimizer_name = None

        self.vae_net = None  # variational autoencoder network for pretraining
        self.vae_trainer = None
        self.vae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

        self.vae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }

    def set_vae(self, net_name):
        """Builds the variational autoencoder network for pretraining."""
        self.net_name = net_name
        self.vae_net = build_autoencoder(self.net_name)  # VAE for pretraining

    def set_network(self, net_name):
        """Builds the neural network."""
        self.net_name = net_name
        self.net = build_network(net_name, ae_net=self.vae_net)  # full M1+M2 model

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the Semi-Supervised Deep Generative model on the training data."""

        self.optimizer_name = optimizer_name

        self.trainer = SemiDeepGenerativeTrainer(alpha=self.alpha, optimizer_name=optimizer_name, lr=lr,
                                                 n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                                 weight_decay=weight_decay, device=device,
                                                 n_jobs_dataloader=n_jobs_dataloader)
        self.net = self.trainer.train(dataset, self.net)
        self.results['train_time'] = self.trainer.train_time

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Semi-Supervised Deep Generative model on the test data."""

        if self.trainer is None:
            self.trainer = SemiDeepGenerativeTrainer(alpha=self.alpha, device=device,
                                                     n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(dataset, self.net)
        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def pretrain(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains a variational autoencoder (M1) for the Semi-Supervised Deep Generative model."""

        # Train
        self.vae_optimizer_name = optimizer_name
        self.vae_trainer = VAETrainer(optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                      lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader)
        self.vae_net = self.vae_trainer.train(dataset, self.vae_net)
        # Get train results
        self.vae_results['train_time'] = self.vae_trainer.train_time

        # Test
        self.vae_trainer.test(dataset, self.vae_net)
        # Get test results
        self.vae_results['test_auc'] = self.vae_trainer.test_auc
        self.vae_results['test_time'] = self.vae_trainer.test_time

    def save_model(self, export_model):
        """Save a Semi-Supervised Deep Generative model to export_model."""

        net_dict = self.net.state_dict()
        torch.save({'net_dict': net_dict}, export_model)

    def load_model(self, model_path):
        """Load a Semi-Supervised Deep Generative model from model_path."""

        model_dict = torch.load(model_path)
        self.net.load_state_dict(model_dict['net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    def save_vae_results(self, export_json):
        """Save variational autoencoder results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.vae_results, fp)
