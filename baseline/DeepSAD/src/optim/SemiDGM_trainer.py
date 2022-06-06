from baseline.DeepSAD.src.base.base_trainer import BaseTrainer
from baseline.DeepSAD.src.base.base_dataset import BaseADDataset
from baseline.DeepSAD.src.base.base_net import BaseNet
from baseline.DeepSAD.src.optim.variational import SVI, ImportanceWeightedSampler
from baseline.DeepSAD.src.utils.misc import binary_cross_entropy
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np


class SemiDeepGenerativeTrainer(BaseTrainer):

    def __init__(self, alpha: float = 0.1, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        self.alpha = alpha

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device
        net = net.to(self.device)

        # Use importance weighted sampler (Burda et al., 2015) to get a better estimate on the log-likelihood.
        sampler = ImportanceWeightedSampler(mc=1, iw=1)
        elbo = SVI(net, likelihood=binary_cross_entropy, sampler=sampler)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, labels, semi_targets, _ = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)

                # Get labeled and unlabeled data and make labels one-hot
                inputs = inputs.view(inputs.size(0), -1)
                x = inputs[semi_targets != 0]
                u = inputs[semi_targets == 0]
                y = labels[semi_targets != 0]
                if y.nelement() > 1:
                    y_onehot = torch.Tensor(y.size(0), 2).to(self.device)  # two labels: 0: normal, 1: outlier
                    y_onehot.zero_()
                    y_onehot.scatter_(1, y.view(-1, 1), 1)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                if y.nelement() < 2:
                    L = torch.tensor(0.0).to(self.device)
                else:
                    L = -elbo(x, y_onehot)
                U = -elbo(u)

                # Regular cross entropy
                if y.nelement() < 2:
                    classication_loss = torch.tensor(0.0).to(self.device)
                else:
                    # Add auxiliary classification loss q(y|x)
                    logits = net.classify(x)
                    eps = 1e-8
                    classication_loss = torch.sum(y_onehot * torch.log(logits + eps), dim=1).mean()

                # Overall loss
                loss = L - self.alpha * classication_loss + U  # J_alpha

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device
        net = net.to(self.device)

        # Use importance weighted sampler (Burda et al., 2015) to get a better estimate on the log-likelihood.
        sampler = ImportanceWeightedSampler(mc=1, iw=1)
        elbo = SVI(net, likelihood=binary_cross_entropy, sampler=sampler)

        # Testing
        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, _, idx = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                idx = idx.to(self.device)

                # All test data is considered unlabeled
                inputs = inputs.view(inputs.size(0), -1)
                u = inputs
                y = labels
                y_onehot = torch.Tensor(y.size(0), 2).to(self.device)  # two labels: 0: normal, 1: outlier
                y_onehot.zero_()
                y_onehot.scatter_(1, y.view(-1, 1), 1)

                # Compute loss
                L = -elbo(u, y_onehot)
                U = -elbo(u)

                logits = net.classify(u)
                eps = 1e-8
                classication_loss = -torch.sum(y_onehot * torch.log(logits + eps), dim=1).mean()

                loss = L + self.alpha * classication_loss + U  # J_alpha

                # Compute scores
                scores = logits[:, 1]  # likelihood/confidence for anomalous class as anomaly score

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')
