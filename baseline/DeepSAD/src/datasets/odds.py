from torch.utils.data import DataLoader, Subset
from baseline.DeepSAD.src.base.base_dataset import BaseADDataset
from baseline.DeepSAD.src.base.odds_dataset import ODDSDataset
from .preprocessing import create_semisupervised_setting

import torch


class ODDSADDataset(BaseADDataset):

    def __init__(self, data):
        super().__init__(self)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = (0,)
        self.outlier_classes = (1,)

        # Get training set
        self.train_set = ODDSDataset(data=data, train=True)
        # Get testing set
        self.test_set = ODDSDataset(data=data, train=False)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):

        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)

        return train_loader, test_loader
