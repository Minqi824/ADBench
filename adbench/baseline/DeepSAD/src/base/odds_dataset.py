from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.datasets.utils import download_url

import os
import torch
import pandas as pd
import numpy as np
import os


class ODDSDataset(Dataset):
    """
    ODDSDataset class for datasets_cc from Outlier Detection DataSets (ODDS): http://odds.cs.stonybrook.edu/

    Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, data, train=True):
        super(Dataset, self).__init__()
        self.train = train

        if self.train:
            self.data = torch.tensor(data['X_train'], dtype=torch.float32)
            self.targets = torch.tensor(data['y_train'], dtype=torch.int64)
        else:
            self.data = torch.tensor(data['X_test'], dtype=torch.float32)
            self.targets = torch.tensor(data['y_test'], dtype=torch.int64)

        # self.semi_targets = torch.zeros_like(self.targets)
        self.semi_targets = self.targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)
