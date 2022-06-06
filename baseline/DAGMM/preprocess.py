import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import pickle as pl



class KDDCupData:
    def __init__(self, data_dir, mode):
        """Loading the data for train and test."""
        data = np.load(data_dir, allow_pickle=True)

        labels = data["kdd"][:,-1]
        features = data["kdd"][:,:-1]
        #In this case, "atack" has been treated as normal data as is mentioned in the paper
        normal_data = features[labels==0] 
        normal_labels = labels[labels==0]

        n_train = int(normal_data.shape[0]*0.5)
        ixs = np.arange(normal_data.shape[0])
        np.random.shuffle(ixs)
        normal_data_test = normal_data[ixs[n_train:]]
        normal_labels_test = normal_labels[ixs[n_train:]]

        if mode == 'train':
            self.x = normal_data[ixs[:n_train]]
            self.y = normal_labels[ixs[:n_train]]
        elif mode == 'test':
            anomalous_data = features[labels==1]
            anomalous_labels = labels[labels==1]
            self.x = np.concatenate((anomalous_data, normal_data_test), axis=0)
            self.y = np.concatenate((anomalous_labels, normal_labels_test), axis=0)

    def __len__(self):
        """Number of images in the object dataset."""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Return a sample from the dataset."""
        return np.float32(self.x[index]), np.float32(self.y[index])



def get_KDDCup99(args, data_dir='./data/kdd_cup.npz'):
    """Returning train and test dataloaders."""
    train = KDDCupData(data_dir, 'train')
    dataloader_train = DataLoader(train, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    
    test = KDDCupData(data_dir, 'test')
    dataloader_test = DataLoader(test, batch_size=args.batch_size, 
                              shuffle=False, num_workers=0)
    return dataloader_train, dataloader_test