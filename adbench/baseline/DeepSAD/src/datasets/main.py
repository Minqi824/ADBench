# from .mnist import MNIST_Dataset
# from .fmnist import FashionMNIST_Dataset
# from .cifar10 import CIFAR10_Dataset
from .odds import ODDSADDataset


def load_dataset(data, train=True):
    """Loads the dataset."""

    # for tabular data
    dataset = ODDSADDataset(data=data, train=train)

    return dataset
