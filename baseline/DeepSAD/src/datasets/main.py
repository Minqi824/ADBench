# from .mnist import MNIST_Dataset
# from .fmnist import FashionMNIST_Dataset
# from .cifar10 import CIFAR10_Dataset
from .odds import ODDSADDataset


def load_dataset(data):
    """Loads the dataset."""
    #代码中暂不支持DeepSAD部署在CV数据集中,之后会更新

    # if dataset == 'mnist':
    #     dataset = MNIST_Dataset(root=data_path,
    #                             normal_class=normal_class,
    #                             known_outlier_class=known_outlier_class,
    #                             n_known_outlier_classes=n_known_outlier_classes,
    #                             ratio_known_normal=ratio_known_normal,
    #                             ratio_known_outlier=ratio_known_outlier,
    #                             ratio_pollution=ratio_pollution)
    #
    # elif dataset == 'fmnist':
    #     dataset = FashionMNIST_Dataset(root=data_path,
    #                                    normal_class=normal_class,
    #                                    known_outlier_class=known_outlier_class,
    #                                    n_known_outlier_classes=n_known_outlier_classes,
    #                                    ratio_known_normal=ratio_known_normal,
    #                                    ratio_known_outlier=ratio_known_outlier,
    #                                    ratio_pollution=ratio_pollution)
    #
    # elif dataset == 'cifar10':
    #     dataset = CIFAR10_Dataset(root=data_path,
    #                               normal_class=normal_class,
    #                               known_outlier_class=known_outlier_class,
    #                               n_known_outlier_classes=n_known_outlier_classes,
    #                               ratio_known_normal=ratio_known_normal,
    #                               ratio_known_outlier=ratio_known_outlier,
    #                               ratio_pollution=ratio_pollution)

    #tabular data
    dataset = ODDSADDataset(data=data)

    return dataset
