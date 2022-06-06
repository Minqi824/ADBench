import numpy as np
from tqdm import tqdm
import torch

from myutils import Utils

utils = Utils()

'''
from the original paper, when implement stratified random sampling
batch / 2 is from the (u,u) pair, batch / 4 is from the (a,a) pair and batch / 2 is from the (a,u) pair
where u is the unlabeled data and a is the labeled anomalies
'''

def sampler_pairs(X_train_tensor, y_train, epoch, batch_num, batch_size, s_a_a, s_a_u, s_u_u):
    '''
    X_train_tensor: the input X in the torch.tensor form
    y_train: label in the numpy.array form

    batch_num: generate how many batches in one epoch
    batch_size: the batch size
    '''
    data_loader_X = []
    data_loader_y = []

    index_a = np.where(y_train == 1)[0]
    index_u = np.where(y_train == 0)[0]

    for i in range(batch_num):  # i.e., drop_last = True
        index = []

        # 分别是(a,a); (a,u); (u,u)共6部分样本
        for j in range(6):
            # generate unique seed and set seed
            seed = utils.unique(epoch, i)
            seed = utils.unique(seed, j)
            utils.set_seed(seed)

            if j < 3:  # 其中batch size // 4与原论文中一致
                index_sub = np.random.choice(index_a, batch_size // 4, replace=True)
                index.append(list(index_sub))

            if j == 3:
                index_sub = np.random.choice(index_u, batch_size // 4, replace=True)  # unlabel部分可以变为False
                index.append(list(index_sub))

            if j > 3:
                index_sub = np.random.choice(index_u, batch_size // 2, replace=True)  # unlabel部分可以变为False
                index.append(list(index_sub))

        # index[0] + index[1] = (a,a), batch / 4
        # index[2] + index[2] = (a,u), batch / 4
        # index[4] + index[5] = (u,u), batch / 2
        index_left = index[0] + index[2] + index[4]
        index_right = index[1] + index[3] + index[5]

        X_train_tensor_left = X_train_tensor[index_left]
        X_train_tensor_right = X_train_tensor[index_right]

        # generate label
        y_train_new = np.append(np.repeat(s_a_a, batch_size // 4), np.repeat(s_a_u, batch_size // 4))
        y_train_new = np.append(y_train_new, np.repeat(s_u_u, batch_size // 2))
        y_train_new = torch.from_numpy(y_train_new).float()

        # shuffle
        index_shuffle = np.arange(len(y_train_new))
        index_shuffle = np.random.choice(index_shuffle, len(index_shuffle), replace=False)

        X_train_tensor_left = X_train_tensor_left[index_shuffle]
        X_train_tensor_right = X_train_tensor_right[index_shuffle]
        y_train_new = y_train_new[index_shuffle]

        # save
        data_loader_X.append([X_train_tensor_left, X_train_tensor_right])  # 注意left和right顺序
        data_loader_y.append(y_train_new)

    return data_loader_X, data_loader_y