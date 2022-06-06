import numpy as np
import pandas as pd
import random
import scipy.io
import os
import mat73
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
from sklearn.mixture import GaussianMixture

from copulas.multivariate import VineCopula
from copulas.univariate import GaussianKDE

from myutils import Utils

# to do: leverage the categorical feature in the current datasets
# currently, data generator only supports for generating the binary classification datasets
class DataGenerator():
    def __init__(self, seed:int=42, dataset:str=None, test_size:float=0.3,
                 generate_duplicates=False, n_samples_threshold=1000, show_statistic=False):
        '''
        Only global parameters should be provided in the DataGenerator instantiation

        seed: seed for reproducible experimental results
        dataset: dataset name
        test_size: testing data size
        '''

        self.seed = seed
        self.dataset = dataset
        self.test_size = test_size

        # 当数据量不够时, 是否生成重复样本
        self.generate_duplicates = generate_duplicates
        self.n_samples_threshold = n_samples_threshold

        # 是否展示统计结果
        self.show_statistic = show_statistic

        # myutils function
        self.utils = Utils()


    '''
    By default, generator will generate real-world datasets.
    However, the realistic synthetic data can be generated, which follows the paper
    "Benchmarking Unsupervised Outlier Detection with Realistic Synthetic Data"

    realistic_synthetic_mode can be either local or global based on the original paper
    alpha and percentage are two parameters cited in the paper
    '''
    def generate_realistic_synthetic(self, X, y, realistic_synthetic_mode, alpha:int, percentage:float):
        '''
        Currently, three types of realistic synthetic outliers can be generated:
        1. local outliers: where normal data follows the GMM distribuion, and anomalies follow the GMM distribution with bigger covariance
        2. dependency outliers: where normal data follows the vine coupula distribution, and anomalies follow the independent distribution captured by GaussianKDE
        3. global outliers: where normal data follows the GMM distribuion, and anomalies follow the uniform distribution
        '''

        if realistic_synthetic_mode in ['local', 'cluster', 'dependency', 'global']:
            pass
        else:
            raise NotImplementedError


        # the number of normal data and anomalies
        pts_n = len(np.where(y == 0)[0])
        pts_a = len(np.where(y == 1)[0])

        # only use the normal data to fit the model
        X = X[y == 0]
        y = y[y == 0]

        # generate the synthetic normal data
        if realistic_synthetic_mode in ['local', 'cluster', 'global']:
            # select the best n_components based on the BIC value
            metric_list = []
            n_components_list = list(np.arange(1, 10))

            for n_components in n_components_list:
                gm = GaussianMixture(n_components=n_components, random_state=self.seed).fit(X)
                metric_list.append(gm.bic(X))

            best_n_components = n_components_list[np.argmin(metric_list)]

            # refit based on the best n_components
            gm = GaussianMixture(n_components=best_n_components, random_state=self.seed).fit(X)

            # generate the synthetic normal data
            X_synthetic_normal = gm.sample(pts_n)[0]

        # we found that copula function may occur error in some datasets
        elif realistic_synthetic_mode == 'dependency':
            # sampling the feature since copulas method may spend too long to fit
            if X.shape[1] > 50:
                idx = np.random.choice(np.arange(X.shape[1]), 50, replace=False)
                X = X[:, idx]

            copula = VineCopula('center') # default is the C-vine copula
            copula.fit(pd.DataFrame(X))

            # sample to generate synthetic normal data
            X_synthetic_normal = copula.sample(pts_n).values

        else:
            pass

        # generate the synthetic abnormal data
        if realistic_synthetic_mode == 'local':
            # generate the synthetic anomalies (local outliers)
            gm.covariances_ = alpha * gm.covariances_
            X_synthetic_anomalies = gm.sample(pts_a)[0]

        elif realistic_synthetic_mode == 'cluster':
            # generate the clustering synthetic anomalies
            gm.means_ = alpha * gm.means_
            X_synthetic_anomalies = gm.sample(pts_a)[0]

        elif realistic_synthetic_mode == 'dependency':
            X_synthetic_anomalies = np.zeros((pts_a, X.shape[1]))

            # using the GuassianKDE for generating independent feature
            for i in range(X.shape[1]):
                kde = GaussianKDE()
                kde.fit(X[:, i])
                X_synthetic_anomalies[:, i] = kde.sample(pts_a)

        elif realistic_synthetic_mode == 'global':
            # generate the synthetic anomalies (global outliers)
            X_synthetic_anomalies = []

            for i in range(X_synthetic_normal.shape[1]):
                low = np.min(X_synthetic_normal[:, i]) * (1 + percentage)
                high = np.max(X_synthetic_normal[:, i]) * (1 + percentage)

                X_synthetic_anomalies.append(np.random.uniform(low=low, high=high, size=pts_a))

            X_synthetic_anomalies = np.array(X_synthetic_anomalies).T

        else:
            pass

        X = np.concatenate((X_synthetic_normal, X_synthetic_anomalies), axis=0)
        y = np.append(np.repeat(0, X_synthetic_normal.shape[0]),
                      np.repeat(1, X_synthetic_anomalies.shape[0]))

        return X, y


    '''
    Here we also consider the robustness of baseline models, where three types of noise can be added
    1. Duplicated anomalies, which should be added to training and testing set, respectively
    2. Irrelevant features, which should be added to both training and testing set
    3. Anomaly contamination, which should be only added to the training set
    4. Label contamination, which should be only added to the training set
    '''
    def add_duplicated_anomalies(self, X, y, duplicate_times:int):
        if duplicate_times <= 1:
            pass
        else:
            # index of normal and anomaly data
            idx_n = np.where(y==0)[0]
            idx_a = np.where(y==1)[0]

            # # generate duplicated anomalies
            # pts_a = len(idx_a)
            # idx_a = np.random.choice(idx_a, ceil(pts_a / duplicate_times))
            # idx_a = np.random.choice(idx_a, pts_a)
            #
            # idx = np.append(idx_n, idx_a); random.shuffle(idx)
            # X = X[idx]; y = y[idx]

            # generate duplicated anomalies
            idx_a = np.random.choice(idx_a, int(len(idx_a) * duplicate_times))

            idx = np.append(idx_n, idx_a); random.shuffle(idx)
            X = X[idx]; y = y[idx]

        return X, y

    def add_irrelevant_features(self, X, y, noise_ratio:float):
        # adding uniform noise
        if noise_ratio == 0.0:
            pass
        else:
            noise_dim = int(noise_ratio / (1 - noise_ratio) * X.shape[1])
            if noise_dim > 0:
                X_noise = []
                for i in range(noise_dim):
                    idx = np.random.choice(np.arange(X.shape[1]), 1)
                    X_min = np.min(X[:, idx])
                    X_max = np.max(X[:, idx])

                    X_noise.append(np.random.uniform(X_min, X_max, size=(X.shape[0], 1)))

                # concat the irrelevant noise feature
                X_noise = np.hstack(X_noise)
                X = np.concatenate((X, X_noise), axis=1)
                # shuffle the dimension
                idx = np.random.choice(np.arange(X.shape[1]), X.shape[1], replace=False)
                X = X[:, idx]

        return X, y

    def remove_anomaly_contamination(self, idx_unlabeled_anomaly, contam_ratio:float):
        idx_unlabeled_anomaly = np.random.choice(idx_unlabeled_anomaly, int(contam_ratio * len(idx_unlabeled_anomaly)), replace=False)
        return idx_unlabeled_anomaly

    def add_label_contamination(self, X, y, noise_ratio:float):
        if noise_ratio == 0.0:
            pass
        else:
            # # here we mainly consider the situation where labeled anomalies have contamination, i.e., unlabeled normal data
            # idx_n = np.where(y==0)[0]
            # idx_a = np.where(y==1)[0]
            #
            # noise_num = int(noise_ratio / (1 - noise_ratio) * len(idx_a))
            # # here replace is set to true for we observe that some datasets the anomalies are even more than normal data
            # idx_n_noise = np.random.choice(idx_n, noise_num, replace=True)
            # y[idx_n_noise] = 1

            # here we consider the label flips situation: a label is randomly filpped to another class with probability p (i.e., noise ratio)
            idx_flips = np.random.choice(np.arange(len(y)), int(len(y) * noise_ratio), replace=False)
            y[idx_flips] = 1 - y[idx_flips] # change 0 to 1 and 1 to 0

        return X, y

    def generator(self, la=None, at_least_one_labeled=False,
                  realistic_synthetic_mode=None, alpha:int=5, percentage:float=0.1,
                  noise_type=None, duplicate_times:int=2, contam_ratio=1.00, noise_ratio:float=0.05):
        '''
        la: labeled anomalies, can be either the ratio of labeled anomalies or the number of labeled anomalies
        at_least_one_labeled: whether to guarantee at least one labeled anomalies in the training set
        '''

        # set seed for reproducible results
        self.utils.set_seed(self.seed)

        # transfer different file format to the numpy array
        if self.dataset in ['annthyroid', 'cardio', 'mammography', 'musk', 'optdigits', 'pendigits',
                            'satimage-2', 'speech', 'thyroid', 'vowels', 'cover', 'http', 'letter',
                            'mnist', 'satellite', 'shuttle', 'smtp', 'breastw', 'vertebral',
                            'wine']:
            if self.dataset in ['http', 'smtp']:
                data = mat73.loadmat(os.path.join('datasets', self.dataset + '.mat'))
            else:
                data = scipy.io.loadmat(os.path.join('datasets', self.dataset + '.mat'))
            X = data['X']
            y = data['y'].squeeze().astype('int64')

        elif self.dataset in ['Waveform_withoutdupl_v10', 'InternetAds_withoutdupl_norm_19', 'PageBlocks_withoutdupl_09',
                              'SpamBase_withoutdupl_40', 'Wilt_withoutdupl_05', 'Cardiotocography_withoutdupl_22',
                              'WBC_withoutdupl_v10', 'WDBC_withoutdupl_v10', 'WPBC_withoutdupl_norm',
                              'Arrhythmia_withoutdupl_46', 'HeartDisease_withoutdupl_44', 'Hepatitis_withoutdupl_16',
                              'Parkinson_withoutdupl_75', 'Pima_withoutdupl_35', 'Stamps_withoutdupl_09']:
            data = pd.read_csv(os.path.join('datasets', self.dataset + '.csv'))

            data.columns = [_.split("'")[1] for _ in data.columns]
            X = data.drop(['outlier', 'id'], axis=1).values
            y = [_.split("'")[1] for _ in data['outlier'].values]
            y = np.array([0 if _ == 'no' else 1 for _ in y])

        elif self.dataset in ['ALOI_withoutdupl', 'glass_withoutduplicates_normalized',
                              'Ionosphere_withoutdupl_norm', 'Lymphography_withoutdupl_idf']:
            data = pd.read_csv(os.path.join('datasets', self.dataset + '.csv'))

            X = data.drop(['outlier','id'], axis=1).values
            y = np.array([0 if _ == 'no' else 1 for _ in data['outlier'].values])

        elif self.dataset in ['abalone.diff', 'comm.and.crime.diff', 'concrete.diff', 'fault.diff', 'imgseg.diff',
                              'landsat.diff', 'magic.gamma.diff', 'skin.diff', 'yeast.diff']:
            data = pd.read_csv(os.path.join('datasets', self.dataset + '.csv'))
            X = data.drop(['point.id', 'motherset', 'origin', 'original.label', 'diff.score', 'ground.truth'],
                          axis=1).values
            y = np.array([0 if _ == 'nominal' else 1 for _ in data['ground.truth'].values])

        # Credit Card Fraud Detection (CCFD) dataset
        elif self.dataset == 'CCFD':
            data = pd.read_csv(os.path.join('datasets', self.dataset + '.csv'))
            X = data.drop(['Time', 'Class'], axis=1)
            y = data['Class'].values

        # Taiwan Bankruptcy Prediction (TBP) dataset
        elif self.dataset == 'TBP':
            data = pd.read_csv(os.path.join('datasets', self.dataset + '.csv'))
            X = data.drop(['Flag'], axis=1)
            y = data['Flag'].values

        elif self.dataset in ['amazon', 'yelp', 'imdb'] + \
                             ['agnews_' + str(i) for i in range(4)] +\
                             ['FashionMNIST_' + str(i) for i in range(10)] +\
                             ['CIFAR10_' + str(i) for i in range(10)] +\
                             ['SVHN_' + str(i) for i in range(10)]:
            data = np.load(os.path.join('datasets_NLP_CV', self.dataset + '.npz'))
            X = data['X']
            y = data['y']

        else:
            raise NotImplementedError

        # to array
        X = np.array(X)
        y = np.array(y)

        # number of labeled anomalies in the original data
        if type(la) == float:
            if at_least_one_labeled:
                n_labeled_anomalies = ceil(sum(y) * (1 - self.test_size) * la)
            else:
                n_labeled_anomalies = int(sum(y) * (1 - self.test_size) * la)
        elif type(la) == int:
            n_labeled_anomalies = la
        else:
            raise NotImplementedError

        ################################
        # 如果数据集异常占比过高(超过5%), 异常样本抽样
        # if (sum(y) / len(y)) > 0.05:
        #     print(f'数据集{self.dataset}异常占比过大, 正在抽样...')
        #     self.utils.set_seed(self.seed)
        #     idx_n = np.where(y==0)[0]
        #     idx_a = np.where(y==1)[0]
        #     idx_a = np.random.choice(idx_a, int(0.05 * len(idx_n) / 0.95), replace=False)
        #
        #     idx = np.append(idx_n, idx_a)
        #     random.shuffle(idx)
        #
        #     X = X[idx]
        #     y = y[idx]

        # 如果数据集大小不足, 生成duplicate样本:
        if len(y) < self.n_samples_threshold and self.generate_duplicates:
            print(f'数据集{self.dataset}正在生成duplicate samples...')
            self.utils.set_seed(self.seed)
            idx_duplicate = np.random.choice(np.arange(len(y)), self.n_samples_threshold, replace=True)
            X = X[idx_duplicate]
            y = y[idx_duplicate]

        # 如果数据集过大, 整体样本抽样
        if len(y) > 10000:
            print(f'数据集{self.dataset}数据量过大, 正在抽样...')
            self.utils.set_seed(self.seed)
            idx_sample = np.random.choice(np.arange(len(y)), 10000, replace=False)
            X = X[idx_sample]
            y = y[idx_sample]
        ################################

        # whether to generate realistic synthetic outliers
        if realistic_synthetic_mode is not None:
            # 因为dependency outlier生成的时间太长了, 这边加载了提前生成好的样本(大于3000的数据集抽样到了3000)
            if realistic_synthetic_mode == 'dependency':
                print('使用了提前生成的dependency outliers...')
                dataset_dict = np.load('dependency_outlier_large.npz', allow_pickle=True)
                dataset_dict = dataset_dict['dataset'].item()

                if self.dataset in dataset_dict.keys():
                    X, y = dataset_dict[self.dataset]
                else:
                    raise NotImplementedError

            else:
                X, y = self.generate_realistic_synthetic(X, y,
                                                         realistic_synthetic_mode=realistic_synthetic_mode,
                                                         alpha=alpha, percentage=percentage)

        # whether to add different types of noise for testing the robustness of benchmark models
        if noise_type is None:
            pass

        elif noise_type == 'duplicated_anomalies':
            # X, y = self.add_duplicated_anomalies(X, y, duplicate_times=duplicate_times)
            pass

        elif noise_type == 'irrelevant_features':
            X, y = self.add_irrelevant_features(X, y, noise_ratio=noise_ratio)

        elif noise_type == 'anomaly_contamination':
            pass

        elif noise_type == 'label_contamination':
            pass

        else:
            raise NotImplementedError

        print(f'current noise type: {noise_type}')

        # show the statistic
        self.utils.data_description(X=X, y=y)

        # spliting the current data to the training set and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, shuffle=True, stratify=y)

        # we respectively generate the duplicated anomalies for the training and testing set
        if noise_type == 'duplicated_anomalies':
            X_train, y_train = self.add_duplicated_anomalies(X_train, y_train, duplicate_times=duplicate_times)
            X_test, y_test = self.add_duplicated_anomalies(X_test, y_test, duplicate_times=duplicate_times)

        # notice that label contamination can only be added in the training set
        elif noise_type == 'label_contamination':
            X_train, y_train = self.add_label_contamination(X_train, y_train, noise_ratio=noise_ratio)

        # minmax scaling
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # idx of normal samples and unlabeled/labeled anomalies
        idx_normal = np.where(y_train == 0)[0]
        idx_anomaly = np.where(y_train == 1)[0]

        if type(la) == float:
            if at_least_one_labeled:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, ceil(la * len(idx_anomaly)), replace=False)
            else:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, int(la * len(idx_anomaly)), replace=False)
        elif type(la) == int:
            if la > len(idx_anomaly):
                raise AssertionError(f'设置的nla超过了异常样本的总数:{len(idx_anomaly)}')
            else:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, la, replace=False)
        else:
            raise NotImplementedError

        idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)
        # whether to remove the anomaly contamination in the unlabeled data
        if noise_type == 'anomaly_contamination':
            idx_unlabeled_anomaly = self.remove_anomaly_contamination(idx_unlabeled_anomaly, contam_ratio)

        # unlabel data = normal data + unlabeled anomalies (which is considered as contamination)
        idx_unlabeled = np.append(idx_normal, idx_unlabeled_anomaly)

        del idx_anomaly, idx_unlabeled_anomaly

        # the label of unlabeled data is 0, and that of labeled anomalies is 1
        y_train[idx_unlabeled] = 0
        y_train[idx_labeled_anomaly] = 1

        return {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test,
                'n_labeled_anomalies':n_labeled_anomalies}

    # generate out-of-distribution dataset (to do)
    def generator_OOD(self,
                      ID_num: int=2, OOD_ratio: float=0.5,
                      n_samples_train=7000, n_samples_test=3000, anomaly_ratio=0.01):
        '''
        ID_num: number of in-distribution anomaly classes in the training set
        OOD_ratio: the out-of-distribution anomaly ratio in the testing set
        n_samples_train: the training set size
        n_samples_test: the testing set size
        '''

        # kddcup99 data
        data_org = pd.read_csv(os.path.join('datasets_unprocessed', 'kddcup99.csv'))

        #观察所有类别分布
        if self.show_statistic:
            classes = list(data_org['label'].drop_duplicates())
            for i in range(len(classes)):
                print(f"class: {classes[i]}, count: {data_org[data_org['label']==classes[i]].shape[0]}")

        #set seed and shuffle
        self.utils.set_seed(self.seed)
        data = data_org.sample(frac=1).reset_index(drop=True)

        # global parameters
        anomaly_class_pool = data['label'].drop_duplicates().values
        anomaly_class_pool = [_ for _ in anomaly_class_pool if _ != 'normal.' and sum(data['label']==_) >=
                              int((n_samples_train+n_samples_test)*anomaly_ratio)]

        # ID anomaly class pool(从总共的异常类别中抽取几个作为in distribution的异常)
        anomaly_class_ID_pool = list(combinations(anomaly_class_pool, self.ID_num))

        # ID anomaly class, 从所有的ID异常类别中随机抽取一种情况
        anomaly_class_ID = list(anomaly_class_ID_pool[np.random.choice(np.arange(len(anomaly_class_ID_pool)), 1)[0]])
        # OOD anomaly class
        anomaly_class_OOD = list(set(anomaly_class_pool) - set(anomaly_class_ID))

        if self.show_statistic:
            print(f'ID 异常类别: {anomaly_class_ID}; OOD 异常类别: {anomaly_class_OOD}')

        # sampling for the normal samples
        idx_normal = np.where(data['label'] == 'normal.')[0]
        # normal samples of training set
        idx_normal_train = np.random.choice(idx_normal, int(n_samples_train * (1-anomaly_ratio)), replace=False)
        # normal samples of testing set
        idx_normal_test = np.random.choice(np.setdiff1d(idx_normal, idx_normal_train),
                                           int(n_samples_test * (1-anomaly_ratio)), replace=False)

        # in the training set, all anomalies are in-distribution anomaly class
        # in the testing set, half of the anomalies are in-distribution and another half are out-of-distribution

        # the index of in-distribution anomalies
        idx_anomalies_ID = []
        for c in anomaly_class_ID:
            idx_anomalies_ID.extend(list(np.where(data['label'] == c)[0]))

        # the index of out-of-distribution anomalies
        idx_anomalies_OOD = []
        for c in anomaly_class_OOD:
            idx_anomalies_OOD.extend(list(np.where(data['label'] == c)[0]))

        # to array
        idx_anomalies_ID = np.array(idx_anomalies_ID)
        idx_anomalies_OOD = np.array(idx_anomalies_OOD)

        # training set anomalies
        idx_anomalies_ID_train = np.random.choice(idx_anomalies_ID, int(n_samples_train*anomaly_ratio), replace=False)

        idx_anomalies_ID_test = np.random.choice(np.setdiff1d(idx_anomalies_ID, idx_anomalies_ID_train),
                                                 int(n_samples_test * anomaly_ratio * (1 - self.OOD_ratio)), replace=False)
        idx_anomalies_OOD_test =np.random.choice(idx_anomalies_OOD,
                                                 int(n_samples_test * anomaly_ratio * self.OOD_ratio), replace=False)

        idx_train = np.concatenate([idx_normal_train, idx_anomalies_ID_train])
        idx_test = np.concatenate([idx_normal_test, idx_anomalies_ID_test, idx_anomalies_OOD_test])

        # combine
        y_train = np.append(np.repeat(0, len(idx_normal_train)),
                            np.repeat(1, len(idx_anomalies_ID_train)))
        X_train = data.drop(['label'], axis=1).values[idx_train]

        y_test = np.append(np.repeat(0, len(idx_normal_test)),
                           np.repeat(1, len(idx_anomalies_ID_test) + len(idx_anomalies_OOD_test)))
        X_test = data.drop(['label'], axis=1).values[idx_test]

        # shuffle idx
        self.utils.set_seed(self.seed)
        idx_train = np.random.choice(np.arange(len(idx_train)), len(idx_train), replace=False)
        idx_test = np.random.choice(np.arange(len(idx_test)), len(idx_test), replace=False)

        X_train = X_train[idx_train]; y_train = y_train[idx_train]
        X_test = X_test[idx_test]; y_test = y_test[idx_test]

        # minmax scaling
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

