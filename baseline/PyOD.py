from myutils import Utils
import numpy as np

#add the baselines from the pyod package
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.combination import aom
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.lscp import LSCP
from pyod.models.mad import MAD
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from pyod.models.rod import ROD
from pyod.models.sod import SOD
from pyod.models.sos import SOS
from pyod.models.vae import VAE
from pyod.models.auto_encoder_torch import AutoEncoder
from pyod.models.so_gaal import SO_GAAL
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.xgbod import XGBOD
from pyod.models.deep_svdd import DeepSVDD


class PYOD():
    def __init__(self, seed, model_name, tune=False):
        '''
        :param seed: seed for reproducible results
        :param model_name: model name
        :param tune: if necessary, tune the hyper-parameter based on the validation set constructed by the labeled anomalies
        '''
        self.seed = seed
        self.utils = Utils()

        self.model_name = model_name
        self.model_dict = {'IForest':IForest, 'OCSVM':OCSVM, 'ABOD':ABOD, 'CBLOF':CBLOF, 'COF':COF, 'AOM':aom,
                           'COPOD':COPOD, 'ECOD':ECOD,  'FeatureBagging':FeatureBagging, 'HBOS':HBOS, 'KNN':KNN,
                           'LMDD':LMDD, 'LODA':LODA, 'LOF':LOF, 'LOCI':LOCI, 'LSCP':LSCP, 'MAD':MAD,
                           'MCD':MCD, 'PCA':PCA, 'ROD':ROD, 'SOD':SOD, 'SOS':SOS, 'VAE':VAE, 'DeepSVDD': DeepSVDD,
                           'AutoEncoder': AutoEncoder, 'SOGAAL': SO_GAAL, 'MOGAAL': MO_GAAL,'XGBOD': XGBOD}

        self.tune = tune

    def grid_hp(self, model_name):
        '''
        define the hyper-parameter search grid for different unsupervised mdoel
        '''

        param_grid_dict = {'IForest': [10, 50, 100, 500], # n_estimators, default=100
                           'OCSVM': ['linear', 'poly', 'rbf', 'sigmoid'], # kernel, default='rbf',
                           'ABOD': [3, 5, 7, 9], # n_neighbors, default=5
                           'CBLOF': [4, 6, 8, 10], # n_clusters, default=8
                           'COF': [5, 10, 20, 50], # n_neighbors, default=20
                           'AOM': None,
                           'COPOD': None,
                           'ECOD': None,
                           'FeatureBagging': [3, 5, 10, 20], # n_estimators, default=10
                           'HBOS': [3, 5, 10, 20], # n_bins, default=10
                           'KNN': [3, 5, 10, 20], # n_neighbors, default=5
                           'LMDD': ['aad', 'var', 'iqr'], # dis_measure, default='aad'
                           'LODA': [3, 5, 10, 20], # n_bins, default=10
                           'LOF': [5, 10, 20, 50], # n_neighbors, default=20
                           'LOCI': [0.1, 0.25, 0.5, 0.75], # alpha, default=0.5
                           'LSCP': [3, 5, 10, 20], # n_bins, default=10
                           'MAD': None,
                           'MCD': None,
                           'PCA': [0.25, 0.5, 0.75, None], # n_components
                           'ROD': None,
                           'SOD': [5, 10, 20, 50], # n_neighbors, default=20
                           'SOS': [2.5, 4.5, 7.5, 10.0], # perplexity, default=4.5
                           'VAE': None,
                           'AutoEncoder': None,
                           'SOGAAL': [10, 20, 50, 100], # stop_epochs, default=20
                           'MOGAAL': [10, 20, 50, 100], # stop_epochs, default=20
                           'XGBOD': None,
                           'DeepSVDD': [20, 50, 100, 200] # epochs, default=100
                           }

        return param_grid_dict[model_name]

    def grid_search(self, X_train, y_train, ratio=None):
        '''
        implement the grid search for unsupervised models and return the best hyper-parameters
        the ratio could be the ground truth anomaly ratio of input dataset
        '''

        # set seed
        self.utils.set_seed(self.seed)
        # get the hyper-parameter grid
        param_grid = self.grid_hp(self.model_name)

        if param_grid is not None:
            # index of normal ana abnormal samples
            idx_a = np.where(y_train==1)[0]
            idx_n = np.where(y_train==0)[0]
            idx_n = np.random.choice(idx_n, int((len(idx_a) * (1-ratio)) / ratio), replace=True)

            idx = np.append(idx_n, idx_a) #combine
            np.random.shuffle(idx) #shuffle

            # valiation set (and the same anomaly ratio as in the original dataset)
            X_val = X_train[idx]
            y_val = y_train[idx]

            # fitting
            metric_list = []
            for param in param_grid:
                try:
                    if self.model_name == 'IForest':
                        model = self.model_dict[self.model_name](n_estimators=param).fit(X_train)

                    elif self.model_name == 'OCSVM':
                        model = self.model_dict[self.model_name](kernel=param).fit(X_train)

                    elif self.model_name == 'ABOD':
                        model = self.model_dict[self.model_name](n_neighbors=param).fit(X_train)

                    elif self.model_name == 'CBLOF':
                        model = self.model_dict[self.model_name](n_clusters=param).fit(X_train)

                    elif self.model_name == 'COF':
                        model = self.model_dict[self.model_name](n_neighbors=param).fit(X_train)

                    elif self.model_name == 'FeatureBagging':
                        model = self.model_dict[self.model_name](n_estimators=param).fit(X_train)

                    elif self.model_name == 'HBOS':
                        model = self.model_dict[self.model_name](n_bins=param).fit(X_train)

                    elif self.model_name == 'KNN':
                        model = self.model_dict[self.model_name](n_neighbors=param).fit(X_train)

                    elif self.model_name == 'LMDD':
                        model = self.model_dict[self.model_name](dis_measure=param).fit(X_train)

                    elif self.model_name == 'LODA':
                        model = self.model_dict[self.model_name](n_bins=param).fit(X_train)

                    elif self.model_name == 'LOF':
                        model = self.model_dict[self.model_name](n_neighbors=param).fit(X_train)

                    elif self.model_name == 'LOCI':
                        model = self.model_dict[self.model_name](alpha=param).fit(X_train)

                    elif self.model_name == 'LSCP':
                        model = self.model_dict[self.model_name](detector_list=[LOF(),LOF()], n_bins=param).fit(X_train)

                    elif self.model_name == 'PCA':
                        model = self.model_dict[self.model_name](n_components=param).fit(X_train)

                    elif self.model_name == 'SOD':
                        model = self.model_dict[self.model_name](n_neighbors=param).fit(X_train)

                    elif self.model_name == 'SOS':
                        model = self.model_dict[self.model_name](perplexity=param).fit(X_train)

                    elif self.model_name == 'SOGAAL':
                        model = self.model_dict[self.model_name](stop_epochs=param).fit(X_train)

                    elif self.model_name == 'MOGAAL':
                        model = self.model_dict[self.model_name](stop_epochs=param).fit(X_train)

                    elif self.model_name == 'DeepSVDD':
                        model = self.model_dict[self.model_name](epochs=param).fit(X_train)

                    else:
                        raise NotImplementedError

                except:
                    metric_list.append(0.0)
                    continue

                try:
                    # model performance on the validation set
                    score_val = model.decision_function(X_val)
                    metric = self.utils.metric(y_true=y_val, y_score=score_val, pos_label=1)
                    metric_list.append(metric['aucpr'])

                except:
                    metric_list.append(0.0)
                    continue

            best_param = param_grid[np.argmax(metric_list)]

        else:
            metric_list = None
            best_param = None

        print(f'The candidate hyper-parameter of {self.model_name}: {param_grid},',
              f' corresponding metric: {metric_list}',
              f' the best candidate: {best_param}')

        return best_param

    def fit(self, X_train, y_train, ratio=None):
        if self.model_name in ['AutoEncoder', 'VAE']:
            # only use the normal samples to fit the model
            idx_n = np.where(y_train==0)[0]
            X_train = X_train[idx_n]
            y_train = y_train[idx_n]

        # selecting the best hyper-parameters of unsupervised model for fair comparison (if labeled anomalies is available)
        if sum(y_train) > 0 and self.tune:
            assert ratio is not None
            best_param = self.grid_search(X_train, y_train, ratio)
        else:
            best_param = None

        print(f'best param: {best_param}')

        # set seed
        self.utils.set_seed(self.seed)

        # fit best on the best param
        if best_param is not None:
            if self.model_name == 'IForest':
                self.model = self.model_dict[self.model_name](n_estimators=best_param).fit(X_train)

            elif self.model_name == 'OCSVM':
                self.model = self.model_dict[self.model_name](kernel=best_param).fit(X_train)

            elif self.model_name == 'ABOD':
                self.model = self.model_dict[self.model_name](n_neighbors=best_param).fit(X_train)

            elif self.model_name == 'CBLOF':
                self.model = self.model_dict[self.model_name](n_clusters=best_param).fit(X_train)

            elif self.model_name == 'COF':
                self.model = self.model_dict[self.model_name](n_neighbors=best_param).fit(X_train)

            elif self.model_name == 'FeatureBagging':
                self.model = self.model_dict[self.model_name](n_estimators=best_param).fit(X_train)

            elif self.model_name == 'HBOS':
                self.model = self.model_dict[self.model_name](n_bins=best_param).fit(X_train)

            elif self.model_name == 'KNN':
                self.model = self.model_dict[self.model_name](n_neighbors=best_param).fit(X_train)

            elif self.model_name == 'LMDD':
                self.model = self.model_dict[self.model_name](dis_measure=best_param).fit(X_train)

            elif self.model_name == 'LODA':
                self.model = self.model_dict[self.model_name](n_bins=best_param).fit(X_train)

            elif self.model_name == 'LOF':
                self.model = self.model_dict[self.model_name](n_neighbors=best_param).fit(X_train)

            elif self.model_name == 'LOCI':
                self.model = self.model_dict[self.model_name](alpha=best_param).fit(X_train)

            elif self.model_name == 'LSCP':
                self.model = self.model_dict[self.model_name](detector_list=[LOF(), LOF()], n_bins=best_param).fit(X_train)

            elif self.model_name == 'PCA':
                self.model = self.model_dict[self.model_name](n_components=best_param).fit(X_train)

            elif self.model_name == 'SOD':
                self.model = self.model_dict[self.model_name](n_neighbors=best_param).fit(X_train)

            elif self.model_name == 'SOS':
                self.model = self.model_dict[self.model_name](perplexity=best_param).fit(X_train)

            elif self.model_name == 'SOGAAL':
                self.model = self.model_dict[self.model_name](stop_epochs=best_param).fit(X_train)

            elif self.model_name == 'MOGAAL':
                self.model = self.model_dict[self.model_name](stop_epochs=best_param).fit(X_train)

            elif self.model_name == 'DeepSVDD':
                self.model = self.model_dict[self.model_name](epochs=best_param).fit(X_train)

            else:
                raise NotImplementedError

        else:
            # unsupervised method would ignore the y labels
            self.model = self.model_dict[self.model_name]().fit(X_train, y_train)

        return self

    # from pyod: for consistency, outliers are assigned with larger anomaly scores
    def predict_score(self, X):
        score = self.model.decision_function(X)
        return score