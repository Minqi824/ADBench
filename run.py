import os
import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import time
import gc
from keras import backend as K

from data_generator import DataGenerator
from myutils import Utils

# unsupervised models
from baseline.PyOD import PYOD
from baseline.DAGMM.run import DAGMM
# semi-supervised models
from baseline.GANomaly.run import GANomaly
from baseline.DeepSAD.src.run import DeepSAD
from baseline.REPEN.run import REPEN
from baseline.DevNet.run import DevNet
from baseline.PReNet.run import PReNet
from baseline.FEAWAD.run import FEAWAD
# fully-supervised models
from baseline.Supervised import supervised
from baseline.FTTransformer.run import FTTransformer

class RunPipeline():
    def __init__(self, suffix:str=None, mode:str='rla', parallel:str=None, NLP_CV=False,
                 generate_duplicates=True, n_samples_threshold=1000,
                 realistic_synthetic_mode:str=None,
                 noise_type=None):
        '''
        :param suffix: saved file suffix (including the model performance result and model weights)
        :param mode: rla or nla —— ratio of labeled anomalies or number of labeled anomalies
        :param parallel: unsupervise, semi-supervise or supervise, choosing to parallelly run the code
        :param NLP_CV: whether to test on the NLP and CV datasets, which are transformed by the pretrained Bert and ResNet18 model
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        :param n_samples_threshold: threshold for generating the above duplicates, if generate_duplicates is False, then datasets with sample size smaller than n_samples_threshold will be dropped
        :param realistic_synthetic_mode: local, global, dependency or cluster —— whether to generate the realistic synthetic anomalies to test different algorithms
        :param noise_type: duplicated_anomalies, irrelevant_features or label_contamination —— whether to test the model robustness
        '''

        # utils function
        self.utils = Utils()

        self.mode = mode
        self.parallel = parallel

        # global parameters
        self.generate_duplicates = generate_duplicates
        self.n_samples_threshold = n_samples_threshold

        self.realistic_synthetic_mode = realistic_synthetic_mode
        self.noise_type = noise_type

        # the suffix of all saved files
        if NLP_CV:
            self.suffix = suffix + '_NLP_CV_' + str(realistic_synthetic_mode) + '_' + str(noise_type) + '_' + self.parallel
        else:
            self.suffix = suffix + '_Tabular_' + str(realistic_synthetic_mode) + '_' + str(noise_type) + '_' + self.parallel

        # whether to test on the NLP and CV datasets
        self.NLP_CV = NLP_CV

        if self.NLP_CV:
            assert self.realistic_synthetic_mode is None
            assert self.noise_type is None

        # data generator instantiation
        self.data_generator = DataGenerator(generate_duplicates=self.generate_duplicates,
                                            n_samples_threshold=self.n_samples_threshold)

        # ratio of labeled anomalies
        if self.noise_type is not None:
            self.rla_list = [1.00]
        else:
            self.rla_list = [0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]

        # number of labeled anomalies
        self.nla_list = [0, 1, 5, 10, 25, 50, 75, 100]
        # seed list
        self.seed_list = list(np.arange(3) + 1)

        if self.noise_type is None:
            pass

        elif self.noise_type == 'duplicated_anomalies':
            self.noise_params_list = [1, 2, 3, 4, 5, 6]

        elif self.noise_type == 'irrelevant_features':
            self.noise_params_list = [0.00, 0.01, 0.05, 0.10, 0.25, 0.50]

        elif self.noise_type == 'label_contamination':
            self.noise_params_list = [0.00, 0.01, 0.05, 0.10, 0.25, 0.50]

        else:
            raise NotImplementedError

        # model_dict (model_name: clf)
        self.model_dict = {}

        # unsupervised algorithms
        if self.parallel == 'unsupervise':
            # from pyod
            for _ in ['IForest', 'OCSVM', 'CBLOF', 'COF', 'COPOD', 'ECOD', 'FeatureBagging', 'HBOS', 'KNN', 'LODA',
                      'LOF', 'LSCP', 'MCD', 'PCA', 'SOD', 'SOGAAL', 'MOGAAL']:
                self.model_dict[_] = PYOD

            # DAGMM
            self.model_dict['DAGMM'] = DAGMM

            # # DeepSVDD (if necessary, the DeepSVDD is only for tensorflow 2.0+)
            # for _ in ['DeepSVDD']:
            #     self.model_dict[_] = PYOD

        # semi-supervised algorithms
        elif self.parallel == 'semi-supervise':
            self.model_dict = {'GANomaly': GANomaly,
                               'DeepSAD': DeepSAD,
                               'REPEN': REPEN,
                               'DevNet': DevNet,
                               'PReNet': PReNet,
                               'FEAWAD': FEAWAD,
                               'XGBOD': PYOD}

        # fully-supervised algorithms
        elif self.parallel == 'supervise':
            # from sklearn
            for _ in ['LR', 'NB', 'SVM', 'MLP', 'RF', 'LGB', 'XGB', 'CatB']:
                self.model_dict[_] = supervised
            for _ in ['ResNet', 'FTTransformer']:
                self.model_dict[_] = FTTransformer
        else:
            raise NotImplementedError

        # We remove the following model for considering the computational cost
        for _ in ['MOGAAL', 'LSCP', 'MCD']:
            if _ in self.model_dict.keys():
                self.model_dict.pop(_)

    # dataset filter for delelting those datasets that do not satisfy the experimental requirement
    def dataset_filter(self):
        # dataset list in the current folder
        dataset_list_org = [os.path.splitext(_)[0] for _ in os.listdir(os.path.join(os.getcwd(), 'datasets'))
                            if os.path.splitext(_)[1] != '']

        dataset_list = []
        dataset_size = []

        for dataset in dataset_list_org:
            add = True
            for seed in self.seed_list:
                self.data_generator.seed = seed
                self.data_generator.dataset = dataset
                data = self.data_generator.generator(la=1.00, at_least_one_labeled=True)

                if not self.generate_duplicates and len(data['y_train']) + len(data['y_test']) < self.n_samples_threshold:
                    add = False

                # elif len(data['y_train']) + len(data['y_test']) > 50000:
                #     add = False

                else:
                    if self.mode == 'nla' and sum(data['y_train']) >= self.nla_list[-1]:
                        pass

                    elif self.mode == 'rla' and sum(data['y_train']) > 0:
                        pass

                    else:
                        add = False

            if add:
                dataset_list.append(dataset)
                dataset_size.append(len(data['y_train']) + len(data['y_test']))
            else:
                print(f"remove the dataset {dataset}")

        # sort datasets by their sample size
        dataset_list = [dataset_list[_] for _ in np.argsort(np.array(dataset_size))]

        return dataset_list

    # model fitting function
    def model_fit(self):
        try:
            # model initialization, if model weights are saved, the save_suffix should be specified
            if self.model_name in ['DevNet', 'FEAWAD', 'REPEN']:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name, save_suffix=self.suffix)
            else:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name)

        except Exception as error:
            print(f'Error in model initialization. Model:{self.model_name}, Error: {error}')
            pass

        try:
            # model fitting, currently most of models are implemented to output the anomaly score
            if self.model_name not in ['DeepSAD', 'ResNet', 'FTTransformer']:
                # fitting
                self.clf = self.clf.fit(X_train=self.data['X_train'], y_train=self.data['y_train'],
                                        ratio=sum(self.data['y_test']) / len(self.data['y_test']))
                # predicting score
                if self.model_name == 'DAGMM':
                    score_test = self.clf.predict_score(self.data['X_train'], self.data['X_test'])
                else:
                    score_test = self.clf.predict_score(self.data['X_test'])
                # performance
                result = self.utils.metric(y_true=self.data['y_test'], y_score=score_test, pos_label=1)
            else:
                result = self.clf.fit2test(self.data)

            K.clear_session()
            print(f"Model: {self.model_name}, AUC-ROC: {result['aucroc']}, AUC-PR: {result['aucpr']}")

            del self.clf
            gc.collect()

        except Exception as error:
            print(f'Error in model fitting. Model:{self.model_name}, Error: {error}')
            result = {'aucroc': np.nan, 'aucpr': np.nan}
            pass

        return result

    # run the experiment
    def run(self):
        #  filteting dataset that do not meet the experimental requirements
        if self.NLP_CV:
            dataset_list = [os.path.splitext(_)[0] for _ in os.listdir(os.path.join(os.getcwd(), 'datasets_NLP_CV'))
                            if os.path.splitext(_)[-1] == '.npz']
        else:
            dataset_list = self.dataset_filter()

        # experimental parameters
        if self.mode == 'nla':
            if self.noise_type is not None:
                experiment_params = list(product(dataset_list, self.nla_list, self.noise_params_list, self.seed_list))
            else:
                experiment_params = list(product(dataset_list, self.nla_list, self.seed_list))
        else:
            if self.noise_type is not None:
                experiment_params = list(product(dataset_list, self.rla_list, self.noise_params_list, self.seed_list))
            else:
                experiment_params = list(product(dataset_list, self.rla_list, self.seed_list))


        print(f'{len(dataset_list)} datasets, {len(self.model_dict.keys())} models')

        # save the results
        df_AUCROC = pd.DataFrame(data=None, index=experiment_params, columns=list(self.model_dict.keys()))
        df_AUCPR = pd.DataFrame(data=None, index=experiment_params, columns=list(self.model_dict.keys()))
        df_time = pd.DataFrame(data=None, index=experiment_params, columns=list(self.model_dict.keys()))

        for i, params in tqdm(enumerate(experiment_params)):
            if self.noise_type is not None:
                dataset, la, noise_param, self.seed = params
            else:
                dataset, la, self.seed = params

            if self.parallel == 'unsupervise' and la != 0.0 and self.noise_type is None:
                continue

            if self.NLP_CV and any([_ in dataset for _ in ['agnews', 'FashionMNIST', 'CIFAR10', 'SVHN']]) and self.seed > 1:
                continue

            print(f'Current experiment parameters: {params}')

            # generate data
            self.data_generator.seed = self.seed
            self.data_generator.dataset = dataset

            try:
                if self.noise_type == 'duplicated_anomalies':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, duplicate_times=noise_param)

                elif self.noise_type == 'irrelevant_features':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, noise_ratio=noise_param)

                elif self.noise_type == 'label_contamination':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, noise_ratio=noise_param)

                else:
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode)

            except Exception as error:
                print(f'Error when generating data: {error}')
                pass
                continue

            for model_name in tqdm(self.model_dict.keys()):
                self.model_name = model_name
                self.clf = self.model_dict[self.model_name]

                # fit model
                start_time = time.time() # starting time
                result = self.model_fit()
                end_time = time.time() # ending time

                # store and save the result (AUC-ROC, AUC-PR and runtime)
                df_AUCROC[model_name].iloc[i] = result['aucroc']
                df_AUCPR[model_name].iloc[i] = result['aucpr']
                df_time[model_name].iloc[i] = round(end_time - start_time, 2)

                df_AUCROC.to_csv(os.path.join(os.getcwd(), 'result', 'AUCROC_' + self.suffix + '.csv'), index=True)
                df_AUCPR.to_csv(os.path.join(os.getcwd(), 'result', 'AUCPR_' + self.suffix + '.csv'), index=True)
                df_time.to_csv(os.path.join(os.getcwd(), 'result', 'Time_' + self.suffix + '.csv'), index=True)

# run the above pipeline for reproducing the results in the paper
pipeline = RunPipeline(suffix='DB_test', parallel='unsupervise', NLP_CV=False,
                       realistic_synthetic_mode=None, noise_type=None)
pipeline.run()