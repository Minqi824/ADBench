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
from baseline.RCCDualGAN.run import RccDualGAN
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
        Only global experimental parameters should be defined in the RunPipeline instantiation

        suffix: saved file suffix (including the saved model performance result and model weights)
        mode: rla or nla —— ratio of labeled anomalies or number of labeled anomalies
        NLP_CV: whether to test different models on the NLP and CV datasets, which are transformed by the pretrained Bert and ResNet18 model

        generate_duplicates: whether to generate duplicated samples when sample size is too small
        n_samples_threshold: threshold for generating the above duplicates

        parallel: dl, pyod or supervise —— choice to parallelly run the model
        realistic_synthetic_mode: local, dependency or global —— whether to use the realistic synthetic dataset to test
        noise type: duplicated_anomalies, irrelevant_features or label_contamination —— whether to test the robustness of different models
        '''

        # my utils function
        self.utils = Utils()

        # global parameters
        self.generate_duplicates = generate_duplicates
        self.n_samples_threshold = n_samples_threshold

        # whether to use the realistic synthetic data instead of real-world datasets?
        self.realistic_synthetic_mode = realistic_synthetic_mode
        # whether to add noise for testing the robustness of baseline models
        self.noise_type = noise_type

        self.parallel = parallel

        # the suffix of all saved files
        if NLP_CV:
            self.suffix = suffix + '_NLP_CV_' + str(realistic_synthetic_mode) + '_' + str(noise_type) + '_' + self.parallel
        else:
            self.suffix = suffix + '_Tabular_' + str(realistic_synthetic_mode) + '_' + str(noise_type) + '_' + self.parallel

        # mode
        self.mode = mode

        # whether to test on the transformed NLP and CV datasets
        self.NLP_CV = NLP_CV

        if self.NLP_CV:
            assert self.realistic_synthetic_mode is None
            assert self.noise_type is None

        # data generator instantiation
        self.data_generator = DataGenerator(generate_duplicates=self.generate_duplicates,
                                            n_samples_threshold=self.n_samples_threshold)

        # ratio of labeled anomalies
        if self.noise_type == 'anomaly_contamination':
            self.rla_list = [0.10]
        elif self.noise_type is not None:
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

        elif self.noise_type == 'anomaly_contamination':
            self.noise_params_list = [0.00, 0.05, 0.10, 0.25, 0.50, 1.00]

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
                               'RCCDualGAN': RccDualGAN,
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

        # 暂时移除MOGAAL和RCC-Dual-GAN, 计算过慢
        # 暂时移除LSCP和MCD模型, 计算过慢
        for _ in ['MOGAAL', 'RCCDualGAN', 'LSCP', 'MCD']:
            if _ in self.model_dict.keys():
                self.model_dict.pop(_)

        # 实际发现某些模型非常慢(且效果不好),已移除: LOCI,LMDD,ROD,SOS
        # 移除ABOD算法: Error: Input contains NaN, infinity or a value too large for dtype('float64').
        # 移除AOM算法: 该算法需要提供各子模型输出的score matrix
        # 移除MAD算法: MAD algorithm is just for univariate data. Got Data with 6 Dimensions
        # 移除DeepSVDD算法: 该算法整合在Pyod中的版本目前是有问题的
        # 移除VAE算法: 该算法在神经元个数大于特征个数时会报错

    # dataset filter for delelting those datasets that do not satisfy the experimental requirement
    def dataset_filter(self):
        # dataset list in the current folder
        dataset_list_org = [os.path.splitext(_)[0] for _ in os.listdir(os.path.join(os.getcwd(), 'datasets'))
                            if os.path.splitext(_)[-1] != '.md']

        # 将不符合标准的数据集筛除
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
                    # nla模式中训练集labeled anomalies个数要超过list中最大的数量
                    if self.mode == 'nla' and sum(data['y_train']) >= self.nla_list[-1]:
                        pass

                    # rla模式中只要训练集labeled anomalies个数超过0即可
                    elif self.mode == 'rla' and sum(data['y_train']) > 0:
                        pass

                    else:
                        add = False

            if add:
                dataset_list.append(dataset)
                dataset_size.append(len(data['y_train']) + len(data['y_test']))
            else:
                print(f"数据集{dataset}被移除")

        # 按照数据集大小进行排序
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

            K.clear_session()  # 实际发现代码会越跑越慢,原因是keras中计算图会叠加,需要定期清除
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


        print(f'共有{len(dataset_list)}个数据集, {len(self.model_dict.keys())}个模型')

        # 记录结果
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

                elif self.noise_type == 'anomaly_contamination':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, contam_ratio=noise_param)

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

                # store and save the result
                df_AUCROC[model_name].iloc[i] = result['aucroc']
                df_AUCPR[model_name].iloc[i] = result['aucpr']
                df_time[model_name].iloc[i] = round(end_time - start_time, 2)

                df_AUCROC.to_csv(os.path.join(os.getcwd(), 'result', 'AUCROC_' + self.suffix + '.csv'), index=True)
                df_AUCPR.to_csv(os.path.join(os.getcwd(), 'result', 'AUCPR_' + self.suffix + '.csv'), index=True)
                df_time.to_csv(os.path.join(os.getcwd(), 'result', 'Time_' + self.suffix + '.csv'), index=True)

# run the experment
pipeline = RunPipeline(suffix='DB_test', parallel='unsupervise', NLP_CV=False,
                       realistic_synthetic_mode=None, noise_type=None)
pipeline.run()