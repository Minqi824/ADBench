Official implementation of _**ADBench**: Anomaly Detection Benchmark_.
****


### Introduction
ADBench is (to our best knowledge) the most comprehensive tabular anomaly detection benchmark.
By analyzing both research needs and deployment requirements in industry,
ADBench covers the experiments with three major angles in anomaly detection:
(_i_) the availability of supervision (e.g., ground truth labels) 
by including 14 unsupervised, 7 semi-supervised, and 9 supervised methods;
(_ii_) algorithm performance under different types of anomalies by 
simulating the environments with 4 types of anomalies; and
(_iii_) algorithm robustness and stability under 3 settings of data corruptions. 
The Figure below provides an overview of our proposed ADBench.

![ADBench](figs/ADBench.png)

### Dependency
The experiment code is written in Python 3 and built on a number of Python packages:  
- scikit-learn==0.20.3 
- pyod==0.9.8 
- Keras==2.3.0 
- tensorflow==1.15.0 
- torch==1.9.0

Batch installation is possible using the supplied "requirements.txt":
```angular2html
pip install -r requirements.txt
```


### Datasets
ADBench includes 55 existing and freshly proposed datasets, as shown in the following Table.  
Among them, 48 widely-used real-world datasets are gathered for model evaluation, which cover many application domains, 
including healthcare (e.g., disease diagnosis), 
audio and language processing (e.g., speech recognition), 
image processing (e.g., object identification), 
finance (e.g., financial fraud detection), etc.  
**Moreover**, as most of these datasets are relatively small, 
we introduce 7 more complex datasets from CV and NLP domains with more samples and richer features in ADBench.
Pretrained models are applied to extract data embedding from NLP and CV datasets to access more complex representation.
For NLP datasets, we use BERT pretrained on the BookCorpus and English Wikipedia to extract the embedding of the [CLS] token.
For CV datasets, we use ResNet18 pretrained on the ImageNet to extract the embedding after the last average pooling layer.

![Daasets](figs/Datasets.png)

### Algorithms
Compared to the previous benchmark studies, we have a larger algorithm collection with
(_i_) latest unsupervised AD algorithms like DeepSVDD and ECOD;
(_ii_) SOTA semi-supervised algorithms, including DeepSAD and DevNet;
(_iii_) latest network architectures like ResNet in computer vision (CV) and Transformer in natural language processing (NLP) domain
---we adapt ResNet and FTTransformer models for tabular AD in the proposed ADBench; and
(_iv_) ensemble learning methods like LightGBM, XGBoost, and CatBoost.
The Figure below shows the algorithms (14 unsupervised, 7 semi-supervised, and 9 supervised algorithms) in ADBench.
% , where we further elaborate in Appx. \ref{appendix:algorithms}.
![Algorithms](figs/Algorithms.png)

|  Model  | Year | Type |  DL  |       Import        |  Source  |
| :-----: | :--------: | :--: | :--: | :-----------------: | :------: |
| [PCA]() | Before 2017 | Unsupervised |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [OCSVM]() | Before 2017  | Unsupervised |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [LOF]() | Before 2017  | Unsupervised |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [CBLOF]() | Before 2017  | Unsupervised |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [COF]() | Before 2017  | Unsupervised |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [HBOS]() | Before 2017  | Unsupervised |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [KNN]() | Before 2017  | Unsupervised |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [SOD]() | Before 2017  | Unsupervised |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [COPOD](https://arxiv.org/abs/2009.09463) | 2020  | Unsupervised |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [ECOD](https://arxiv.org/abs/2201.00382) | 2022  | Unsupervised |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [IForest*]() | Before 2017  | Unsupervised |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [LODA*]() | Before 2017  | Unsupervised |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [DeepSVDD](http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf) | 2018  | Unsupervised |  &check;   | from pyod.models.deep_svdd import DeepSVDD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [DAGMM](https://openreview.net/forum?id=BJJLHbb0-) | 2018  | Unsupervised |  &check;   | from baseline.DAGMM.run import DAGMM | [Link](https://github.com/mperezcarrasco/PyTorch-DAGMM) |
| [GANomaly](https://arxiv.org/abs/1805.06725) | 2018  | Semi-supervised^ |  &check;   | from baseline.GANomaly.run import GANomaly | [Link]() |
| [XGBOD](https://arxiv.org/abs/1912.00290) | 2018  | Semi-supervised |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [DeepSAD](https://arxiv.org/abs/1906.02694) | 2019  | Semi-supervised |  &check;   | from baseline.DeepSAD.src.run import DeepSAD | [Link](https://github.com/lukasruff/Deep-SAD-PyTorch) |
| [REPEN](https://arxiv.org/abs/1806.04808) | 2018  | Weakly-supervised |  &check;   | from baseline.REPEN.run import REPEN | [Link]() |
| [DevNet](https://arxiv.org/abs/1911.08623) | 2019  | Weakly-supervised |  &check;   | from baseline.DevNet.run import DevNet | [Link](https://github.com/GuansongPang/deviation-network) |
| [PReNet](https://arxiv.org/abs/1910.13601) | 2020  | Weakly-supervised |  &check;   | from baseline.PReNet.run import PReNet | [Link]() |
| [FEAWAD](https://arxiv.org/abs/2105.10500) | 2021  | Weakly-supervised |  &check;   | from baseline.FEAWAD.run import FEAWAD | [Link](https://github.com/yj-zhou/Feature_Encoding_with_AutoEncoders_for_Weakly-supervised_Anomaly_Detection/blob/main/FEAWAD.py) |
| [NB]() | Before 2017  | Supervised |  &cross;   | from baseline.Supervised import supervised | [Link]() |
| [SVM]() | Before 2017  | Supervised |  &cross;   | from baseline.Supervised import supervised | [Link]() |
| [MLP]() | Before 2017  | Supervised |  &check;   | from baseline.Supervised import supervised | [Link]() |
| [RF](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) | Before 2017  | Supervised |  &cross;   | from baseline.Supervised import supervised | [Link]() |
| [LGB](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) | NIPS, 2017  | Supervised |  &cross;   | from baseline.Supervised import supervised | [Link](https://lightgbm.readthedocs.io/en/latest/) |
| [XGB](https://arxiv.org/abs/1603.02754) | Before 2017  | Supervised |  &cross;   | from baseline.Supervised import supervised | [Link](https://xgboost.readthedocs.io/en/stable/) |
| [CatB](https://arxiv.org/pdf/1706.09516.pdf) | 2019  | Supervised |  &cross;   | from baseline.Supervised import supervised | [Link](https://xgboost.readthedocs.io/en/stable/) |
| [ResNet](https://arxiv.org/pdf/2106.11959.pdf) | 2019  | Supervised |  &check;   | from baseline.FTTransformer.run import FTTransformer | [Link](https://yura52.github.io/rtdl/stable/index.html) |
| [FTTransformer](https://arxiv.org/pdf/2106.11959.pdf) | 2019  | Supervised |  &check;   | from baseline.FTTransformer.run import FTTransformer | [Link](https://yura52.github.io/rtdl/stable/index.html) |
- '*' denotes that this model is ensembled.
- '^' denotes that this semi-supervised model only uses normal samples in the training stage.



### Results


## Quickly implement ADBench for benchmarking AD algorithms.


  ### Quickly Implement ADBench for Your Customized Algorithm
    run_customized.ipynb
    
  ### Reproduce the Results in Our Papers
    run.py

  ### Supported Benchmark Algorithms (continuous updating...)
