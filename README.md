Official implementation of _**ADBench**: Anomaly Detection Benchmark_. __Please star, watch, and fork ADBench for the active updates!__
****


## Why Do You Need ADBench?

ADBench is (to our best knowledge) **the most comprehensive tabular anomaly detection benchmark**, where we analyze the performance of **30** anomaly detection algorithms on **55** benchmark datasets. By analyzing both research needs and deployment requirements in industry,
ADBench conducts 93,654 experiments with three major angles in anomaly detection:
 1. **the availability of supervision** (e.g., ground truth labels) 
by including 14 unsupervised, 7 semi-supervised, and 9 supervised methods;
 2. **algorithm performance under different types of anomalies** by 
simulating the environments with 4 types of anomalies; and
 3. **algorithm robustness and stability** under 3 settings of data corruptions. 

**Key Takeaways**: Adbench answers many questions for both researchers with interesting findings: 
1. (**!!!**) surprisingly none of the benchmarked unsupervised algorithms is statistically better than others, emphasizing the importance of algorithm selection;
2. (**!!**) with merely 1% labeled anomalies, most semi-supervised methods can outperform the best unsupervised method, justifying the importance of supervision;
3. in controlled environments, we observe that best unsupervised methods for specific types of anomalies are even better than semi- and fully-supervised methods, revealing the necessity of understanding data characteristics;
4. semi-supervised methods show potential in achieving robustness in noisy and corrupted data, possibly due to their efficiency in using labels and feature selection.
5. (**!!!!!**) and many more can be found in our papers (Section 4)

The Figure below provides an overview of our proposed ADBench (see our [paper (to be released)]() for details).

![ADBench](figs/ADBench.png)

----

## How to use ADBench?

We envision three primary usages of ADBench:

- **Have better understanding of anomaly detection algorithms**: please read our [paper (to be released)]() for details
- **Conduct future research on anomaly detection**: we list 4 important future research questions in the paper--see Section 4 to see some thoughts!
- **Access rich algorithm implementation and datasets**: see details below for how to use them
- **Benchmark your anomaly detection algorithms**: see [notebook](https://github.com/Minqi824/ADBench/blob/main/run_customized.ipynb) for instruction.



### Dependency
The experiment code is written in Python 3 and built on a number of Python packages:  
- scikit-learn==0.20.3 
- pyod==0.9.8 
- Keras==2.3.0 (required only for certain deep learning methods)
- tensorflow==1.15.0 (required only for certain deep learning methods)
- torch==1.9.0 (required only for certain deep learning methods)
- rtdl==0.0.13

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

![Dataasets](figs/Datasets.png)

### Algorithms
Compared to the previous benchmark studies, we have a larger algorithm collection with
(_i_) latest unsupervised AD algorithms like DeepSVDD and ECOD;
(_ii_) SOTA semi-supervised algorithms, including DeepSAD and DevNet;
(_iii_) latest network architectures like ResNet in computer vision (CV) and Transformer in natural language processing (NLP) domain
---we adapt ResNet and FTTransformer models for tabular AD in the proposed ADBench; and
(_iv_) ensemble learning methods like LightGBM, XGBoost, and CatBoost.
The Figure below shows the algorithms (14 unsupervised, 7 semi-supervised, and 9 supervised algorithms) in ADBench.
![Algorithms](figs/Algorithms.png)

For each algorithm, we also introduce its specific implementation in the following Table.
|  Model  | Year | Type |  DL  |       Import       |  Source  |
| :-----: | :--------: | :--: | :--: | :-----------------: | :------: |
| [PCA](https://apps.dtic.mil/sti/pdfs/ADA465712.pdf) | Before 2017 | Unsup |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [OCSVM](https://proceedings.neurips.cc/paper/1999/file/8725fb777f25776ffa9076e44fcfd776-Paper.pdf) | Before 2017  | Unsup |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [LOF](https://dl.acm.org/doi/pdf/10.1145/342009.335388) | Before 2017  | Unsup |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [CBLOF](https://www.sciencedirect.com/science/article/abs/pii/S0167865503000035?casa_token=8zegN8osm64AAAAA:mf8lhwsCXHslgL8eYYJUSKJYgSiy42ibf6aMrP-zlaKE5tz_hiy63Olqv_NGAM7Gz21pjCTuMA) | Before 2017  | Unsup |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [COF](https://link.springer.com/chapter/10.1007/3-540-47887-6_53) | Before 2017  | Unsup |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [HBOS](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.401.5686&rep=rep1&type=pdf) | Before 2017  | Unsup |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [KNN](https://dl.acm.org/doi/pdf/10.1145/342009.335437) | Before 2017  | Unsup |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [SOD](https://www.dbs.ifi.lmu.de/~zimek/publications/PAKDD2009/pakdd09-SOD.pdf) | Before 2017  | Unsup |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [COPOD](https://arxiv.org/abs/2009.09463) | 2020  | Unsup |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [ECOD](https://arxiv.org/abs/2201.00382) | 2022  | Unsup |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [IForest†](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest) | Before 2017  | Unsup |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [LODA†](https://link.springer.com/article/10.1007/s10994-015-5521-0) | Before 2017  | Unsup |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [DeepSVDD](http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf) | 2018  | Unsup |  &check;   | from pyod.models.deep_svdd import DeepSVDD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [DAGMM](https://openreview.net/forum?id=BJJLHbb0-) | 2018  | Unsup |  &check;   | from baseline.DAGMM.run import DAGMM | [Link](https://github.com/mperezcarrasco/PyTorch-DAGMM) |
| [GANomaly](https://arxiv.org/abs/1805.06725) | 2018  | Semi |  &check;   | from baseline.GANomaly.run import GANomaly | [Link](https://github.com/samet-akcay/ganomaly) |
| [XGBOD†](https://arxiv.org/abs/1912.00290) | 2018  | Semi |  &cross;   | from baseline.PyOD import PYOD | [Link](https://pyod.readthedocs.io/en/latest/#) |
| [DeepSAD](https://arxiv.org/abs/1906.02694) | 2019  | Semi |  &check;   | from baseline.DeepSAD.src.run import DeepSAD | [Link](https://github.com/lukasruff/Deep-SAD-PyTorch) |
| [REPEN](https://arxiv.org/abs/1806.04808) | 2018  | Semi |  &check;   | from baseline.REPEN.run import REPEN | [Link](https://github.com/GuansongPang/deep-outlier-detection) |
| [DevNet](https://arxiv.org/abs/1911.08623) | 2019  | Semi |  &check;   | from baseline.DevNet.run import DevNet | [Link](https://github.com/GuansongPang/deviation-network) |
| [PReNet](https://arxiv.org/abs/1910.13601) | 2020  | Semi |  &check;   | from baseline.PReNet.run import PReNet | / |
| [FEAWAD](https://arxiv.org/abs/2105.10500) | 2021  | Semi |  &check;   | from baseline.FEAWAD.run import FEAWAD | [Link](https://github.com/yj-zhou/Feature_Encoding_with_AutoEncoders_for_Weakly-supervised_Anomaly_Detection/blob/main/FEAWAD.py) |
| [NB](https://www.cs.unb.ca/~hzhang/publications/FLAIRS04ZhangH.pdf) | Before 2017  | Sup |  &cross;   | from baseline.Supervised import supervised | [Link](https://scikit-learn.org/stable/supervised_learning.html) |
| [SVM](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639) | Before 2017  | Sup |  &cross;   | from baseline.Supervised import supervised | [Link](https://scikit-learn.org/stable/supervised_learning.html) |
| [MLP](https://files.eric.ed.gov/fulltext/ED294889.pdf) | Before 2017  | Sup |  &check;   | from baseline.Supervised import supervised | [Link](https://scikit-learn.org/stable/supervised_learning.html) |
| [RF†](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) | Before 2017  | Sup |  &cross;   | from baseline.Supervised import supervised | [Link](https://scikit-learn.org/stable/supervised_learning.html) |
| [LGB†](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) | NIPS, 2017  | Supervised |  &cross;   | from baseline.Supervised import supervised | [Link](https://lightgbm.readthedocs.io/en/latest/) |
| [XGB†](https://arxiv.org/abs/1603.02754) | Before 2017  | Sup |  &cross;   | from baseline.Supervised import supervised | [Link](https://catboost.ai/en/docs/) |
| [CatB†](https://arxiv.org/pdf/1706.09516.pdf) | 2019  | Sup |  &cross;   | from baseline.Supervised import supervised | [Link](https://xgboost.readthedocs.io/en/stable/) |
| [ResNet](https://arxiv.org/pdf/2106.11959.pdf) | 2019  | Sup |  &check;   | from baseline.FTTransformer.run import FTTransformer | [Link](https://yura52.github.io/rtdl/stable/index.html) |
| [FTTransformer](https://arxiv.org/pdf/2106.11959.pdf) | 2019  | Sup |  &check;   | from baseline.FTTransformer.run import FTTransformer | [Link](https://yura52.github.io/rtdl/stable/index.html) |
- '†' marks ensembling.
- Un-, semi-, and fully-supervised methods are denoted as _unsup_, _semi_ and _sup_, respectively.

### Results in Our Papers
- For complete results of ADBench, please refer to the original paper.
- For reproduce experiment results of ADBench, please run the code in [run.py](run.py).

### Quickly implement ADBench for benchmarking AD algorithms.
We provide an example for quickly implementing ADBench for any customized (AD) algorithms,
as shown in [run_customized.ipynb](run_customized.ipynb).
