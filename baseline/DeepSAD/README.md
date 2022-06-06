# Deep SAD: A Method for Deep Semi-Supervised Anomaly Detection
This repository provides a [PyTorch](https://pytorch.org/) implementation of the *Deep SAD* method presented in our ICLR 2020 paper ”Deep Semi-Supervised Anomaly Detection”.


## Citation and Contact
You find a PDF of the Deep Semi-Supervised Anomaly Detection ICLR 2020 paper on arXiv 
[https://arxiv.org/abs/1906.02694](https://arxiv.org/abs/1906.02694).

If you find our work useful, please also cite the paper:
```
@InProceedings{ruff2020deep,
  title     = {Deep Semi-Supervised Anomaly Detection},
  author    = {Ruff, Lukas and Vandermeulen, Robert A. and G{\"o}rnitz, Nico and Binder, Alexander and M{\"u}ller, Emmanuel and M{\"u}ller, Klaus-Robert and Kloft, Marius},
  booktitle = {International Conference on Learning Representations},
  year      = {2020},
  url       = {https://openreview.net/forum?id=HkgH0TEYwH}
}
```

If you would like get in touch, just drop us an email to [contact@lukasruff.com](mailto:contact@lukasruff.com).


## Abstract
> > Deep approaches to anomaly detection have recently shown promising results over shallow methods on large and complex datasets. Typically anomaly detection is treated as an unsupervised learning problem. In practice however, one may have---in addition to a large set of unlabeled samples---access to a small pool of labeled samples, e.g. a subset verified by some domain expert as being normal or anomalous. Semi-supervised approaches to anomaly detection aim to utilize such labeled samples, but most proposed methods are limited to merely including labeled normal samples. Only a few methods take advantage of labeled anomalies, with existing deep approaches being domain-specific. In this work we present Deep SAD, an end-to-end deep methodology for general semi-supervised anomaly detection. We further introduce an information-theoretic framework for deep anomaly detection based on the idea that the entropy of the latent distribution for normal data should be lower than the entropy of the anomalous distribution, which can serve as a theoretical interpretation for our method. In extensive experiments on MNIST, Fashion-MNIST, and CIFAR-10, along with other anomaly detection benchmark datasets, we demonstrate that our method is on par or outperforms shallow, hybrid, and deep competitors, yielding appreciable performance improvements even when provided with only little labeled data.

## The need for semi-supervised anomaly detection

![fig1](imgs/fig1.png?raw=true "fig1")


## Installation
This code is written in `Python 3.7` and requires the packages listed in `requirements.txt`.

Clone the repository to your machine and directory of choice:
```
git clone https://github.com/lukasruff/Deep-SAD-PyTorch.git
```

To run the code, we recommend setting up a virtual environment, e.g. using `virtualenv` or `conda`:

### `virtualenv`
```
# pip install virtualenv
cd <path-to-Deep-SAD-PyTorch-directory>
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### `conda`
```
cd <path-to-Deep-SAD-PyTorch-directory>
conda create --name myenv
source activate myenv
while read requirement; do conda install -n myenv --yes $requirement; done < requirements.txt
```


## Running experiments
We have implemented the [`MNIST`](http://yann.lecun.com/exdb/mnist/), 
[`Fashion-MNIST`](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/), and 
[`CIFAR-10`](https://www.cs.toronto.edu/~kriz/cifar.html) datasets as well as the classic anomaly detection
benchmark datasets `arrhythmia`, `cardio`, `satellite`, `satimage-2`, `shuttle`, and `thyroid` from the 
Outlier Detection DataSets (ODDS) repository ([http://odds.cs.stonybrook.edu/](http://odds.cs.stonybrook.edu/))
as reported in the paper. 

The implemented network architectures are as reported in the appendix of the paper.

### Deep SAD
You can run Deep SAD experiments using the `main.py` script.    

Here's an example on `MNIST` with `0` considered to be the normal class and having 1% labeled (known) training samples 
from anomaly class `1` with a pollution ratio of 10% of the unlabeled training data (with unknown anomalies from all 
anomaly classes `1`-`9`):
```
cd <path-to-Deep-SAD-PyTorch-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# create folders for experimental output
mkdir log/DeepSAD
mkdir log/DeepSAD/mnist_test

# change to source directory
cd src

# run experiment
python main.py mnist mnist_LeNet ../log/DeepSAD/mnist_test ../data --ratio_known_outlier 0.01 --ratio_pollution 0.1 --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 0 --known_outlier_class 1 --n_known_outlier_classes 1;
```
Have a look into `main.py` for all possible arguments and options.

### Baselines
We also provide an implementation of the following baselines via the respective `baseline_<method_name>.py` scripts:
OC-SVM (`ocsvm`), Isolation Forest (`isoforest`), Kernel Density Estimation (`kde`), kernel Semi-Supervised Anomaly 
Detection (`ssad`), and Semi-Supervised Deep Generative Model (`SemiDGM`).

Here's how to run SSAD for example on the same experimental setup as above:
```
cd <path-to-Deep-SAD-PyTorch-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# create folder for experimental output
mkdir log/ssad
mkdir log/ssad/mnist_test

# change to source directory
cd src

# run experiment
python baseline_ssad.py mnist ../log/ssad/mnist_test ../data --ratio_known_outlier 0.01 --ratio_pollution 0.1 --kernel rbf --kappa 1.0 --normal_class 0 --known_outlier_class 1 --n_known_outlier_classes 1;
```

The autoencoder is provided through Deep SAD pre-training using `--pretrain True` with `main.py`. 
To then run a hybrid approach using one of the classic methods on top of autoencoder features, simply point to the saved
autoencoder model using `--load_ae ../log/DeepSAD/mnist_test/model.tar` and set `--hybrid True`.

To run hybrid SSAD for example on the same experimental setup as above:
```
cd <path-to-Deep-SAD-PyTorch-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# create folder for experimental output
mkdir log/hybrid_ssad
mkdir log/hybrid_ssad/mnist_test

# change to source directory
cd src

# run experiment
python baseline_ssad.py mnist ../log/hybrid_ssad/mnist_test ../data --ratio_known_outlier 0.01 --ratio_pollution 0.1 --kernel rbf --kappa 1.0 --hybrid True --load_ae ../log/DeepSAD/mnist_test/model.tar --normal_class 0 --known_outlier_class 1 --n_known_outlier_classes 1;
```

## License
MIT
