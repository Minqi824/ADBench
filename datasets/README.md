We describe the data sources of 10 CV/NLP datasets as below.
****
For the **CV** datasets, we provide two versions of transformed embedding data
for evaluating tabular-based AD algorithms (codes are available in [colab](https://colab.research.google.com/drive/1tB90CB-BuKDOM3WYV75-WkK6xrMQxQ5M?usp=sharing)):
- [ImageNet-pretrained ResNet-18](https://pytorch.org/hub/pytorch_vision_resnet/) to extract the embedding after the last average pooling layer.
- [ImageNet-pretrained ViT](https://github.com/lukemelas/PyTorch-Pretrained-ViT) to extract the embedding after the last average pooling layer.


For the **NLP** datasets, we provide two versions of transformed embedding data (codes are available in [colab](https://colab.research.google.com/drive/1uMr_5jIqrlP1UL1SlBm7cdO7fmDaEamB?usp=sharing))
- [Pretrained BERT](https://huggingface.co/bert-base-uncased) to extract the embedding of the [CLS] token.
- [Pretrained RoBERTa](https://huggingface.co/roberta-base) to extract the embedding of the [CLS] token.
****
- CIFAR10: https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
- FashionMNIST: https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST
- MNIST-C: https://zenodo.org/record/3239543
- MVTec-AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
- SVHN: https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html#torchvision.datasets.SVHN
****
- Agnews: https://huggingface.co/datasets/ag_news
- Amazon: https://huggingface.co/datasets/amazon_polarity
- Imdb: https://huggingface.co/datasets/imdb
- Yelp: https://huggingface.co/datasets/yelp_review_full
- 20newsgroups: http://qwone.com/~jason/20Newsgroups/

****
Here we demonstrate the MNIST-C and MVTec-AD data embedded the by the ResNet-18 pretrained on the ImageNet dataset.
These transformed (from image/text to tabular) data could be considered as a good baseline for evaluating different AD algorithms.
![MNIST-C](figs/MNIST-C.png)
![MVTec-AD](figs/MVTec-AD.png)