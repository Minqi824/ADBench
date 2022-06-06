# from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
# from .fmnist_LeNet import FashionMNIST_LeNet, FashionMNIST_LeNet_Autoencoder
# from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .mlp import MLP, MLP_Autoencoder
from .vae import VariationalAutoencoder
from .dgm import DeepGenerativeModel, StackedDeepGenerativeModel


#注意此处与源码有不同
#源码是不同数据集有不同的网络结构(which is weird)
#注意bias必须要设为0,否则DeepSAD可能出现mode collapse(原论文中也提及)
def build_network(net_name, input_size ,ae_net=None):
    """Builds the neural network."""
    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    elif net_name == 'fmnist_LeNet':
        net = FashionMNIST_LeNet()

    elif net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    else:
        net = MLP(x_dim=input_size, h_dims=[100, 20], rep_dim=10, bias=False)

    return net

def build_autoencoder(net_name, input_size):
    """Builds the corresponding autoencoder network."""
    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    elif net_name == 'fmnist_LeNet':
        ae_net = FashionMNIST_LeNet_Autoencoder()

    elif net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    else:
        ae_net = MLP_Autoencoder(x_dim=input_size, h_dims=[100, 20], rep_dim=10, bias=False)

    return ae_net
