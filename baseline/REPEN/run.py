import numpy as np
from baseline.REPEN.model import repen
from myutils import Utils


class REPEN():
    '''
    该段代码并非REPEN原来paper的代码,源码好像有问题
    mode: 暂时选的supervised
    known_outliers: 不知道怎么设置,看代码设置了一个比较大的数,根据实验结果来看还比较正常
    '''
    def __init__(self, seed, model_name='REPEN', save_suffix=None,
                 mode:str='supervised', hidden_dim:int=20, batch_size:int=256, nb_batch:int=50, n_epochs:int=30):
        self.utils = Utils()
        self.device = self.utils.get_device()  # get device
        self.seed = seed

        self.MAX_INT = np.iinfo(np.int32).max
        self.MAX_FLOAT = np.finfo(np.float32).max

        # self.sess = tf.Session()
        # K.set_session(self.sess)

        # hyper-parameters
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.nb_batch = nb_batch
        self.n_epochs = n_epochs

        self.save_suffix = save_suffix

    def fit(self, X_train, y_train, ratio=None):
        # initialization the network
        self.utils.set_seed(self.seed)

        # change the model type when no label information is available
        if sum(y_train) == 0:
            self.mode = 'unsupervised'

        # model initialization
        self.model = repen(mode=self.mode, hidden_dim=self.hidden_dim, batch_size=self.batch_size, nb_batch=self.nb_batch,
                           n_epochs=self.n_epochs, known_outliers=1000000, save_suffix=self.save_suffix)


        # fitting
        self.model.fit(X_train, y_train)

        return self

    def predict_score(self, X):
        score = self.model.decision_function(X)
        return score