import numpy as np
from baseline.REPEN.model import repen
from myutils import Utils
import os


class REPEN():
    def __init__(self, seed, model_name='REPEN', save_suffix='test',
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
        if not os.path.exists('baseline/REPEN/model'):
            os.makedirs('baseline/REPEN/model')

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