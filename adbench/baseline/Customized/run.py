from sklearn.linear_model import LogisticRegression
from adbench.myutils import Utils
from adbench.baseline.Customized.fit import fit
from adbench.baseline.Customized.model import LR

class Customized():
    '''
    You should define the following fit and predict_score function
    Here we use the LogisticRegression as an example
    '''
    def __init__(self, seed:int, model_name:str=None):
        self.seed = seed
        self.utils = Utils()
        self.model_name = model_name

    def fit(self, X_train, y_train):
        # Initialization
        self.model = LR(random_state=self.seed)

        # fitting
        self.model = fit(X_train=X_train, y_train=y_train, model=self.model)

        return self

    def predict_score(self, X):
        score = self.model.predict_proba(X)[:, 1]
        return score