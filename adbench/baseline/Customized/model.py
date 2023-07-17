from sklearn.linear_model import LogisticRegression

class LR(LogisticRegression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)