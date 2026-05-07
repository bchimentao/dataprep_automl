class AutoMLRunner:
    def train(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError