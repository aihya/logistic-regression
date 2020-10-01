import numpy as np


class LogisticRegression():
    def __init__(self):
        pass

    def sigmoid(self, vect):
        return 1 / (1 + np.power(np.e, -z))

    def hypothesis(self, X, weights):
        return np.matmul(X, weights, dtype=float)

    def cost(self, X, y, weights):
        m = y.shape[0]
        return 1/m * np.sum(self._cost(X, y, weights))

    def _cost(self, X, y, weights):
        _hypothesis = self.hypothesis(X, weights)
        return -y * np.log(_hypothesis) - (1 - y) * np.log(1 - _hypothesis)
