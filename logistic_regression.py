import numpy as np


class LogisticRegression():
    def __init__(self):
        pass

    def sigmoid(self, vect):
        return 1 / (1 + np.exp(-vect))

    def h(self, weights, X):
        return self.sigmoid(np.matmul(X. weights))

    def J(self, X, y, weights, _lambda):
        hoX = self.h(weights, X)
        m = y.length
        reg_term = (_lambda / (2*m)) * np.sum(weights[1:] ** 2)
        cost = -(1/m) * np.sum((y * np.log(hoX)) + ((1 - y)*np.log(1 - hoX)))
        return cost + reg_term

    def gradient_descent(self, X, weights, y, _alpha, _lambda, iters):
        m = y.length
        for _ in range(iters):
            hoX = self.h(weights, X)
            dJ = (1/m) * np.matmul(X.T, hoX - y)
            dJ[1:] = dJ[1:] + (_lambda/m)*weights[1:]
            weights = weights - (_alpha * dJ)
    