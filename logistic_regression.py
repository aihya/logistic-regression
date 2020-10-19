import numpy as np

class LogisticRegression():
    def __init__(self, normalize=False):
        self._normalize = normalize
        self._weights = None

    def normalize(self, X):
        for col_i in range(X.shape[1]):
            col = X[:, col_i]
            col_mean = np.mean(col)
            col_range = max(col) - min(col)
            for row_i, _ in enumerate(X):
                X[row_i][col_i] = (X[row_i][col_i] - col_mean) / (col_range if col_range != 0 else 1)
        return X

    def sigmoid(self, vect):
        return 1 / (1 + np.exp(-vect))

    def h(self, weights, X):
        return self.sigmoid(np.matmul(X, weights))

    def J(self, X, y, weights, _lambda):
        hoX = self.h(weights, X)
        m = len(y)
        reg_term = (_lambda / (2*m)) * np.sum(weights[1:] ** 2)
        cost = -(1/m) * (np.sum((y * np.log(hoX)) + ((1 - y) * np.log(1 - hoX))))
        return cost + reg_term

    def gradient(self, X, weights, y, _alpha, _lambda, iters):
        m = len(y)
        for _ in range(iters):
            hoX = self.h(weights, X)
            dJ = (1/m) * np.matmul(X.T, hoX - y)
            if _lambda != 0:
                dJ[1:] = dJ[1:] + (_lambda/m)*weights[1:]
            weights = weights - (_alpha * dJ)
            print("Iteration: \x1b[33m{}\x1b[0m, Cost \x1b[36m{}\x1b[0m\r".format(_+1, self.J(X, y, weights, _lambda)), end='')
        print("\nFinal Cost {}".format(self.J(X, y, weights, _lambda)))
        print(weights)
        return weights
    
    def fit(self, X, y, lr=0.1, reg_factor=1, iters=100000):
        if (self._normalize):
            X = self.normalize(X)
        X = np.insert(X, 0, 1, axis=1)
        classes = np.unique(y)
        thetas = []
        for c in classes:
            binary = np.where(y == c, 1, 0)
            theta = np.random.rand(X.shape[1])
            thetas.append(self.gradient(X, theta, binary, lr, reg_factor, iters).T)
        self._weights = np.array(thetas)

    def predict(self, X, weights):
        X = np.insert(X, 0, 1, axis=1)
        pred = [np.argmax(p) for p in self.sigmoid(np.matmul(X, weights.T))]
        return pred
