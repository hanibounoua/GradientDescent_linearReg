import numpy as np

class linearRegression:

    def __init__(self, learningRate, n_iter):
        self.learningRate = learningRate
        self.n_iter = n_iter

        self.omega = None
        self.b = None

    def fit(self, X, y):
        if len(X.shape) == 1:
            n_sample, n_features = X[..., np.newaxis].shape
        else:
            n_sample, n_features = X.shape
        self.omega = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iter):
            for ind, x_i in enumerate(X):
                self.omega += (2/n_sample) * self.learningRate * x_i * (y[ind] - np.dot(self.omega, x_i) - self.b)
                self.b += (2/n_sample) * self.learningRate * (y[ind] - np.dot(self.omega, x_i) - self.b)

    def predict(self, X):
        n_sample = X.shape[0]
        y = np.zeros(n_sample)
        for ind, x_i in enumerate(X):
            y[ind] = np.dot(self.omega, x_i) + self.b
        return y
