'''
Author: Maciej Kaczkowski
15.04-xx.04.2021
'''


import numpy as np
import pandas as pd


class NaiveBayesClassifier:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # init mean, var, priors
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for c in self.classes:
            X_c = X[c == y]
            self.mean[c, :] = X_c.mean(axis=0)
            self.var[c, :] = X_c.var(axis=0)

    def predict(self):
        pass

    def predict_one(self):
        pass
