'''
Author: Maciej Kaczkowski
15.04-xx.04.2021
'''


import numpy as np
import pandas as pd


class NaiveBayesClassifier:

    def fit(self, X, y):
        #TODO: cleanup
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.features = X.columns
        n_classes = len(self.classes)
        self.num_classes = n_classes
        self.num_features = n_features
        # print(n_classes)
        # print(self.classes)
        # print(n_features)
        # print(n_samples)

        # init mean, var, priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        # print(self.mean.size)
        # print(self.var.size)
        # print(self.priors.size)

        for c in self.classes:
            X_c = X[c == y]
            # print(X_c)
            # print(type(X_c))
            self._mean[c-1, :] = X_c.mean(axis=0)
            self._var[c-1, :] = X_c.var(axis=0)
            self._priors[c-1] = X_c.shape[0] / float(n_samples)

        # print('mean:\n')
        # print(self._mean)
        # print('var:\n')
        # print(self._var)
        # print('priors:\n')
        # print(self._priors)

    def predict(self, X):
        X_matrix = X.to_numpy()
        y_pred = [self.predict_single_item(x) for x in X_matrix]
        # print(y_pred)
        return y_pred

    def predict_single_item(self, x):
        posteriors = []

        for cls_idx, c in enumerate(self.classes):
            prior = np.log(self._priors[cls_idx])
            class_conditional = np.sum(np.log(self._pdf(cls_idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    #TODO: fix using notes
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        # print('var, mean, x:')
        # print(var, mean, x)
        numerator = np.exp(- (x-mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

