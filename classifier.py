"""
Author: Maciej Kaczkowski
15.04-29.04.2021
"""


import numpy as np


class NaiveBayesClassifier:

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.features = X.columns
        n_classes = len(self.classes)
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for class_ in self.classes:
            X_class_ = X[class_ == y]
            self._mean[class_-1, :] = X_class_.mean(axis=0)
            self._var[class_-1, :] = X_class_.var(axis=0)
            self._priors[class_-1] = X_class_.shape[0] / float(n_samples)

    def predict(self, X):
        X_matrix = X.to_numpy()
        y_pred = [self.predict_single_item(x) for x in X_matrix]
        return y_pred

    def predict_single_item(self, x):
        posteriors = []

        for cls_idx, class_ in enumerate(self.classes):
            prior = np.log(self._priors[cls_idx])
            class_conditional = np.sum(np.log(self.get_pdf(cls_idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def get_pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x-mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

