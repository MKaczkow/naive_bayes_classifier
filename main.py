"""
Author: Maciej Kaczkowski
15.04-29.04.2021
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from classifier import NaiveBayesClassifier


def get_accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def split_dataset(dataset: pd.DataFrame, train_frac):
    train = dataset.sample(frac=train_frac, random_state=300660)
    test = dataset.drop(train.index)
    return train.drop(columns='class'), test.drop(columns='class'), \
           train['class'], test['class']


# reading clean dataset
main_df = pd.read_csv(r'seeds_dataset_clean.txt', header=None, sep='\t')
main_df.columns = ['area', 'perimeter', 'compactness', 'kernel length',
                    'kernel width', 'asymmetry coef.', 'groove length', 'class']

#TODO: reading non-fully-clean dataset

# reading non-clean dataset (not yet fully functional)
# train_df = pd.read_csv(r'seeds_dataset.txt', header=None, sep='\n', skipinitialspace=True)
# train_df = train_df[0].str.split('\t', expand=True)

nbc = NaiveBayesClassifier()


# finding best train/(train+test) ratio
train_fractions = np.linspace(start=0.1, stop=0.9, num=17)
prediction_accuracies = np.zeros((17, 1))

for idx, train_frac in enumerate(train_fractions):
    X_train, X_test, y_train, y_test = split_dataset(main_df, train_frac=train_frac)
    # alternatively sklearn.model_selection.train_test_split can be used
    nbc.fit(X_train, y_train)
    predictions = nbc.predict(X_test)
    prediction_accuracies[idx] = get_accuracy(y_test, predictions)

best_train_fraction = train_fractions[np.argmax(prediction_accuracies)]
plt.plot(train_fractions, prediction_accuracies)
plt.xlabel('train_fraction')
plt.ylabel('prediction_accuracy')


#TODO: add other metrics (recall, confusion matrix, etc.)


plt.show()
