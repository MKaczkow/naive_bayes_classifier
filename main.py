"""
Author: Maciej Kaczkowski
15.04-29.04.2021
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from classifier import NaiveBayesClassifier


def calculate_metrics(y_test, y_pred):
    cnf_mat = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f_score = f1_score(y_test, y_pred, average='macro')
    print('Accuracy: {}\nPrecision: {}\nRecall: {}\nF1_score: {}'.format(
        acc, precision, recall, f_score))
    return cnf_mat


def split_dataset(dataset: pd.DataFrame, train_frac):
    train = dataset.sample(frac=train_frac, random_state=300660)
    test = dataset.drop(train.index)
    return train.drop(columns='class'), test.drop(columns='class'), \
           train['class'], test['class']


# reading clean dataset
main_df = pd.read_csv(r'seeds_dataset_clean.txt', header=None, sep='\t')
main_df.columns = ['area', 'perimeter', 'compactness', 'kernel length',
                    'kernel width', 'asymmetry coef.', 'groove length', 'class']


nbc = NaiveBayesClassifier()


# finding best train/(train+test) ratio
train_fractions = np.linspace(start=0.1, stop=0.9, num=17)
prediction_accuracies = np.zeros((17, 1))

for idx, train_frac in enumerate(train_fractions):
    X_train, X_test, y_train, y_test = split_dataset(main_df, train_frac=train_frac)
    # alternatively sklearn.model_selection.train_test_split can be used
    nbc.fit(X_train, y_train)
    predictions = nbc.predict(X_test)
    prediction_accuracies[idx] = accuracy_score(y_test, predictions)

best_train_fraction = train_fractions[np.argmax(prediction_accuracies)]


# plotting prediction_accuracy(train_fractions)
plt.figure(1)
plt.plot(train_fractions, prediction_accuracies)
plt.title('Finding best train/(train+test) ratio')
plt.xlabel('train_fraction')
plt.ylabel('prediction_accuracy')


# plotting confusion matrix and classification metrics
X_train, X_test, y_train, y_test = split_dataset(main_df, train_frac=best_train_fraction)
nbc.fit(X_train, y_train)
predictions = nbc.predict(X_test)

plt.figure(2)
print("\nNormal dataset metrics: ")
cnf_mat = calculate_metrics(y_test, predictions)
sns.heatmap(cnf_mat, annot=True, fmt='g')
plt.title('Normal dataset confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')


# checking if shuffling data makes any difference
shuffled_df = main_df.sample(frac=1)
X_train, X_test, y_train, y_test = split_dataset(shuffled_df, train_frac=best_train_fraction)
nbc.fit(X_train, y_train)
predictions = nbc.predict(X_test)

plt.figure(3)
print("\nShuffled dataset metrics: ")
cnf_mat = calculate_metrics(y_test, predictions)
sns.heatmap(cnf_mat, annot=True, fmt='g')
plt.title('Shuffled dataset confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')


plt.show()
