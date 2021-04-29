'''
Author: Maciej Kaczkowski
15.04-xx.04.2021
'''


import pandas as pd
import numpy as np
import scipy.stats


def pdf(x):
    mean = 11.3
    var = 0.1
    numerator = np.exp(- (x - mean) ** 2 / (2 * var))
    denominator = np.sqrt(2 * np.pi * var)
    return numerator / denominator


print(scipy.stats.norm(11.3, 0.01).pdf(11.31))
print(pdf(11.31))

# Dummy DataFrame
df = pd.DataFrame({'col1': [1,2,3], 'col2': [4,5,6]})
# Extract second row (index: 1)
print(df.iloc[1].to_numpy())
