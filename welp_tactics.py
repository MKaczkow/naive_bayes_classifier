'''
Author: Maciej Kaczkowski
15.04-xx.04.2021
'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# reading clean dataset
main_df = pd.read_csv(r'seeds_dataset_clean.txt', header=None, sep='\t')
main_df.columns = ['area', 'perimeter', 'compactness', 'kernel length',
                    'kernel width', 'asymmetry coef.', 'groove length', 'class']


plt.figure(1)
sns.scatterplot(data=main_df, x='asymmetry coef.', y='compactness', hue='class')

plt.figure(2)
sns.scatterplot(data=main_df, x='asymmetry coef.', y='groove length', hue='class')

plt.figure(3)
sns.scatterplot(data=main_df, x='asymmetry coef.', y='kernel length', hue='class')

plt.figure(4)
sns.scatterplot(data=main_df, x='asymmetry coef.', y='kernel width', hue='class')

plt.figure(5)
sns.scatterplot(data=main_df, x='asymmetry coef.', y='perimeter', hue='class')

plt.figure(6)
sns.scatterplot(data=main_df, x='asymmetry coef.', y='area', hue='class')

plt.figure(7)
sns.scatterplot(data=main_df, y='compactness.', x='groove length', hue='class')

plt.figure(8)
sns.scatterplot(data=main_df, y='compactness', x='kernel width', hue='class')

plt.figure(9)
sns.scatterplot(data=main_df, y='compactness', x='kernel length', hue='class')

plt.figure(9)
sns.scatterplot(data=main_df, y='compactness', x='area', hue='class')

plt.figure(10)
sns.scatterplot(data=main_df, y='compactness', x='perimeter', hue='class')

plt.figure(11)
sns.scatterplot(data=main_df, y='groove length', x='kernel width', hue='class')

plt.figure(12)
sns.scatterplot(data=main_df, y='groove length', x='kernel length', hue='class')

plt.figure(13)
sns.scatterplot(data=main_df, x='groove length', y='area', hue='class')

plt.figure(14)
sns.scatterplot(data=main_df, x='groove length', y='perimeter', hue='class')

plt.figure(15)
sns.scatterplot(data=main_df, x='kernel length', y='kernel width', hue='class')

plt.figure(16)
sns.scatterplot(data=main_df, x='kernel length', y='area', hue='class')

plt.figure(17)
sns.scatterplot(data=main_df, x='kernel length', y='perimeter', hue='class')

plt.figure(18)
sns.scatterplot(data=main_df, x='area', y='perimeter', hue='class')


plt.show()
