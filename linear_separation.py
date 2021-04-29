"""
Author: Maciej Kaczkowski
15.04-29.04.2021
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# reading clean dataset
main_df = pd.read_csv(r'seeds_dataset_clean.txt', header=None, sep='\t')
main_df.columns = ['area', 'perimeter', 'compactness', 'kernel length',
                    'kernel width', 'asymmetry coef.', 'groove length', 'class']


# printing all plots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
sns.scatterplot(ax=axes[0, 0], data=main_df, x='asymmetry coef.', y='compactness', hue='class', legend=True)
sns.scatterplot(ax=axes[0, 1], data=main_df, x='asymmetry coef.', y='groove length', hue='class', legend=True)
sns.scatterplot(ax=axes[1, 0], data=main_df, x='asymmetry coef.', y='kernel length', hue='class', legend=True)
sns.scatterplot(ax=axes[1, 1], data=main_df, x='asymmetry coef.', y='kernel width', hue='class', legend=True)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
sns.scatterplot(ax=axes[0, 0], data=main_df, x='asymmetry coef.', y='perimeter', hue='class', legend=True)
sns.scatterplot(ax=axes[0, 1], data=main_df, x='asymmetry coef.', y='area', hue='class', legend=True)
sns.scatterplot(ax=axes[1, 0], data=main_df, x='compactness', y='groove length', hue='class', legend=True)
sns.scatterplot(ax=axes[1, 1], data=main_df, x='compactness', y='kernel width',  hue='class', legend=True)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
sns.scatterplot(ax=axes[0, 0], data=main_df, y='compactness', x='kernel length', hue='class', legend=True)
sns.scatterplot(ax=axes[0, 1], data=main_df, y='compactness', x='area', hue='class', legend=True)
sns.scatterplot(ax=axes[1, 0], data=main_df, y='compactness', x='perimeter', hue='class', legend=True)
axes[1, 1].set_visible(False)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
sns.scatterplot(ax=axes[0, 0], data=main_df, x='groove length', y='kernel width',  hue='class', legend=True)
sns.scatterplot(ax=axes[0, 1], data=main_df, x='groove length', y='kernel length',  hue='class', legend=True)
sns.scatterplot(ax=axes[1, 0], data=main_df, x='groove length', y='area', hue='class', legend=True)
sns.scatterplot(ax=axes[1, 1], data=main_df, x='groove length', y='perimeter', hue='class', legend=True)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
sns.scatterplot(ax=axes[0, 0], data=main_df, x='kernel length', y='kernel width', hue='class', legend=True)
sns.scatterplot(ax=axes[0, 1], data=main_df, x='kernel length', y='area', hue='class', legend=True)
sns.scatterplot(ax=axes[1, 0], data=main_df, x='kernel length', y='perimeter', hue='class', legend=True)
axes[1, 1].set_visible(False)


fig, axes = plt.subplots(2, 2, figsize=(10, 10))
sns.scatterplot(ax=axes[0, 0], data=main_df, x='kernel width', y='perimeter', hue='class', legend=True)
sns.scatterplot(ax=axes[0, 1], data=main_df, x='kernel width', y='area', hue='class', legend=True)
sns.scatterplot(ax=axes[1, 0], data=main_df, x='area', y='perimeter', hue='class', legend=True)
axes[1, 1].set_visible(False)


# printing only the most promising plots
plt.figure(7)
sns.scatterplot(data=main_df, x='groove length', y='kernel length',  hue='class', legend=True)

plt.figure(8)
sns.scatterplot(data=main_df, x='perimeter', y='area', hue='class', legend=True)

plt.figure(9)
sns.scatterplot(data=main_df, x='groove length', y='area', hue='class', legend=True)


plt.show()
