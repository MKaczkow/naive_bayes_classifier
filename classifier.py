'''
Author: Maciej Kaczkowski
15.04-xx.04.2021
'''


import pandas as pd


# reading clean dataset
main_df = pd.read_csv(r'seeds_dataset_clean.txt', header=None, sep='\t')

# reading non-clean dataset (not yet fully functional)
# train_df = pd.read_csv(r'seeds_dataset.txt', header=None, sep='\n', skipinitialspace=True)
# train_df = train_df[0].str.split('\t', expand=True)

# pre-training operations
main_df.columns = ['area', 'perimeter', 'compactness', 'kernel length',
                    'kernel width', 'asymmetry coef.', 'groove length', 'class']

print(main_df.head().to_string())
print(main_df.info())
print(main_df.describe())
