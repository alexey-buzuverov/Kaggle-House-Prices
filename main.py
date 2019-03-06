# Kaggle House Prices

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

# Load data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# Remove 2 houses which are outliers from the train set

train = train.drop([523, 1298])

# Build joint train+test dataframe
test['SalePrice'] = np.nan
all = pd.concat([train, test])
all.to_csv('all.csv')

train['TotalArea'] = train.apply(lambda s: s['GrLivArea'] + s['TotalBsmtSF'], axis=1)
all['TotalArea'] = all.apply(lambda s: s['GrLivArea'] + s['TotalBsmtSF'], axis=1)
# all[['Id','OverallQual','YearBuilt','GrLivArea','TotalBsmtSF','TotalArea','SalePrice']].to_csv('all_cut.csv')

# Basic EDA
# train.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000), c = 'black');
# train.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000), c = 'black');
train.plot.scatter(x='TotalArea', y='SalePrice', ylim=(0,800000), c = 'black');
# sns.distplot(train['SalePrice'], color = 'black');
# sns.boxplot(x='OverallQual', y='SalePrice', data=train);
# sns.boxplot(x='YearBuilt', y='SalePrice', data=train);
# plt.xticks(rotation=90);
# sns.distplot(train['SalePrice'], color = 'black');

# corr_mat = train.corr()
# cols = corr_mat.nlargest(25, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(train[cols].values.T)
# sns.set(font_scale=1)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', cmap='gist_gray_r',
#                  annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()