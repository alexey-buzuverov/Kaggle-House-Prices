# Kaggle House Prices

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from fancyimpute import KNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, KFold, cross_val_score, GridSearchCV

# Load data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# Build joint train+test dataframe
test['SalePrice'] = np.nan
train.loc[:,'MSSubClass'] = train['MSSubClass'].astype('object')
test.loc[:,'MSSubClass'] = test['MSSubClass'].astype('object')
all = pd.concat([train, test] , ignore_index = True)

### Define featue groups ###
cont_feat = ['LotArea','GrLivArea','1stFlrSF','2ndFlrSF','LowQualFinSF','TotalBsmtSF',\
            'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','GarageArea','MasVnrArea',
            'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','LotFrontage','MiscVal']

num_feat = ['TotRmsAbvGrd','BedroomAbvGr','KitchenAbvGr','FullBath','HalfBath','BsmtFullBath','BsmtHalfBath',\
            'GarageCars','Fireplaces']

QC_feat = ['OverallQual','OverallCond','ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',\
           'KitchenQual','GarageQual','GarageCond','PoolQC','Fence','FireplaceQu']

type_feat = ['MSSubClass','BldgType','HouseStyle','Functional','LotConfig','LotShape','LandContour','LandSlope',\
             'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',\
             'BsmtFinType1','BsmtFinType2','BsmtExposure','Utilities','Heating','CentralAir','Electrical',\
             'GarageType','GarageFinish','MiscFeature','SaleType','SaleCondition']

neighb_feat = ['MSZoning','Street','Alley','Neighborhood','Condition1','Condition2','PavedDrive']

time_feat = ['YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold']

# To .csv
all.to_csv('csv/all.csv')
all[['Id','SalePrice'] + cont_feat].to_csv('csv/all_cont.csv')
all[['Id','SalePrice'] + num_feat].to_csv('csv/all_num.csv')
all[['Id','SalePrice'] + QC_feat].to_csv('csv/all_QC.csv')
all[['Id','SalePrice'] + type_feat].to_csv('csv/all_type.csv')
all[['Id','SalePrice'] + neighb_feat].to_csv('csv/all_neighb.csv')
all[['Id','SalePrice'] + time_feat].to_csv('csv/all_time.csv')

##### Feature cleaning #####

### Filter columns containing NAs ###
counts_na = all.isnull().sum()[all.isnull().sum() > 0].sort_values(ascending=False)
all[np.append('Id',counts_na.index.values)].to_csv('csv/all_na.csv')

# PoolQC
all[all['PoolQC'].isnull()].to_csv('csv/all_pool_na.csv')
all.loc[all['PoolArea'] == 0, 'PoolQC'] = 'NA'
# 2600, 2504, 2421 missing discrete

# MiscFeature
all[all['MiscFeature'].isnull()].to_csv('csv/all_misc_na.csv')
all.loc[all['MiscVal'] == 0, 'MiscFeature'] = 'NA'

# Alley
all[all['Alley'].isnull()].to_csv('csv/all_alley_na.csv')
all.loc[all['Alley'].isnull(), 'Alley'] = 'NA'

# Fence
all[all['Fence'].isnull()].to_csv('csv/all_fence_na.csv')
all.loc[all['Fence'].isnull(), 'Fence'] = 'NA'

# FireplaceQu
all[all['FireplaceQu'].isnull()].to_csv('csv/all_fireplace_na.csv')
all.loc[all['FireplaceQu'].isnull(), 'FireplaceQu'] = 'NA'

# LotFrontage
all[all['LotFrontage'].isnull()].to_csv('csv/all_lotfr_na.csv')
# 486 NAs continuous

# Garage
all[all['GarageYrBlt'].isnull() | all['GarageFinish'].isnull() | all['GarageQual'].isnull()
     | all['GarageCond'].isnull() | all['GarageType'].isnull()].to_csv('csv/all_garage_na.csv')

def get_features_garage(row):
    if pd.isnull(row['GarageType']):
         features = pd.Series(['NA', row['YearBuilt'], 'NA', 'NA', 'NA'],
                              index=['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond'])
    else:
         features = row[['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']]
         features['GarageYrBlt'] = row['YearBuilt']
    return features

# Perform corrections
all.loc[all['Id'] == 2127, 'GarageYrBlt'] = all.loc[all['Id'] == 2127, 'YearRemodAdd']
all.loc[all['Id'] == 2577, ['GarageType', 'GarageCars', 'GarageArea']] = [np.nan, 0, 0]
all[['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']] = \
     all.apply(lambda s: get_features_garage(s), axis=1)
# 2127 discrete

# Basement
all[all['BsmtQual'].isnull() | all['BsmtCond'].isnull() | all['BsmtFinType1'].isnull()
     | all['BsmtFinType2'].isnull() | all['BsmtExposure'].isnull()].to_csv('csv/all_basement_na.csv')
all.loc[all['Id'] == 2121, ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF'] ] = [0, 0, 0, 0]
all.loc[all['Id'].isin([2121, 2189]), ['BsmtFullBath', 'BsmtHalfBath']] = [0, 0]
all.loc[all['TotalBsmtSF'] == 0, ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure']] = \
    ['NA', 'NA', 'NA', 'NA', 'NA']
# 8 NAs discrete

# Masonry veneer
all[all['MasVnrArea'].isnull() | all['MasVnrType'].isnull()].to_csv('csv/all_masvnr_na.csv')
all.loc[all['MasVnrArea'].isnull(), ['MasVnrArea', 'MasVnrType']] = [0, 'NA']
all.loc[all['MasVnrType'] == 'None', 'MasVnrType'] = 'NA'
# 2611 discrete

# Rest of variables
all[all['MSZoning'].isnull() | all['Functional'].isnull() | all['Utilities'].isnull() | \
    all['SaleType'].isnull() | all['KitchenQual'].isnull() | all['Exterior1st'].isnull() | \
    all['Exterior2nd'].isnull() | all['Electrical'].isnull()].to_csv('csv/all_rest_na.csv')
# 10 NAs discrete

### Find features with very few different values in train set ###
train = all.loc[all['SalePrice'].notnull()]
test = all.loc[all['SalePrice'].isnull()]
desc_train = train.apply(lambda s: s.value_counts().iloc[0], axis = 0).sort_values(ascending=False)
desc_test = test.drop('SalePrice', axis = 1).apply(lambda s: s.value_counts().iloc[0], axis = 0).\
    sort_values(ascending=False)
# Utilities, Street, PoolArea, PoolQC

### Find features values with very few occurencies in train set ###
asc_train = train[type_feat + neighb_feat].apply(lambda s: s.value_counts().iloc[-1], axis = 0)\
    .sort_values(ascending=True)
asc_test = test[type_feat + neighb_feat].apply(lambda s: s.value_counts().iloc[-1], axis = 0).\
    sort_values(ascending=True)

# Less than 5 counts #
# Functional: Sev;
# Utilities: NoSeWa;
# Electrical: Mix, FuseP;
# Heating: Floor, OthW, Wall;
# RoofMatl: Roll, Metal, Membran
# RoofStype: Shed
# Exterior1st: AsphShn, CBlock, ImStucc, Stone, BrkComm;
# Exterior2nd: CBlock, Other, AsphShn
# MiscFeature: TenC, Othr, Gar2
# Condition1: RRNe
# Condition2: PosN, RRAe, RRAn, PosA, Artery, RRNn
# SaleType: Con, Oth, CWD
# SaleCondition: AdjLand
# Foundation: Wood
# MSSubClass: 40
# LotConfig: FR3
# Neighborhood: Blueste

# Between 5 and 9 counts #
# Functional: Maj2;
# Heating: Grav
# RoofMatl: WdShngl, WdShake
# RoofStyle: Mansard
# Exterior2nd: Brk Cmn
# Condition1: RRNn, PosA
# Condition2: Feedr
# SaleType: ConLI, ConLw, ConLD
# Foundation: Stone
# Neighborhood: NPkVill
# Street
# GarageType: 2Types, CarPort
# HouseStyle: 2.5Fin
# LotShape: IR3

### Find feature values which are not present in both train and test sets ###
train_dummy = pd.get_dummies(train)
test_dummy = pd.get_dummies(test)
train_vs_test = set( train_dummy.columns ) - set( test_dummy.columns )
test_vs_train = set( test_dummy.columns ) - set( train_dummy.columns )

# {'Exterior1st_Stone', 'RoofMatl_Metal', 'MiscFeature_TenC',
# 'RoofMatl_Roll', 'Heating_Floor', 'Exterior2nd_Other', 'PoolQC_Fa',
# 'Utilities_NoSeWa', 'Condition2_RRAe', 'Condition2_RRAn', 'Heating_OthW',
# 'GarageQual_Ex', 'Electrical_Mix', 'Exterior1st_ImStucc', 'RoofMatl_Membran',
# 'Condition2_RRNn', 'HouseStyle_2.5Fin'}
# {'MSSubClass_150'}

##### Feature Encoding #####

# Substitution: Quality
qual_dict = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
qual_list = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',
           'KitchenQual','GarageQual','GarageCond','PoolQC','FireplaceQu']
all.loc[:, qual_list] = all[qual_list].applymap(lambda s: qual_dict[s] if pd.notnull(s) else s)

# Substitution: the rest
catord_dict = {
    'Fence': {'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},
    'Functional': {'Typ': 0, 'Min1': 1, 'Min2': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6, 'Sal': 7},
    'LotShape': {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3},
    'LandSlope': {'Gtl': 0, 'Mod': 1, 'Sev': 2},
    'MasVnrType': {'NA': 0, 'BrkCmn': 0, 'BrkFace': 1, 'Stone': 2},
    'BsmtExposure': {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'BsmtFinType1': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'BsmtFinType2': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'CentralAir': {'N': 0, 'Y': 1},
    'GarageFinish': {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
    'Street': {'Grvl': 0, 'Pave': 1},
    'Alley': {'NA': 0, 'Grvl': 1, 'Pave': 2},
    'PavedDrive': {'N': 0, 'P': 1, 'Y': 2}
    }
catord_feat = ['Fence','Functional','LotShape','LandSlope','MasVnrType','BsmtExposure','BsmtFinType1',
              'BsmtFinType2','CentralAir','GarageFinish','Street','Alley','PavedDrive']
for feat in catord_feat:
    all.loc[:, feat] = all[feat].apply(lambda s: catord_dict[feat][s] if pd.notnull(s) else s)

### Missing value imputation ###

# Categorical using mode
catna_feat = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'Electrical', 'SaleType']
for feat in catna_feat:
    all.loc[all[feat].isnull(), feat] = all[feat].mode().iloc[0]
all.loc[all['MiscFeature'].isnull(), 'MiscFeature'] = 'Gar2'

# Numerical using KNN
na_feat = ['PoolQC', 'LotFrontage', 'GarageQual', 'GarageCond', 'GarageFinish',
           'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType2', 'MasVnrType', 'KitchenQual', 'Functional']

knn_feat = cont_feat + num_feat + QC_feat + catord_feat
all_knn = all.loc[:, knn_feat]
scaler = StandardScaler()
scaler.fit(all_knn.fillna(value = 0))
all_knn.loc[:,na_feat] = all_knn[na_feat].fillna(value = -1)
all_knn.loc[:, :] = scaler.transform(all_knn)
all_knn.loc[:,na_feat] = all_knn[na_feat].apply(lambda s: s.apply(lambda e: np.nan if e == s.min() else e), axis = 0)
all_knn_i = pd.DataFrame(data=KNN(k=3, verbose = False).fit_transform(all_knn),
                            columns=all_knn.columns, index=all_knn.index)
all.loc[:, knn_feat] = scaler.inverse_transform(all_knn_i).round(0)

# Cross-check
# for col in all.columns:
#    if all[col].isnull().any(): print(col)

### Outlier removal ###
all.drop(all[all['Id'].isin([524, 1299])].index, inplace = True) # Outlier removal

### Feature engineering ###

# Age: ignoring negative (sold before finished) ages
all['Age'] = all.apply(lambda s: max(s['YrSold'] - s['YearBuilt'], 0), axis = 1)
all['AgeRemod'] = all.apply(lambda s: max(s['YrSold'] - s['YearRemodAdd'], 0), axis = 1)
all['AgeGarage'] = all.apply(lambda s: max(s['YrSold'] - s['GarageYrBlt'], 0), axis = 1)
all['Remodeled'] = all.apply(lambda s: 1 if s['YearRemodAdd'] - s['YearBuilt'] > 1 else 0, axis = 1)

# Area and basement
all['TotLivArea'] = all.apply(lambda s: s['GrLivArea'] + s['TotalBsmtSF'], axis=1)

# Number of rooms
all['TotBathAbvGrd'] = all.apply(lambda s: s['FullBath'] + 0.5*s['HalfBath'], axis=1)
all['TotBath'] = all.apply(lambda s: s['TotBathAbvGrd'] + s['BsmtFullBath'] + 0.5*s['BsmtHalfBath'], axis=1)

### Feature removal ###
all = all.drop(['Utilities'], axis = 1)

### Dummy Encoding and traget variable trainsformation ###
all = pd.get_dummies(all)
all.loc[all['SalePrice'].notnull(),'SalePrice'] = \
    all.loc[all['SalePrice'].notnull()].apply(lambda s: np.log(s['SalePrice']), axis=1)
all.to_csv('all_check.csv')

### Prepare for modeling ###
X_train = all.loc[all['SalePrice'].notnull()].drop(['Id','SalePrice'], axis=1)
y_train = all.loc[all['SalePrice'].notnull(),'SalePrice']
X_test = all.loc[all['SalePrice'].isnull()].drop(['Id','SalePrice'], axis=1)
test_Id = all.loc[all['SalePrice'].isnull(),'Id']

### Estimation: random forest ###

# Cross-validation
# cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
# rf_grid = {'n_estimators': [100],
#            'max_depth': [4,8,12],
#            'min_samples_split': [4,8,12],
#            'min_samples_leaf': [4,8,12]}
# rf_search = GridSearchCV(estimator = RandomForestRegressor(), param_grid = rf_grid,
#               cv = cv, refit=True, n_jobs=1)
# rf_search.fit(X_train, y_train)
# rf_best = rf_search.best_estimator_
# print("Accuracy CV: {}, std: {}, with params {}"
#        .format(np.sqrt(rf_search.best_score_), rf_search.cv_results_['std_test_score'][rf_search.best_index_],
#                rf_search.best_params_))

# Single run: Random Forest
# RF_reg = RandomForestRegressor( n_estimators = 1000,
#                             max_depth = 20,
#                             min_samples_split = 8,
#                             min_samples_leaf = 4 )
# RF_reg.fit(X_train, y_train)

# Single run: XGB
XGBr = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=7200,
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)
XGBr.fit(X_train, y_train)

# Test set accuracy
alg_best = XGBr
y_hat_train = alg_best.predict(X_train)
print('Accuracy Train: {}'
        .format(np.sqrt( metrics.mean_squared_error(y_train, y_hat_train)) ))

### Submission ###
y_hat_test = np.exp(alg_best.predict(X_test))
submission = pd.DataFrame({
    'Id': test_Id,
    'SalePrice': y_hat_test
})
submission.to_csv('submission.csv', index=False)

### Helper functions ###

# for i, row in all.iterrows():
#     if row['GarageYrBlt'] < row['YearRemodAdd']: print(row['Id'], row['GarageYrBlt'], row['YearRemodAdd'])
# for i, row in all.iterrows():
#     if row['YearRemodAdd'] - row['YearBuilt'] == 3: print(row['Id'], row['YearBuilt'], row['YearRemodAdd'])

# train.loc[train['Remodeled'] == 1].groupby('AgeRemod').mean()['SalePrice'].plot(c = 'black')
# train.loc[train['Remodeled'] == 0].groupby('AgeRemod').mean()['SalePrice'].plot(c = 'grey')
# pearsonr(train.loc[train['Remodeled'] == 1]['SalePrice'], train.loc[train['Remodeled'] == 1]['AgeRemod'])

# (all['GrLivArea'] - all['1stFlrSF'] - all['2ndFlrSF'] - all['LowQualFinSF'] != 0).any()
# (all['TotalBsmtSF'] - all['BsmtFinSF1'] - all['BsmtFinSF2'] - all['BsmtUnfSF'] != 0).any()
# (all['TotRmsAbvGrd'] - all['BedroomAbvGr'] - all['KitchenAbvGr'] != 0).any()
# all['GarageSize'] = all.apply(lambda s: np.sqrt(s['GarageCars']*s['GarageArea']), axis = 1)
# train.plot.scatter(x='GarageSize', y='SalePrice', ylim=(0,800000), c = 'black')

### Correlatons ###

corr_mat = train.corr(method='pearson')
cols = corr_mat.nlargest(25, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', cmap='gist_gray_r',
                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

### Print statistics ###
# for col in train_f[num_feat]: print(col, train_f[num_feat][col].unique())
# for col in all[type_feat]: print(col, all[type_feat][col].unique())
# train.groupby('Heating')['SalePrice'].agg(['mean','count','size']).sort_values(by = 'mean', ascending=False)

# Basic EDA
# train.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000), c = 'black')
# train.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000), c = 'black')
# train.plot.scatter(x='TotalArea', y='SalePrice', c = 'black');
# sns.distplot(train['SalePrice'], color = 'black');
# sns.boxplot(x='OverallQual', y='SalePrice', data=train);
# sns.boxplot(x='YearBuilt', y='SalePrice', data=train);
# plt.xticks(rotation=90);
# sns.distplot(train['SalePrice'], color = 'black');