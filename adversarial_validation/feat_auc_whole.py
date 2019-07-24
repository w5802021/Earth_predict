import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, metrics
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

PATH = 'E:/kaggle/kaggle-Lanl_Earthquake_Prediction/input/masters-final-project'
train = pd.read_csv(PATH + '/train_x.csv')
# train_y = pd.read_csv(PATH + '/input/masters-final-project/train_y.csv')
test = pd.read_csv(PATH + '/test_x.csv')   #.drop(['seg_id'],axis=1)

# features = train.columns

sele = pd.read_csv('E:/kaggle/kaggle-Lanl_Earthquake_Prediction/feat_val/feat_auc_master865.csv').T
sele.columns = ['mean','std']
sele = sele.dropna()
sele = sele.sort_values('mean',ascending=False)
sele = sele[(sele['mean'] < 0.7) & (sele['mean'] > 0)]
sele = list(sele.index)

features = sele
# train[sele].to_csv('train69.csv')
# test[sele].to_csv('test69.csv')

# train['classic_sta_lta1_mean_0'].loc[~np.isfinite(train['classic_sta_lta1_mean_0'])] = train['classic_sta_lta1_mean_0'].loc[np.isfinite(train['classic_sta_lta1_mean_0'])].mean()
# scaler = StandardScaler()
# train.drop(labels=['seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
# scaler.fit(train)
# scaled_train_X = pd.DataFrame(scaler.transform(train), columns=train.columns)
# pcol = []
# pcor = []
# pval = []
# y = train_y['time_to_failure'].values
#
# for col in scaled_train_X.columns:
#     pcol.append(col)
#     pcor.append(abs(pearsonr(scaled_train_X[col], y)[0]))
#     pval.append(abs(pearsonr(scaled_train_X[col], y)[1]))
#
# df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
# df.sort_values(by=['cor', 'pval'], inplace=True, ascending=False)   #pval值粗略地表示不相关系统产生具有Pearson相关性的
#                                                                     # 数据集的概率至少与从这些数据集计算的数据集一样极端。
#                                                                     # p值并不完全可靠，但对于大于500左右的数据集可能是合理的。
# df = df.reset_index(drop=True)
# df.dropna(inplace=True)
# df = df.iloc[: 500]      #改动  取根据pearson系数最大的前500个特征

train['target'] = 0
test['target'] = 1

train_test = pd.concat([train, test], axis =0)
target = train_test['target'].values
#
# sele = pd.read_csv(PATH + '/feat_val/feat_auc_master865.csv').T
# sele.columns = ['mean','std']
# sele = sele.dropna()
# sele = sele.sort_values('mean',ascending=False)
# sele = sele[(sele['mean'] < 0.58) & (sele['mean'] > 0)]
# sele = list(sele.index)
# cols = [x for x in list(df['col']) if x in sele]

# sele = pd.read_csv('E:/kaggle/kaggle-Lanl_Earthquake_Prediction/feat_val/998cv_rank2.csv',header=None)
# sele.columns = ['feat','cv']
# sele = sele[(sele['cv'] <=3) & (sele['cv'] > 0)]
# cols = list(sele.feat)

# features = cols
# out1 = train[cols]
# out2 = test[cols]
#
# out1.to_csv(PATH + '/input/masters-final-project/train_x_filter.csv',index=False)
# out2.to_csv(PATH + '/input/masters-final-project/test_x_filter.csv',index=False)

param = {'num_leaves': 50,
         'min_data_in_leaf': 30,
         'objective':'binary',
         'max_depth': 5,
         'learning_rate': 0.006,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 27,
         "metric": 'auc',
         "verbosity": -1}

folds = KFold(n_splits=5, shuffle=True, random_state=15)
res = pd.DataFrame(columns =['res'],index = [0,1])

oof = np.zeros(len(train_test))
val_aucs = []

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_test.values, target)):
    print("fold n°{}".format(fold_))
    # print('Using feature',features)

    trn_data = lgb.Dataset(train_test.iloc[trn_idx][features], label=target[trn_idx])
    val_data = lgb.Dataset(train_test.iloc[val_idx][features], label=target[val_idx])

    num_round = 30000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1400)

    oof[val_idx] = clf.predict(train_test.iloc[val_idx][features], num_iteration=clf.best_iteration)
    val_aucs.append(roc_auc_score(target[val_idx], oof[val_idx]))


res.ix[0, :] = np.mean(val_aucs)
res.ix[1, :] = np.std(val_aucs)
res.to_csv('feat_auc11.csv')