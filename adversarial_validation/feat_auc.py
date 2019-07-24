import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, metrics
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

PATH = 'E:/kaggle/kaggle-Lanl_Earthquake_Prediction/input/masters-final-project'

train = pd.read_csv(PATH + '/train_x.csv').drop(['seg_id','seg_start','seg_end'],axis=1)

test = pd.read_csv(PATH + '/test_x.csv')    #.drop(['seg_id'],axis=1)

# sele = pd.read_csv(PATH + '/feat_val/feat_auc.csv').T
# sele.columns = ['mean','std']
# sele = sele[sele['mean'] <= 0.63]

features = train.columns

train['target'] = 0
test['target'] = 1

train_test = pd.concat([train, test], axis =0)

target = train_test['target'].values

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

folds = KFold(n_splits=3, shuffle=True, random_state=15)
res = pd.DataFrame(columns = features,index = [0,1])

for i in tqdm(range(len(features))):
    feat = features[i]
    oof = np.zeros(len(train_test))
    val_aucs = []
    pan = train_test[feat].unique()
    if len(pan) <= 30:continue

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_test.values, target)):
        print("fold n°{}".format(fold_))
        print('Using feature',feat)

        trn_data = lgb.Dataset(pd.DataFrame(train_test.iloc[trn_idx][feat],columns=[feat]), label=target[trn_idx])
        val_data = lgb.Dataset(pd.DataFrame(train_test.iloc[val_idx][feat],columns=[feat]), label=target[val_idx])

        num_round = 30000
        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1400)

        oof[val_idx] = clf.predict(train_test.iloc[val_idx][[feat]], num_iteration=clf.best_iteration)
        val_aucs.append(roc_auc_score(target[val_idx], oof[val_idx]))

    res.ix[0, feat] = np.mean(val_aucs)
    res.ix[1, feat] = np.std(val_aucs)
    res.to_csv('feat_auc_master865.csv')