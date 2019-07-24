import os
import time
import warnings
import traceback
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
import scipy.signal as sg
import multiprocessing as mp
from scipy.signal import hann
from scipy.signal import hilbert
from scipy.signal import convolve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from GBDT import myhyperopt
import matplotlib.pyplot as plt
from GBDT import feature_selector

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from tqdm import tqdm

warnings.filterwarnings("ignore")

SIG_LEN = 150000
NUM_SEG_PER_PROC = 4000
NUM_THREADS = 6

NY_FREQ_IDX = 75000  # the test signals are 150k samples long, Nyquist is thus 75k.
CUTOFF = 18000
MAX_FREQ_IDX = 20000
FREQ_STEP = 2500

PATH = "E:/kaggle/kaggle-Lanl_Earthquake_Prediction/"

def join_mp_build():
    train_X = pd.read_csv(PATH + 'input/masters-final-project/train_x_%d.csv' % 0)
    train_y = pd.read_csv(PATH + 'input/masters-final-project/train_y_%d.csv' % 0)
    test_X = pd.read_csv(PATH + 'input/masters-final-project/test_x.csv')
    for i in tqdm(range(1, NUM_THREADS)):
        temp = pd.read_csv(PATH + 'input/masters-final-project/train_x_%d.csv' % i)
        train_X = train_X.append(temp)
        temp = pd.read_csv(PATH + 'input/masters-final-project/train_y_%d.csv' % i)
        train_y = train_y.append(temp)
    return train_X, train_y, test_X

def data_augmentation(train,aug_ratio=0.5):
    a = np.arange(0, train.shape[1])      #
    # initialise aug dataframe - remember to set dtype!
    train_aug = pd.DataFrame(index=train.index, columns=train.columns, dtype='float64')

    for i in tqdm(range(0, len(train))):              #样本数量4194
        # ratio of features to be randomly sampled
        AUG_FEATURE_RATIO = aug_ratio
        # to integer count
        AUG_FEATURE_COUNT = np.floor(train.shape[1] * AUG_FEATURE_RATIO).astype('int16')

        # randomly sample half of columns that will contain random values  #如果是ndarray数组，随机样本在该数组获取（取数据元素）,如果是整型数据随机样本生成类似np.arange(n)
        aug_feature_index = np.random.choice(train.shape[1], AUG_FEATURE_COUNT, replace=False)
        aug_feature_index.sort()

        # obtain indices for features not in aug_feature_index            #
        feature_index = np.where(np.logical_not(np.in1d(a, aug_feature_index)))[0]

        # first insert real values for features in feature_index           #将未被增强的特征按原始数据保存下来
        train_aug.iloc[i, feature_index] = train.iloc[i, feature_index]

        # random row index to randomly sampled values for each features        #从原始样本中随机抽取与增强特征数量相同的样本 （即增强的特征只作用于部分样本）
        rand_row_index = np.random.choice(len(train), len(aug_feature_index), replace=True)

        # for each feature being randomly sampled, extract value from random row in train
        for n, j in enumerate(aug_feature_index):                     #即增强的特征  只更新 n 个样本 n=增强特征数
            train_aug.iloc[i, j] = train.iloc[rand_row_index[n], j]

    return train_aug   #过采样产生的数据，将其与train_x拼接融合

def scale_fields(train_X, test_X):
    try:
        train_X.drop(labels=['seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
    except:
        pass
    print('start scaler')
    scaler = StandardScaler()
    scaler.fit(train_X)
    scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)
    scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)
    return scaled_train_X, scaled_test_X


def xgb_trimmed_model(scaled_train_X, scaled_test_X, train_y):
    maes = []
    rmses = []
    tr_maes = []
    tr_rmses = []
    submission = pd.read_csv(PATH + 'input/sample_submission.csv', index_col='seg_id')

    pcol = []
    pcor = []
    pval = []
    y = train_y['time_to_failure'].values

    for col in scaled_train_X.columns:
        pcol.append(col)
        pcor.append(abs(pearsonr(scaled_train_X[col], y)[0]))
        pval.append(abs(pearsonr(scaled_train_X[col], y)[1]))

    df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
    df.sort_values(by=['cor', 'pval'], inplace=True, ascending=False)   #pval值粗略地表示不相关系统产生具有Pearson相关性的
                                                                        # 数据集的概率至少与从这些数据集计算的数据集一样极端。
                                                                        # p值并不完全可靠，但对于大于500左右的数据集可能是合理的。
    df = df.reset_index(drop=True)
    df.dropna(inplace=True)
    df = df.iloc[: 500]      #改动  取根据pearson系数最大的前500个特征

    drop_cols = []
    for col in scaled_train_X.columns:
        if col not in df['col'].tolist():
            drop_cols.append(col)

    scaled_train_X.drop(labels=drop_cols, axis=1, inplace=True)
    scaled_test_X.drop(labels=drop_cols, axis=1, inplace=True)

    ############################数据增强####################################

    # train_X_aug = data_augmentation(scaled_train_X)
    # train_y_aug = data_augmentation(train_y)
    # scaled_train_X = pd.concat([scaled_train_X, train_X_aug])
    # train_y = pd.concat([train_y, train_y_aug])

    #######################################################################

    predictions = np.zeros(len(scaled_test_X))
    preds_train = np.zeros(len(scaled_train_X))
    feature_importance_df = pd.DataFrame()


    print('shapes of train and test:', scaled_train_X.shape, scaled_test_X.shape)

    n_fold = 8
    folds = KFold(n_splits = n_fold, shuffle = False, random_state=42)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        strLog = "fold {}".format(fold_)
        print(strLog)
        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

        model = xgb.XGBRegressor(n_estimators = 1000,
                                 learning_rate = 0.005,
                                 max_depth = 12,
                                 subsample = 0.9,
                                 colsample_bytree = 0.3,
                                 reg_lambda = 1.0,  # seems best within 0.5 of 2.0
                                 # gamma=1,
                                 random_state = 777 + fold_,
                                 n_jobs = -1,
                                 verbosity = 2)
        model.fit(X_tr, y_tr)

        # predictions
        preds = model.predict(scaled_test_X)  # , num_iteration=model.best_iteration_)
        predictions += preds / folds.n_splits
        preds = model.predict(scaled_train_X)  # , num_iteration=model.best_iteration_)
        preds_train += preds / folds.n_splits
        preds = model.predict(X_val)  # , num_iteration=model.best_iteration_)


        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = scaled_train_X.columns
        fold_importance_df["importance"] = model.feature_importances_[:len(scaled_train_X.columns)]
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        # mean absolute error
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)                   #验证集误差
        maes.append(mae)

        # root mean squared error
        rmse = mean_squared_error(y_val, preds)
        print('RMSE: %.6f' % rmse)
        rmses.append(rmse)

        # training for over fit
        preds = model.predict(X_tr)  # , num_iteration=model.best_iteration_)
        mae = mean_absolute_error(y_tr, preds)
        print('Tr MAE: %.6f' % mae)                 #训练集误差

        tr_maes.append(mae)
        rmse = mean_squared_error(y_tr, preds)
        print('Tr RMSE: %.6f' % rmse)
        tr_rmses.append(rmse)

    print('MAEs', maes)
    print('MAE mean: %.6f' % np.mean(maes))
    print('RMSEs', rmses)
    print('RMSE mean: %.6f' % np.mean(rmses))

    print('Tr MAEs', tr_maes)
    print('Tr MAE mean: %.6f' % np.mean(tr_maes))
    print('Tr RMSEs', tr_rmses)
    print('Tr RMSE mean: %.6f' % np.mean(tr_rmses))

    submission.time_to_failure = predictions
    submission.to_csv('./Xgboost_result/submission_xgb_pearson_6fold_shuffle.csv')  # index needed, it is seg id
    pr_tr = pd.DataFrame(data=preds_train, columns=['time_to_failure'], index=range(0, preds_train.shape[0]))
    pr_tr.to_csv(r'./Xgboost_result/preds_tr_xgb_pearson_6fold_shuffle.csv', index=False)
    feature_importance_df.to_csv('xgboost_feat_import.csv')

    print('Train shape: {}, Test shape: {}, Y shape: {}'.format(scaled_train_X.shape, scaled_test_X.shape, train_y.shape))

def lgb_trimmed_model(scaled_train_X, scaled_test_X, train_y):
    maes = []
    rmses = []
    tr_maes = []
    tr_rmses = []
    submission = pd.read_csv(PATH + 'input/sample_submission.csv', index_col='seg_id')

    pcol = []
    pcor = []
    pval = []
    y = train_y['time_to_failure'].values

    for col in scaled_train_X.columns:
        pcol.append(col)
        pcor.append(abs(pearsonr(scaled_train_X[col], y)[0]))
        pval.append(abs(pearsonr(scaled_train_X[col], y)[1]))
    df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
    df.sort_values(by=['cor', 'pval'], inplace=True, ascending=False)
    df.dropna(inplace=True)

    #########################由pearson系数过滤特征##########################
    df = df.iloc[: 500]  # 改动  取根据pearson系数最大的前500个特征

    drop_cols = []
    for col in scaled_train_X.columns:
        if col not in df['col'].tolist():
            drop_cols.append(col)

    scaled_train_X.drop(labels=drop_cols, axis=1, inplace=True)
    scaled_test_X.drop(labels=drop_cols, axis=1, inplace=True)
    ############################数据增强####################################

    # train_X_aug = data_augmentation(scaled_train_X)
    # train_y_aug = data_augmentation(train_y)
    # scaled_train_X = pd.concat([scaled_train_X, train_X_aug])
    # train_y = pd.concat([train_y, train_y_aug])

    #######################################################################

    predictions = np.zeros(len(scaled_test_X))
    preds_train = np.zeros(len(scaled_train_X))

    print('shapes of train and test:', scaled_train_X.shape, scaled_test_X.shape)

    n_fold = 6
    folds = KFold(n_splits=n_fold, shuffle=False, random_state=42)
    # params = myhyperopt.quick_hyperopt(scaled_train_X, train_y, 'lgbm', 2500)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        strLog = "fold {}".format(fold_)
        print(strLog)

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

        params = {'num_leaves': 21,
                  'min_data_in_leaf': 20,
                  'objective': 'regression',
                  'max_depth': -1,
                  'learning_rate': 0.001,
                  "boosting": "gbdt",
                  "feature_fraction": 0.91,
                  "bagging_freq": 1,
                  "bagging_fraction": 0.91,
                  "bagging_seed": 42,
                  "metric": 'mae',
                  "lambda_l1": 0.1,
                  "verbosity": -1,
                  "random_state": 42}

        model = lgb.LGBMRegressor(**params, n_estimators=60000, n_jobs=-1)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',
                  verbose=1000, early_stopping_rounds=200)

        # predictions
        preds = model.predict(scaled_test_X)  # , num_iteration=model.best_iteration_)
        predictions += preds / folds.n_splits
        preds = model.predict(scaled_train_X)  # , num_iteration=model.best_iteration_)
        preds_train += preds / folds.n_splits
        preds = model.predict(X_val)  # , num_iteration=model.best_iteration_)

        # mean absolute error
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)  # 验证集误差
        maes.append(mae)

        # root mean squared error
        rmse = mean_squared_error(y_val, preds)
        print('RMSE: %.6f' % rmse)
        rmses.append(rmse)

        # training for over fit
        preds = model.predict(X_tr)  # , num_iteration=model.best_iteration_)
        mae = mean_absolute_error(y_tr, preds)
        print('Tr MAE: %.6f' % mae)  # 训练集误差

        tr_maes.append(mae)
        rmse = mean_squared_error(y_tr, preds)
        print('Tr RMSE: %.6f' % rmse)
        tr_rmses.append(rmse)

    print('MAEs', maes)
    print('MAE mean: %.6f' % np.mean(maes))
    print('RMSEs', rmses)
    print('RMSE mean: %.6f' % np.mean(rmses))

    print('Tr MAEs', tr_maes)
    print('Tr MAE mean: %.6f' % np.mean(tr_maes))
    print('Tr RMSEs', tr_rmses)
    print('Tr RMSE mean: %.6f' % np.mean(tr_rmses))

    submission.time_to_failure = predictions
    submission.to_csv('./Lightgbm_result/submission_lgb_pearson_6fold_shuffle.csv')  # index needed, it is seg id
    pr_tr = pd.DataFrame(data=preds_train, columns=['time_to_failure'], index=range(0, preds_train.shape[0]))
    pr_tr.to_csv(r'./Lightgbm_result/preds_tr_lgb_pearson_6fold_shuffle.csv', index=False)

    print('Train shape: {}, Test shape: {}, Y shape: {}'.format(scaled_train_X.shape, scaled_test_X.shape, train_y.shape))


if __name__ == '__main__':
    train_X, train_y, test_X = join_mp_build()
    train_X['classic_sta_lta1_mean_0'].loc[~np.isfinite(train_X['classic_sta_lta1_mean_0'])] = train_X['classic_sta_lta1_mean_0'].loc[np.isfinite(train_X['classic_sta_lta1_mean_0'])].mean()
    scaled_train_X, scaled_test_X = scale_fields(train_X,test_X)

    xgb_trimmed_model(scaled_train_X, scaled_test_X, train_y)