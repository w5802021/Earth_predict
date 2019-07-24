import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
import scipy.signal as sg
import multiprocessing as mp
from scipy.signal import hanning as hann
from scipy.signal import hilbert
from scipy.signal import convolve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from tqdm import tqdm

import gc
import os

import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy import stats
# from sklearn.svm import NuSVR, SVR
# from catboost import CatBoostRegressor
# from sklearn.kernel_ridge import KernelRidge
# from scipy.signal import hilbert, convolve
# from scipy.signal import hanning as hann
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold,cross_val_score
# from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
# from sklearn.model_selection import GridSearchCV
# from tqdm import tqdm

# from GBDT import myhyperopt

# settings
warnings.filterwarnings('ignore')
# np.random.seed(2019)
PATH = "E:/kaggle/kaggle-Lanl_Earthquake_Prediction/"

clean_idx = [x for x in range(4194,4651)]

# logger
def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger()

def read_train_data(nrows=4096):
    # load data
    logger.info('Start read data')
    train_df = pd.read_csv(PATH + 'input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64},nrows=nrows)
    return train_df

def load_feature_data():
    logger.info('Start read feature data')

    # train_X = pd.DataFrame()
    train_features = pd.read_csv(PATH + 'input/lanl-features/train_features.csv')
    train_features1 = pd.read_csv(PATH + 'input/lanl-features/train_features_denoised.csv')

    train_mfcc_features = pd.read_csv(PATH + 'input/lanl-features/mfcc_train40.csv')

    train_X = pd.concat([train_features,train_features1], axis=1)  #train_mfcc_features

    # train_X = train_features1      #.ix[0:4193,:]

    scaler = StandardScaler()
    scaler.fit(train_X)
    scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)

    # test_X = pd.DataFrame()
    test_features = pd.read_csv(PATH + 'input/lanl-features/test_features.csv')
    test_features1 = pd.read_csv(PATH + 'input/lanl-features/test_features_denoised.csv')

    test_mfcc_features = pd.read_csv(PATH + 'input/lanl-features/mfcc_test40.csv')
    mfcccol = list(train_mfcc_features.columns)

    test_X = pd.concat([test_features,test_features1], axis=1)  #test_mfcc_features.drop(['seg_id'], axis=1)
    # test_X = test_features1

    scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)

    train_y = pd.read_csv(PATH + 'input/lanl-features/y_1419.csv')    #.ix[0:4193]
    submission = pd.read_csv(PATH + 'input/sample_submission.csv', index_col='seg_id')

    return scaled_train_X, scaled_test_X, train_y,submission,mfcccol

def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def classic_sta_lta(x, length_sta, length_lta):
    sta = np.cumsum(x ** 2)
    # Convert to float
    sta = np.require(sta, dtype=np.float)
    # Copy for LTA
    lta = sta.copy()
    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta
    # Pad zeros
    sta[:length_lta - 1] = 0
    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    return sta / lta

def create_features(seg_id, seg, X):
    xc = pd.Series(seg['acoustic_data'].values)
    zc = np.fft.fft(xc)

    X.loc[seg_id, 'mean'] = xc.mean()
    X.loc[seg_id, 'std'] = xc.std()
    X.loc[seg_id, 'max'] = xc.max()
    X.loc[seg_id, 'min'] = xc.min()

    # FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    X.loc[seg_id, 'Rmean'] = realFFT.mean()
    X.loc[seg_id, 'Rstd'] = realFFT.std()
    X.loc[seg_id, 'Rmax'] = realFFT.max()
    X.loc[seg_id, 'Rmin'] = realFFT.min()
    X.loc[seg_id, 'Imean'] = imagFFT.mean()
    X.loc[seg_id, 'Istd'] = imagFFT.std()
    X.loc[seg_id, 'Imax'] = imagFFT.max()
    X.loc[seg_id, 'Imin'] = imagFFT.min()
    X.loc[seg_id, 'Rmean_last_5000'] = realFFT[-5000:].mean()
    X.loc[seg_id, 'Rstd__last_5000'] = realFFT[-5000:].std()
    X.loc[seg_id, 'Rmax_last_5000'] = realFFT[-5000:].max()
    X.loc[seg_id, 'Rmin_last_5000'] = realFFT[-5000:].min()
    X.loc[seg_id, 'Rmean_last_15000'] = realFFT[-15000:].mean()
    X.loc[seg_id, 'Rstd_last_15000'] = realFFT[-15000:].std()
    X.loc[seg_id, 'Rmax_last_15000'] = realFFT[-15000:].max()
    X.loc[seg_id, 'Rmin_last_15000'] = realFFT[-15000:].min()

    X.loc[seg_id, 'mean_change_abs'] = np.mean(np.diff(xc))
    X.loc[seg_id, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(xc) / xc[:-1]))[0])
    X.loc[seg_id, 'abs_max'] = np.abs(xc).max()
    # X.loc[seg_id, 'abs_min'] = np.abs(xc).min()

    X.loc[seg_id, 'std_first_50000'] = xc[:50000].std()
    X.loc[seg_id, 'std_last_50000'] = xc[-50000:].std()
    X.loc[seg_id, 'std_first_10000'] = xc[:10000].std()
    X.loc[seg_id, 'std_last_10000'] = xc[-10000:].std()

    X.loc[seg_id, 'avg_first_50000'] = xc[:50000].mean()
    X.loc[seg_id, 'avg_last_50000'] = xc[-50000:].mean()
    X.loc[seg_id, 'avg_first_10000'] = xc[:10000].mean()
    X.loc[seg_id, 'avg_last_10000'] = xc[-10000:].mean()

    X.loc[seg_id, 'min_first_50000'] = xc[:50000].min()
    X.loc[seg_id, 'min_last_50000'] = xc[-50000:].min()
    X.loc[seg_id, 'min_first_10000'] = xc[:10000].min()
    X.loc[seg_id, 'min_last_10000'] = xc[-10000:].min()

    X.loc[seg_id, 'max_first_50000'] = xc[:50000].max()
    X.loc[seg_id, 'max_last_50000'] = xc[-50000:].max()
    X.loc[seg_id, 'max_first_10000'] = xc[:10000].max()
    X.loc[seg_id, 'max_last_10000'] = xc[-10000:].max()

    X.loc[seg_id, 'max_to_min'] = xc.max() / np.abs(xc.min())
    X.loc[seg_id, 'max_to_min_diff'] = xc.max() - np.abs(xc.min())
    X.loc[seg_id, 'count_big'] = len(xc[np.abs(xc) > 500])
    X.loc[seg_id, 'sum'] = xc.sum()

    X.loc[seg_id, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(xc[:50000]) / xc[:50000][:-1]))[0])
    X.loc[seg_id, 'mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(xc[-50000:]) / xc[-50000:][:-1]))[0])
    X.loc[seg_id, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(xc[:10000]) / xc[:10000][:-1]))[0])
    X.loc[seg_id, 'mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(xc[-10000:]) / xc[-10000:][:-1]))[0])

    X.loc[seg_id, 'q95'] = np.quantile(xc, 0.95)
    X.loc[seg_id, 'q99'] = np.quantile(xc, 0.99)
    X.loc[seg_id, 'q05'] = np.quantile(xc, 0.05)
    X.loc[seg_id, 'q01'] = np.quantile(xc, 0.01)

    X.loc[seg_id, 'abs_q95'] = np.quantile(np.abs(xc), 0.95)
    X.loc[seg_id, 'abs_q99'] = np.quantile(np.abs(xc), 0.99)
    X.loc[seg_id, 'abs_q05'] = np.quantile(np.abs(xc), 0.05)
    X.loc[seg_id, 'abs_q01'] = np.quantile(np.abs(xc), 0.01)

    X.loc[seg_id, 'trend'] = add_trend_feature(xc)
    X.loc[seg_id, 'abs_trend'] = add_trend_feature(xc, abs_values=True)
    X.loc[seg_id, 'abs_mean'] = np.abs(xc).mean()
    X.loc[seg_id, 'abs_std'] = np.abs(xc).std()

    X.loc[seg_id, 'mad'] = xc.mad()
    X.loc[seg_id, 'kurt'] = xc.kurtosis()
    X.loc[seg_id, 'skew'] = xc.skew()
    X.loc[seg_id, 'med'] = xc.median()

    X.loc[seg_id, 'Hilbert_mean'] = np.abs(hilbert(xc)).mean()
    X.loc[seg_id, 'Hann_window_mean'] = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()
    X.loc[seg_id, 'classic_sta_lta1_mean'] = classic_sta_lta(xc, 500, 10000).mean()
    X.loc[seg_id, 'classic_sta_lta2_mean'] = classic_sta_lta(xc, 5000, 100000).mean()
    X.loc[seg_id, 'classic_sta_lta3_mean'] = classic_sta_lta(xc, 3333, 6666).mean()
    X.loc[seg_id, 'classic_sta_lta4_mean'] = classic_sta_lta(xc, 10000, 25000).mean()
    X.loc[seg_id, 'Moving_average_700_mean'] = xc.rolling(window=700).mean().mean(skipna=True)
    X.loc[seg_id, 'Moving_average_1500_mean'] = xc.rolling(window=1500).mean().mean(skipna=True)
    X.loc[seg_id, 'Moving_average_3000_mean'] = xc.rolling(window=3000).mean().mean(skipna=True)
    X.loc[seg_id, 'Moving_average_6000_mean'] = xc.rolling(window=6000).mean().mean(skipna=True)
    ewma = pd.Series.ewm
    X.loc[seg_id, 'exp_Moving_average_300_mean'] = (ewma(xc, span=300).mean()).mean(skipna=True)
    X.loc[seg_id, 'exp_Moving_average_3000_mean'] = ewma(xc, span=3000).mean().mean(skipna=True)
    X.loc[seg_id, 'exp_Moving_average_30000_mean'] = ewma(xc, span=6000).mean().mean(skipna=True)
    no_of_std = 2
    X.loc[seg_id, 'MA_700MA_std_mean'] = xc.rolling(window=700).std().mean()
    X.loc[seg_id, 'MA_700MA_BB_high_mean'] = (
            X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()
    X.loc[seg_id, 'MA_700MA_BB_low_mean'] = (
            X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()
    X.loc[seg_id, 'MA_400MA_std_mean'] = xc.rolling(window=400).std().mean()
    X.loc[seg_id, 'MA_400MA_BB_high_mean'] = (
            X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()
    X.loc[seg_id, 'MA_400MA_BB_low_mean'] = (
            X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()
    X.loc[seg_id, 'MA_1000MA_std_mean'] = xc.rolling(window=1000).std().mean()

    X.loc[seg_id, 'iqr'] = np.subtract(*np.percentile(xc, [75, 25]))
    X.loc[seg_id, 'q999'] = np.quantile(xc, 0.999)
    X.loc[seg_id, 'q001'] = np.quantile(xc, 0.001)
    X.loc[seg_id, 'ave10'] = stats.trim_mean(xc, 0.1)

    for windows in [10, 100, 1000]:
        x_roll_std = xc.rolling(windows).std().dropna().values
        x_roll_mean = xc.rolling(windows).mean().dropna().values

        X.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        X.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        X.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        X.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        X.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        X.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        X.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        X.loc[seg_id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        X.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        X.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        X.loc[seg_id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        X.loc[seg_id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        X.loc[seg_id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        X.loc[seg_id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        X.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        X.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        X.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        X.loc[seg_id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        X.loc[seg_id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        X.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        X.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

def feature_engineering(train_df):
    # features engineering
    logger.info('Features engineering')
    rows = 150000
    segments = int(np.floor(train_df.shape[0] / rows))
    print("Number of segments: ", segments)
    train_X = pd.DataFrame(index=range(segments), dtype=np.float64)
    train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])

    # process train data
    logger.info('Process train data')
    for seg_id in range(segments):
        seg = train_df.iloc[seg_id * rows:seg_id * rows + rows]
        create_features(seg_id, seg, train_X)
        train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]

    scaler = StandardScaler()
    scaler.fit(train_X)
    scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)

    # process test data
    logger.info('Process test data')
    submission = pd.read_csv(PATH + 'input/sample_submission.csv', index_col='seg_id')
    test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)

    for seg_id in test_X.index:
        seg = pd.read_csv(PATH + 'input/test/' + seg_id + '.csv')
        create_features(seg_id, seg, test_X)
    scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)

    del train_df
    gc.collect()

    scaled_train_X.to_csv(PATH + 'GBDT/feature_extra/train_feature.csv', index=False)
    scaled_test_X.to_csv(PATH + 'GBDT/feature_extra/test_feature.csv', index=False)
    train_y.to_csv(PATH + 'GBDT/feature_extra/train_y.csv', index=False)

    return scaled_train_X, scaled_test_X, train_y, submission

def data_augmentation(train,aug_ratio=0.1):
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

def run_model_xgb(scaled_train_X,scaled_test_X, train_y,feature_col):
    logger.info('Run xgb model')
    n_fold = 10
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    train_columns = feature_col
    random_seed = 4126
    predictions = np.zeros(len(scaled_test_X))
    preds_train = np.zeros(len(scaled_train_X))

    maes = []
    tr_maes = []

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        strLog = "fold {}".format(fold_)
        print(strLog)

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

        clf = xgb.XGBRegressor(n_estimators=10000,
                               learning_rate=0.1,
                               max_depth=6,
                               subsample=0.9,
                               colsample_bytree=0.67,
                               reg_lambda=1.0, # seems best within 0.5 of 2.0
                               # gamma=1,
                               random_state=random_seed,
                               n_jobs=12,
                               verbosity=-1)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(scaled_test_X)  # , num_iteration=model.best_iteration_)
        predictions += preds / folds.n_splits
        preds = clf.predict(scaled_train_X)  # , num_iteration=model.best_iteration_)
        preds_train += preds / folds.n_splits

        preds = clf.predict(X_tr)
        mae = mean_absolute_error(y_tr, preds)
        print('Tr MAE: %.6f' % mae)
        maes.append(mae)


        preds = clf.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)
        tr_maes.append(mae)

def run_model_lgbm(params,scaled_train_X, scaled_test_X, train_y,feature_col):
    logger.info('Run lgbm model')
    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state = 2013)
    train_columns = feature_col

    scores = []

    oof = np.zeros(len(scaled_train_X))
    predictions = np.zeros(len(scaled_test_X))
    feature_importance_df = pd.DataFrame()

    # run model
    # a = folds.split(scaled_train_X, train_y.values)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print("fold {}".format(fold_))
        logger.info("fold {}".format(fold_))

        val_idx = val_idx[val_idx<4195]

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

        clf = lgb.LGBMRegressor(**params, n_estimators = 200000, n_jobs=-1)

        clf.fit(X_tr, y_tr,eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',verbose=1000,early_stopping_rounds=400)

        oof[val_idx] = clf.predict(X_val, num_iteration=clf.best_iteration_)
        # feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = train_columns
        fold_importance_df["importance"] = clf.feature_importances_[:len(train_columns)]
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        # predictions
        predictions += clf.predict(scaled_test_X, num_iteration=clf.best_iteration_) / folds.n_splits
        scores.append(mean_absolute_error(y_val, oof[val_idx]))

    # feature_importance_df = feature_importance_df.groupby(['Feature']).mean()
    # feature_importance_df.to_csv('E:/kaggle/kaggle-Lanl_Earthquake_Prediction/data_sample/feature_importance.csv')
    strLog = 'CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores))
    print(strLog)
    logger.info(strLog)

    return oof,predictions, feature_importance_df,mean_absolute_error(train_y.values, oof)

def plt_feature_importance(feature_importance_df):
    logger.info('Plot feature importance')
    cols = (feature_importance_df[["Feature", "importance"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:200].index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]
    # d_features = best_features.groupby(['Feature']).mean()
    # d_features.to_csv('best_features.csv')
    plt.figure(figsize=(14, 26))
    sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (averaged over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')

def submit(submission, predictions,i):
    # submission
    logger.info('Submisison')
    submission.time_to_failure = predictions
    submission.to_csv('./Lightgbm_result/submission_' + str(i) + '_auc.csv', index=True)

def main(nrows=None):
    # train_df = read_train_data(nrows)
    # scaled_train_X, scaled_test_X, train_y, submission = feature_engineering(train_df)
    scaled_train_X, scaled_test_X, train_y, submission,mfcccols = load_feature_data()
    # feature_importance_df = pd.read_csv('feature_importance.csv')
    #
    # cols = list((feature_importance_df[["Feature", "importance"]]
    #                      .groupby("Feature")
    #                      .mean()
    #                      .sort_values(by="importance", ascending=False)[:].index))[:200]
    # cols.extend(scaled_train_X.columns[-20:])
    # dd = pd.read_excel(PATH + 'input/lanl-features/features_select2.xlsx',header=None)
    # dd.columns = ['feat']
    # cols = list(dd['feat']) + list(mfcccols)

    # best_feat = pd.read_csv('best_features.csv')
    # best_feat1 = best_feat[best_feat['importance'] >= 1]
    # cols1 = list(best_feat1['feat'])

    # sele = pd.read_csv(PATH + '/feat_val/feat_auc_now998.csv').T
    # sele.columns = ['mean', 'std']
    # sele = sele.sort_values('mean', ascending=False)
    # selec = sele.copy()
    # sele_hold = sele[sele['mean'] < 0.8]
    # #
    # # # hold_auc = pd.Series(list(sele_hold.index))
    # # # hold_auc.to_csv('E:/kaggle/kaggle-Lanl_Earthquake_Prediction/feat_val/hold_feat.csv',index=False)
    # #
    # sele_hold = list(sele_hold.index)
    # cols = sele_hold

    # sele = pd.read_csv('E:/kaggle/kaggle-Lanl_Earthquake_Prediction/feat_val/998cv_rank2.csv',header=None)
    # sele.columns = ['feat','cv']
    # sele = sele[(sele['cv'] <=3) & (sele['cv'] > 0)]
    # cols = list(sele.feat)
    # scaled_train_X = scaled_train_X[cols]
    # scaled_test_X = scaled_test_X[cols]

    l = [0.5]
    res = []

    train_X_aug = data_augmentation(scaled_train_X)
    train_y_aug = data_augmentation(train_y)
    train_all = pd.concat([scaled_train_X, train_X_aug])
    y_all = pd.concat([train_y, train_y_aug])
    # train_all.to_csv('E:\kaggle\kaggle-Lanl_Earthquake_Prediction\input\lanl-features/train_aug_0.5.csv',index=False)
    # y_all.to_csv('E:\kaggle\kaggle-Lanl_Earthquake_Prediction\input\lanl-features/y_aug_0.5.csv',index=False)
    # train_all = pd.read_csv('E:\kaggle\kaggle-Lanl_Earthquake_Prediction\input\lanl-features/train_aug_0.5.csv').drop(['target'],axis=1)
    # y_all = pd.read_csv('E:\kaggle\kaggle-Lanl_Earthquake_Prediction\input\lanl-features/y_aug_0.5.csv')

    # lgbm_params = myhyperopt.quick_hyperopt(train_all, y_all, 'lgbm', 2500)

    lgbm_params = {'num_leaves':60,
                  'min_data_in_leaf': 79,
                  'objective': 'gamma',
                  'max_depth': -1,
                  'learning_rate': 0.02,
                  "boosting": "gbdt",
                  "bagging_freq": 5,
                  "bagging_fraction": 0.8126672064208567,
                  "bagging_seed": 1024,
                  "metric": 'mae',
                  "verbosity": -1,
                  'reg_alpha': 0.1302650970728192,
                  'reg_lambda': 0.3603427518866501,
                  'feature_fraction': 0.2,
                   'colsample_bytree': 1.0
                  }

    oof,predictions, feature_importance,mae = run_model_lgbm(lgbm_params,train_all, scaled_test_X, y_all,train_all.columns)
    # run_model_xgb(scaled_train_X,scaled_test_X, train_y,scaled_train_X.columns)
    submit(submission, predictions,'2000feat_aug0.5')
    # # res.append(mae)
    # plt_feature_importance(feature_importance)
    plt.figure(figsize=(16, 8))
    plt.plot(train_y, color='g', label='y_train')
    plt.plot(oof[:4194], color='r', label='lgb')
    plt.legend()
    plt.title('Predictions vs actual')
    plt.show()
    # print(res)

if __name__ == "__main__":
    main()
