from tqdm import tqdm as tqdm
from numpy import random
from keras.layers import (BatchNormalization,Flatten,Convolution1D,Activation,Input,Dense,LSTM)
from tsfresh.feature_extraction import feature_calculators
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import Sequence, to_categorical
from sklearn.metrics import mean_absolute_error
from keras.callbacks import ReduceLROnPlateau
from keras import losses, models, optimizers
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
import lightgbm as lgb
import seaborn as sns
import random as rn
import pandas as pd
import numpy as np
import scipy as sp
import itertools
import warnings
import librosa
import pywt
import os
import gc
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

#每段地震的起始及发生的数据索引
etq_meta = [{"start":0,         "end":5656574},
            {"start":5656574,   "end":50085878},
            {"start":50085878,  "end":104677356},
            {"start":104677356, "end":138772453},
            {"start":138772453, "end":187641820},
            {"start":187641820, "end":218652630},
            {"start":218652630, "end":245829585},
            {"start":245829585, "end":307838917},
            {"start":307838917, "end":338276287},
            {"start":338276287, "end":375377848},
            {"start":375377848, "end":419368880},
            {"start":419368880, "end":461811623},
            {"start":461811623, "end":495800225},
            {"start":495800225, "end":528777115},
            {"start":528777115, "end":585568144},
            {"start":585568144, "end":621985673},
            {"start":621985673, "end":629145480}]

PATH = 'E:/kaggle/kaggle-Lanl_Earthquake_Prediction/'
np.random.seed(1337)
noise = np.random.normal(0, 0.5, 150_000)

def normalize(X_train, X_valid, X_test, normalize_opt, excluded_feat):
    feats = [f for f in X_train.columns if f not in excluded_feat]
    if normalize_opt != None:
        if normalize_opt == 'min_max':
            scaler = preprocessing.MinMaxScaler()
        elif normalize_opt == 'robust':
            scaler = preprocessing.RobustScaler()
        elif normalize_opt == 'standard':
            scaler = preprocessing.StandardScaler()
        elif normalize_opt == 'max_abs':
            scaler = preprocessing.MaxAbsScaler()
        scaler = scaler.fit(X_train[feats])
        X_train[feats] = scaler.transform(X_train[feats])
        X_valid[feats] = scaler.transform(X_valid[feats])
        X_test[feats] = scaler.transform(X_test[feats])

    return X_train, X_valid, X_test

def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')

def denoise_signal_simple(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    # univeral threshold
    uthresh = 10
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')


def feature_gen(z):
    X = pd.DataFrame(index=[0], dtype=np.float64)

    # Add noise, subtract median to remove bias from mean/median as time passes in the experiment
    # Save the result as a new segment, z
    z = z + noise
    z = z - np.median(z)

    # Save denoised versions of z
    den_sample = denoise_signal(z)
    den_sample_simple = denoise_signal_simple(z)

    # Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(z)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_denoise_simple = librosa.feature.mfcc(den_sample_simple)
    mfcc_mean_denoise_simple = mfcc_denoise_simple.mean(axis=1)  # 0-19

    # Spectral contrast
    lib_spectral_contrast_denoise_simple = librosa.feature.spectral_contrast(den_sample_simple).mean(axis=1)  # 0-6
    lib_spectral_contrast = librosa.feature.spectral_contrast(z).mean(axis=1)  # 0-6

    # Neural network features
    X['NN_zero_crossings_denoise'] = len(np.where(np.diff(np.sign(den_sample)))[0])
    X['NN_LGBM_percentile_roll20_std_50'] = np.percentile(pd.Series(z).rolling(20).std().dropna().values, 50)
    X['NN_q95_roll20_std'] = np.quantile(pd.Series(z).rolling(20).std().dropna().values, 0.95)
    X['NN_LGBM_mfcc_mean4'] = mfcc_mean[4]
    X['NN_lib_spectral_contrast0'] = lib_spectral_contrast[0]
    X['NN_num_peaks_3_denoise'] = feature_calculators.number_peaks(den_sample, 3)
    X['NN_mfcc_mean_denoise_simple2'] = mfcc_mean_denoise_simple[2]
    X['NN_mfcc_mean5'] = mfcc_mean[5]
    X['NN_mfcc_mean2'] = mfcc_mean[2]
    X['NN_mfcc_mean_denoise_simple5'] = mfcc_mean_denoise_simple[5]
    X['NN_absquant95'] = np.quantile(np.abs(z), 0.95)
    X['NN_median_roll50_std_denoise_simple'] = np.median(
        pd.Series(den_sample_simple).rolling(50).std().dropna().values)
    X['NN_mfcc_mean_denoise_simple1'] = mfcc_mean_denoise_simple[1]
    X['NN_quant99'] = np.quantile(z, 0.99)
    X['NN_lib_zero_cross_rate_denoise_simple'] = librosa.feature.zero_crossing_rate(den_sample_simple)[0].mean()
    X['NN_fftr_max_denoise'] = np.max(pd.Series(np.abs(np.fft.fft(den_sample)))[0:75000])
    X['NN_abssumgreater15'] = np.sum(abs(z[np.where(abs(z) > 15)]))
    X['NN_LGBM_mfcc_mean18'] = mfcc_mean[18]
    X['NN_lib_spectral_contrast_denoise_simple2'] = lib_spectral_contrast_denoise_simple[2]
    X['NN_fftr_sum'] = np.sum(pd.Series(np.abs(np.fft.fft(z)))[0:75000])
    X['NN_mfcc_mean_denoise_simple10'] = mfcc_mean_denoise_simple[10]

    # Extra features only LGBM used.
    X['LGBM_num_peaks_2_denoise_simple'] = feature_calculators.number_peaks(den_sample_simple, 2)
    X['LGBM_autocorr5'] = feature_calculators.autocorrelation(pd.Series(z), 5)

    # Windowed fast fourier transformations
    fftrhann20000 = np.sum(np.abs(np.fft.fft(np.hanning(len(z)) * z)[:20000]))
    fftrhann20000_denoise = np.sum(np.abs(np.fft.fft(np.hanning(len(z)) * den_sample)[:20000]))
    fftrhann20000_diff_rate = (fftrhann20000 - fftrhann20000_denoise) / fftrhann20000

    X['LGBM_fftrhann20000_diff_rate'] = fftrhann20000_diff_rate

    return X

# create train and test sets
def parse_sample(sample, start):
    delta = feature_gen(sample['acoustic_data'].values)
    delta['start'] = start
    delta['target'] = sample['time_to_failure'].values[-1]
    return delta

def sample_train_gen(df, segment_size=150_000, indices_to_calculate=[0]):
    result = Parallel(n_jobs=1, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")(
        delayed(parse_sample)(df[int(i): int(i) + segment_size], int(i))
        for i in tqdm(indices_to_calculate))
    data = [r.values for r in result]
    data = np.vstack(data)
    X = pd.DataFrame(data, columns=result[0].columns)
    X = X.sort_values("start")
    return X

def parse_sample_test(seg_id):
    sample = pd.read_csv('../input/test/' + seg_id + '.csv', dtype={'acoustic_data': np.int32})
    delta = feature_gen(sample['acoustic_data'].values)
    delta['seg_id'] = seg_id
    return delta

def sample_test_gen():
    X = pd.DataFrame()
    submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
    result = Parallel(n_jobs=1, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")(
        delayed(parse_sample_test)(seg_id) for seg_id in tqdm(submission.index))
    data = [r.values for r in result]
    data = np.vstack(data)
    X = pd.DataFrame(data, columns=result[0].columns)
    return X

def train_lgbm_model(train_sample,targets,test,submission):
    oof = np.zeros(len(train_sample))
    prediction = np.zeros(len(submission))

    n_fold = 3
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=1337)
    kf = list(kf.split(np.arange(len(train_sample))))

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for seed in seeds:
        print('Seed', seed)
        for fold_n, (train_index, valid_index) in enumerate(kf):
            print('Fold', fold_n)

            trn_data = lgb.Dataset(train_sample.iloc[train_index], label=targets.iloc[train_index])
            val_data = lgb.Dataset(train_sample.iloc[valid_index], label=targets.iloc[valid_index])

            params = {'num_leaves': 4,
                      'min_data_in_leaf': 5,
                      'objective': 'fair',
                      'max_depth': -1,
                      'learning_rate': 0.01,
                      "boosting": "gbdt",
                      'boost_from_average': True,
                      "feature_fraction": 0.9,
                      "bagging_freq": 1,
                      "bagging_fraction": 0.5,
                      "bagging_seed": 0,
                      "metric": 'mae',
                      "verbosity": -1,
                      'max_bin': 500,
                      'reg_alpha': 0,
                      'reg_lambda': 0,
                      'seed': seed,
                      'n_jobs': -1
                      }

            clf = lgb.train(params, trn_data, 1000000, valid_sets=[trn_data, val_data], verbose_eval=1000,
                            early_stopping_rounds=1000)

            oof[valid_index] += clf.predict(train_sample.iloc[valid_index], num_iteration=clf.best_iteration)
            prediction += clf.predict(test, num_iteration=clf.best_iteration) / n_fold

    oof_LGBM = oof / len(seeds)
    pred_LGBM = prediction / len(seeds)

    print('\nMAE for LGBM ', mean_absolute_error(targets, oof_LGBM))
    return oof_LGBM,pred_LGBM


# Define Neural Network architecture
def get_model(train_sample):

    inp = Input(shape=(1, train_sample.shape[1]))
    x = BatchNormalization()(inp)
    x = LSTM(128, return_sequences=True)(x)  # LSTM as first layer performed better than Dense.
    x = Convolution1D(128, (2), activation='relu', padding="same")(x)
    x = Convolution1D(84, (2), activation='relu', padding="same")(x)
    x = Convolution1D(64, (2), activation='relu', padding="same")(x)

    x = Flatten()(x)

    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)

    # outputs  (3输出的神经网络)
    ttf = Dense(1, activation='relu', name='regressor')(x)  # Time to Failure
    tsf = Dense(1)(x)  # Time Since Failure
    classifier = Dense(1, activation='sigmoid')(x)  # Binary for TTF<0.5 seconds

    model = models.Model(inputs=inp, outputs=[ttf, tsf, classifier])
    opt = optimizers.Nadam(lr=0.008)

    # We are fitting to 3 targets simultaneously: Time to Failure (TTF), Time Since Failure (TSF), and Binary for TTF<0.5 seconds
    # We weight the model to optimize heavily for TTF
    # Optimizing for TSF and Binary TTF<0.5 helps to reduce overfitting, and helps for generalization.
    model.compile(optimizer=opt, loss=['mae', 'mae', 'binary_crossentropy'], loss_weights=[8, 1, 1], metrics=['mae'])
    return model

def train_nn_model(train_sample,targets,test,submission):
    n = 8  # number of NN runs
    oof_final = np.zeros(len(train_sample))
    sub_final = np.zeros(len(submission))
    i = 0

    while i < n:
        print('Running Model ', i + 1)

        oof = np.zeros(len(train_sample))
        prediction = np.zeros(len(submission))

        n_fold = 3
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=1337)
        kf = list(kf.split(np.arange(len(train_sample))))

        for fold_n, (train_index, valid_index) in enumerate(kf):
            # define training and validation sets
            train_x = train_sample.iloc[train_index]           # training set
            valid_x = train_sample.iloc[valid_index]           # validation set

            train_y_ttf = targets['target'].iloc[train_index]  # training target(Time to Failure)
            valid_y_ttf = targets['target'].iloc[valid_index]  # validation target(Time to Failure)

            train_y_tsf = targets['tsf'].iloc[train_index]     # training target(Time Since Failure)
            valid_y_tsf = targets['tsf'].iloc[valid_index]     # validation target(Time Since Failure)

            train_y_clf = targets['binary'].iloc[train_index]  # training target(Binary for TTF<0.5 Secs)
            valid_y_clf = targets['binary'].iloc[valid_index]  # validation target(Binary for TTF<0.5 Secs)

            # apply min max scaler on training, validation data
            train_x, valid_x, test_scaled = normalize(train_x.copy(), valid_x.copy(), test.copy(), 'min_max', [])

            # Reshape training,validation,test data for fitting
            # 1904*1*21
            train_x = train_x.values.reshape(train_x.shape[0], 1, train_x.shape[1])
            # 953*1*21
            valid_x = valid_x.values.reshape(valid_x.shape[0], 1, valid_x.shape[1])
            test_scaled = test_scaled.values.reshape(test_scaled.shape[0], 1, test_scaled.shape[1])

            # obtain Neural Network Instance
            model = get_model(train_sample)

            # setup Neural Network callbacks
            cb_checkpoint = ModelCheckpoint("model.hdf5", monitor='val_regressor_mean_absolute_error',
                                            save_weights_only=True, save_best_only=True, period=1)

            cb_Early_Stop = EarlyStopping(monitor='val_regressor_mean_absolute_error', patience=20)

            cb_Reduce_LR = ReduceLROnPlateau(monitor='val_regressor_mean_absolute_error', factor=0.5, patience=5,
                                             verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

            callbacks = [cb_checkpoint, cb_Early_Stop, cb_Reduce_LR]  # define callbacks set

            ### NN seeds setup- Start
            os.environ['PYTHONHASHSEED'] = '0'
            np.random.seed(1234)
            rn.seed(1234)
            tf.set_random_seed(1234)
            session_conf = tf.ConfigProto(allow_soft_placement=True)
            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            K.set_session(sess)
            ### NN seeds setup- End
            model.fit(train_x, [train_y_ttf, train_y_tsf, train_y_clf],
                      epochs=1000, callbacks=callbacks
                      , batch_size=256, verbose=0,
                      validation_data=(valid_x, [valid_y_ttf, valid_y_tsf, valid_y_clf]))

            model.load_weights("model.hdf5")

            oof[valid_index] += model.predict(valid_x)[0].ravel()
            prediction += model.predict(test_scaled)[0].ravel() / n_fold

            K.clear_session()

        # Obtain the MAE for this run.
        model_score = mean_absolute_error(targets['target'], oof)

        # Sometimes, the NN performs very badly. This happens if the weights are initialized poorly.
        # If the MAE is < 2, then the model has performed correctly, and we will use it in the average.
        if model_score < 2:
            print('MAE: ', model_score, ' Averaged')
            oof_final += oof / n
            sub_final += prediction / n
            i += 1  # Increase i, so we know that we completed a successful run.

        # If the MAE is >= 2, then the NN has performed badly.
        # We will reject this run in the average.
        else:
            print('MAE: ', model_score, ' Not Averaged')

    print('\nMAE for NN: ', mean_absolute_error(targets['target'], oof_final))
    return oof_final, sub_final

def sub(oof_LGBM,sub_LGBM,oof_nn,sub_nn,submission):
    MIN_VALUE = 0.1

    # Correct LGBM predictions
    oof_LGBM[oof_LGBM < MIN_VALUE] = MIN_VALUE
    sub_LGBM[sub_LGBM < MIN_VALUE] = MIN_VALUE

    # Correct NN predictions
    oof_nn[oof_nn < MIN_VALUE] = MIN_VALUE
    sub_nn[sub_nn < MIN_VALUE] = MIN_VALUE

    oof_geomean = (oof_LGBM * oof_nn) ** (1 / 2)
    sub_geomean = (sub_LGBM * sub_nn) ** (1 / 2)

    submission['time_to_failure'] = sub_LGBM
    print(submission.head())
    submission.to_csv('submission_LGBM.csv')

    submission['time_to_failure'] = sub_nn
    print(submission.head())
    submission.to_csv('submission_NN.csv')

    submission['time_to_failure'] = sub_geomean
    print(submission.head())
    submission.to_csv('submission_geomean.csv')


def main():
    # raw = pd.read_csv(PATH + 'input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
    # indices_to_calculate = [i*150_000 for i in range(0,4194,1)]
    # indices_to_calculate = raw.index.values[::150_000][:-1]
    # train = sample_train_gen(raw, indices_to_calculate=indices_to_calculate)
    # train.to_csv('train_feat.csv',index = False)
    # del raw
    # gc.collect()
    # test = sample_test_gen()
    # test.to_csv('test_feat.csv',index = False)

    train = pd.read_csv('train_feat.csv')
    test =pd.read_csv('test_feat.csv')
    #对每段seg对应的所属地震编号
    for i, etq in enumerate(etq_meta):
        train.loc[(train['start'] + 150_000 >= etq["start"]) & (train['start'] <= etq["end"] - 150_000), "eq"] = i
    #对17段地震段编号后，筛除地震段编号TTF时间较小的地震段
    train_sample = train[train["eq"].isin([0,1,2,4,7,9,10,11,13,14])]

    del train
    gc.collect()
    train_sample = train_sample.reset_index(drop=True)
    targets = train_sample[['target', 'start']]
    #将target列-target列对应的数组往下移动一行的值
    targets['tsf'] = targets['target'] - targets['target'].shift(1).fillna(0)
    targets['tsf'] = np.where(targets['tsf'] > 1.5, targets['tsf'], 0)
    targets['tsf'].iloc[0] = targets['target'].iloc[0]

    temp_max = 0
    for i in tqdm(range(targets.shape[0])):
        if targets['tsf'].iloc[i] > 0:
            temp_max = targets['tsf'].iloc[i]
        else:
            targets['tsf'].iloc[i] = temp_max

    targets['tsf'] = targets['tsf'] - targets['target']

    # 创建flag定义目标变量TTF<0.5s
    # This will be used in the NN as an additional objective
    target = targets['target'].copy().values
    target[target >= 0.5] = 1
    target[target < 0.5] = 0
    target = 1 - target

    targets['binary'] = target
    #targets中 ttf、tsf、binary作为三个目标传送给神经网络进行多目标优化
    del target
    gc.collect()

    submission = pd.read_csv(PATH + 'input/sample_submission.csv', index_col='seg_id')
    train_sample.drop(['start', 'target', 'eq'], axis=1, inplace=True)
    test.drop(['seg_id'], axis=1, inplace=True)
    test = test.astype('float64')

    #LGBM训练
    LGBM_feats = [feat for feat in train_sample.columns if 'LGBM' in feat]
    train_sample_lgbm = train_sample[LGBM_feats]
    test_lgbm = test[LGBM_feats]
    oof_LGBM,sub_LGBM = train_lgbm_model(train_sample_lgbm, targets['target'], test_lgbm, submission)

    #NN模型训练
    NN_feats = [feat for feat in train_sample.columns if 'NN' in feat]
    train_sample_nn = train_sample[NN_feats]
    test_nn = test[NN_feats]
    oof_nn,sub_nn = train_nn_model(train_sample_nn,targets,test_nn,submission)

    sub(oof_LGBM, sub_LGBM, oof_nn, sub_nn, submission)

if __name__ == '__main__':
    main()