import gc
import os
import time
import logging
import datetime
import warnings
import numpy as np
import pandas as pd

import re
import os
from tqdm import tqdm
import librosa, librosa.display

import builtins

random_seed = 4126
import matplotlib.pyplot as plt

cast = {
    'acoustic_data': 'int',
    'time_to_failure': 'float'
}

def df_fragments(path, length, skip=1):
    with open(path, 'r') as f:
        m = {}
        cols = []
        count = 0
        index = 0
        for line in f:
            if len(cols) == 0:
                for col in line.strip("\n\r ").split(','):
                    cols.append(col)
                continue
            if count == 0:
                for col in cols:
                    m[col] = []
            if index % skip == 0:
                for j, cell in enumerate(line.strip("\n\r ").split(',')):
                    col = cols[j]
                    m[col].append(getattr(builtins, cast[col])(cell))
            count += 1
            if count == length:
                if index % skip == 0:
                    yield pd.DataFrame(m)
                index += 1
                count = 0

def count_rows(path):
    with open(path, 'r') as f:
        i = -1
        for _ in f:
            i += 1
        return i

def main():

    PATH = 'E:/kaggle/kaggle-Lanl_Earthquake_Prediction'
    print('counting total...')
    total = 629145480
    print('total: {}'.format(total))

    print('generating train data...')
    fragment_size = 150000
    skip = 1

    mfcc_ttf_map = {}

    for df in tqdm(df_fragments(PATH + '/input/train.csv', length=fragment_size, skip=skip), total=(total // fragment_size) // skip):  #df为从0-620m每150k取段

        mfcc = librosa.feature.mfcc(df['acoustic_data'].values.astype('float32'),sr=50000,n_mfcc=40)
        mfcc_mean = mfcc.mean(axis=1)
        for i, each_mfcc_mean in enumerate(mfcc_mean):
            key = 'mfcc_{}'.format(i)
            if key not in mfcc_ttf_map:
                mfcc_ttf_map[key] = []
            mfcc_ttf_map[key].append(each_mfcc_mean)
        key = 'time_to_failure'
        if key not in mfcc_ttf_map:
            mfcc_ttf_map[key] = []
        mfcc_ttf_map[key].append(df.iloc[-1][key])

    mfcc_ttf_df = pd.DataFrame(mfcc_ttf_map)
    fname = 'mfcc_train40.csv'
    mfcc_ttf_df.to_csv(fname, index=False)
    print('saved {}.'.format(fname))

    print('generating test features...')
    test_dir = PATH + '/input/test'
    test_map = {}
    for f in tqdm(os.listdir(test_dir)):
        path = test_dir + '/' + f
        df = pd.read_csv(path, delimiter=',',
                         error_bad_lines=False)
        mfcc = librosa.feature.mfcc(df['acoustic_data'].values.astype('float32'),n_mfcc=40)
        mfcc_mean = mfcc.mean(axis=1)
        for i, each_mfcc_mean in enumerate(mfcc_mean):
            key = 'mfcc_{}'.format(i)
            if key not in test_map:
                test_map[key] = []
            test_map[key].append(each_mfcc_mean)
        key = 'seg_id'
        if key not in test_map:
            test_map[key] = []
        test_map[key].append(re.sub('.csv$', '', f))
    test_df = pd.DataFrame(test_map)
    test_csv = 'mfcc_test40.csv'
    test_df.to_csv(test_csv, index=False)
    print('saved {}'.format(test_csv))

if __name__ == '__main__':
    main()