import gc
import os
import time
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

PATH = 'E:/kaggle/kaggle-Lanl_Earthquake_Prediction/'

train_X = pd.read_csv(PATH + 'input/masters-final-project/train_x_%d.csv' % 0)
train_y = pd.read_csv(PATH + 'input/masters-final-project/train_y_%d.csv' % 0)
test_X = pd.read_csv(PATH + 'input/masters-final-project/test_x.csv')
for i in range(1, 6):
    temp = pd.read_csv(PATH + 'input/masters-final-project/train_x_%d.csv' % i)
    train_X = train_X.append(temp)
    temp = pd.read_csv(PATH + 'input/masters-final-project/train_y_%d.csv' % i)
    train_y = train_y.append(temp)
uniq = train_X.groupby(['seg_start']).count()

a = 1