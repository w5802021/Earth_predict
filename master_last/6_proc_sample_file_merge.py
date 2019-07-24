
import pandas as pd


from tqdm import tqdm



SIG_LEN = 150000
NUM_SEG_PER_PROC = 4000
NUM_THREADS = 6

NY_FREQ_IDX = 75000  # the test signals are 150k samples long, Nyquist is thus 75k.
CUTOFF = 18000
MAX_FREQ_IDX = 20000
FREQ_STEP = 2500

PATH = "E:/kaggle/kaggle-Lanl_Earthquake_Prediction/master_last/"

def join_mp_build():
    train_X = pd.read_csv(PATH + 'train_x_%d.csv' % 0)
    train_y = pd.read_csv(PATH + 'train_y_%d.csv' % 0)
    # test_X = pd.read_csv(PATH + 'master_last/test_x.csv')
    for i in tqdm(range(1, NUM_THREADS)):
        temp = pd.read_csv(PATH + 'train_x_%d.csv' % i)
        train_X = train_X.append(temp)
        temp = pd.read_csv(PATH + 'train_y_%d.csv' % i)
        train_y = train_y.append(temp)
    train_X.to_csv(PATH + 'train_x.csv', index=False)
    train_y.to_csv(PATH + 'train_y.csv',index=False)

join_mp_build()
