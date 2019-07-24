import numpy as np
import pandas as pd
import os
from tqdm import tqdm
# Define model
from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU, Dropout, TimeDistributed, LSTM, CuDNNLSTM
from keras.optimizers import adam, RMSprop
from keras.callbacks import ModelCheckpoint
# Fix seeds
from numpy.random import seed

seed(639)
from tensorflow import set_random_seed

set_random_seed(5944)

from numpy.random import seed

seed(639)
from tensorflow import set_random_seed

set_random_seed(5944)

PATH = "E:/kaggle/kaggle-Lanl_Earthquake_Prediction/"

float_data = pd.read_csv(PATH + "input/train.csv",
                         dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}).values

def extract_features(z):
    return np.c_[z.mean(axis=1),z.min(axis=1),z.max(axis=1),z.std(axis=1)]   #相当concat(axis=1)  np.r_ concat(axis=0)

def create_X(x, last_index=None, n_steps=150, step_length=1000):
    if last_index == None:
        last_index = len(x)

    assert last_index - n_steps * step_length >= 0

    # Reshaping and approximate standardization with mean 5 and std 3.
    temp = (x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1) - 5) / 3    # 总段长150k
                                                                                  # 共产生150个小段 每段长1000计算统计特征
    return np.c_[extract_features(temp),
                 extract_features(temp[:, :100]),  # 最前100个
                 extract_features(temp[:, :10]),   # 最前10个
                 extract_features(temp[:, :3]),    # 最前3个
                 extract_features(temp[:, -step_length // 10:]),    #最后100个
                 extract_features(temp[:, -step_length // 100:]),   #最后10个
                 extract_features(temp[:, -step_length // 300:])]   #最后3个

n_features = create_X(float_data[0:150000]).shape[1]
print("Our RNN is based on %i features" % n_features)

def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000):   #训练样本 测试样本 生成器
    if max_index is None:
        max_index = len(data) - 1

    while True:
        rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)

        samples = np.zeros((batch_size, n_steps, n_features))
        targets = np.zeros(batch_size, )

        for j, row in enumerate(rows):

            samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)  #
            targets[j] = data[row - 1, 1]
        yield samples, targets

batch_size = 32

second_earthquake = 50085877
float_data[second_earthquake, 1]

train_gen = generator(float_data, batch_size = batch_size)
valid_gen = generator(float_data, batch_size = batch_size, max_index = second_earthquake)   #验证集

print("LSTM feat nums %i features"% n_features)

cb = [ModelCheckpoint("model.hdf5", save_best_only=True, period=3)]

# The LSTM architecture
model = Sequential()
# First LSTM layer with Dropout regularisation
model.add(CuDNNLSTM(units=50, return_sequences=True, input_shape=(None,n_features)))
model.add(Dropout(0.2))
# Second LSTM layer
model.add(CuDNNLSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Third LSTM layer
model.add(CuDNNLSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Fourth LSTM layer
model.add(CuDNNLSTM(units=50))
model.add(Dropout(0.2))
# The output layer
model.add(Dense(units=1))

model.summary()

model.compile(optimizer='rmsprop',loss='mae')

history = model.fit_generator(train_gen,
                              steps_per_epoch=1000,
                              epochs=100,
                              verbose=2,
                              callbacks=cb,
                              validation_data=valid_gen,
                              validation_steps=200)

import matplotlib.pyplot as plt

def perf_plot(history, what='loss'):
    x = history.history[what]
    val_x = history.history['val_' + what]
    epochs = np.asarray(history.epoch) + 1

    plt.plot(epochs, x, 'bo', label="Training " + what)
    plt.plot(epochs, val_x, 'b', label="Validation " + what)
    plt.title("Training and validation " + what)
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig('train_produre.png')
    return None

perf_plot(history)  # Extra Layer

# Load submission file
submission = pd.read_csv(PATH + 'input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})
x = None
# Load each test data, create the feature matrix, get numeric prediction
for i, seg_id in enumerate(tqdm(submission.index)):
  #  print(i)
    seg = pd.read_csv(PATH + 'input/test/' + seg_id + '.csv')
    x = seg['acoustic_data'].values[:]
    #print(x.shape)
    submission.time_to_failure[i] = (model.predict(np.expand_dims(create_X(x), 0)))

# Save
submission.to_csv(PATH + 'result/submission_LSTM.csv')