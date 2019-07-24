import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

PATH = 'E:\kaggle\kaggle-Lanl_Earthquake_Prediction'

Xtrain = pd.read_csv(PATH + '/data_sample/train_features.csv')
Ytrain = pd.read_csv(PATH + '/data_sample/y.csv')
Xtest = pd.read_csv(PATH + '/data_sample/test_features.csv')

scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = pd.DataFrame(scaler.transform(Xtrain), columns=Xtrain.columns)

submission = pd.read_csv(PATH + '/input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

X_train, X_val, Y_train, Y_val = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=5)

#Model parameters

kernel_init = 'he_normal'
input_size = len(Xtrain.columns)

# Neural Network #
# Model architecture: A very simple shallow Neural Network
model = Sequential()
model.add(Dense(32, input_dim = input_size))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('linear'))

#compile the model
optim = optimizers.Adam(lr = 0.001)
model.compile(loss = 'mean_absolute_error', optimizer = optim)

#Callbacks
# csv_logger = CSVLogger('log.csv', append=True, separator=';')
best_model = ModelCheckpoint("model_5000+.hdf5", save_best_only=True, period=3)
restore_best = EarlyStopping(monitor='val_loss', verbose=2, patience=100, restore_best_weights=True)

model.fit(x=X_train, y=Y_train, batch_size=64, epochs=2000, verbose=2, callbacks=[best_model], validation_data=(X_val,Y_val))
### Neural Network End ###

nn_predictions = model.predict(Xtest, verbose = 2, batch_size = 64)
submission['time_to_failure'] = nn_predictions
submission.to_csv(PATH + 'result/submission_NN5000+.csv')