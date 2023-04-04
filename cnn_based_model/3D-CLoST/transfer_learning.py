
from __future__ import print_function
import os
import _pickle as pickle
import numpy as np
import math
import h5py
import time
import json
from bayes_opt import BayesianOptimization

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from src.model import build_model
import src.metrics as metrics
from src.datasets import Parking
from src.evaluation import evaluate
from cache_utils import cache, read_cache

def save_to_csv(score, csv_name):
    if not os.path.isfile(csv_name):
        if os.path.isdir('results') is False:
            os.mkdir('results')
            with open(csv_name, 'a', encoding="utf-8") as file:
                file.write('rmse_tot,'
                           'mae_tot'
                           )
                file.write("\n")
                file.close()
        with open(csv_name, 'a', encoding="utf-8") as file:
            file.write(f'{score[0]},{score[1]}')
            file.write("\n")
            file.close()
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:  # Currently, memory growth needs to be the same across GPUs
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)  # Memory growth must be set before GPUs have been initialized

np.random.seed(1234)
tf.random.set_seed(1234)

# parameters
DATAPATH = '../../results_one_month/feat_CNN_mean_time_1h.h5'
task = 'mean_time'
# nb_epoch = 150  # number of epoch at training stage
# nb_epoch_cont = 150  # number of epoch at training (cont) stage
# batch_size = 64  # batch size
T = 24  # number of time intervals in one day
CACHEDATA = False  # cache data or NOT

len_closeness = len_c = 2  # length of closeness dependent sequence
len_period = len_p = 0  # length of peroid dependent sequence
len_trend = len_t = 1  # length of trend dependent sequence
# len_cpt = [[2,0,1]]
# batch_size = [16, 64]  # batch size
# lr = [0.0015, 0.00015]  # learning rate
# lstm = [350, 500]
# lstm_number = [2, 3]

nb_flow = 1  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test,
days_test = 10
len_test = T*days_test
len_val = len_test

map_height, map_width = 54, 43  # grid size

#path_cache = os.path.join(DATAPATH, 'CACHE', '3D-CLoST')  # cache path
#if CACHEDATA and os.path.isdir(path_cache) is False:
#    os.mkdir(path_cache)

# load data
print("loading data...")
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)


# load data
print("loading data...")
X_train, Y_train, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask = Parking.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_c, len_period=len_p, len_trend=len_t, len_test=len_test,
        len_val=len_val, preprocess_name='preprocessing_parking.pkl', meta_data=True, datapath=DATAPATH, task=task)

# load best parameter

params_fname = f'3dclost_parking_{task}_best_params.json'
with open(os.path.join('results', params_fname), 'r') as f:
     params = json.load(f)

# build model
lstm = 350 if params['lstm'] < 1 else 500
lstm_number = int(params['lstm_number'])
batch_size = 16 if params['batch_size'] < 2 else 64
lr = round(params['lr'],5)
# build model
model = build_model('BJ', X_train,  Y_train, conv_filt=64, kernel_sz=(2,3,3), 
                mask=mask, lstm=lstm, lstm_number=lstm_number, add_external_info=True,
                lr=lr, save_model_pic=None, nb_flow = nb_flow)


## single-step-prediction no TL
nb_epoch = 150
hyperparams_name = f'3dclost_transferlearning_{task}'
fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
history = model.fit(X_train, Y_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    validation_data=(X_test, Y_test),
                    callbacks=[model_checkpoint],
                    verbose=2)

# predict
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'No_transfer_learning_{task}.csv')
save_to_csv(score, csv_name)


## TL without re-training
# load weights
model_fname = 'Parking_mean_time_0.c2.p0.t1.lstm_350.lstmnumber_2.lr_0.00033.batchsize_16.best.h5'
model.load_weights(os.path.join('MODEL', model_fname))

# predict
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'Transfer_learning_{task}_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = f'transfer_learning_{task}.h5'
h5 = h5py.File(fname, 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()

## TL with re-training

hyperparams_name = f'Transfer_learning_{task}'
fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
history = model.fit(X_train, Y_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    validation_data=(X_test, Y_test),
                    callbacks=[model_checkpoint],
                    verbose=2)

# evaluate after training
model.load_weights(fname_param)
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred, mmn)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'Transfer_learning_training_{task}_results.csv')
save_to_csv(score, csv_name)

# save real vs predicted
fname = f'transfer_learning_{task}_trained.h5'
h5 = h5py.File(fname, 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()