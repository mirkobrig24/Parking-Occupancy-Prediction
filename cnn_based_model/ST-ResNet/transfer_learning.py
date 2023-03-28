
from __future__ import print_function
import os
import pickle
import numpy as np
import time
import h5py
import math
from sklearn.model_selection import ParameterGrid
from bayes_opt import BayesianOptimization
import json

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from deepst.models.STResNet import stresnet
import deepst.metrics as metrics
from deepst.datasets import Parking
from deepst.evaluation import evaluate

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
# parameters
DATAPATH = '../../results_one_month/feat_CNN_sosta_media_contemporanea_1h.h5'
task = 'sosta_media_contemporanea'
nb_epoch = 150  # number of epoch at training stage
# nb_epoch_cont = 150  # number of epoch at training (cont) stage
batch_size = [16, 32, 64]  # batch size
T = 24  # number of time intervals in one day (For 8H is 3 and for 1H is 24)
CACHEDATA = False  # cache data or NOT

lr = [0.001, 0.001]  # learning rate
len_closeness = 12  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 0  # length of trend dependent sequence
#len_cpt = [[12,2,2]]
nb_residual_unit = [2,4,6]   # number of residual units

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


# load best parameter

params_fname = f'stresnet_{task}_best_params.json'
with open(os.path.join('results', params_fname), 'r') as f:
     params = json.load(f)

residual_units = int(params['residual_units']) * 2
batch_size = 2 ** int(params['batch_size'])
seq_len = int(params['seq_len'])
len_period = int(params['len_period'])
len_trend = int(params['len_trend'])
lr = round(params['lr'],5)

def build_model(len_closeness, len_period, len_trend, nb_flow, map_height, map_width,
                external_dim, nb_residual_unit, bn, bn2=False, save_model_pic=False, lr=0.00015):
    c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    model = stresnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                     external_dim=external_dim, nb_residual_unit=nb_residual_unit, bn=bn, bn2=bn2)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    # model.summary()
    if (save_model_pic):
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='park_model.png', show_shapes=True)

    return model

X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = Parking.load_data(
        T=T, nb_flow=nb_flow, len_closeness=seq_len, len_period=len_period, len_trend=len_trend, len_test=len_test,
        meta_data=True, meteorol_data=False, holiday_data=False, datapath=DATAPATH)

    # build model
tf.keras.backend.set_image_data_format('channels_first')
model = build_model(seq_len, len_period, len_trend, nb_flow, map_height,
                        map_width, external_dim, residual_units,
                        bn=True,
                        bn2=True,
                        save_model_pic=False,
                        lr=lr
                        )


## single-step-prediction no TL
nb_epoch = 150
hyperparams_name = f'PREDCNN_transfer_learning_{task}'
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
model_fname = 'parking_sosta_media_contemporanea_0.c4.p3.t1.resunits_4.lr_0.0005.batchsize_8.best.h5'
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