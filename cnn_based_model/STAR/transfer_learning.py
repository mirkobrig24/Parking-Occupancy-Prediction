# Attenzione nessun fattore esterno!


from __future__ import print_function
import os
import _pickle as pickle
import numpy as np
import math
import h5py
import json
import time
from sklearn.model_selection import ParameterGrid

from keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from bayes_opt import BayesianOptimization

from star.model import *
import star.metrics as metrics
from star import Parking
from star.evaluation import evaluate

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
nb_epoch = 1  # number of epoch at training stage
# nb_epoch_cont = 150  # number of epoch at training (cont) stage
batch_size = [16, 32, 64]  # batch size
T = 24  # number of time intervals in one day
CACHEDATA = False  # cache data or NOT

lr = 0.001  # learning rate
len_c = 1  # length of closeness dependent sequence
len_p = 0  # length of peroid dependent sequence
len_t = 0  # length of trend dependent sequence
#len_cpt = [[1,0,0]]
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

path_cache = os.path.join(DATAPATH, 'CACHE', 'STAR')  # cache path
path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

def build_model(len_c, len_p, len_t, nb_flow, map_height, map_width,
                external_dim, nb_residual_unit, bn, bn2=False, save_model_pic=False, lr=0.001):

    c_conf = (len_c, nb_flow, map_height,
              map_width) if len_c > 0 else None
    p_conf = (len_p, nb_flow, map_height,
              map_width) if len_p > 0 else None
    t_conf = (len_t, nb_flow, map_height,
              map_width) if len_t > 0 else None
    model = STAR(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                     external_dim=external_dim, nb_residual_unit=nb_residual_unit, bn=bn, bn2=bn2)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    #model.summary()
    #exit()
    if (save_model_pic):
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='park_model.png', show_shapes=True)

    return model
fname = os.path.join(path_cache, 'Parking_{}_P{}_T{}.h5'.format(len_c, len_p, len_t))


def data(T, nb_flow, len_c, len_p, len_t, len_test, len_val, DATAPATH):
    X_train, Y_train,X_test, Y_test, mmn, external_dim, \
    timestamp_train_all,  timestamp_test = Parking.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_c, len_period=len_p, len_trend=len_t, len_test=len_test,
        len_val=len_val, meta_data=False, meteorol_data=False, holiday_data=False, datapath=DATAPATH)
    return  X_train, Y_train, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_test

params_fname = f'star_{task}_best_params.json'
with open(os.path.join('results', params_fname), 'r') as f:
    params = json.load(f)


## single-step-prediction no TL
nb_epoch = 150
hyperparams_name = f'PREDCNN_transfer_learning_{task}'
residual_units = int(params['residual_units']) * 2
batch_size = 2 ** int(params['batch_size'])
#kernel_size = int(kernel_size)
lr = round(params['lr'],5)
len_c = int(params['len_c'])
len_p = int(params['len_p'])
len_t = int(params['len_t'])
X_train, Y_train, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_test = data(T, nb_flow, len_c, len_p, len_t, len_test, len_val, DATAPATH)
# build model
tf.keras.backend.set_image_data_format('channels_first')
model = build_model(len_c, len_p, len_t, nb_flow, map_height,
                    map_width, external_dim, residual_units,
                    bn=True,
                    bn2=True,
                    save_model_pic=False,
                    lr=lr
                    )

## single-step-prediction no TL
nb_epoch = 150
hyperparams_name = f'star_transfer_learning_{task}'
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
model_fname = 'parking_mean_time_0.c8.p5.t1.resunits_6.lr_0.0007.batchsize_8.best.h5'
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