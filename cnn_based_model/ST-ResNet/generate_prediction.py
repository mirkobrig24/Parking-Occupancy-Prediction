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
from deepst.datasets import TaxiNYC
from deepst.evaluation import evaluate



# Configurazione per estrarre predetti e reali per il pomeriggio
def estrazione_pomeriggio(len_test,x):
    for i in range(0, (len_test), 24):
        if i == 0:
            b = x[i:i+8]
        else:
            b = np.concatenate((b, x[i:i+8] ), axis=0)
    return b

# Configurazione per estrarre predetti e reali per la mattina
def estrazione_mattina(len_test,x):
    for i in range(16, (len_test), 24):
        if i == 16:
            b = x[i:i+8]
        else:
            b = np.concatenate((b, x[i:i+8] ), axis=0)
    return b


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:  # Currently, memory growth needs to be the same across GPUs
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)  # Memory growth must be set before GPUs have been initializ

# parameters
DATAPATH = '../results/input_1h.h5'
nb_epoch = 200  # number of epoch at training stage
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
days_test = 122
len_test = int((T*days_test) * 0.2)
len_val = 2*len_test

map_height, map_width = 54, 43  # grid size

path_cache = os.path.join(DATAPATH, 'CACHE', 'ST-ResNet')  # cache path
path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

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


params_fname = f'stresnet_parking_best_params_{T}.json'
with open(os.path.join('results', params_fname), 'r') as f:
    params = json.load(f)

residual_units=params['residual_units']
                #lr=params['lr'],
batch_size=params['batch_size']
seq_len=params['seq_len']
len_period=params['len_period']
len_trend =params['len_trend']



residual_units = int(residual_units) * 2
batch_size = 2 ** int(batch_size)
seq_len = int(seq_len)
len_period = int(len_period)
len_trend = int(len_trend)
    # kernel_size = int(kernel_size)
lr = 0.001 #round(lr,5)

X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = TaxiNYC.load_data(
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
# load weights
model_fname = '/home/gpu2/Documenti/Parking-Occupancy-Prediction/ST-ResNet/MODEL/parking.c7.p2.t2.resunits_2.lr_0.001.batchsize_8.best.h5'
model.load_weights( model_fname)

Y_pred = model.predict(X_test)  # compute predictions


# save real vs predicted
#fname = 'ST_parking_realVSpred.h5'
#h5 = h5py.File(fname, 'w')
#h5.create_dataset('Y_real', data=Y_test)
#h5.create_dataset('Y_pred', data=Y_pred)
#h5.create_dataset('timestamps', data=timestamp_test)
#h5.create_dataset('max', data=mmn._max)
#h5.close()

print('score fascia mattutina')

Y_test1 = estrazione_mattina(len_test, Y_test)
Y_pred1 = estrazione_mattina(len_test, Y_pred)

score = evaluate(Y_test1, Y_pred1, mmn, rmse_factor=1)  # evaluate performance
print('Test rmse: %.6f mape: %.6f ' %
          (score[0], score[1]))


print('score fascia pomeridiana')
Y_test2 = estrazione_pomeriggio(len_test, Y_test)
Y_pred2 = estrazione_pomeriggio(len_test, Y_pred)

score = evaluate(Y_test2, Y_pred2, mmn, rmse_factor=1)  # evaluate performance
print('Test rmse: %.6f mape: %.6f ' %
          (score[0], score[1]))
