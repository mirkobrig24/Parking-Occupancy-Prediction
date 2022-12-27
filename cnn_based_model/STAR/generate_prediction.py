'''
Questo codice permette di estrapolare 

'''

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
from star import TaxiNYC
from star.evaluation import evaluate

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

# Configurazione per estrarre predetti e reali per la notte
def estrazione_notte(x):
    print(len(x))
    for i in range(0, len(x), 24):
        if i == 0:
            b = x[i:i+6]
            b = np.concatenate((b, x[i+22:i+25]), axis = 0)
        else:
            b = np.concatenate((b, x[i:i+6]), axis = 0)
            b = np.concatenate((b, x[i+22:i+25]), axis = 0)
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
days_test = 122
len_test = int((T*days_test) * 0.2)
len_val = 2*len_test

map_height, map_width = 54, 43  # grid size

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
fname = os.path.join(path_cache, 'TaxiNYC_C{}_P{}_T{}.h5'.format(len_c, len_p, len_t))

#if os.path.exists(fname) and CACHEDATA:
#    X_train_all, Y_train_all, X_train, Y_train, \
#    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
#    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = read_cache(
#        fname)
#    print("load %s successfully" % fname)
#else:
def data(T, nb_flow, len_c, len_p, len_t, len_test, len_val, DATAPATH):
    X_train_all, Y_train_all,X_test, Y_test, mmn, external_dim, \
    timestamp_train_all,  timestamp_test = TaxiNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_c, len_period=len_p, len_trend=len_t, len_test=len_test,
        len_val=len_val, meta_data=True, meteorol_data=False, holiday_data=False, datapath=DATAPATH)
    return  X_train_all, Y_train_all, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_test

params_fname = f'star_parking_best_params_{T}.json'
with open(os.path.join('results', params_fname), 'r') as f:
     params = json.load(f)

residual_units=params['residual_units']
batch_size=params['batch_size']
len_c = params['len_c']
len_p = params['len_p']
len_t = params['len_t']
print()

residual_units = int(residual_units) * 2
batch_size = 2 ** int(batch_size)
    #kernel_size = int(kernel_size)
lr = 0.001 #round(lr,5)
len_c = int(len_c)
len_p = int(len_p)
len_t = int(len_t)

print(residual_units)
print(batch_size)
print(len_c)

    #len_p = int(len_p)
    #len_t = int(len_t)
X_train_all, Y_train_all, X_test, Y_test, mmn, external_dim, \
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

# load weights
model_fname = '/home/gpu2/Documenti/Parking-Occupancy-Prediction/STAR/MODEL/parking.c8.p5.t2.resunits_6.lr_0.001.batchsize_8.best.h5'
model.load_weights( model_fname)

Y_pred = model.predict(X_test)  # compute predictions

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

print('score fascia notte')
Y_test3 = estrazione_notte(len_test, Y_test)
Y_pred3 = estrazione_notte(len_test, Y_pred)

score = evaluate(Y_test3, Y_pred3, mmn, rmse_factor=1)  # evaluate performance
print('Test rmse: %.6f mape: %.6f ' %
          (score[0], score[1]))


# save real vs predicted
#fname = 'STAR_parking_realVSpred.h5'
#h5 = h5py.File(fname, 'w')
#h5.create_dataset('Y_real', data=Y_test)
#h5.create_dataset('Y_pred', data=Y_pred)
#h5.create_dataset('timestamps', data=timestamp_test)
#h5.create_dataset('max', data=mmn._max)
#h5.close()