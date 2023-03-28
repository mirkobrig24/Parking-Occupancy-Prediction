from ST3DNet import *
import pickle
from utils import *
import os
import json
import time
from bayes_opt import BayesianOptimization
import math
import h5py
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import ParameterGrid

from evaluation import evaluate

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

np.random.seed(1234)
tf.random.set_seed(1234)

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


# parameters
# parameters
T = 24  # number of time intervals in one day
# lr = [0.00015, 0.00035]  # learning rate
# lr = 0.00002  # learning rate
len_closeness = len_c =  6  # length of closeness dependent sequence
len_period = len_p = 0  # length of peroid dependent sequence
len_trend = len_t = 1  # length of trend dependent sequence
# nb_residual_unit = [4,5,6]   # number of residual units
nb_flow = 1  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test,
days_test = 10
len_test = T*days_test
len_val = len_test
map_height, map_width = 54, 43  # grid size
nb_epoch = 150  # number of epoch at training stage

path_result = 'RET'
path_model = 'MODEL'
# parameters
task = 'sosta_media_contemporanea'

#path_cache = os.path.join(DATAPATH, 'CACHE', '3D-CLoST')  # cache path
#if CACHEDATA and os.path.isdir(path_cache) is False:
#    os.mkdir(path_cache)

filename = os.path.join('CACHE', f'{task}_one_month', 'Parking_c%d_p%d_t%d'%(len_closeness, len_period, len_trend))
f = open(filename, 'rb')
X_train = pickle.load(f)
Y_train = pickle.load(f)
X_test = pickle.load(f)
Y_test = pickle.load(f)
mmn = pickle.load(f)
external_dim = pickle.load(f)
timestamp_train = pickle.load(f)
timestamp_test = pickle.load(f)


for i in X_train:
    print(i.shape)
Y_train = mmn.inverse_transform(Y_train)  # X is MaxMinNormalized, Y is real value
Y_test = mmn.inverse_transform(Y_test)
c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None

t_conf = (len_trend, nb_flow, map_height,
          map_width) if len_trend > 0 else None


# load best parameter

params_fname = f'st3dnet_Parking_{task}_best_params.json'
with open(os.path.join('results', params_fname), 'r') as f:
     params = json.load(f)

residual_units = int(params['residual_units'])
batch_size = 16 * int(params['batch_size'])
    # kernel_size = int(kernel_size)
lr = round(params['lr'],5)

model = ST3DNet(c_conf=c_conf, t_conf=t_conf, external_dim=external_dim,
                        nb_residual_unit=residual_units)
adam = Adam(lr=lr)
model.compile(loss='mse', optimizer=adam, metrics=[rmse])
hyperparams_name = 'Parking_{}_{}.c{}.p{}.t{}.resunits_{}.lr_{}.batchsize_{}'.format(
      task, i, len_c, len_p, len_t, residual_units,
        lr, batch_size)


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
model_fname = 'Parking_sosta_media_contemporanea_0.c6.p0.t1.resunits_6.lr_0.00042.batchsize_16.best.h5'
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