import numpy as np
import time
import os
import json
import pickle as pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import tensorflow as tf
from keras import backend as K
import h5py

from utils import cache, read_cache
from src import Parking3d
from src.evaluation import evaluate
from src import streednet 
models_dict = {
    'STREED-Net': streednet
}


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


model_name = 'STREED-Net'

# parameters

DATAPATH = '../../results_one_month/feat_CNN_mean_time_1h.h5'
task = 'mean_time'
nb_epoch = 150  # number of epoch at training stage
T = 24  # number of time intervals in one day
CACHEDATA = False  # cache data or NOT

len_closeness = 4  # length of closeness dependent sequence
len_period = 2  # length of peroid dependent sequence
len_trend = 0  # length of trend dependent sequence

nb_flow = 1  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test,
days_test = 10
len_test = T*days_test
len_val = len_test

map_height, map_width = 54, 43  # grid size

#path_cache = os.path.join(DATAPATH, 'CACHE', '3D-CLoST')  # cache path
#if CACHEDATA and os.path.isdir(path_cache) is False:
#    os.mkdir(path_cache)

cache_folder = 'STREED-Net' #if model_name in ['model3', 'model3attention', 'model3resunit', 'model3resunit_attention', 'model3resunit_doppia_attention'] else 'Autoencoder'
path_cache = os.path.join(DATAPATH, 'CACHE', cache_folder)  # cache path
path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)
if os.path.isdir('results') is False:
    os.mkdir('results')

# load data
print("loading data...")
fname = os.path.join(path_cache, 'Parking_C{}_P{}_T{}.h5'.format(
    len_closeness, len_period, len_trend))
if os.path.exists(fname) and CACHEDATA:
    X_train, Y_train, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = read_cache(
        fname, 'preprocessing_bj.pkl')
    print("load %s successfully" % fname)
else:
    #if (model_name.startswith('model3')):
    #else:
    #    load_data = TaxiBJ.load_data
    X_train, Y_train, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = Parking3d.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        len_val=len_val, preprocess_name='preprocessing_parking.pkl', meta_data=True, meteorol_data=False, holiday_data=False, datapath=DATAPATH)
    if CACHEDATA:
        cache(fname, X_train, Y_train, X_train, Y_train, X_val, Y_val, X_test, Y_test,
              external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test)

print(external_dim)
print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])


# load best parameter

params_fname = f'STREED-Net_Parking_{task}_params.json'
with open(os.path.join('results', params_fname), 'r') as f:
     params = json.load(f)

encoder_blocks = int(params['encoder_blocks'])
batch_size = 16 * int(params['batch_size'])
kernel_size = int(params['kernel_size'])
lr = round(params['lr'],5)
filters = [64,64,64,64,16]
# build model
m = models_dict[model_name]
model = m.build_model(
    len_closeness, len_period, len_trend, nb_flow, map_height, map_width,
    external_dim=external_dim, lr=lr,
    encoder_blocks=encoder_blocks,
    filters=filters,
    kernel_size=kernel_size,
    num_res=2,
    # save_model_pic=f'TaxiBJ_{model_name}'
)

## single-step-prediction no TL

hyperparams_name = f'streed_net_transfer_learning_{task}'
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
model_fname = 'STREED-Net.TaxiNYC_mean_time_0.c4.p2.t0.encoderblocks_2.kernel_size_4.lr_0.00086.batchsize_16.best.h5'
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