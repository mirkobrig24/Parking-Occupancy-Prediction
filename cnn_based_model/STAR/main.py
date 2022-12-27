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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:  # Currently, memory growth needs to be the same across GPUs
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)  # Memory growth must be set before GPUs have been initializ


np.random.seed(1234)
tf.random.set_seed(1234)

# parameters
DATAPATH = '../../results/feat_sosta_media_contemporanea_1h.h5'
task = 'sosta_media_contemporanea'
nb_epoch = 150  # number of epoch at training stage
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
fname = os.path.join(path_cache, 'Parking_{}_P{}_T{}.h5'.format(len_c, len_p, len_t))

#if os.path.exists(fname) and CACHEDATA:
#    X_train_all, Y_train_all, X_train, Y_train, \
#    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
#    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = read_cache(
#        fname)
#    print("load %s successfully" % fname)
#else:
def data(T, nb_flow, len_c, len_p, len_t, len_test, len_val, DATAPATH):
    X_train_all, Y_train_all,X_test, Y_test, mmn, external_dim, \
    timestamp_train_all,  timestamp_test = Parking.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_c, len_period=len_p, len_trend=len_t, len_test=len_test,
        len_val=len_val, meta_data=False, meteorol_data=False, holiday_data=False, datapath=DATAPATH)
    return  X_train_all, Y_train_all, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_test


#print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])

def train_model(lr, batch_size, residual_units, len_c, len_p, len_t, save_results=False, i=''): #lr
    # get discrete parameters
    residual_units = int(residual_units) * 2
    batch_size = 2 ** int(batch_size)
    #kernel_size = int(kernel_size)
    lr = round(lr,5)
    len_c = int(len_c)
    len_p = int(len_p)
    len_t = int(len_t)
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
    hyperparams_name = 'parking_{}_{}.c{}.p{}.t{}.resunits_{}.lr_{}.batchsize_{}'.format(
        task, i, len_c, len_p, len_t, residual_units,
        lr, batch_size)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=25, mode='min')
    # lr_callback = LearningRateScheduler(lrschedule)
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    # train model
    print("training model...")
    ts = time.time()
    if (i):
        print(f'Iteration {i}')
        np.random.seed(i * 18)
        tf.random.set_seed(i * 18)

    history = model.fit(X_train_all, Y_train_all,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        validation_data=(X_test, Y_test),
                        # callbacks=[early_stopping, model_checkpoint],
                        # callbacks=[model_checkpoint, lr_callback],
                        callbacks=[model_checkpoint],
                        verbose=2)
    model.save_weights(os.path.join(
        'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    # evaluate
    model.load_weights(fname_param)
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))

    if (save_results):
        print('evaluating using the model that has the best loss on the valid set')
        model.load_weights(fname_param)  # load best weights for current iteration

        Y_pred = model.predict(X_test)  # compute predictions

        score = evaluate(Y_test, Y_pred, mmn, rmse_factor=1)  # evaluate performance

        # save to csv
        csv_name = os.path.join('results', f'parking_{task}_results.csv')
        if not os.path.isfile(csv_name):
            if os.path.isdir('results') is False:
                os.mkdir('results')
            with open(csv_name, 'a', encoding="utf-8") as file:
                file.write('iteration,'
                           'rmse_tot,'
                           'mae_tot'
                           )
                file.write("\n")
                file.close()
        with open(csv_name, 'a', encoding="utf-8") as file:
            file.write(f'{i},{score[0]},{score[1]}')
            file.write("\n")
            file.close()
        K.clear_session()
    
    # bayes opt is a maximization algorithm, to minimize validation_loss, return 1-this
    bayes_opt_score = 1.0 - score[0]

    return bayes_opt_score

# bayesian optimization
optimizer = BayesianOptimization(f=train_model,
                                 pbounds={'residual_units': (1, 3.999), # *2
                                          'lr': (0.0001, 0.001),
                                          'batch_size': (3, 5.999), # *2
                                          'len_c': (1, 8.999),
                                          'len_p': (1, 5.999),
                                          'len_t': (1, 2.999)
                                          #   'kernel_size': (3, 5.999)
                                 },
                                 verbose=2)

#optimizer.maximize(init_points=2, n_iter=5)

# training-test-evaluation iterations with best params
#if os.path.isdir('results') is False:
#    os.mkdir('results')
#targets = [e['target'] for e in optimizer.res]
#bs_fname = f'bs_parking_{task}.json'
#with open(os.path.join('results', bs_fname), 'w') as f:
#    json.dump(optimizer.res, f, indent=2)
#best_index = targets.index(max(targets))
#params = optimizer.res[best_index]['params']
# save best params
params_fname = f'star_{task}_best_params.json'
#with open(os.path.join('results', params_fname), 'w') as f:
#    json.dump(params, f, indent=2)
with open(os.path.join('results', params_fname), 'r') as f:
    params = json.load(f)

# iterations with best params
for i in range(0, 5):
    train_model(residual_units=params['residual_units'],
                lr=params['lr'],
                batch_size=params['batch_size'],
                len_c = params['len_c'],
                len_p = params['len_p'],
                len_t = params['len_t'],
                # kernel_size=params['kernel_size'],
                save_results=True,
                i=i)
