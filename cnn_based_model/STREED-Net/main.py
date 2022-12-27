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

from utils import cache, read_cache
from src import Parking3d
from src.evaluation import evaluate
from src import streednet 
models_dict = {
    'STREED-Net': streednet
}

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

# Nome modello
model_name = 'STREED-Net'


# parameters
DATAPATH = '../../results/feat_sosta_media_contemporanea_1h.h5'
task = 'sosta_media_contemporanea'

nb_epoch = 150  # number of epoch at training stage
T = 24  # number of time intervals in one day
CACHEDATA = False  # cache data or NOT

len_closeness = 4  # length of closeness dependent sequence
len_period = 2  # length of peroid dependent sequence
len_trend = 0  # length of trend dependent sequence

nb_flow = 1  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test,
days_test = 122
len_test = int((T*days_test) * 0.2)
len_val = 2*len_test

map_height, map_width = 54, 43  # grid size

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
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = read_cache(
        fname, 'preprocessing_bj.pkl')
    print("load %s successfully" % fname)
else:
    #if (model_name.startswith('model3')):
    #else:
    #    load_data = TaxiBJ.load_data
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = Parking3d.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        len_val=len_val, preprocess_name='preprocessing_parking.pkl', meta_data=True, meteorol_data=False, holiday_data=False, datapath=DATAPATH)
    if CACHEDATA:
        cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
              external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test)

print(external_dim)
print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])


def train_model(encoder_blocks, lr, batch_size, kernel_size, save_results=False, i=''):
    # get discrete parameters
    encoder_blocks = int(encoder_blocks)
    batch_size = 16 * int(batch_size)
    kernel_size = int(kernel_size)
    lr = round(lr,5)

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
    # model.summary()
    hyperparams_name = '{}.TaxiNYC_{}_{}.c{}.p{}.t{}.encoderblocks_{}.kernel_size_{}.lr_{}.batchsize_{}'.format(
        model_name, task, i, len_closeness, len_period, len_trend, encoder_blocks,
        kernel_size, lr, batch_size)
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
                        verbose=0)
    model.save_weights(os.path.join(
        'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    # evaluate
    model.load_weights(fname_param)
    score = model.evaluate(
        X_test, Y_test, batch_size=128 , verbose=0) # batch_size=Y_test.shape[0]
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    if (save_results):
        print('evaluating using the model that has the best loss on the valid set')
        model.load_weights(fname_param)  # load best weights for current iteration

        Y_pred = model.predict(X_test)  # compute predictions

        score = evaluate(Y_test, Y_pred, mmn, rmse_factor=1)  # evaluate performance

        # save to csv
        csv_name = os.path.join('results', f'{model_name}_Parking_{task}_results.csv')
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
                              pbounds={'encoder_blocks': (2, 2),
                                       'lr': (0.0001, 0.001),
                                       'batch_size': (1, 3.999), # *16
                                       'kernel_size': (3, 4.999)
                              },
                              verbose=2)

#bs_fname = f'bs_parking_{task}.json'
#logger = JSONLogger(path="./results/" + bs_fname)
#optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

#optimizer.maximize(init_points=2, n_iter=5)


# New optimizer is loaded with previously seen points
# load_logs(optimizer, logs=["./results/" + bs_fname], reset = False)
# optimizer.maximize(init_points=10, n_iter=10)

# training-test-evaluation iterations with best params

#targets = [e['target'] for e in optimizer.res]
#bs_fname = f'bs_parking_{task}.json'
#with open(os.path.join('results', bs_fname), 'w') as f:
#    json.dump(optimizer.res, f, indent=2)
#best_index = targets.index(max(targets))
#params = optimizer.res[best_index]['params']
# save best params
params_fname = f'{model_name}_Parking_{task}_params.json'
#with open(os.path.join('results', params_fname), 'w') as f:
#    json.dump(params, f, indent=2)
with open(os.path.join('results', params_fname), 'r') as f:
    params = json.load(f)
for i in range(0, 5):
    train_model(encoder_blocks=params['encoder_blocks'],
                lr=params['lr'],
                batch_size=params['batch_size'],
                kernel_size=params['kernel_size'],
                save_results=True,
                i=i)
