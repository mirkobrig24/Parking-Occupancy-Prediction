import pandas as pd
import sklearn
import tensorflow
from sklearn.metrics import mean_squared_error,mean_absolute_error
from copy import copy
import numpy as np
import os
import h5py
from pandas import read_csv
from pandas import datetime
from prophet import Prophet
import datetime
from tensorflow.keras.metrics import (
    MeanAbsolutePercentageError,
    RootMeanSquaredError
)
import logging
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True
import timeit


def rmse(y_true, y_pred):
    m = RootMeanSquaredError()
    m.update_state(y_true, y_pred)
    return m.result().numpy()

def mape(y_true, y_pred):
    #idx = y_true > 10 for 8H sample
    idx = y_true > 0 # for 1H sample
    m = MeanAbsolutePercentageError()
    m.update_state(y_true[idx], y_pred[idx])
    #m.update_state(y_true, y_pred)
    return m.result().numpy()

def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = np.array(f['data'])
    timestamps = np.array(f['date'])
    f.close()
    return data, timestamps

def remove_incomplete_days(data, timestamps, T=48, h0_23=False):
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    first_timestamp_index = 0 if h0_23 else 1
    last_timestamp_index = T-1 if h0_23 else T
    while i < len(timestamps):
        if int(timestamps[i][8:]) != first_timestamp_index:
            i += 1
        elif i+T-1 < len(timestamps) and int(timestamps[i+T-1][8:]) == last_timestamp_index:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps

def save_to_csv(model_name, dataset_name, score):
    csv_name = f'{model_name}_results.csv'
    if not os.path.isfile(csv_name):
        with open(csv_name, 'a', encoding = "utf-8") as file:
            file.write('dataset,''rsme,''mape')
            file.write("\n")
            file.close()
    with open(csv_name, 'a', encoding = "utf-8") as file:
        file.write(f'{dataset_name},{score[0]},{score[1]}'
                )
        file.write("\n")
        file.close()
        
def evaluate(y_true, y_pred):

    score = []

    score.append(rmse(y_true, y_pred))
    score.append(mape(y_true, y_pred))
    
    print(
        f'rmse_total: {score[0]}\n'
        f'mape_total: {score[1]}\n'
    )
    return score


#############  45 - 54 ##############################

def prop_prediction(data, T, len_test):
    train_data, test_data = data[:-len_test], data[-len_test:]
    num_rows, num_columns = data.shape[2], data.shape[3]

    prediction_shape = (len_test, data.shape[1], data.shape[2], data.shape[3])
    predicted_data = np.empty(prediction_shape)

    for flow in [0]:                                                                   
        for row in (list(range(num_rows))[50:52]):
            for column in range(num_columns):
                history_region = [x[flow][row][column] for x in train_data]
                history_region = np.array(history_region)
                #print(len(history_region))
                dti = pd.date_range("2013-01-01", periods=len(history_region), freq="H").to_pydatetime()
                d = {'ds': dti, 'y': history_region}
                df = pd.DataFrame(data=d)
                #print(df.tail())
                start = timeit.default_timer()
                for i in range(len_test):

                    if (sum(history_region) == 0):                              
                        yhat = 0
                        
                    else:
                        model=Prophet()
                        model_fit=model.fit(df)
                        d_test = pd.date_range(df.iloc[-1].ds, periods=2, freq="H")
                        d_test=d_test[1:].to_pydatetime()
                        future=pd.DataFrame(data={'ds':d_test})
                        forecast = model.predict(future)
                        
                    predicted_data[i][flow][row][column] = forecast.loc[0,'yhat']
                    obs = test_data[i][flow][row][column]
                    new=pd.DataFrame({'ds':future.iloc[0].ds, 'y': obs}, index=[len(df)])
                    df = pd.concat([df.loc[:],new]).reset_index(drop=True)
                    df=df.drop(0)
                print(f'flow {flow}, region {row}x{column}')
                stop = timeit.default_timer()

                print('Time: ', stop - start)

    return predicted_data[:, :, 50:52, :]

def prop_prediction_parking():
    DATAPATH = '../results/feat_sosta_media_contemporanea_1h.h5'
    nb_flow = 1 # i.e. inflow and outflow
    T = 24 # number timestamps per day
    #len_test = T * 10 # number of timestamps to predict (ten days)
    days_test = 122
    len_test = int((T*days_test) * 0.2)
    # load data
    fname = os.path.join(DATAPATH)
    print("file name: ", fname)
    data, timestamps = load_stdata(fname)                                              
    # print(timestamps)
    # remove a certain day which does not have 24 timestamps
    #data, timestamps = remove_incomplete_days(data, timestamps, T)                     
    #data = data[:, :nb_flow]
    data = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])                
    data = data[:, :nb_flow, :, :]
    data[data < 0] = 0.
    print('data shape: ' + str(data.shape))
    print(data.max())
    # make predictions
    predicted_data = prop_prediction(data, T, len_test)

    # evaluate
    print('Evaluating on Parking')
    real_data = data[-len_test:, :, 50:52, :]
    score = evaluate(real_data, predicted_data)                                         

    # save to csv
    save_to_csv('_50_51', 'parking', score)                                               



if __name__ == '__main__':
    prop_prediction_parking()