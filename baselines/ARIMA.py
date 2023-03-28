from copy import copy
import numpy as np
import os

from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima.model import ARIMA

from utils import load_stdata, remove_incomplete_days, evaluate, save_to_csv

def arima_prediction(data, T, len_test):
    train_data, test_data = data[:-len_test], data[-len_test:]
    num_rows, num_columns = data.shape[2], data.shape[3]

    prediction_shape = (len_test, data.shape[1], data.shape[2], data.shape[3])
    predicted_data = np.empty(prediction_shape)

    for flow in [0]:
        for row in range(num_rows):
            for column in range(num_columns):
                history_region = [x[flow][row][column] for x in train_data]
                history_region = np.array(history_region)

                for i in range(len_test):
                    if (sum(history_region) == 0):
                        yhat = 0
                    else:
                        model = ARIMA(history_region, order=(1,0,0))
                        model_fit = model.fit()
                        output = model_fit.forecast()
                        yhat = output[0]
                    predicted_data[i][flow][row][column] = yhat
                    obs = test_data[i][flow][row][column]
                    history_region = np.append(history_region, obs)
                    history_region = np.delete(history_region, 0)
                print(f'flow {flow}, region {row}x{column}')
    
    return predicted_data

def arima_prediction_parking():
    #DATAPATH = '../results/feat_sosta_media_contemporanea_1h.h5'
    DATAPATH = '../results_one_month/feat_CNN_sosta_media_contemporanea_1h.h5'
    nb_flow = 1 # i.e. inflow and outflow
    T = 24 # number timestamps per day
    #len_test = T * 10 # number of timestamps to predict (ten days)
    #days_test = 122
    #len_test = int((T*days_test) * 0.2)
    # load data
    
    days_test = 10  
    len_test = T*days_test

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
    predicted_data = arima_prediction(data, T, len_test)

    # evaluate
    print('Evaluating on Parking')
    real_data = data[-len_test:]
    score = evaluate(real_data, predicted_data)

    # plot real vs prediction data of a region
    # plot_region_data(real_data, predicted_data, (13,3), 0)

    # save to csv
    save_to_csv('ARIMA_sosta_media_one_month', 'parking', score)
    # ARIMA_sosta_media_new ---> elimino ogni volta la prima osservazione


if __name__ == '__main__':
    arima_prediction_parking()
