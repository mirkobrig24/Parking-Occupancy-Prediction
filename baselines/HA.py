from copy import copy
import numpy as np
import os
import datetime

from utils import (
    load_stdata, remove_incomplete_days, evaluate, plot_region_data, save_to_csv
)

def get_day_of_week(timestamp):
    date_string = timestamp.decode("utf-8")[:-6]
    day_of_week = datetime.datetime.strptime(date_string, '%Y-%m-%d').strftime('%A')
    return day_of_week

def ha_prediction(data, timestamps, T, len_test):
    num_timestamps = len(data)

    # estraggo i dati di train. solo questi e le previsioni già effettuate
    # vengono usate per predire i nuovi valori
    train_data = list(data[:-len_test])

    predicted_data = []
    # loop su tutti i timestamp del test_set
    for i in range(num_timestamps-len_test, num_timestamps):
        # prendo tutti i timestamps corrispondenti alla stessa ora e allo stesso giorno
        # e faccio la media. Problema: ci sono dei giorni mancanti nel dataset
        # step = T * 7
        # start_idx = i % step
        # historical_data = [data_all[t] for t in range(start_idx, i, step)]

        # provo a usare semplicemente tutti i giorni precedenti alla stessa ora
        # step = T
        # start_idx = i % step
        # historical_data = [data_all[t] for t in range(start_idx, i, step)]
        # prediction = np.mean(historical_data, axis=0).astype(int)
        # predicted_data.append(prediction)

        # possibile soluzione: converto il corrispondete timestamp in giorno della
        # settimana e vedo se corrisponde
        # Ad esempio se T=24 e i=500, prendo i seguenti timestamp:
        # [20, 44, 68, 92, 116, 140, 164, 188, 212, 236, 260, 284, 308, 332, 356, 380, 404, 428, 452, 476]
        # e li considero solo se appartengono allo stesso giorno della settimana
        # del timestamp i 
        current_day_of_week = get_day_of_week(timestamps[i])
        step = T
        start_idx = i % step
        historical_data = [
            train_data[t] for t in range(start_idx, i, step) if get_day_of_week(timestamps[t]) == current_day_of_week
        ]
        prediction = np.mean(historical_data, axis=0).astype(int)
        train_data.append(prediction)
        predicted_data.append(prediction)

    predicted_data = np.asarray(predicted_data)
    print('prediction shape: ' + str(predicted_data.shape))
    return predicted_data

def ha_prediction_parking():
    DATAPATH = '../results/feat_sosta_media_contemporanea_1h.h5'
    nb_flow = 1 # i.e. inflow and outflow
    T = 24 # number timestamps per day
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
    predicted_data = ha_prediction(data, timestamps, T, len_test)

    # evaluate
    print('Evaluating on Parking')
    real_data = data[-len_test:]
    score = evaluate(real_data, predicted_data)

    # save to csv
    save_to_csv('HA', 'parking', score)



if __name__ == '__main__':
    ha_prediction_parking()
