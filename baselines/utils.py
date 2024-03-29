import h5py
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
from matplotlib import pyplot
import os
from keras import backend as K
from tensorflow.keras.metrics import (
    MeanAbsolutePercentageError,
    RootMeanSquaredError
)

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


def evaluate(y_true, y_pred):

    score = []

    score.append(rmse(y_true, y_pred))
    score.append(mape(y_true, y_pred))
    
    print(
        f'rmse_total: {score[0]}\n'
        f'mape_total: {score[1]}\n'
    )
    return score

def plot_region_data(real_data, predicted_data, region, flow):
    # region deve essere una lista o tupla di 2 elementi
    # flow deve essere 0 (inflow) o 1 (outflow)
    row, column = region[0], region[1]

    real_data_region = [x[flow][row][column] for x in real_data]
    predicted_data_region = [x[flow][row][column] for x in predicted_data]

    pyplot.plot(real_data_region)
    pyplot.plot(predicted_data_region, color='red')
    pyplot.legend(['real', 'predicted'])
    pyplot.show()

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