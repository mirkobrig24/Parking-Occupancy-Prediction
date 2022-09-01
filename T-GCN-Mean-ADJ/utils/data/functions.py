import numpy as np
import pandas as pd
import torch
import pickle
from .dataset_generation import load_data
import h5py

def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path, header=None)
    feat = np.array(feat_df, dtype=dtype)
    feat = feat.T
    return feat

def load_features_2(fname):
    f = h5py.File(fname, 'r')
    data = np.array(f['data'])
    timestamps = np.array(f['date'])
    f.close()
    return data

def load_adjacency_matrix(adj_path, dtype=np.float32):
    with open(adj_path, 'rb') as f:
        adj_df = pickle.load(f)

    adj = np.array(adj_df.todense(), dtype=dtype)
    return adj

'''
def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj
'''

def transform(X, max):
    X = 1. * X / max
    X = X * 2. - 1.
    return X

def generate_torch_datasets_2(T, nb_flow, seq_len, len_period, len_trend, len_test,datapath):
    print(T)
    T = T[0] if type(T) is tuple else T
    nb_flow = nb_flow[0] if type(nb_flow) is tuple else nb_flow
    len_period = len_period[0] if type(len_period) is tuple else len_period
    len_trend = len_trend[0] if type(len_trend) is tuple else len_trend
    len_test = len_test[0] if type(len_test) is tuple else len_test

    train_X, train_Y, test_X, test_Y, mmn= load_data(T=T, nb_flow=nb_flow, len_closeness=seq_len, len_period=len_period, len_trend=len_trend, len_test=len_test,
        datapath=datapath)
    if seq_len > 1:
        train_X = torch.squeeze(torch.FloatTensor(train_X))
        test_X = torch.squeeze(torch.FloatTensor(test_X))
    train_dataset = torch.utils.data.TensorDataset(
        train_X, torch.FloatTensor(train_Y)
    )

    print('---TRAIN-DATASET-LEN---', len(train_dataset))
    print('---TRAIN_X---', train_X.size())
    print('---TRAIN_Y---', torch.FloatTensor(train_Y).size())
    print('------------------------------------')
    
    test_dataset = torch.utils.data.TensorDataset(
        test_X, torch.FloatTensor(test_Y)
    )

    print('---TEST-DATASET-LEN---', len(test_dataset))
    print('---TEST_X---', test_X.size())
    print('---TEST_Y---', torch.FloatTensor(test_Y).size())
    print('------------------------------------')
    return train_dataset, test_dataset


def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i : i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i : i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )

    print('---TRAIN-DATASET-LEN---', len(train_dataset))
    print('---TRAIN_X---', torch.FloatTensor(train_X).size())
    print('---TRAIN_Y---', torch.FloatTensor(train_Y).size())
    print('------------------------------------')
    
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )

    print('---TEST-DATASET-LEN---', len(test_dataset))
    print('---TEST_X---', torch.FloatTensor(test_X).size())
    print('---TEST_Y---', torch.FloatTensor(test_Y).size())
    print('------------------------------------')
    
    return train_dataset, test_dataset
