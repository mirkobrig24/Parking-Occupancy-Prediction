import torch
import torchmetrics
from tensorflow.keras.metrics import (
    RootMeanSquaredError,
    MeanAbsolutePercentageError,
    MeanAbsoluteError
)
import numpy as np

def denormalize(x, mmn):
    return x * mmn / 2.

def inverse_transform(X, mmn):
    X = (X + 1.) / 2.
    X = 1. * X * mmn 
    return X

def rmse(y_true, y_pred, mmn):
    rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(y_pred, y_true))
    return denormalize(rmse, mmn)

def mape(y_true, y_pred, mmn):
    y_true = inverse_transform(y_true, mmn).cpu()
    y_pred = inverse_transform(y_pred, mmn).cpu()
    #idx = y_true > 10 # for 8H sample
    idx = y_true > 0 # for 1H sample
    m = MeanAbsolutePercentageError()
    #print(y_true[idx].size())
    # Ci sono alcune sample che hanno zero y_true > 0
    if y_true[idx].size()[0] != 0:
        m.update_state(y_true[idx], y_pred[idx])
#    else:
#        skip
    #print('valoreeee')
    #print(m.result().numpy())
    #mean_abs_percentage_error = torchmetrics.MeanAbsolutePercentageError()
    #print(mean_abs_percentage_error(y_pred[idx],y_true[idx]))
    return m.result().numpy()
    
    #mean_abs_percentage_error = torchmetrics.MeanAbsolutePercentageError().to('cuda')
    #return mean_abs_percentage_error(y_pred[idx], y_true[idx])#.numpy()

def accuracy(pred, y):
    """
    :param pred: predictions
    :param y: ground truth
    :return: accuracy, defined as 1 - (norm(y - pred) / norm(y))
    """
    return 1 - torch.linalg.norm(y - pred, "fro") / torch.linalg.norm(y, "fro")


def r2(pred, y):
    """
    :param y: ground truth
    :param pred: predictions
    :return: R square (coefficient of determination)
    """
    return 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(pred)) ** 2)


def explained_variance(pred, y):
    return 1 - torch.var(y - pred) / torch.var(y)
