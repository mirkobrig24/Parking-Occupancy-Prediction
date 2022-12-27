from tensorflow.keras.metrics import (
    RootMeanSquaredError,
    MeanAbsolutePercentageError
)
import numpy as np

def evaluate(y_true, y_pred, mmn, rmse_factor=1):
    def denormalize(x, mmn):
        return x * (mmn._max - mmn._min) / 2.
    
    def inverse_transform(X, mmn):
        X = (X + 1.) / 2.
        X = 1. * X * (mmn._max - mmn._min) + mmn._min
        return X
    
    def split_flow(X):
        inflow = X[:,0]
        outflow = X[:,1]
        return inflow, outflow
    
    def rmse(y_true, y_pred):
        m_factor = rmse_factor
        m = RootMeanSquaredError()
        m.update_state(y_true, y_pred)
        return denormalize(m.result().numpy(), mmn) * m_factor
    
    def mape(y_true, y_pred):
        y_true = inverse_transform(y_true, mmn)
        y_pred = inverse_transform(y_pred, mmn)
        idx = y_true > 0
        m = MeanAbsolutePercentageError()
        m.update_state(y_true[idx], y_pred[idx])
        return m.result().numpy()
        # return np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx])) * 100
    


    score = []


    score.append(rmse(y_true, y_pred))
    score.append(mape(y_true, y_pred))


    print(
        f'rmse_total: {score[0]}\n'
        f'mape_total: {score[1]}\n'
    )
    return score
