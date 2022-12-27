from tensorflow.keras.metrics import (
    RootMeanSquaredError,
    MeanAbsolutePercentageError,
    MeanAbsoluteError
)
import numpy as np

def evaluate(y_true, y_pred, rmse_factor=1):

    def rmse(y_true, y_pred):
        m_factor = rmse_factor
        m = RootMeanSquaredError()
        m.update_state(y_true, y_pred)
        return m.result().numpy() * m_factor
    
    def mape(y_true, y_pred):
        idx = y_true > 0

        m = MeanAbsolutePercentageError()
        print('dimensioni: ', y_true.shape)
        print('dimensioni: ', y_pred.shape)

        m.update_state(y_true[idx], y_pred[idx])
        return m.result().numpy()


    score = []

    score.append(rmse(y_true, y_pred))
    score.append(mape(y_true, y_pred))


    print(
        f'rmse_total: {score[0]}\n'
        f'mape_total: {score[1]}\n'
    )
    return score

