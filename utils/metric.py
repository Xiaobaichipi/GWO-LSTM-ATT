from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def metric(pred, true):
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    mspe = np.mean(np.square((true - pred) / true)) * 100
    r2 = r2_score(true, pred)
    return mae, mse, rmse, mape, mspe, r2
