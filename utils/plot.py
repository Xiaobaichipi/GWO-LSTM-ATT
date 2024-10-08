import os
import matplotlib.pyplot as plt
import time

import numpy as np


def plot(pred, true, count, scaler_y, seq_len, pred_len, file):
    pred = scaler_y.inverse_transform(pred)
    true = scaler_y.inverse_transform(true)

    name = 'GWO-LSTM-ATT'
    folder_path = f'pictures/{file}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = f'{folder_path}{name}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = f'{folder_path}{seq_len}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(f'data/{file}/{name}/gwo_pred_seq{seq_len}_pred{pred_len}.npy', pred)
    np.save(f'data/{file}/{name}/gwo_true_seq{seq_len}_pred{pred_len}.npy', true)
    plt.figure(figsize=(10, 6))
    plt.plot(pred.reshape(-1), label='Predicted values')
    plt.plot(true.reshape(-1), label='Actual values')
    plt.ylabel('values(m)')
    plt.legend()
    plt.title(f'GWO-LSTM-ATT test Stock Price Prediction')
    plt.ylabel('Stock Price')
    time1 = time.gmtime()
    time2 = time.strftime("%Y-%m-%d-%H-%M-%S", time1)
    plt.savefig(f'{folder_path}GWO_LSTM_ATT结果_seq{seq_len}_pre{pred_len}_{time2}_{count}.png',
                dpi=300)
    plt.show()
