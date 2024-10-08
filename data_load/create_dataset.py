import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

def create_dataset(x, y, seq_len, pred_len, device, test='test'):
    x_seq, y_seq = [], []
    number = len(x) - seq_len - pred_len + 1
    if pred_len == 0:
        for i in range(number):
            x_end = i + seq_len
            # y_end = x_end + pred_len
            x_seq.append(x[i:x_end])
            y_seq.append(y[i:x_end])
    else:
        for i in range(number):
            x_end = i + seq_len
            y_end = x_end + pred_len
            x_seq.append(x[i:x_end])
            y_seq.append(y[x_end:y_end])  # 1:1, 是空
    td_x = np.asarray(x_seq)
    td_y = np.asarray(y_seq)
    input_tensor = torch.tensor(td_x, dtype=torch.float32).to(device)
    label_tensor = torch.tensor(td_y, dtype=torch.float32).to(device)
    train_data = TensorDataset(input_tensor, label_tensor)
    batch_size = input_tensor.shape[0]

    shuffle = False
    if test == 'train':
        shuffle = True
    data_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    return data_loader

def read_data(file, scaler_x, scaler_y):
    data = pd.read_csv(f'data/{file}.csv')
    fdata_path = f'data/{file}/'
    if not os.path.exists(fdata_path):
        os.makedirs(fdata_path)

    # 特征与目标获取
    features_cols = ['date','humidity','wind_speed','meanpressure']
    target_cols = ['meantemp']
    features = data[features_cols]
    target = data[target_cols]

    # 归一化
    features = scaler_x.fit_transform(features)
    target = scaler_y.fit_transform(target.values.reshape(-1, 1))

    # 数据集切分
    train_input, test_input, train_label, test_label = train_test_split(features, target, test_size=0.146, shuffle=False)
    return train_input, test_input, train_label, test_label