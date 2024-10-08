import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from metrics import metric
import matplotlib.pyplot as plt
import time
import os
import random
from utils.utils import file_choise, _seed
from data_load.create_dataset import create_dataset, read_data
from exp.exp_main import Build_model


def train(count, name, seq_len, pred_len):
    time_now = time.time()
    num_epochs = 200
    train_steps = len(train_loader)
    fdim = -1
    for epoch in range(num_epochs):
        models.train()
        train_loss = []
        epoch_time = time.time()
        iter_count = 0
        for i, (input_feature, input_label) in enumerate(train_loader):
            optimizer.zero_grad()
            input_feature = input_feature.permute(1, 0, 2)  # 新的LSTM
            # input_label = input_label.permute(1, 0)         # 新的LSTM
            outputs = models(input_feature)
            loss = criterion(outputs, input_label)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            iter_count += 1
            if (i + 1) % 10 == 0:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((num_epochs - epoch) * train_steps - i)
                iter_count = 0
                time_now = time.time()
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)

        print('Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}'.format(
            epoch + 1, train_steps, train_loss))
    test_loss = _test(count)
    path = f'checkpoints/{name}/{seq_len}_{pred_len}_{count}.pth'
    torch.save(models.state_dict(), path)
    return test_loss


def _test(count):
    models.eval()
    pred = []
    true = []
    with torch.no_grad():
        for i, (input_feature, input_label) in enumerate(test_loader):
            input_feature = input_feature.permute(1, 0, 2)    # 新的LSTM
            # input_label = input_label.permute(1, 0)           # 新的LSTM
            predicted = models(input_feature).cpu().detach().numpy()
            pred.append(predicted[:, 0])
            # pred.append(predicted[0, :])    # 新LSTM
            true.append(input_label[:, 0].cpu().detach().numpy())
            # true.append(input_label[0, :].cpu().detach().numpy())     # 新LSTM
    preds = np.concatenate(pred, axis=0)
    trues = np.concatenate(true, axis=0)
    mae, mse, rmse, mape, mspe, r2 = metric(preds, trues)
    print('mse:{0:.4f}, mae:{1:.4f}, rmse:{2:.4f}, mape:{3:.4f}, mspe:{4:.4f}, r2:{5:.4f}'.format(mse, mae, rmse, mape,
                                                                                                  mspe, r2))
    f = open("result.txt", 'a')
    f.write(f'{name}' + "  \n")
    f.write(
        'mse:{0:.4f}, mae:{1:.4f}, rmse:{2:.4f}, mape:{3:.4f}, mspe:{4:.4f}, r2:{5:.4f}'.format(mse, mae, rmse, mape,
                                                                                                mspe, r2))
    f.write('\n')
    f.write('\n')
    f.close()
    plot(preds.reshape(-1, 1), trues.reshape(-1, 1), count)
    return mae, mse, rmse, mape, mspe, r2


def plot(pred, true, count):
    pred = scaler_y.inverse_transform(pred)
    true = scaler_y.inverse_transform(true)
    # result save
    folder_path = f'pictures/{file}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = f'{folder_path}{name}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = f'{folder_path}{seq_len}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path_1 = f'{fcsv_path}{name}_pred_seq{seq_len}_pred{pred_len}/'
    if not os.path.exists(folder_path_1):
        os.makedirs(folder_path_1)
    np.save(f'{folder_path_1}Prediction_count{count}.npy', pred)
    np.save(f'{folder_path_1}True_count{count}.npy', true)
    plt.figure(figsize=(10, 6))
    plt.plot(pred.reshape(-1), label='Predicted values')
    plt.plot(true.reshape(-1), label='Actual values')
    plt.ylabel('values(m)')
    plt.legend()
    plt.title(f'{name}-{file} Stock Price Prediction')
    plt.ylabel('Stock Price')
    plt.savefig(f'{folder_path}{name}_seq{seq_len}_pre{pred_len}_{count}.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    """
    当下要修改的内容，包括数据处理部分，以及保证自动化处理的连贯性
    """
    _seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for j in range(1):
        # file = file_choise(j)
        file = 'DailyDelhiClimate'
        # 数据加载
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        train_input, test_input, train_label, test_label = read_data(file, scaler_x, scaler_y)
        # 序列长度
        seq_len_list = [7]  # 跑过的5, 10, 20, 60, 120, 240, 30, 40, 50
        """
        GRU: 5, 10, 20, 60, 120, 240
        """
        # seq_len_list = [5]
        pred_len = 7

        # 模型参数
        input_size = train_input.shape[1]
        hidden_size = 200
        num_layers = 2
        output_dim = pred_len

        model_names = ['GRU']
        # model_names = ['LSTM']

        # 创建DataFrame数据结构
        lt_data = pd.DataFrame(
            {"Model": [], "seq_len": [], "pred_len": [], "mae": [], "mse": [], "rmse": [], "mape": [], "mspe": [],
             "r2": []}
        )
        csv_path = f'data/{file}/'
        for name in model_names:
            # 不同长度的10次训练
            for seq_len in seq_len_list:
                # if pred_len != 1:
                #     output_dim = pred_len
                # else:
                #     output_dim = train_label.shape[1]

                models = Build_model(name, input_size, hidden_size, output_dim, num_layers, device)._build_model()
                criterion = nn.MSELoss().to(device)
                optimizer = torch.optim.Adam(models.parameters(), lr=0.01)
                score = []  # 存储评估指标
                # 训练时序数据
                train_loader = create_dataset(train_input, train_label.flatten(), seq_len, pred_len, device, 'train')

                # 测试时序数据
                test_loader = create_dataset(test_input, test_label.flatten(), seq_len, pred_len, device)

                fcsv_path = f'data/{file}/{name}/'
                if not os.path.exists(fcsv_path):
                    os.makedirs(fcsv_path)

                for i in range(1):
                    _metric = train(i + 1, name, seq_len, pred_len)
                    torch.cuda.empty_cache()
                    score.append(_metric)

                score = np.asarray(score)
                new_data = pd.DataFrame(
                    {"Model": name, "seq_len": seq_len, "pred_len": pred_len, "mae": score[:, 0], "mse": score[:, 1],
                     "rmse": score[:, 2], "mape": score[:, 3],
                     "mspe": score[:, 4],
                     "r2": score[:, 5]})
                lt_data = pd.concat([lt_data, new_data], ignore_index=True)
        lt_data.to_csv(f'{csv_path}{file}_{pred_len}_result.csv', encoding='utf-8-sig')
    """
    闫总对工作有啥指示没？
    没有
    """
