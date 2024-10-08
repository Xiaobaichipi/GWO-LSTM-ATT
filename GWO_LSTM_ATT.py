import os

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from data_load.create_dataset import create_dataset, read_data
from models.GWO import GreyWolfOptimizer
from models.LSTM_ATT import Model
from utils.print_color import Colors, colored_print
import gc
from utils.utils import file_choise, _seed

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10.5

if __name__ == '__main__':
    _seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for j in range(5):
        file = file_choise(j)
        # 数据加载
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        train_input, test_input, train_label, test_label = read_data(file, scaler_x, scaler_y)

        seq_len_list = [5, 10 ,20]  # 已经跑过的5, 10 ,20
        # seq_len_list = [1]
        dim = 3
        lb = [10, 1, 0.001]
        """
        小王：人家给的？但是显示是3个维度呀，比这之前的多了一个内容
        """
        ub = [300, 5, 0.1]
        num_wolves = 20  # 20
        max_iter = 50

        name = 'GWO-LSTM-ATT'
        fcsv_path = f'data/{file}/{name}/'
        if not os.path.exists(fcsv_path):
            os.makedirs(fcsv_path)
        pred_len = 5
        for seq_len in seq_len_list:

            input_size = train_input.shape[1]
            if pred_len != 1:
                output_dim = pred_len
            else:
                output_dim = train_label.shape[1]
            # 训练时序数据
            train_loader = create_dataset(train_input, train_label.flatten(), seq_len, pred_len, device, 'train')

            # 测试时序数据
            test_loader = create_dataset(test_input, test_label.flatten(), 5, 5, device)

            params = []  # 存储超参数
            times = []  # 存储时间
            score = []  # 存储评估指标
            for i in range(1):
                colored_print(Colors.BLUE,
                              f'<---------------------------------------------第{i + 1}次更新开始--------------------------------------------->')
                gwo = GreyWolfOptimizer(dim, lb, ub, num_wolves, max_iter, Model, input_size, output_dim, device,
                                        scaler_y, seq_len, pred_len, train_loader, test_loader, file)
                best_params, best_time, best_score = gwo.optimize()
                torch.cuda.empty_cache()
                params.append(best_params)
                times.append(best_time)
                score.append(best_score)
                mae, mse, rmse, mape, mspe, r2 = best_score
                print("Best parameters: hidden_size:{} | num_layers:{} | lr:{}".format(best_params[0], best_params[1],
                                                                                       best_params[2]))
                print("mae:{} | mse:{} | rmse:{} | mape:{} | mspe:{} | r2:{}".format(mae, mse, rmse, mape, mspe, r2))

                f = open("GL.txt", 'a')
                f.write(
                    'now:{} | hidden_size:{} | num_layers:{} | lr:{} | iter'.format(best_time, best_params[0],
                                                                                    best_params[1],
                                                                                    best_params[2]))
                f.write("mse:{} | mae:{} | rmse:{} | mape:{} | mspe:{} | r2:{}".format(mse, mae, rmse, mape, mspe, r2))
                f.write('\n')
                f.close()
                colored_print(Colors.BLUE,
                              f'<---------------------------------------------第{i + 1}次更新结束--------------------------------------------->')

            params = np.asarray(params)
            times = np.asarray(times)
            score = np.asarray(score)
            # 写入文件
            gwo_data = pd.DataFrame(
                {"hidden_size": params[:, 0], "num_layers": params[:, 1], "lr": params[:, 2], "timestamp": times[:],
                 "mae": score[:, 0], "mse": score[:, 1], "rmse": score[:, 2], "mape": score[:, 3], "mspe": score[:, 4],
                 "r2": score[:, 5]})
            gwo_data.to_csv(f'{fcsv_path}gwo_seq{seq_len}_pred{pred_len}_result.csv', encoding='utf-8-sig')
            # 使用垃圾回收
            gc.collect()
