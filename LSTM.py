import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from metrics import metric
import matplotlib.pyplot as plt
import time
from data_load.create_dataset import create_dataset
from models.LSTM import Model
from utils.utils import file_choise, _seed


def calculate_flux_peak(data):
    # 逐行取每行数据的最大值作为峰值
    flux_peaks = np.max(data, axis=1)
    return flux_peaks


def train(scaler_x, scaler_y, train_loader, models,
          optimizer, seq_len, criterion, test_input_tensor,
          test_label, pre_features):
    # 训练
    time_now = time.time()
    num_epochs = 300
    train_steps = len(train_loader)

    for epoch in range(num_epochs):
        models.train()
        train_loss = []
        epoch_time = time.time()
        iter_count = 0
        for i, (input_feature, input_label) in enumerate(train_loader):
            optimizer.zero_grad()
            input_label = input_label.permute(1, 0)
            input_feature = input_feature.permute(1, 0, 2)
            outputs = models(input_feature)
            outputs = torch.clamp(outputs, min=0)
            outputs = outputs.view(seq_len, -1)
            loss = criterion(outputs, input_label)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            # print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
            iter_count += 1
            if (i + 1) % 10 == 0:
                # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((num_epochs - epoch) * train_steps - i)
                # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
        torch.cuda.empty_cache()
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)

        print('Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}'.format(
            epoch + 1, train_steps, train_loss))
    test_label1 = scaler_y.inverse_transform(test_label)
    test_loss = _test(scaler_x, scaler_y, models, test_input_tensor, test_label)  # 传递epoch + 1作为count给_test函数

    predict(models, scaler_x, scaler_y, pre_features)
    # 载入模型参数
    path = f'checkpoints/LSTM.pth'
    torch.save(models.state_dict(), path)  # 保存模型参数

    return test_loss


def _test(scaler_x, scaler_y, models, test_input_tensor, test_label):
    # 测试
    models.eval()
    # test_label = test_label.reshape
    predicted = models(test_input_tensor.reshape(1, -1, 5)).cpu().detach().numpy()
    predicted = predicted.reshape(-1, 1)
    mae, mse, rmse, mape, mspe, r2 = metric(predicted, test_label)
    test_label1 = scaler_y.inverse_transform(test_label)
    print('mse:{0:.4f}, mae:{1:.4f}, rmse:{2:.4f}, mape:{3:.4f}, mspe:{4:.4f}, r2:{5:.4f}'.format(mse, mae, rmse, mape,
                                                                                                  mspe, r2))
    f = open("result.txt", 'a')
    f.write('LSTM' + "  \n")
    f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, r2:{}'.format(mse, mae, rmse, mape, mspe, r2))
    f.write('\n')
    f.write('\n')
    f.close()
    plot(predicted.reshape(1, -1), test_label, scaler_x, scaler_y)
    return mae, mse, rmse, mape, mspe, r2


def predict(models, scaler_x, scaler_y, pre_features):
    val_output = models(pre_features.reshape(1, -1, 5)).cpu().detach().numpy()
    val_output = val_output.reshape(-1, 1)
    pred = scaler_y.inverse_transform(val_output)
    plt.figure(figsize=(10, 6))
    plt.plot(pred.reshape(-1), label='Predicted values')
    plt.ylabel('values(m)')
    plt.legend()
    # 设置标题
    plt.title(f'Predict')
    # 设置X、Y轴标签
    plt.ylabel('Ci sunhao')
    # 显示图像
    plt.show()
    data = pd.DataFrame({'output': pred.reshape(-1)})
    data.to_csv('data/huaweibei/fujian4.csv', encoding='utf-8')


def plot(pred, true, scaler_x, scaler_y):
    pred = scaler_y.inverse_transform(pred)
    true = scaler_y.inverse_transform(true)
    np.save(f'data/meantemp/LSTM_pred_seq{1}_pred{0}.npy', pred)
    np.save(f'data/meantemp/LSTM_true_seq{1}_pred{0}.npy', true)
    plt.figure(figsize=(10, 6))
    plt.plot(pred.reshape(-1), label='Predicted values')
    plt.plot(true.reshape(-1), label='Actual values')
    plt.ylabel('values(m)')
    plt.legend()
    # 设置标题
    plt.title(f'LSTM test Stock Price Prediction')
    # 设置X、Y轴标签
    plt.ylabel('Stock Price')
    # plt.xlabel('天数')
    # plt.savefig(f'pictures/LSTM/{seq_len}/LSTM结果_seq{seq_len}_pre{pred_len}_{count}.png', dpi=300)
    # 显示图像
    plt.show()


def main():
    _seed()
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据加载
    train_data = pd.read_csv('data/huaweibei/hebing.csv')
    test_data = pd.read_csv('data/huaweibei/fujian3.csv')

    # 提取B_max
    flux_data_train = train_data.iloc[:, 5:].values  # 假设前五列为其他信息，后面的列是磁通密度数据
    flux_data_test = test_data.iloc[:, 5:].values

    flux_peak_train = calculate_flux_peak(flux_data_train)
    flux_peak_test = calculate_flux_peak(flux_data_test)

    # 构建新数据集
    df_train_new = pd.DataFrame({
        '温度': train_data['温度，oC'],
        '频率': train_data['频率，Hz'],
        '励磁波形': train_data['励磁波形'],
        '材料类别': train_data['材料类别'],
        '磁通密度峰值': flux_peak_train,
        '磁芯损耗': train_data['磁芯损耗，w/m3']  # 目标值
    })

    df_test_new = pd.DataFrame({
        '温度': test_data['温度，oC'],
        '频率': test_data['频率，Hz'],
        '励磁波形': test_data['励磁波形'],
        '材料类别': test_data['磁芯材料'],
        '磁通密度峰值': flux_peak_test
    })

    # 分离特征和目标
    features = df_train_new.drop(columns=['磁芯损耗'])
    target = df_train_new['磁芯损耗']

    # 归一化特征和目标
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    features = scaler_x.fit_transform(features)
    pre_feature = scaler_x.fit_transform(df_test_new)
    pre_features = torch.tensor(pre_feature, dtype=torch.float32).to(device)
    target = scaler_y.fit_transform(target.values.reshape(-1, 1))
    target_1 = scaler_y.inverse_transform(target)
    # 分割数据集
    train_input, test_input, train_label, test_label = train_test_split(features, target, test_size=0.2, shuffle=False)

    test_label_1 = scaler_y.inverse_transform(test_label)
    test_label_2 = scaler_y.inverse_transform(test_label)
    # 设定seq_len和pred_len
    seq_len_list = [1]
    # seq_len_list = [6]
    pred_len = 0

    # 模型初始化
    input_size = features.shape[1]
    hidden_size = 200
    num_layers = 2
    output_dim = 1
    models = Model(input_size, output_dim, hidden_size, num_layers).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(models.parameters(), lr=0.01)
    for seq_len in seq_len_list:
        # 创建时间序列数据集
        train_loader = create_dataset(train_input, train_label.flatten(), seq_len, pred_len, device)

        # 转换为张量数据
        test_input_tensor = torch.tensor(test_input, dtype=torch.float32).to(device)
        test_label_tensor = torch.tensor(test_label, dtype=torch.float32).to(device)

        score = []  # 存储评估指标
        _metric = train(scaler_x, scaler_y, train_loader,
                        models, optimizer, seq_len, criterion,
                        test_input_tensor, test_label, pre_features)
        torch.cuda.empty_cache()
        score.append(_metric)
        score = np.asarray(score)
        lt_data = pd.DataFrame(
            {"mae": score[:, 0], "mse": score[:, 1], "rmse": score[:, 2], "mape": score[:, 3], "mspe": score[:, 4],
             "r2": score[:, 5]})
        lt_data.to_csv(f'data/huaweibei/LSTM_seq{seq_len}_pred{pred_len}_result.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    main()

    # # 使用真实数据进行验证
    # val_data = pd.read_csv('data/test_final.csv')
