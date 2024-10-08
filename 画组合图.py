import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 10.4

# 文件夹路径
# pred = 5
pred_list = [5, 10, 20]
# base_folder = f'data/{pred}'
# folders = ['000063', '600519', '600877', '601318', '601398']
# _name = ['中兴通讯', '贵州茅台', '电科芯片', '中国平安', '工商银行']
folders = ['600877']
_name = ['电科芯片']
# 文件名列表


# 创建一个3x2的网格
fig = plt.figure(constrained_layout=True, figsize=(10, 6))
gs = fig.add_gridspec(2, 2)

# 用于存储图例的lines和labels
lines = []
labels = ['GWO-LSTM-ATT', 'LSTM-ATT', 'LSTM', 'GRU', 'True']
file = '/data/10/600877/gwo_pred_seq5_pred5.npy'

# 遍历每个文件夹并在子图上绘制数据
# for i, folder in enumerate(folders):
for i, pred in enumerate(pred_list):
    # 生成完整的文件夹路径
    # folder_path = f'{base_folder}/{folder}'
    folder_path = f'./data/{pred}/{folders[0]}'

    # 创建子图
    ax = fig.add_subplot(gs[i // 2, i % 2])
    # ax.set_title(f'{_name[i]} ({folder})')
    ax.set_title(f'{_name[0]} ({pred}步长)')

    filenames = [
        f'gwo_pred_seq{pred}_pred5.npy',
        'LSTM_ATT_Prediction_count.npy',
        'LSTM_Prediction_count.npy',
        'GRU_Prediction_count.npy',
        'True_count1.npy'
    ]
    # 绘制每个.npy文件的数据
    for j, filename in enumerate(filenames):
        file_path = f'{folder_path}/{filename}'
        data = np.load(file_path)
        line, = ax.plot(data, label=labels[j])
        if i == 0:  # 只添加一次图例label，避免重复
            lines.append(line)

# 在第3行第2列放置图例
ax_legend = fig.add_subplot(gs[1, 1])
ax_legend.axis('off')  # 隐藏坐标轴
ax_legend.legend(lines, labels, loc='center', fontsize=20, borderaxespad=2)

plt.savefig(f'pictures/组合图.png', dpi=600)
plt.show()
