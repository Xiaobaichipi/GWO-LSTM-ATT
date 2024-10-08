import os

from jiajia.utils.utils import file_choise

path = './data/'
names = ['LSTM', 'GRU', 'LSTM_ATT']
pred_len = 5
seq_list = [5, 10, 20]
for seq_len in seq_list:

    for i in range(5):
        file = file_choise(i)

        for name in names:
            # path = f'./data/{file}/{name}/{name}_pred_seq{seq_len}_pred{pred_len}/'
            # F:\XiaoWang\jiajia\data\5\600877
            path = f'./data/{seq_len}/{file}/'
            old_name = f'{name}_Prediction_count'
            new_name = f'{name}_Prediction_count.npy'

            name1 = f'{path}{old_name}'
            name2 = f'{path}{new_name}'
            # 判断有没有这个文件
            if os.path.exists(name1):
                os.rename(name1, name2)
                print(old_name, '======>', new_name)
            else:
                continue

            # 设置新文件名
