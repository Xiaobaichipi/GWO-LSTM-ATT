import torch
import random
import numpy as np

def _seed():
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
def file_choise(data_name):
    # 文件名
    file = 'null'
    if data_name == 0:
        print('银行->工商银行->0\n')
        file = '601398'
    elif data_name == 1:
        print('饮料制造->贵州茅台->1\n')
        file = '600519'
    elif data_name == 2:
        print('保险->中国平安->2\n')
        file = '601318'
    elif data_name == 3:
        print('通信设备->中兴通讯->3\n')
        file = '000063'
    elif data_name == 4:
        print('芯片->电科芯片->4\n')
        file = '600877'
    else:
        print('你的选项是错的，没有这个数据集，程序结束')

    return file

def calculate_percentage_increase(original_value, new_value):
    # 计算增量
    increase = new_value - original_value

    # 计算提升百分比
    percentage_increase = (increase / original_value) * 100

    return percentage_increase