import torch
import numpy as np
import torch.nn as nn
from utils.print_color import Colors, colored_print
from utils.plot import plot
from exp.exp_main import Exp_main
import time
import gc


class GreyWolfOptimizer:
    def __init__(self, dim, lb, ub, num_wolves, max_iter, Model,
                 input_size, output_size, device,
                 scaler_y, seq_len, pred_len, train_loader,
                 test_loader, file):
        self.dim = dim
        self.lb = torch.tensor(lb)
        self.ub = torch.tensor(ub)
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.Model = Model
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.scaler_y = scaler_y
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.file = file

    def optimize(self):
        print(
            '<---------------------------------------------开始灰狼算法--------------------------------------------->')
        # 1. 初始化灰狼群体的位置
        wolves = [self.lb + (self.ub - self.lb) * np.random.rand(self.dim) for _ in range(self.num_wolves)]
        # wolves = torch.tensor(wolves, dtype=torch.float32)
        patience = 3
        count = 0
        min_fitness = 65532
        _time = str(0)
        _matrix = [0, 0, 0, 0, 0, 0]
        indices = []
        for iteration in range(self.max_iter):
            print("Epoch: {}".format(iteration + 1))
            # 2. 计算每只灰狼的适应度
            print(
                '<---------------------------------------------计算每只灰狼的适应度--------------------------------------------->')
            results = [self.obj_function(wolf.numpy(), iteration, min_fitness) for wolf in wolves]
            results = np.asarray(results)
            fitness, _time1, _matric1 = results[:, 0], results[:, 1], results[:, 2]
            fitness = fitness.astype(float)
            fitness = torch.tensor(fitness)

            # 3. 将fitness置为负值
            neg_fitness = -fitness

            # 4. 根据torch.topk(fitness, k=3)选取3个最大的值
            topk = torch.topk(neg_fitness, k=5)
            indices = topk.indices
            tmp = fitness[indices[0]]  # 获取其中最小的
            if tmp < min_fitness:
                colored_print(Colors.GREEN
                              ,
                              '<---------------------------------------------更新成功--------------------------------------------->')
                min_fitness = tmp
                _time = _time1[indices[0]]
                _matrix = _matric1[indices[0]]
                count = 0
            else:
                colored_print(Colors.RED
                              ,
                              f'<---------------------------------------------更新失败{count}次--------------------------------------------->')
                count += 1

            if count >= patience:
                break

            # 5. 将alpha, beta, delta取list[0,1,2]
            print(
                '<---------------------------------------------更新五个最优位置--------------------------------------------->')
            alpha, beta, delta, xiaowang, jiajia = wolves[indices[0]], wolves[indices[1]], wolves[indices[2]], wolves[
                indices[3]], wolves[indices[4]]

            # 6. 根据alpha, beta, delta的对应下标，获取对应的超参数组合
            a = 2 - iteration * (2 / self.max_iter)
            print(
                '<---------------------------------------------开始更新粒子--------------------------------------------->')
            for i in range(len(wolves)):
                A1, A2, A3, A4, A5 = 2 * a * torch.rand(self.dim) - a, 2 * a * torch.rand(
                    self.dim) - a, 2 * a * torch.rand(
                    self.dim) - a, 2 * a * torch.rand(self.dim) - a, 2 * a * torch.rand(self.dim) - a
                C1, C2, C3, C4, C5 = 2 * torch.rand(self.dim), 2 * torch.rand(self.dim), \
                                     2 * torch.rand(self.dim), 2 * torch.rand(self.dim), 2 * torch.rand(self.dim)

                D_alpha = torch.abs(C1 * alpha - wolves[i])
                D_beta = torch.abs(C2 * beta - wolves[i])
                D_delta = torch.abs(C3 * delta - wolves[i])
                D_xiaowang = torch.abs(C4 * xiaowang - wolves[i])
                D_jiajia = torch.abs(C4 * jiajia - wolves[i])

                X1 = alpha - A1 * D_alpha
                X2 = beta - A2 * D_beta
                X3 = delta - A3 * D_delta
                X4 = xiaowang - A4 * D_xiaowang
                X5 = jiajia - A5 * D_jiajia

                # 更新位置并确保在边界内
                wolves[i] = torch.clamp((X1 + X2 + X3 + X4 + X5) / 5, self.lb, self.ub)

                print('更新第{}个粒子 | 参数为:{}-{}-{}'.format(i + 1, wolves[i][0], wolves[i][1], wolves[i][2]))
            print(
                '<---------------------------------------------结束更新粒子--------------------------------------------->')
        print(
            '<---------------------------------------------更新粒子结束--------------------------------------------->')
        best_wolf = wolves[indices[0]].numpy()
        # best_fitness, _ = self.obj_function(best_wolf, iteration, min_fitness, state='test')
        return best_wolf, _time, _matrix

    def obj_function(self, params, count, min_score):
        hidden_size = int(params[0])
        num_layers = int(params[1])
        lr = params[2]

        print('hidden_size:{} | num_layers:{} | lr:{}'.format(hidden_size, num_layers, lr))

        models = self.Model(self.input_size, hidden_size, self.output_size, num_layers).to(self.device)
        optimizer = torch.optim.Adam(models.parameters(), lr=lr)
        criterion = nn.MSELoss().to(self.device)

        exp = Exp_main(count, models, optimizer, criterion, self.train_loader,
                       self.test_loader, self.input_size)
        """
        count, models, optimizer, criterion, train_loader,
                 test_input_tensor, input_size, test_label
        """

        _metric, pred, true = exp._train()
        mae, mse, rmse, mape, mspe, r2 = _metric
        _metric = np.asarray(_metric)
        # 将 R² 转为负值以便于优化器最小化
        neg_r2 = -r2

        # 删除模型和优化器
        del models
        del optimizer

        # 使用垃圾回收
        gc.collect()

        # 清空 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 加权目标函数，优先考虑MSE
        weight_mse = 1 / 2
        # weight_neg_r2 = 0.01
        weight = 1 / 2
        # weighted_sum = weight_mse * mse + weight_neg_r2 * neg_r2 + weight * mae + weight * rmse
        weighted_sum = weight_mse * mse + weight * mae
        time1 = time.gmtime()
        time2 = time.strftime("%Y%m%d%H%M%S", time1)
        time_numeric = int(time2)

        if (weighted_sum < min_score) and (count != 0):
            plot(pred.reshape(-1, 1), true.reshape(-1, 1), count, self.scaler_y, self.seq_len, self.pred_len, self.file)
            # pred, true, count, scaler_y, seq_len, pred_len
            f = open("data/GWO_LSTM.txt", 'a')
            f.write(
                'now:{} | hidden_size:{} | num_layers:{} | lr:{} | iteration:{}|'.format(time_numeric, hidden_size,
                                                                                         num_layers, lr,
                                                                                         count))
            colored_print(Colors.BLUE,
                          '<---------------------------------------------写入成功--------------------------------------------->')
            f.write("mse:{} | mae:{} | rmse:{} | mape:{} | mspe:{} | r2:{}".format(mse, mae, rmse, mape, mspe, r2))
            f.write('\n')
            f.close()

        # 删除不再需要的变量
        del mae, mse, rmse, mape, mspe, r2
        del time1, time2, weight_mse, weight

        return weighted_sum, time_numeric, _metric
