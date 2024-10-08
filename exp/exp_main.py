import numpy as np
import torch
import torch.nn as nn
from utils.metric import metric
from exp.exp_basic import Exp_Basic
from models import _gru, LSTM, LSTM_ATT


class Exp_main(Exp_Basic):
    def __init__(self, count, models, optimizer, criterion, train_loader,
                 test_loader, input_size):
        self.count = count
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.input_size = input_size
        self.models = models

    def _train(self):
        num_epochs = 50
        # train_steps = len(self.train_loader)
        mse = 65532
        patience = 0
        for epoch in range(num_epochs):
            # self.models.train()
            train_loss = []
            for i, (input_feature, input_label) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                outputs = self.models(input_feature)
                loss = self.criterion(outputs, input_label)
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()
                if loss.item() < mse:
                    patience = 0
                    mse = loss.item()
                else:
                    if patience == 3:
                        break
                    else:
                        patience += 1

        test_metrics, pred, true = self._test()
        path = f'checkpoints/{self.count}.pth'
        torch.save(self.models.state_dict(), path)

        return test_metrics, pred, true

    def _test(self):
        # self.models.eval()
        self.models.eval()
        pred = []
        true = []
        with torch.no_grad():
            for i, (input_feature, input_label) in enumerate(self.test_loader):
                predicted = self.models(input_feature).cpu().detach().numpy()
                pred.append(predicted[:, 0])
                true.append(input_label[:, 0].cpu().detach().numpy())
        preds = np.concatenate(pred, axis=0)
        trues = np.concatenate(true, axis=0)
        mae, mse, rmse, mape, mspe, r2 = metric(preds, trues)
        print('mse:{0:.4f}, mae:{1:.4f}, rmse:{2:.4f}, mape:{3:.4f}, mspe:{4:.4f}, r2:{5:.4f}'.format(mse, mae, rmse,
                                                                                                      mape, mspe, r2))
        _metric = [mae, mse, rmse, mape, mspe, r2]
        return _metric, preds,  trues


# class Build_model():
class Build_model:
    def __init__(self, model_name, input_size, hidden_size, output_dim, num_layers, device):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.device = device
        self.model_name = model_name

    def _build_model(self):
        model_dict = {
            'GRU': _gru,
            'LSTM': LSTM,
            'LSTM_ATT': LSTM_ATT
        }
        # model = model_dict[self.model_name].Model(self.input_size, self.hidden_size, self.output_dim, self.num_layers).to(self.device)
        model = model_dict[self.model_name].Model(self.input_size, self.output_dim, self.hidden_size, self.num_layers).to(self.device)      # 新LSTM
        return model
