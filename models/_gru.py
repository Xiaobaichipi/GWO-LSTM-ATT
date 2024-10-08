import torch
import torch.nn as nn


class Model(nn.Module):
    """
    suggest: keep seq_len = label_len = pred_len
    """

    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(Model, self).__init__()
        self.out_dim = out_dim
        self.gru = nn.GRU(inp_dim, mid_dim, num_layers=2, batch_first=True)
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression

    def forward(self, x):
        h_0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size) \
            .to(x.device)
        y = self.gru(x)[0]  # y, (h, c) = self.rnn(x)
        seq_len, batch_size, hid_dim = y.shape
        y = y[-1, :, :]
        # y = y.view(-1, hid_dim)
        y = self.reg(y)
        # y = y.view(self.out_dim, batch_size)
        return y
