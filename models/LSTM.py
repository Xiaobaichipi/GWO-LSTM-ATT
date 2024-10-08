import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(Model, self).__init__()
        self.out_dim = out_dim
        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            # nn.ReLU(),
            # nn.Linear(mid_dim, mid_dim),
            nn.Sigmoid(),
            nn.Linear(mid_dim, out_dim),
            nn.Sigmoid()
        )  # regression

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)
        seq_len, batch_size, hid_dim = y.shape
        y = y[-1, :, :]
        # y = y.view(-1, hid_dim)
        y = self.reg(y)
        # y = y.view(self.out_dim, batch_size)
        return y

    """
    PyCharm Crtl+click nn.LSTM() jump to code of PyTorch:
    Examples::
        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)     # 5个时间步，也就是每个时间序列的长度是5,3表示一共有3个时间序列，10表示每个序列在每个时间步的维度是10
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """

    def output_y_hc(self, x, hc):

        y, hc = self.rnn(x, hc)  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, hc
