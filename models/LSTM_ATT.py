import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_outputs):
        attention_weights = torch.tanh(self.attention(lstm_outputs))
        attention_weights = self.context(attention_weights).squeeze(-1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_sum = torch.sum(lstm_outputs * attention_weights.unsqueeze(-1), dim=1)
        return weighted_sum


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, num_layers):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        attn_out = self.attention(lstm_out)
        output = self.fc(attn_out)
        return output
