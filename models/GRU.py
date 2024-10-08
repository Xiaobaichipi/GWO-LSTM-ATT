import torch
import torch.nn as nn


class Model(nn.Module):
    """
    suggest: keep seq_len = label_len = pred_len
    """

    def __init__(self, input_size, hidden_size, output_dim, num_layers):
        super(Model, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size) \
            .to(x.device)

        output, _ = self.gru(x, h_0)
        output = self.fc(output[:, -1, :])  # Use the last output of the sequence
        return output
