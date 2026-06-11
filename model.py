import torch
import torch.nn as nn

# ------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size = 12, hidden_size = 64, num_layers = 2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, dropout = 0.2)
        self.fc = nn.Linear(hidden_size, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out  # Weighted the loss, self.sigmoid(out) no longer needed