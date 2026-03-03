import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, lstm_size, fc_size):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(2 * input_size, lstm_size, num_layers=1, batch_first=True, dropout=0.5)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(lstm_size, fc_size)
        self.act = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(fc_size, 1)
    
    def forward(self, x1, x2):
        h = torch.concat((x1, x2), dim=2) # (N, L, I) -> (N, L, 2 * I)
        h = self.lstm1(h)[0][:, -1, :] # (N, L, H) -> (N, H)
        h = self.dropout1(h)
        x = self.fc1(h)
        x = self.act(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

class AvgLSTM(nn.Module):
    def __init__(self, input_size, lstm_size, fc_size):
        super(AvgLSTM, self).__init__()
        self.lstm = LSTM(input_size, lstm_size, fc_size)

    def forward(self, x1, x2):
        path1_p = self.lstm(x1, x2)
        path2_p = self.lstm(x2, x1)
        return (path1_p + (1 - path2_p)) / 2