import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, input_size):
        super(Linear, self).__init__()
        # self.fc = nn.Linear(2 * input_size, 1)
        self.fc = nn.Linear(input_size, 1)
    
    def forward(self, x1, x2):
        # x = torch.sub((x1, x2), dim=1)
        x = torch.sub(x1, x2)
        return torch.sigmoid(self.fc(x))

class DNN(nn.Module):
    def __init__(self, input_size, sizes = [], dropout = 0.4):
        super(DNN, self).__init__()
        sizes.insert(0, 2 * input_size)
        sizes.append(1)
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i != len(sizes) - 2:
                layers.append(nn.PReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.net(x)
        return torch.sigmoid(x)
    
class Copy2021(nn.Module):
    def __init__(self, input_size):
        super(Copy2021, self).__init__()
        self.fc1 = nn.Linear(2 * input_size, 64)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return torch.sigmoid(x)