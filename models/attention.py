import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, game_size, opp_size, attention_sizes=[], attention_dropout=0.25, hidden_sizes=[], hidden_dropout=0.4):
        super(Attention, self).__init__()
        self.game_size = game_size
        self.opp_size = opp_size
        attention_sizes.insert(0, opp_size)
        attention_sizes.append(1)
        layers = []
        for i in range(len(attention_sizes) - 1):
            layers.append(nn.Linear(attention_sizes[i], attention_sizes[i + 1]))
            if i != len(attention_sizes) - 2:
                layers.append(nn.PReLU())
            layers.append(nn.Dropout(float(attention_dropout)))
        self.attention = nn.Sequential(*layers)
        hidden_sizes.insert(0, game_size)
        hidden_sizes.append(1)
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if i != len(hidden_sizes) - 2:
                layers.append(nn.PReLU())
            layers.append(nn.Dropout(float(hidden_dropout)))
        self.prediction_head = nn.Sequential(*layers)
    
    def forward(self, x1, x2):
        # todo mask non existent games (sum of games is 0)
        x1_games, x1_opp = torch.split(x1, [self.game_size, self.opp_size], dim = 2) # (batch_size, num_games, game_size + opp_size)

        x2_games, x2_opp = torch.split(x2, [self.game_size, self.opp_size], dim = 2)
        attention1 = self.attention(x1_opp)
        attention2 = self.attention(x2_opp)
        attention1 = torch.softmax(attention1, dim = 1) # (batch_size, num_games, 1)
        attention2 = torch.softmax(attention2, dim = 1)
        x1_embed = torch.bmm(attention1.transpose(1, 2), x1_games).squeeze(1)
        x2_embed = torch.bmm(attention2.transpose(1, 2), x2_games).squeeze(1)
        # print(attention1)
        # print(x1_embed)

        return torch.sigmoid(self.prediction_head(torch.sub(x1_embed, x2_embed)))