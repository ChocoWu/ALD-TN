import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Classification(nn.Module):
    def __init__(self, hidden_size, num_class):
        super(Classification, self).__init__()

        self.hidden_size = hidden_size
        self.num_class = num_class
        self.fc = nn.Linear(self.hidden_size * 4, self.num_class)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, pri_enc, share_enc=None):
        # share_enc = [batch_size, max_sent, input_dim]
        # pri_enc = [batch_size, max_sent, input_dim]

        # share_enc = torch.sum(pri_enc, dim=1)
        # pri_enc = torch.sum(pri_enc, dim=1)
        # share_enc = [batch_size, input_dim]
        # pri_enc = [batch_size, input_dim]

        output = torch.cat((pri_enc, share_enc), dim=1)
        output = self.fc(output)
        # output = [batch_size, num_class]

        return output

