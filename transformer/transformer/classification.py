import torch
import torch.nn as nn
import torch.nn.functional as F


class Classification(nn.Module):
    def __init__(self, input_dim, num_class):
        super(Classification, self).__init__()

        self.linear = nn.Linear(input_dim, num_class)

    def forward(self, x):
        batch_size = x.size(0)
        # x = x.contiguous().view(batch_size, -1)
        x = torch.max(x, dim=1)[0]
        x = self.linear(x)
        # x = F.softmax(x, dim=1)
        return x
