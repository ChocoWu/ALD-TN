import torch
import torch.nn as nn
from .EncoderLayer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim,  n_layers, n_heads, pf_dim, dropout, device, max_length=200):
        super().__init__()

        # input_dim : embedding layer output size

        self.device = device

        self.linear = nn.Linear(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, input_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len, input dim]
        # src_mask = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, src len]

        src = self.dropout((src * self.scale) + self.pos_embedding(pos))
        # src = [batch size, src len, input dim]

        src = self.linear(src)

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src, attention = layer(src, src_mask)

        # src = [batch size, src len, hid dim]
        # attention = [batch size, n heads, src len, src len]

        return src, attention
