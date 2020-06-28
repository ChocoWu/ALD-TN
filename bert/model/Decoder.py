import torch
import torch.nn as nn
from .DecoderLayer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device,
                 max_length=100, embedding=None, trg_pad_idx=0, enc_out_dim=1536):
        super().__init__()

        # input_dim : embedding layer output size
        # output_dim : target vocabulary size

        self.device = device
        self.trg_pad_idx = trg_pad_idx
        self.linear = nn.Linear(input_dim, hid_dim)
        self.enc_linear = nn.Linear(enc_out_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, input_dim)
        if embedding is not None:
            self.tok_embedding = nn.Embedding.from_pretrained(embedding)
        else:
            self.tok_embedding = nn.Embedding(output_dim, input_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        # temp = torch.ones((trg_len, trg_len), device=self.device)
        # temp = torch.tril(temp)
        # trg_sub_mask = temp.bool()

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).to(torch.uint8)

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, trg, enc_src, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, bert_dim*2]
        # src_mask = [batch size, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        trg_mask = self.make_trg_mask(trg)
        # trg_mask = [batch size, trg len]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        # trg = self.dropout((trg * self.scale) + self.pos_embedding(pos))
        # trg = [batch size, trg len, input dim]

        trg = self.linear(trg)
        # trg = [batch size, trg len, hid dim]

        enc_src = self.enc_linear(enc_src)
        # trg = [batch size, src len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        # output = [batch size, trg len, output dim]

        return output, attention

