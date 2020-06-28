import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo
import numpy as np


class TrgEmbeddingLayer(nn.Module):
    def __init__(self, w_vocab_size, c_vocab_size,
                 w_embedding_size, c_embedding_size,
                 w_embedding=None, c_embedding=None,
                 options_file=None, weights_file=None,
                 update_embedding=True, model_type='word',
                 input_dropout_p=0.2, trg_pad_idx=0, device=None):
        super(TrgEmbeddingLayer, self).__init__()

        self.model_type = model_type
        self.trg_pad_idx = trg_pad_idx
        self.embedding_size = w_embedding_size
        self.device = device
        if w_embedding is not None:
            self.w_embedding = nn.Embedding.from_pretrained(w_embedding)
        else:
            self.w_embedding = nn.Embedding(w_vocab_size, w_embedding_size)
        self.w_embedding.weight.requires_grad = update_embedding

        if c_embedding is not None:
            self.c_embedding = nn.Embedding.from_pretrained(c_embedding)
        else:
            self.c_embedding = nn.Embedding(c_vocab_size, c_embedding_size)
        self.c_embedding.weight.requires_grad = update_embedding

        self.input_dropout = nn.Dropout(input_dropout_p)
        self.elmo = Elmo(options_file, weights_file, 1)

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

    def forward(self, input_var, c_input=None):
        batch_size = input_var.size(0)
        # mask = input_var.ne(0).byte()
        # # mask = [batch size, seq len]
        # length = mask.sum(1)
        # max_word = max(length)

        max_word = input_var.size(1)
        tgt_mask = self.make_trg_mask(input_var)
        # temp = None
        # if len(mask.size()) > 2:
        #     # word_mask = mask.view(-1, mask.size(2))
        #     # sent_mask = mask.sum(2).ne(0).byte()
        #     max_sent = input_var.size(1)
        #     max_word = input_var.size(2)
        #
        #     temp = torch.zeros(batch_size, mask.size(1) * mask.size(2), dtype=torch.int64).cuda()
        #     for i in range(mask.size(0)):
        #         for j in range(mask.size(1)):
        #             for k in range(mask.size(2)):
        #                 if input_var[i][j][k] != 0:
        #                     temp[i][30 * j + k] = input_var[i][j][k]
        #                 else:
        #                     break
        #     if input_var.is_cuda:
        #         temp.cuda()
        #     tgt_mask = self.make_trg_mask(temp)
        #     # mask = temp.ne(0).byte()
        #
        # else:
        #     max_word = input_var.size(1)
        #     temp = input_var
        #     tgt_mask = self.make_trg_mask(temp)

        if self.model_type == 'elmo' and c_input is not None:
            # embedded = self.w_embedding(input_var)
            # embedded = embedded.view(-1, max_word, self.embedding_size)
            word = c_input[:, :max_word, :].view(-1, max_word, 50)
            embeddings = self.elmo(word)
            elmo_embeddings = embeddings['elmo_representations'][0]

            # embedded = torch.cat((embedded, elmo_embeddings), 2)
            embedded = self.input_dropout(elmo_embeddings)
        else:
            embedded = self.w_embedding(input_var)
            embedded = embedded.view(-1, max_word, self.embedding_size)
            embedded = self.input_dropout(embedded)

        return embedded, tgt_mask
