import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo
import numpy as np


class SrcEmbeddingLayer(nn.Module):
    def __init__(self, w_vocab_size, c_vocab_size,
                 w_embedding_size, c_embedding_size,
                 w_embedding=None, c_embedding=None,
                 options_file=None, weights_file=None,
                 update_embedding=True, model_type='word',
                 input_dropout_p=0.2, src_pad_idx=0):
        super(SrcEmbeddingLayer, self).__init__()

        self.model_type = model_type
        self.src_pad_idx = src_pad_idx
        self.embedding_size = w_embedding_size
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

    def make_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def forward(self, input_var, c_input=None):
        batch_size = input_var.size(0)
        # mask = input_var.ne(0).byte()
        # word_mask, sent_mask = None, None
        max_word = input_var.size(1)
        src_mask = self.make_src_mask(input_var)
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
        #     src_mask = self.make_src_mask(temp)
        #     # mask = temp.ne(0).byte()
        #
        # else:
        #     max_word = input_var.size(1)
        #     temp = input_var
        #     src_mask = self.make_src_mask(temp)

        if self.model_type == 'elmo' and c_input is not None:
            # use [Glove, ELMo]
            embedded = self.w_embedding(input_var)
            embedded = embedded.view(-1, max_word, self.embedding_size)
            word = c_input[:, :max_word, :].view(-1, max_word, 50)
            embeddings = self.elmo(word)
            elmo_embeddings = embeddings['elmo_representations'][0]

            embedded = torch.cat((embedded, elmo_embeddings), 2)
            embedded = self.input_dropout(embedded)
        else:
            # directly use GLove
            embedded = self.w_embedding(input_var)
            embedded = embedded.view(-1, max_word, self.embedding_size)
            embedded = self.input_dropout(embedded)

        return embedded, src_mask
