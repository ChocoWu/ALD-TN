import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo
import numpy as np


class Embedding(nn.Module):
    def __init__(self, w_vocab_size, c_vocab_size,
                 w_embedding_size, c_embedding_size,
                 w_embedding=None, c_embedding=None,
                 options_file=None, weights_file=None,
                 update_embedding=True, model_type='word',
                 input_dropout_p=0):
        super(Embedding, self).__init__()

        self.model_type = model_type
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

    def forward(self, input_var, c_input=None, use_type='word'):
        batch_size = input_var.size(0)
        mask = input_var.ne(0).byte()
        word_mask, sent_mask = None, None
        max_sent, max_word = 1, 0
        temp = None
        if len(mask.size()) > 2:
            word_mask = mask.view(-1, mask.size(2))
            sent_mask = mask.sum(2).ne(0).byte()
            max_sent = input_var.size(1)
            max_word = input_var.size(2)


            temp = torch.zeros(batch_size, mask.size(1)*mask.size(2), dtype=torch.int64).cuda()
            for i in range(mask.size(0)):
                for j in range(mask.size(1)):
                    for k in range(mask.size(2)):
                        if input_var[i][j][k] != 0:
                            temp[i][30*j+k] = input_var[i][j][k]
                        else:
                            break
            if input_var.is_cuda:
                temp.cuda()
            mask = temp.ne(0).byte()

        else:
            max_word = input_var.size(1)
            temp = input_var

        if self.model_type == 'elmo':
            embedded = self.w_embedding(input_var)
            share_enc_embedded = self.w_embedding(temp)
            embedded = self.input_dropout(embedded)
            embedded = embedded.view(-1, max_word, self.embedding_size)
            word = c_input.view(-1, max_word, 50)
            share_enc_word = c_input.view(-1, max_word * max_sent, 50)
            embeddings = self.elmo(word)
            share_enc_embeddings = self.elmo(share_enc_word)
            elmo_embeddings = embeddings['elmo_representations'][0]
            share_enc_elmo_embeddings = share_enc_embeddings['elmo_representations'][0]
            # c_encode = c_encode.squeeze().view(batch_size, max_word, -1)

            embedded = torch.cat((embedded, elmo_embeddings), 2)
            share_enc_embedded = torch.cat((share_enc_embedded, share_enc_elmo_embeddings), 2)

        elif self.model_type == 'char':
            embedded = self.w_embedding(input_var)
            share_enc_embedded = self.w_embedding(temp)
            embedded = self.input_dropout(embedded)
            share_enc_embedded = self.input_dropout(share_enc_embedded)
            c_encode = self.char_encode(c_input)
            c_encode = c_encode.squeeze().view(batch_size, max_word, -1)
            embedded = torch.cat((embedded, c_encode), 2)
        # print(c_encode.size())
        else:
            embedded = self.w_embedding(input_var)
            share_enc_embedded = self.w_embedding(temp)
            embedded = self.input_dropout(embedded)
            share_enc_embedded = self.input_dropout(share_enc_embedded)
            embedded = embedded.view(-1, max_word, self.embedding_size)

        return embedded, share_enc_embedded, mask, word_mask, sent_mask
