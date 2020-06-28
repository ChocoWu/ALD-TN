#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import LSTM


class Hi_Attention(nn.Module):
    def __init__(self, config):
        super(Hi_Attention, self).__init__()
        self.config = config
        # self.vocab_size = config.vocab_size
        self.embedding_size = config.w_embedding_size
        self.dropout_prob = config.dropout_prob
        self.num_class = config.num_class
        self.max_sent = config.max_sent
        self.max_word = config.max_word

        self.w_num_layer = config.w_num_layer
        self.w_hidden_size = config.w_hidden_size
        self.w_atten_size = config.w_atten_size
        self.w_is_bidirectional = config.w_is_bidirectional
        self.w_dropout_prob = config.w_dropout_prob

        self.s_num_layer = config.s_num_layer
        self.s_hidden_size = config.s_hidden_size
        self.s_atten_size = config.s_atten_size
        self.s_is_bidirectional = config.s_is_bidirectional
        self.s_dropout_prob = config.s_dropout_prob

        self.model_type = config.model_type

        if self.w_is_bidirectional:
            self.w_num_directions = 2
        else:
            self.w_num_directions = 1

        if self.s_is_bidirectional:
            self.s_num_directions = 2
        else:
            self.s_num_directions = 1

        if self.model_type == 'elmo':
            self.w_input_size = self.embedding_size + 512
        else:
            self.w_input_size = self.embedding_size

        self.word_atten = LSTM(self.w_input_size, self.w_hidden_size, self.w_num_layer,
                               self.w_dropout_prob, self.w_is_bidirectional, self.w_atten_size)
        self.s_input_size = self.w_num_directions * self.w_hidden_size
        self.sent_atten = LSTM(self.s_input_size, self.s_hidden_size, self.s_num_layer,
                               self.s_dropout_prob, self.s_is_bidirectional, self.s_atten_size)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.linear = nn.Linear(self.s_hidden_size * self.s_num_directions, 300)

    def forward(self, input, word_mask, sent_mask, label=None):
        """

        :param input: [batch_size*max_sent, max_word, input_dim]
        :param word_mask: [batch_size * max_sent, max_word]
        :param sent_mask: [batch_sie, max_sent]
        :return:
            [batch_size, num_class]
        """
        if label is not None:
            w_x, word_weights, _ = self.word_atten(input, word_mask, label=label)
            # x = [batch size * max_sent, max_word, w_hidden_size * w_num_directions]
            # word_weights = [batch size * max_sent, max_word]
            x = self.dropout(w_x)

            x = torch.sum(x, dim=1)
            # x = [batch size * max_sent, w_hidden_size * w_num_directions]

            dim = x.size(1)
            x = x.view(-1, self.max_sent, dim)
            # x = [batch size, max_sent, w_hidden_size * w_num_directions]

            x, sent_weights, h_n = self.sent_atten(x, sent_mask, label=label)
            # x = [batch_size, max_sent, s_hidden_size * s_num_directions]
            # sent_weights = [batch_size, max_sent]
            # h_n = [batch_size, num_layers, s_num_directions * s_hidden_size]
            x = self.dropout(x)

            # x = torch.sum(x, dim=1)
            # x = [batch_size, s_hidden_size * s_num_directions]

            x = self.linear(x)
            # x = [batch_size, max_sent, 300]
        else:
            w_x, word_weights, _ = self.word_atten(input, word_mask)
            # x = [batch size * max_sent, max_word, w_hidden_size * w_num_directions]
            # word_weights = [batch size * max_sent, max_word]
            x = self.dropout(w_x)

            x = torch.sum(x, dim=1)
            # x = [batch size * max_sent, w_hidden_size * w_num_directions]

            dim = x.size(1)
            x = x.view(-1, self.max_sent, dim)
            # x = [batch size, max_sent, w_hidden_size * w_num_directions]

            x, sent_weights, h_n = self.sent_atten(x, sent_mask)
            # x = [batch_size, max_sent, s_hidden_size * s_num_directions]
            # sent_weights = [batch_size, max_sent]
            # h_n = tuple([batch_size, num_layers*s_num_directions, s_hidden_size])
            x = self.dropout(x)
            # x = [batch_size, max_sent, s_hidden_size * s_num_directions]

            # x = torch.sum(x, dim=1)
            # x = [batch_size, s_hidden_size * s_num_directions]

            # x = self.linear(x)
            # x = [batch_size, 300]

        return x, h_n, word_weights, sent_weights


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train_file", type = str, default = "./data/english/agr_en_train.csv")
#     parser.add_argument("--valid_file", type = str, default = "./data/english/agr_en_dev.csv")
#     parser.add_argument("--test_file", type = str, default = "./data/english/agr_en_fb_test.csv")
#     # parser.add_argument("--tag_format", type = str, choices = ["bio", "bmes"], default = "bio")
#
#     parser.add_argument("--save_dir", type = str, default = "./checkpoint/")
#     parser.add_argument("--log_dir", type = str, default = "./log/")
#     parser.add_argument("--config_path", type = str, default = "./checkpoint/config.pt")
#     parser.add_argument("--continue_train", type = bool, default = False, help = "continue to train model")
#     parser.add_argument("--pretrain_embedding", type = bool, default = True)
#     parser.add_argument("--embedding_file", type = str, default = "./data/glove.840B.300d.txt")
#
#     parser.add_argument("--seed", type = int, default = 123, help = "seed for random")
#     parser.add_argument("--batch_size", type = int, default = 64, help = "number of batch size")
#     parser.add_argument("--epochs", type = int, default = 100, help = "number of epochs")
#     parser.add_argument("--embedding_size", type = int, default = 300)
#     parser.add_argument("--lr", type = float, default = 1e-3, help = "learning rate of adam")
#     parser.add_argument("--weight_decay", type = float, default = 1e-5, help = "weight decay of adam")
#     parser.add_argument("--patience", type = int, default = 8)
#     parser.add_argument("--freeze", type = int, default = 10)
#     parser.add_argument("--num_class", type = int, default = 3)
#     parser.add_argument("--dropout_prob", type = float, default = 0.5)
#     parser.add_argument("--max_sent", type = int, default = 6)
#     parser.add_argument("--max_word", type = int, default = 35)
#
#     parser.add_argument("--w_hidden_size", type = int, default = 150)
#     parser.add_argument("--w_num_layer", type = int, default = 1, help = "number of layers")
#     parser.add_argument("--w_atten_size", type = int, default = 300)
#     parser.add_argument("--w_is_bidirectional", type = bool, default = True)
#     parser.add_argument("--w_dropout_prob", type = float, default = 0.5)
#
#     parser.add_argument("--s_hidden_size", type = int, default = 150)
#     parser.add_argument("--s_num_layer", type = int, default = 1, help = "number of layers")
#     parser.add_argument("--s_atten_size", type = int, default = 300)
#     parser.add_argument("--s_is_bidirectional", type = bool, default = True)
#     parser.add_argument("--s_dropout_prob", type = float, default = 0.5)
#
#     parser.add_argument("--is_use_char", type = bool, default = False)
#     parser.add_argument("--char_encode_type", type = str, default = 'lstm')
#     parser.add_argument("--c_embedding_size", type = int, default = 50)
#     parser.add_argument("--c_hidden_size", type = int, default = 20)
#     parser.add_argument("--c_num_layer", type = int, default = 1)
#     parser.add_argument("--c_is_bidirectional", type = bool, default = True)
#     parser.add_argument("--c_dropout_prob", type = float, default = 0.5)
#     parser.add_argument("--c_kernel_size", type = list, default = [2])
#     parser.add_argument("--num_filter", type = int, default = 2)
#
#     parser.add_argument("--use_gpu", type = bool, default = True)
#     args = parser.parse_args()
#     args.vocab = None
#     # train_dataset, valid_dataset, test_dataset, vocab = data_utils.load_data(args, args.vocab)
#     # args.vocab = vocab
#     args.vocab_size = 100
#     args.n_tags = 3
#     args.alphabet_size = 36
#
#     model = Hi_Attention(args)
#     print(model)
#     for name, param in model.named_parameters():
#         print(name, param.size())
