#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, dropout_prob, is_bidirectional, attention_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.dropout_prob = dropout_prob
        self.is_directonal = is_bidirectional
        self.attention_size = attention_size

        if self.is_directonal:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layer,
                            dropout=self.dropout_prob,
                            bidirectional=self.is_directonal,
                            batch_first=True)
        self.drop = nn.Dropout(self.dropout_prob)
        self.linear_1 = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size)
        # self.linear_2 = nn.Linear(self.hidden_size * self.num_directions * self.num_layer, self.attention_size)
        self._reset_parameter()

        self.l_conv = nn.Conv1d(3, 3, 3, padding = 1)

    def _reset_parameter(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def _masking(self, score, mask, score_mask_value=-np.inf):
        """
        def masking(scores, sequence_lengths, score_mask_value=tf.constant(-np.inf)):
            score_mask = tf.sequence_mask(sequence_lengths, maxlen=tf.shape(scores)[1])
            score_mask_values = score_mask_value * tf.ones_like(scores)
            return tf.where(score_mask, scores, score_mask_values)
        如果没有被mask 则从score中选择，否则使用-np.inf进行填充
        :param score: alpha 参数
        :param mask: 二维结构
        :return:
        """
        score_mask_values = score_mask_value * torch.ones_like(score)
        return torch.where(mask, score, score_mask_values)

    def attention_net(self, lstm_output, h, mask):
        """

        :param lstm_output:  batch_size, seq_len, hidden_size * num_directions
        :param h: batch_size, num_layers * num_directions, hidden_size
        :return:
        """
        if self.is_directonal:
            h = h.view(-1, self.num_layer, self.num_directions*self.hidden_size)
            # h =  [batch_size, num_layers, num_directions * hidden_size]
            h = h[:, h.size(1)-1, :].unsqueeze(1)
            # h = [batch size, 1, num_directions * hidden_size]
            # h = self.linear_1(h)
            # h = [batch size, 1, hidden_size]

        # output = self.linear_1(lstm_output)
        # output = [batch_size, seq_len, hidden_size]

        attn = torch.bmm(lstm_output, h.transpose(1, 2)).squeeze(2)
        # attn = [batch_size, seq_len]
        if mask is not None:
            attn = attn * (mask.float())
        attn = attn / (attn.sum(1, keepdim=True) + 1e-13)
        # attn = [batch_size, seq_len]
        attn = attn.unsqueeze(2)
        # attn = [batch_size, seq_len, 1]
        attn = attn.repeat(1, 1, self.num_directions*self.hidden_size)
        # attn = [batch_size, seq_len, hidden_size]

        output = torch.mul(lstm_output, attn)
        # output = [batch_size, seq_len, hidden_size]

        # hidden = torch.reshape(h, [-1, self.num_layer * self.num_directions * self.hidden_size])
        # hidden = self.linear_2(hidden)  # [batch_size, attention_size]
        # # print(hidden.size())
        # atten_lstm = torch.tanh(self.linear_1(lstm_output))  # [batch_size, seq_len, attention_size]
        # # print(atten_lstm.size())
        # # [batch_size, seq_len, attention_size] * [batch_size, attention_size, 1] -> [batch_size, seq_len, 1]
        # # -> [batch_size, seq_len]
        # atten_weights = torch.bmm(atten_lstm, hidden.unsqueeze(2)).squeeze(2)
        # soft_attn_weights = F.softmax(atten_weights, 1)  # [batch_size, seq_len]
        # w = soft_attn_weights * (mask.float())
        # weights = w / (w.sum(1, keepdim=True) + 1e-13)
        #
        # new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), weights.unsqueeze(2)).squeeze(2)

        return output, attn[:, :, 0]

    def pooling_net(self, lstm_out):
        """
        max_pooling
        :param lstm_out:
        :return:
        """
        return torch.max(lstm_out, dim = 1)

    def label_attention(self, lstm_output, label, label_embedding_size, attention_size, mask):
        """

        :param lstm_output:  batch_size, seq_len, hidden_size * num_directions
        :param label: batch_size, embedding_size
        :param label_embedding_size: if use_char 340 else 1324
        :param attention_size: hidden_size * num_directions
        :return:
        """
        w = nn.Parameter(torch.randn(label_embedding_size, attention_size))
        b = nn.Parameter(torch.zeros(300))
        label_embedding = torch.tanh(F.linear(label, w, b))  # [3, attention_size]
        # [batch_size, seq_len, hidden_size * num_directions] -> [batch_size, seq_len, attention_size]
        atten_lstm = torch.tanh(self.linear_1(lstm_output))
        # [batch_size, seq_len, attention_size] * [attention_size, 3] -> [batch_size, seq_len, 3]
        label_embedding = label_embedding.cuda()
        G = torch.matmul(atten_lstm, label_embedding.transpose(0, 1))
        c = torch.norm(label_embedding, dim=1)
        v = torch.norm(lstm_output, dim=2)
        g = torch.matmul(v.unsqueeze(2), c.unsqueeze(0))
        G = torch.div(G, g)  # [batch_size, seq_len, 3]

        # to further capture the relative spatial information among consecutive words, using cnn and activate function
        # 使用conv的结构捕获空间结构
        # u = F.relu(self.l_conv(G.transpose(1, 2)))  # [batch_size, seq_len,  3]
        # calculate the attention weights
        m = torch.max(G, dim=2)[0]
        m = F.softmax(m, 1)
        w = m * (mask.float())
        weights = w / (w.sum(1, keepdim = True) + 1e-13)

        output = torch.bmm(lstm_output.transpose(1, 2), weights.unsqueeze(2)).squeeze(2)

        return output, weights

    def forward(self, x, mask, label=None):
        """

        :param x: batch_size, seq_len, input_dim
        :param mask: batch_size, seq_len
        :return:
            batch_size, num_directions * hidden_size
        """
        # weights = 0
        # length = mask.sum(1)
        # sorted_length, idx = length.sort(0, descending=True)
        # x = x[idx]
        #
        # x = nn.utils.rnn.pack_padded_sequence(x, sorted_length, batch_first=True)
        lstm_output, (h_n, c_n) = self.lstm(x)
        # h_n = [batch_size, num_layers*num_directions, hidden_size]
        # c_n = [batch_size, num_layers*num_directions, hidden_size]

        # lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        # print('lstm_output size', lstm_output.size())
        #
        # _, idx = idx.sort(0, descending=False)
        # lstm_output = lstm_output[idx]
        label_embedding_size = 300
        if label is None:
            output, weights = self.attention_net(lstm_output, h_n, mask)
            # output = [batch_size, seq_len, hidden_size]
            # weights = [batch_size, seq_len]

        else:
            output, weights = self.label_attention(lstm_output, label, label_embedding_size, self.attention_size, mask)

        return output, weights, (h_n, c_n)


# if __name__ == '__main__':
    # input = torch.LongTensor([[[2, 3, 4, 0, 0], [5, 6, 7, 9, 10]], [[2, 3, 4, 0, 0], [0, 0, 0, 0, 0]]])
    # mask = input.ne(0).byte()
    # mask = mask.view(-1, mask.size(2))
    # print('mask', mask)
    # # print(mask)
    # # word_mask = mask.reshape(-1)
    # # print(word_mask)
    # # sent_mask = mask.sum(2)
    # # print(sent_mask)
    # # sent_mask = sent_mask.ne(0).byte()
    # # print(sent_mask)
    # model = LSTM(10, 15, 1, 0.1, True, 10)
    # embedding = nn.Embedding(20, 10)
    # x = embedding(input)
    # print(x.size())
    # print('embedding x', x.size())
    # x = x.view(-1, 5, 10)
    # output = model(x, mask)
    # print(output.size())

