import torch
import torch.nn as nn
import torch.nn.functional as F


class Multi_Task(nn.Module):
    """ sequence-to-sequence architecture with configurable encoder decoder and classification.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        classification (Classification): object of Classification
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: norm_input, norm_lengths, norm_target, teacher_forcing_ratio, class_input, class_y
        - **norm_input** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **class_input** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **norm_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **norm_target** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **class_y** (list, optional): list of sequences, whose length is the batch size and within which
          each label is a list of token IDs. This information is forwarded to the classification network.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: norm_result(decoder_outputs, decoder_hidden, ret_dict), class_result(classification, attn)
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.
        - **classification** (batch, num_class): batch-length list of tensors with size(num_class) containing the
          outputs of the classification networks
        - **attn**: (batch) optional, if use attention in classification network, we will get the attention weights of each word
          in one sentence,

    """
    def __init__(self, embedding_layer, encoder, decoder, classification,
                 class_encoder=None, norm_encoder=None, decode_function=F.log_softmax, opt=None):
        super(Multi_Task, self).__init__()
        self.embedding_layer = embedding_layer
        self.encoder = encoder
        self.decoder = decoder
        self.class_encoder = class_encoder
        self.norm_encoder = norm_encoder
        self.classification = classification
        self.decode_function = decode_function

        # self.input_size = 200

        self.linear = nn.Linear(opt.s_hidden_size * 2, 2)
        self.linear_1 = nn.Linear(opt.w_hidden_size * 2, opt.w_hidden_size)
        self.linear_2 = nn.Linear(opt.w_hidden_size * 2 * opt.num_layer, opt.w_hidden_size)

    def flatten_parameters(self):
        self.encoder.encode_word.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
        self.norm_encoder.encode_word.flatten_parameters()
        self.class_encoder.word_atten.lstm.flatten_parameters()
        self.class_encoder.sent_atten.lstm.flatten_parameters()

    def get_share_feature(self, share_feature):
        feature = torch.sum(share_feature, dim=1)
        share_features = self.linear(feature)
        return feature, share_features

    def get_private_feature(self, private_feature):
        feature = torch.sum(private_feature, dim=1)
        # private_feature = self.linear(feature)
        return feature

    def forward(self, input_variable, c_inputs, target=None, teacher_forcing_ratio=0, train_type=None):
        # input_variable = [batch_size, max_sent, max_word]
        # c_inputs = [batch_size, max_sent, max_word, 50]
        # target = [batch_size, length] or [batch_size, 1]

        if train_type == 'pre_train':
            embedded, share_enc_embedded, mask, word_mask, sent_mask = self.embedding_layer(input_variable, c_inputs)
            # embedded = [batch_size*max_sent, max_word, embedding_dim]
            # share_enc_embedded = [batch_size*max_sent, max_word, embedding_dim]
            # mask = [batch_size*max_sent, max_word]
            # word_mask = [batch_size*max_sent, max_word]
            # sent_mask = [batch_size, max_sent]

            share_encoder_outputs, share_encoder_hidden, _, _ = self.encoder(embedded, word_mask, sent_mask)
            share_feature, feature = self.get_share_feature(share_encoder_outputs)
            # share_feature = [batch_size, opt.s_hidden_size * 2]
            # feature = [batch_size, 2]

            pri_encoder_outputs, pri_encoder_hidden, _, _ = self.norm_encoder(embedded, word_mask, sent_mask)
            # share_encoder_hidden/pri_encoder_hidden = [batch_size, num_layers*2, hidden_size]
            # share_encoder_outputs/pri_encoder_hidden = [batch_size, max_sent, opt.s_hidden_size * 2]
            private_feature = torch.sum(pri_encoder_outputs, dim=1)

            encoder_outputs = torch.cat([share_encoder_outputs, pri_encoder_outputs], dim=2)
            encoder_hidden = tuple([torch.cat([share_encoder_hidden[i], pri_encoder_hidden[i]], dim=2) for i in range(len(share_encoder_hidden))])

            result = self.decoder(inputs=target,
                                  encoder_hidden=encoder_hidden,
                                  encoder_outputs=encoder_outputs,
                                  function=self.decode_function,
                                  teacher_forcing_ratio=teacher_forcing_ratio)
            return result, share_feature, private_feature, feature
        else:
            embedded, share_enc_embedded, mask, word_mask, sent_mask = self.embedding_layer(input_variable, c_inputs)

            share_encoder_outputs, share_encoder_hidden, _, _ = self.encoder(embedded, word_mask, sent_mask)
            share_feature, feature = self.get_share_feature(share_encoder_outputs)

            pri_encoder_outputs, pri_encoder_hidden, word_weights, sent_weights = self.class_encoder(embedded, word_mask, sent_mask)
            private_feature = torch.sum(pri_encoder_outputs, dim=1)

            result = self.classification(share_feature, private_feature)
            return result, share_feature, private_feature, feature
