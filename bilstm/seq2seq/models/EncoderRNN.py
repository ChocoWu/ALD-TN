import torch.nn as nn
import torch


class EncoderRNN(nn.Module):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)

    """

    def __init__(self, w_embedding_size, w_hidden_size,
                 c_embedding_size, c_hidden_size,dropout_p=0.5,
                 n_layers=1, variable_lengths=False, use_type='word'):
        super(EncoderRNN, self).__init__()

        self.use_type = use_type

        self.variable_lengths = variable_lengths

        self.encode_char = nn.LSTM(input_size=c_embedding_size,
                                   hidden_size=c_hidden_size,
                                   bidirectional=True,
                                   dropout=dropout_p,
                                   num_layers=n_layers,
                                   batch_first=True)
        self.max_pool = nn.MaxPool1d(c_hidden_size * 2, stride=1)
        if self.use_type == 'elmo':
            encoder_word_input_size = w_embedding_size + 512
        elif self.use_type == 'char':
            encoder_word_input_size = w_embedding_size + 20
        else:
            encoder_word_input_size = w_embedding_size
        self.encode_word = nn.LSTM(input_size=encoder_word_input_size,
                                   hidden_size=w_hidden_size,
                                   bidirectional=True,
                                   dropout=dropout_p,
                                   num_layers=n_layers,
                                   batch_first=True)
        # self.input_dropout = nn.Dropout(input_dropout_p)

    def char_encode(self, c_input):
        max_word = c_input.size(2)
        c_input = c_input.view(-1, max_word)
        mask = c_input.ne(0).byte()
        length = mask.sum(1)
        sorted_legth, idx = length.sort(0, descending=True)
        c_input = self.c_embedding(c_input)
        c_input = self.input_dropout(c_input)
        # c_input = c_input[idx]
        # c_input = nn.utils.rnn.pack_padded_sequence(c_input, sorted_legth, batch_first=True)
        h, _ = self.encode_char(c_input)
        # h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        # _, idx = idx.sorted(0, descending=False)
        # h = h[idx]
        h = self.max_pool(h)
        return h

    def forward(self, embedding_input, mask):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """

        # length = mask.sum(1)
        # sorted_lendth, idx = length.sort(0, descending=True)
        #
        # embedded = embedding_input[idx]
        # if self.variable_lengths:
        #     embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_lendth, batch_first=True)
        # print(embedded)
        output, hidden = self.encode_word(embedding_input)
        # if self.variable_lengths:
        #     output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # _, idx = idx.sort(0, descending=False)
        # output = output[idx]
        return output, hidden
