import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import string


class Vocabulary(object):
    W_PAD, W_UNK = '<pad>', '<unk>'
    C_PAD, C_UNK = '<pad>', '<unk>'
    SYM_SOS = '<sos>'
    SYM_EOS = '<eos>'
    alphabet = string.printable

    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.word2count = {}

        self.char2id = {}
        self.id2char = {}
        self.char2count = {}

        self.tag2id = {}
        self.id2tag = {}
        self.tag2count = {}

        self._init_vocab()

    def _init_vocab(self):
        for word in [self.W_PAD, self.W_UNK, self.SYM_SOS, self.SYM_EOS]:
            self.word2id[word] = len(self.word2id)
            self.id2word[self.word2id[word]] = word
            self.word2count[word] = 1

        for c in [self.C_PAD, self.C_UNK]:
            self.char2id[c] = len(self.char2id)
            self.id2char[self.char2id[c]] = c
        for c in self.alphabet:
            self.char2id[c] = len(self.char2id)
            self.id2char[self.char2id[c]] = c
            self.char2count[c] = 0

    def add_sentence(self, sentence):
        for word in sentence:
            if word in self.word2id:
                self.word2count[word] += 1
            else:
                self.word2id[word] = len(self.word2id)
                self.id2word[self.word2id[word]] = word
                self.word2count[word] = 1

    def add_word(self, word):
        for char in word:
            if char not in self.char2id:
                self.char2id[char] = len(self.char2id)
                self.id2char[self.char2id[char]] = char

    def add_tags(self, tags):
        for tag in tags:
            if tag not in self.tag2id:
                self.tag2id[tag] = len(self.tag2id)
                self.id2tag[self.tag2id[tag]] = tag

    @property
    def n_words(self):
        return len(self.word2id)

    @property
    def n_chars(self):
        return len(self.char2id)

    @property
    def n_tags(self):
        return len(self.tag2id)

    def word_to_id(self, word):
        return self.word2id.get(word, self.word2id[self.W_UNK])

    def id_to_word(self, id):
        return self.id2word[id]

    def char_to_id(self, char):
        return self.char2id.get(char, self.char2id[self.C_UNK])

    def id_to_char(self, id):
        return self.id2char[id]

    def tag_to_id(self, tag):
        return self.tag2id[tag]

    def id_to_tag(self, id):
        return self.id2tag[id]


class OurDataset(Dataset):
    def __init__(self, word_inputs, char_inputs=None, labels=None, max_word=200, max_char=50):
        # assert len(char_inputs) == len(word_inputs) and len(word_inputs) == len(outputs)
        assert len(word_inputs) == len(labels)
        self.char_inputs = char_inputs
        self.word_inputs = word_inputs
        self.labels = labels
        self.max_word = max_word
        self.max_char = max_char

        # self.collate_fn = lambda x: [self.pad_seq(d, True, 200) for _, d in pd.DataFrame(x).iteritems()]

    def __getitem__(self, item):
        if self.char_inputs is None:
            return torch.tensor(self.word_inputs[item], dtype=torch.long), \
                   torch.tensor(self.labels[item], dtype=torch.long)
        # elif len(self.char_inputs) == 0:
        #     return torch.LongTensor(self.word_inputs[item]), \
        #            torch.LongTensor(self.labels[item])
        else:
            return torch.tensor(self.word_inputs[item], dtype=torch.long), \
                   torch.tensor(self.char_inputs[item], dtype=torch.long), \
                   torch.tensor(self.labels[item], dtype=torch.long)

    def __len__(self):
        return len(self.word_inputs)

    def collate_fn(self, batch):
        """
        Pad a batch, if the longest in the sequence does not exceed max word,
        then pad according to the longest sequence,
        if it exceeds, then pad according to max word
        :param batch:
        :return:
        """
        src, char_inputs, trg = zip(*batch)
        src, src_max_len = self.pad_word_sequence(src, True, self.max_word)
        char_inputs = self.pad_char_sequence(char_inputs, True, src_max_len, self.max_char)
        trg, tgt_max_len = self.pad_word_sequence(trg, True, self.max_word)
        return src, char_inputs, trg

    def pad_word_sequence(self, sequences, batch_first, max_word=200, padding_value=0):
        m_word = max([len(sent) for sent in sequences])
        if m_word < max_word:
            return pad_sequence(sequences, batch_first), m_word
        else:
            max_size = sequences[0].size()
            trailing_dims = max_size[1:]
            m_word = max_word
            if batch_first:
                out_dims = (len(sequences), m_word) + trailing_dims
            else:
                out_dims = (m_word, len(sequences)) + trailing_dims

            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                if batch_first:
                    if length > m_word:
                        out_tensor[i, :m_word - 1, ...] = tensor[:m_word - 1]
                        out_tensor[i, max_word - 1, ...] = tensor[-1]
                    else:
                        out_tensor[i, :length, ...] = tensor
                else:
                    if length > m_word:
                        out_tensor[:m_word - 1, i, ...] = tensor[:m_word - 1]
                        out_tensor[m_word - 1, i, ...] = tensor[-1]
                    else:
                        out_tensor[:length, i, ...] = tensor

            return out_tensor, m_word

    def pad_char_sequence(self, sequences, batch_first, max_word, max_char, padding_value=0):
        """
        针对的是word_sentence_level的pad
        :param sequence: [batch_size, max_sent_num, max_word_num]
        :param batch_first:
        :param pad_value: 默认为0
        :return:
        """

        if batch_first:
            out_dims = (len(sequences), max_word, max_char)
        else:
            out_dims = (max_word, max_char, len(sequences))

        # out_tensor = sequences[0][0].data.new(*out_dims).fill_(padding_value)
        out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
        # print(out_tensor)
        for i, tensors in enumerate(sequences):
            tensors = tensors[:max_word]
            for j, tensor in enumerate(tensors):
                length = len(tensor)
                if length >= max_char:
                    if batch_first:
                        out_tensor[i, j, :max_char, ...] = tensor[:max_char]
                    else:
                        out_tensor[j, :max_char, i, ...] = tensor[:max_char]
                else:
                    if batch_first:
                        out_tensor[i, j, :length, ...] = tensor
                    else:
                        out_tensor[j, :length, i, ...] = tensor

        return out_tensor
