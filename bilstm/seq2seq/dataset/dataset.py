import torch
from torch.utils.data import Dataset
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
    def __init__(self, word_inputs, char_inputs=None, labels=None):
        # assert len(char_inputs) == len(word_inputs) and len(word_inputs) == len(outputs)
        assert len(word_inputs) == len(labels)
        self.char_inputs = char_inputs
        self.word_inputs = word_inputs
        self.labels = labels

        # self.collate_fn = lambda x: [pad_sequence(d, True) for _, d in pd.DataFrame(x).iteritems()]

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
