import pandas as pd
import numpy as np
import json
from transformer.dataset.dataset import OurDataset, Vocabulary
from transformer.util.utils import *
from allennlp.modules.elmo import batch_to_ids


def load_data_with_diff_vocab(inputFile, src_vocab, tgt_vocab, max_word=200, max_char=50, data_type ='norm'):
    """
    use different to load data, src_vocab, tgt_vocab, src_vocab including the lexnorm2015 and aggressive dataset
    :param inputFile:
    :param src_vocab:
    :param tgt_vocab:
    :param max_word:
    :param max_char:
    :param data_type:
    :return:
    """
    char_inputs = []
    word_inputs = []
    outputs = []
    src, tgt = pickle.load(open(inputFile, 'rb'))
    for s, t in zip(src, tgt):
        elmo_id = batch_to_ids(s)
        elmo_id = elmo_id.view(-1, 50)
        tokens = ['<sos>'] + s + ['<eos>']
        word_input = [src_vocab.word_to_id(word) for word in tokens]
        if data_type == 'norm':
            output = [tgt_vocab.word_to_id(tgt_vocab.SYM_SOS)]
            output.extend([tgt_vocab.word_to_id(word) for word in t])
            output.append(tgt_vocab.word_to_id(tgt_vocab.SYM_EOS))
        else:
            output = [tgt_vocab.tag_to_id(t)]
        char_inputs.append(elmo_id)
        word_inputs.append(word_input)
        outputs.append(output)
    dataset = OurDataset(word_inputs, char_inputs, outputs, max_word, max_char)
    return dataset


def build_src_tgt_vocab(inputFile, src_vocabFile, tgt_vocabFile, label_list):
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src, tgt = pickle.load(open(inputFile, 'rb'))
    for s, t in zip(src, tgt):
        src_vocab.add_sentence(s)
        tgt_vocab.add_sentence(t)
        [src_vocab.add_word(word) for word in s]
        [tgt_vocab.add_word(word) for word in t]

    tgt_vocab.add_tags(label_list)
    save_to_pickle(src_vocabFile, src_vocab)
    save_to_pickle(tgt_vocabFile, tgt_vocab)

    return src_vocab, tgt_vocab


def load_embedding(filename, embedding_size, vocab):
    embeddings_index = {}
    with open(filename, encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    scale = np.sqrt(3.0 / embedding_size)
    embedding = np.random.uniform(-scale, scale, (vocab.n_words, embedding_size))

    for word, vector in embeddings_index.items():
        if word in vocab.word2id:
            embedding[vocab.word2id[word]] = vector

    return embedding

