# coding=utf-8

import time
from sklearn import metrics
import random
import numpy as np

import torch
import sys
import pickle
import string
import pandas as pd
import json
import csv
from .tokenizer import *
import emoji


def get_device(gpu_id):
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu, not recommend")
    return device, n_gpu


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def classifiction_metric(preds, labels, label_list):
    """ 分类任务的评价指标， 传入的数据需要是 numpy 类型的 """

    acc = metrics.accuracy_score(labels, preds)

    labels_list = [i for i in range(len(label_list))]

    report = metrics.classification_report(labels, preds, labels=labels_list, target_names=label_list, digits=5,
                                           output_dict=True)

    if len(label_list) > 2:
        auc = 0.5
    else:
        auc = metrics.roc_auc_score(labels, preds)
    return acc, report, auc


def accuracy(y_true, y_pred, eos):
    total = 0
    correct = 0
    assert len(y_true) == len(y_pred)
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            if y_pred[i][j] == eos:
                break
            elif y_true[i][j] == y_pred[i][j]:
                correct += 1
                total += 1
            else:
                total += 1
    return correct / total


def evaluate_metrics(x, y_true, y_pred, pad, output_vocab, tokenizer):
    """

    :param x: the original input [batch_size, seq_len]
    :param y_true: normalization input [batch_size, seq_len]
    :param y_pred: predicted normalization input [batch_size, [seq_len]
    :param pad: end of sentence
    :param output_vocab:
    :param tokenizer:
    :return:
    """
    correct_norm = 0.0
    total_norm = 0.0
    total_nsw = 0.0
    p, r, f1 = 0.0, 0.0, 0.0
    assert len(x) == len(y_pred) == len(y_true)

    for src, pred, gold in zip(x, y_pred, y_true):
        try:
            i = 0
            while i != src.index(pad) and i < len(gold):
                if output_vocab.id2word[pred[i]] != tokenizer.convert_ids_to_tokens(src[i]) and output_vocab.id2word[gold[i]] == output_vocab.id2word[pred[i]]:
                    correct_norm += 1
                if output_vocab.id2word[gold[i]] != tokenizer.convert_ids_to_tokens(src[i]):
                    total_nsw += 1
                if output_vocab.id2word[pred[i]] != tokenizer.convert_ids_to_tokens(src[i]):
                    total_norm += 1
                i += 1
        except AssertionError:
            print("Invalid data format")
            sys.exit(1)
    # calc p, r, f
    p = correct_norm / total_norm
    r = correct_norm / total_nsw
    if p != 0 and r != 0:
        f1 = (2 * p * r) / (p + r)
    return p, r, f1


def build_src_tgt_vocab(inputFile, src_vocabFile, tgt_vocabFile, label_list):
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src, tgt = pickle.load(open(inputFile, 'rb'))

    # df = pd.read_csv(inputFile, names=['src', 'tgt'])
    for s, t in zip(src, tgt):
        # print(tgt)
        # src = tokenize(str(src))
        # tgt = tokenize(tgt)
        src_vocab.add_sentence(s)
        tgt_vocab.add_sentence(t)
        [src_vocab.add_word(word) for word in s]
        [tgt_vocab.add_word(word) for word in t]

    tgt_vocab.add_tags(label_list)
    save_to_pickle(src_vocabFile, src_vocab)
    save_to_pickle(tgt_vocabFile, tgt_vocab)

    return src_vocab, tgt_vocab


def load_from_pickle(path):
    file = open(path, 'rb')
    obj = pickle.load(file)
    file.close()

    return obj


def save_to_pickle(path, obj):
    file = open(path, 'wb')
    pickle.dump(obj, file)
    file.close()

    return 1


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


def replace_symbol(text):
    """
    replace HASHTAG MENTION URL PERCENTAGE EMAIL NUM DATE PHONE TIME MONEY
    :param text:
    :return:
    """
    result = text.lower()
    result = emoji.demojize(result)
    result = re.sub(RE_HASHTAG, 'HASHTAG', result)
    result = re.sub(RE_MENTION, 'MENTION', result)
    result = re.sub(RE_URL, 'URL', result)
    result = re.sub(RE_PERCENTAGE, 'PERCENTAGE', result)
    result = re.sub(RE_EMAIL, 'EMAIL', result)
    result = re.sub(RE_NUM, 'NUM', result)
    result = re.sub(RE_DATE, 'DATE', result)
    result = re.sub(RE_PHONE, 'PHONE', result)
    result = re.sub(RE_TIME, 'TIME', result)
    result = re.sub(RE_MONEY, 'MONEY', result)
    # result = re.sub(r'[\"\'\|\*\/,.?!]+', '', result)
    return result


def decode_emoji(text):
    """

    :param text:
    :return:
    """
    return emoji.demojize(text)


def process_lexnorm2015(sourcefile, targetfile):
    """
    lexnorm2015 the format of the data:
    {
        "tid": "",
        "index": "",
        "output": [],
        "input": []
    }
    :param sourcefile: source filename .json
    :param targetfile: The name of the file to be saved, the saved file format is csv
    :return:
    """
    with open(sourcefile) as f:
        source_file = json.load(f)
        print('the total of source data is ', len(source_file))
        with open(targetfile, 'a') as f:
            csv_write = csv.writer(f)
            for data in source_file:
                output = '|'.join(data['output']).strip()
                input = '|'.join(data['input']).strip()
                output = replace_symbol(output)
                input = replace_symbol(input)
                csv_write.writerow([input, output])

    print("Lexnorm2015 data, Done!!!")


def process_trac(sourcefile, targetfile):
    """
    process the data from TRAC, the format of the data is csv(id, context, label)
    :param sourcefile:
    :param targetfile:
    :return:
    """
    df = pd.read_csv(sourcefile, names=['id', 'context', 'label'])
    print("the total number of TRAC dataset is ", len(df))
    df['encode_context'] = df.apply(lambda s: replace_symbol(str(s.context)), axis=1)
    print(df['encode_context'][0])
    print(df['label'][0])
    df.to_csv(targetfile, columns=['encode_context', 'label'], index=False, header=False)


def merage_dataset(sf1, sf2, tf1):
    """
    :param sf1: source file 1 'src' 'tgt'
    :param sf2: source file 2 'context' 'label'
    :param tf1: target file 'src' 'tgt' 'context' 'label'
    :return:
    """
    df1 = pd.read_csv(sf1, names=['src', 'tgt'])
    df2 = pd.read_csv(sf2, names=['src', 'tgt'])
    length = len(df2)
    print(len(df1))
    print('the length of meraged data is ', length)
    data = pd.concat([df1, df2], axis=0)
    data.to_csv(tf1, index=False, header=False)


if __name__ == '__main__':
    # process_lexnorm2015
    process_lexnorm2015('../../data/lexnorm2015/train_data.json', "../../data/train_lexnorm2015.csv")
    process_lexnorm2015('../../data/lexnorm2015/test_truth.json', "../../data/test_lexnorm2015.csv")

    # merage data
    merage_dataset('../../data/train_lexnorm2015.csv', '../../data/agr_en_train.csv', '../../data/dict.csv')
