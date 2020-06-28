import os
import sys
import csv
import pandas as pd

from .Classifier_utils import InputExample, convert_examples_to_features, convert_features_to_tensors


def read_tsv(filename):
    with open(filename, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t")
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def read_csv(filename):
    # data = pd.read_csv(filename, encoding='utf-8', delimiter=",", names=['sentence', 'label'])
    # return list(data['sentence']), list(data['label'])
    with open(filename, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=",")
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def load_tsv_dataset(filename, set_type):
    """
    文件内数据格式: sentence  label
    """
    examples = []
    lines = read_tsv(filename)
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = i
        text_a = list(line[0])
        label = list(line[1])
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def load_csv_dataset(filename, set_type):
    """
    文件内数据格式: sentence  label
    """
    examples = []
    lines = read_csv(filename)
    for (i, line) in enumerate(zip(lines)):
        # if i == 0:
        #     continue
        guid = i
        text_a = line[0][0]
        label = line[0][1]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def load_data(data_dir, tokenizer, max_length, batch_size, data_type,
              label_list, format_type=1, filename='train.csv', trg_vocab=None):
    if format_type == 0:
        load_func = load_tsv_dataset
    if format_type == 1:
        load_func = load_csv_dataset

    examples = load_func(os.path.join(data_dir, filename), data_type)
    features = convert_examples_to_features(
        examples, label_list, max_length, tokenizer, trg_vocab)

    dataloader = convert_features_to_tensors(features, batch_size, data_type)

    examples_len = len(examples)

    return dataloader, examples_len
