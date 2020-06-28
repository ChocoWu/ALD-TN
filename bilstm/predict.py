import argparse
import pandas as pd
import numpy as np
import random

import torch

from seq2seq.models import TopKDecoder
from seq2seq.models.multi_task import Multi_Task
from seq2seq.evaluator.predictor import Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.utils import *
from seq2seq.util.data_processing import load_data, load_data_with_diff_vocab
from torch.utils.data import DataLoader


CLASS_TRAIN = './data/dev_norm.csv'
CLASS_DEV = './data/agr_en_dev.csv'
CLASS_TEST_FB = './data/agr_en_fb_test.csv'
CLASS_TEST_TW = './data/agr_en_tw_test.csv'


parser = argparse.ArgumentParser()
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment/checkpoints/',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level', default='info', help='Logging level.')
parser.add_argument('--src_vocab_path', type=str, default='./data/src_vocab.pt')
parser.add_argument('--tgt_vocab_path', type=str, default='./data/tgt_vocab.pt')
parser.add_argument('--embedding_path', type=str, default='./data/embedding_300.pt')
parser.add_argument('--pretrain_file', type=str, default='../../glove.840B.300d.txt')
parser.add_argument('--w_embedding_size', type=int, default=300)
parser.add_argument('--c_embedding_size', type=int, default=50)
parser.add_argument('--w_hidden_size', type=int, default=150)
parser.add_argument('--c_hidden_size', type=int, default=50)
parser.add_argument('--max_word', type=int, default=30)
parser.add_argument('--max_sent', type=int, default=6)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_layer', type=int, default=1)
parser.add_argument('--num_class', type=int, default=3)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--use_type', type=str, default='elmo')
parser.add_argument('--class_batch_size', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
opt = parser.parse_args()

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

checkpoint = Checkpoint().load(opt.expt_dir)
model = checkpoint.model

beam_search = Multi_Task(model.embedding_layer, model.encoder, TopKDecoder(model.decoder, 3),
                         model.classification, model.class_encoder, model.norm_encoder, opt=opt)
# Multi_Task(multi_task.encoder, TopKDecoder(multi_task.decoder, 3), multi_task.classification)


if torch.cuda.is_available():
    beam_search = beam_search.cuda()


input_vocab = load_from_pickle(opt.src_vocab_path)
output_vocab = load_from_pickle(opt.tgt_vocab_path)

predictor = Predictor(beam_search, input_vocab, output_vocab)
# inp_seq = ["This was largely accounted for by seed under 9 years old , about 90% of which is viable .",
#            "MENTION MENTION weddings in the summer in Aruba ofc u guys r my bridesmaids"]
# inp_seq = "MENTION MENTION weddings in the summer in Aruba ofc u guys r my bridesmaids"
# seq = predictor.predict(inp_seq.split())
# print(" ".join(seq[:-1]))
# assert " ".join(seq[:-1]) == inp_seq[::-1]

# class_train = load_data(CLASS_TRAIN, opt.max_sent, opt.max_word, input_vocab, output_vocab, use_type=opt.use_type)
# class_dev = load_data(CLASS_DEV, opt.max_sent, opt.max_word, input_vocab, output_vocab, use_type=opt.use_type)
# class_fb_test = load_data(CLASS_TEST_FB, opt.max_sent, opt.max_word, input_vocab, output_vocab, use_type=opt.use_type)
# class_tw_test = load_data(CLASS_TEST_TW, opt.max_sent, opt.max_word, input_vocab, output_vocab, use_type=opt.use_type)
class_train = load_data_with_diff_vocab(CLASS_TRAIN, input_vocab, output_vocab, type='norm')
class_train_loader = DataLoader(class_train, batch_size=opt.class_batch_size, shuffle=True)
# class_dev_loader = DataLoader(class_dev, batch_size=opt.class_batch_size, shuffle=True)
# class_fb_test_loader = DataLoader(class_fb_test, batch_size=opt.class_batch_size, shuffle=True)
# class_tw_test_loader = DataLoader(class_tw_test, batch_size=opt.class_batch_size, shuffle=True)
# test = pd.read_csv('./data/agr_en_train_1.csv', names=['src', 'label'])
# test = list(test['src'])
seq = predictor.predict_n(class_train_loader, class_train.__len__())
with open('data/result.txt', 'w') as f:
    for line in seq:
        print(line)
        f.write(line)
        f.write('\n')