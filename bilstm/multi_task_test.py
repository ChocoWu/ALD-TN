import os
import argparse
import logging
import dill
import time
import pandas as pd
import numpy as np
import random

import torch
import torchtext
from torch.optim import Adam
from torch.utils.data import DataLoader

from .seq2seq.trainer.multi_task_train import MultiTaskTrainer
from .seq2seq.models import EncoderRNN, DecoderRNN, TopKDecoder
from .seq2seq.models.multi_task import Multi_Task
from .seq2seq.models.Classification import Classification
from .seq2seq.models.Embedding_Layer import Embedding
from .seq2seq.loss.loss import Perplexity
from .seq2seq.evaluator.predictor import Predictor
from .seq2seq.util.utils import *
from .seq2seq.util.data_processing import build_src_tgt_vocab, load_data_with_diff_vocab, load_embedding, load_data
from .seq2seq.hi_attn.model import Hi_Attention


DIC = './data/dict.csv'
NORM_TRAIN = './data/train_norm_1.csv'
NORM_DEV = './data/dev_norm_1.csv'
CLASS_TRAIN = './data/agr_en_train_1.csv'
CLASS_DEV = './data/agr_en_dev_1.csv'
CLASS_TEST_FB = './data/agr_en_fb_test_1.csv'
CLASS_TEST_TW = './data/agr_en_tw_test_1.csv'


parser = argparse.ArgumentParser()
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment/',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--options_file', type=str, default='./data/elmo_2x2048_256_2048cnn_1xhighway_options.json')
parser.add_argument('--weights_file', type=str, default='./data/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5')
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
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--num_layer', type=int, default=1)
parser.add_argument('--num_class', type=int, default=3)
parser.add_argument('--round1', type=int, default=7)
parser.add_argument('--round2', type=int, default=5)
parser.add_argument('--norm_batch_size', type=int, default=16)
parser.add_argument('--class_batch_size', type=int, default=32)
parser.add_argument('--norm_epochs', type=int, default=10)
parser.add_argument('--class_epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='elmo')
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=42)

# parser.add_argument("--w_hidden_size", type=int, default=150)
parser.add_argument("--w_num_layer", type=int, default=1, help="number of layers")
parser.add_argument("--w_atten_size", type=int, default=300)
parser.add_argument("--w_is_bidirectional", type=bool, default=True)
parser.add_argument("--w_dropout_prob", type=float, default=0.5)

parser.add_argument("--s_hidden_size", type=int, default=150)
parser.add_argument("--s_num_layer", type=int, default=1, help="number of layers")
parser.add_argument("--s_atten_size", type=int, default=300)
parser.add_argument("--s_is_bidirectional", type=bool, default=True)
parser.add_argument("--s_dropout_prob", type=float, default=0.5)

opt = parser.parse_args()

logger = get_logger(opt.expt_dir + "Train_{}.txt".format(time.strftime("%m-%d_%H-%M-%S")))
logger.info(opt)

logger.info('start loading vocabulary......')
if os.path.exists(opt.src_vocab_path) and os.path.exists(opt.tgt_vocab_path):
    input_vocab = load_from_pickle(opt.src_vocab_path)
    output_vocab = load_from_pickle(opt.tgt_vocab_path)
else:
    input_vocab,  output_vocab = build_src_tgt_vocab(DIC, opt.src_vocab_path, opt.tgt_vocab_path)
opt.vocab_size = input_vocab.n_words
opt.alphabet_size = input_vocab.n_chars
logger.info('number of words in input vocabulary is {}, number of words in output vocabulary is {}'.format(input_vocab.n_words, output_vocab.n_words))
logger.info('loading vocabulary done.')

logger.info('start loading dataset ......')
norm_train = load_data(NORM_TRAIN, opt.max_sent, opt.max_word, input_vocab, output_vocab, model_type=opt.model_type)
norm_dev = load_data(NORM_DEV, opt.max_sent, opt.max_word, input_vocab, output_vocab, model_type=opt.model_type)
class_train = load_data(CLASS_TRAIN, opt.max_sent, opt.max_word, input_vocab, output_vocab, model_type=opt.model_type, data_type='class')
class_dev = load_data(CLASS_DEV, opt.max_sent, opt.max_word, input_vocab, output_vocab, model_type=opt.model_type, data_type='class')
class_fb_test = load_data(CLASS_TEST_FB, opt.max_sent, opt.max_word, input_vocab, output_vocab, model_type=opt.model_type, data_type='class')
class_tw_test = load_data(CLASS_TEST_TW, opt.max_sent, opt.max_word, input_vocab, output_vocab, model_type=opt.model_type, data_type='class')
norm_train_loader = DataLoader(norm_train, batch_size=opt.norm_batch_size, shuffle=True)
norm_dev_loader = DataLoader(norm_dev, batch_size=opt.norm_batch_size, shuffle=True)
class_train_loader = DataLoader(class_train, batch_size=opt.class_batch_size, shuffle=True)
class_dev_loader = DataLoader(class_dev, batch_size=opt.class_batch_size, shuffle=True)
class_fb_test_loader = DataLoader(class_fb_test, batch_size=opt.class_batch_size, shuffle=False)
class_tw_test_loader = DataLoader(class_tw_test, batch_size=opt.class_batch_size, shuffle=False)
logger.info('loading dataset done.')

logger.info('start loading embedding ......')
if os.path.exists(opt.embedding_path):
    embedding = load_from_pickle(opt.embedding_path)
    embedding = torch.FloatTensor(embedding)
else:
    embedding = load_embedding(opt.pretrain_file, opt.w_embedding_size, input_vocab)
    save_to_pickle(opt.embedding_path, embedding)
    embedding = torch.FloatTensor(embedding)
logger.info('loading embedding done.')

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

weight = torch.ones(output_vocab.n_words)
pad = output_vocab.word_to_id(output_vocab.W_PAD)
loss = Perplexity(weight, pad)

if opt.use_cuda:
    loss.cuda()

multi_task = None
if not opt.resume:
    embedding_input = Embedding(input_vocab.n_words, input_vocab.n_chars,
                                opt.w_embedding_size, opt.c_embedding_size,
                                w_embedding=embedding,
                                options_file=opt.options_file, weights_file=opt.weights_file, model_type=opt.model_type)
    # hierarchical encoder
    share_encoder = Hi_Attention(opt)
    class_encoder = Hi_Attention(opt)
    norm_encoder = Hi_Attention(opt)
    # bilstm encoder
    # share_encoder = EncoderRNN(opt.w_embedding_size, opt.w_hidden_size,
    #                            opt.c_embedding_size, opt.c_hidden_size,
    #                            variable_lengths = True, use_type = opt.use_type)
    # class_encoder = EncoderRNN(opt.w_embedding_size, opt.w_hidden_size,
    #                            opt.c_embedding_size, opt.c_hidden_size,
    #                            variable_lengths=True, use_type=opt.use_type)
    # norm_encoder = EncoderRNN(opt.w_embedding_size, opt.w_hidden_size,
    #                           opt.c_embedding_size, opt.c_hidden_size,
    #                           variable_lengths = True, use_type = opt.use_type)

    decoder = DecoderRNN(output_vocab.n_words, opt.max_word*opt.max_sent, opt.s_hidden_size * 4,
                         dropout_p=0.2, use_attention=True,
                         bidirectional_encoder=True,
                         rnn_cell='lstm',
                         eos_id=output_vocab.word_to_id(output_vocab.SYM_EOS),
                         sos_id=output_vocab.word_to_id(output_vocab.SYM_SOS))
    classification = Classification(opt.s_hidden_size, opt.num_class)
    multi_task = Multi_Task(embedding_input, share_encoder, decoder, classification, class_encoder, norm_encoder,
                            opt=opt)

if opt.use_cuda:
    multi_task.cuda()

logger.info(multi_task)

# norm_train_loader = None
# norm_dev_loader = None
# class_train_loader = None
# class_dev_loader = None
# class_fb_test_loader = None
# class_tw_test_loader = None
train = [norm_train_loader, class_train_loader]
dev = [norm_dev_loader, class_dev_loader]
test = [class_fb_test_loader, class_tw_test_loader]

# train
t = MultiTaskTrainer(config=opt, loss=loss, norm_batch_size=opt.norm_batch_size, class_batch_size=opt.class_batch_size,
                     print_every=100, expt_dir=opt.expt_dir, logger=logger, use_cuda=opt.use_cuda, random_seed=opt.seed)

multi_task = t.train(multi_task, train,
                     round1=opt.round1, round2=opt.round2, norm_epochs=opt.norm_epochs, class_epochs=opt.class_epochs,
                     dev_data=dev, test_data=test,
                     optimizer=None,
                     teacher_forcing_ratio=0.5,
                     resume=opt.resume,
                     lr=opt.lr)

beam_search = Multi_Task(multi_task.embedding_layer, multi_task.share_encoder, TopKDecoder(multi_task.decoder, 3),
                         multi_task.classification, multi_task.class_encoder, multi_task.norm_encoder,
                         opt=opt)


if torch.cuda.is_available():
    beam_search = beam_search.cuda()

predictor = Predictor(beam_search, input_vocab, output_vocab)
# inp_seq = ["This was largely accounted for by seed under 9 years old , about 90% of which is viable .",
#            "MENTION MENTION weddings in the summer in Aruba ofc u guys r my bridesmaids"]
# inp_seq = "MENTION MENTION weddings in the summer in Aruba ofc u guys r my bridesmaids"
# seq = predictor.predict(inp_seq.split())
# print(" ".join(seq[:-1]))
# assert " ".join(seq[:-1]) == inp_seq[::-1]
test = pd.read_csv('./data/agr_en_train_1.csv', names=['src'])
test = list(test['src'])
seq = predictor.predict_n(test, len(test))
with open('data/result.txt', 'w') as f:
    for line in seq:
        print(line)
        f.write(line)
        f.write('\n')

