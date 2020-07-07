import torch
import torch.nn as nn
# import ticker
import argparse
import numpy as np
import random
import time
import os
import math
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader
from allennlp.modules.elmo import batch_to_ids
from transformer.Encoder import Encoder
from transformer.Decoder import Decoder
from transformer.multitask import MultiTask
from transformer.classification import Classification
from transformer.TrgEmbeddingLayer import TrgEmbeddingLayer
from transformer.SrcEmbeddingLayer import SrcEmbeddingLayer
from transformer.util.tokenizer import tokenize
from transformer.util.data_processing import load_embedding, load_data_with_diff_vocab, build_src_tgt_vocab
from transformer.util.utils import get_logger, load_from_pickle, save_to_pickle, evaluate_metrics, accuracy


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)


def train(model, iterator, optimizer, norm_criterion, adv_criterion, class_criterion,
          clip, device, adv_weight=0.001, diff_weight=0.005, data_type='norm', model_type='word',
          is_adv_loss=True):
    model.train()
    epoch_loss = 0.0

    for idx, batch in enumerate(iterator):
        src = batch[0]
        src_char = batch[1]
        trg = batch[2]
        if device.type == 'cuda':
            src = src.cuda()
            if model_type == 'word':
                src_char = None
            else:
                src_char = src_char.cuda()
            trg = trg.cuda()

        optimizer.zero_grad()

        if data_type == 'norm':
            output, _, _, _, share_feature, pri_feature, feature, _ = model(src, trg[:, :-1], src_char=src_char,
                                                                            data_type=data_type, model_type=model_type)

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]
            # share_feature = [batch size, trg len, hid dim]
            # pri_feature = [batch size, trg len, hid dim]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = norm_criterion(output, trg)

            if is_adv_loss:
                # adversarial loss
                target = torch.ones(src.size(0), dtype=torch.long)
                if device.type == 'cuda':
                    target = target.cuda()
                adv_loss = adv_criterion(feature, target)
                # normalization different loss
                batch_size = share_feature.size(0)
                share_feature = share_feature.contiguous().view(batch_size, -1)
                pri_feature = pri_feature.contiguous().view(batch_size, -1)
                share_features = share_feature - torch.mean(share_feature, dim=0)
                pri_feature = pri_feature - torch.mean(pri_feature, dim=0)

                correlation_matrix = torch.mm(share_features, torch.transpose(pri_feature, 0, 1))
                diff_loss = torch.mean(torch.pow(correlation_matrix, 2))

                loss = loss + adv_loss * adv_weight + diff_loss * diff_weight

        elif data_type == 'class':
            output, _, _, share_feature, pri_feature, feature, _ = model(src, trg, src_char=src_char,
                                                                         data_type=data_type,
                                                                         model_type=model_type)
            loss = class_criterion(output, trg.squeeze())

            if is_adv_loss:
                # adversarial loss
                target = torch.zeros(src.size(0), dtype=torch.long)
                if device.type == 'cuda':
                    target = target.cuda()
                adv_loss = adv_criterion(feature, target)

                # classification different loss
                batch_size = share_feature.size(0)
                share_feature = share_feature.contiguous().view(batch_size, -1)
                pri_feature = pri_feature.contiguous().view(batch_size, -1)
                share_features = share_feature - torch.mean(share_feature, dim=0)
                pri_feature = pri_feature - torch.mean(pri_feature, dim=0)

                correlation_matrix = torch.mm(share_features, torch.transpose(pri_feature, 0, 1))
                diff_loss = torch.mean(torch.pow(correlation_matrix, 2))

                loss = loss + adv_loss * adv_weight + diff_loss * diff_weight

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        if idx % 200 == 0:
            logger.info('train loss {}/{} : {}'.format(idx, len(iterator), loss.item()))
    avg_loss = epoch_loss / len(iterator)
    logger.info(f'{data_type} Train Loss: {avg_loss:.3f} | Train PPL: {math.exp(avg_loss):7.3f}')

    return avg_loss


def evaluate(model, iterator, optimizer, norm_criterion, adv_criterion, class_criterion, device,
             adv_weight=0.001, diff_weight=0.005, data_type='norm', model_type='word', is_adv_loss=True):
    model.eval()

    epoch_loss = 0
    src_label = []
    pred_label = []
    gold_label = []
    share_attns = []
    pri_attns = []
    with torch.no_grad():
        for idx, batch in enumerate(iterator):
            src = batch[0]
            src_char = batch[1]
            trg = batch[2]
            if device.type == 'cuda':
                src = src.cuda()
                if model_type == 'word':
                    src_char = None
                else:
                    src_char = src_char.cuda()
                trg = trg.cuda()

            optimizer.zero_grad()

            if data_type == 'norm':
                output, _, share_attn, pri_attn, share_feature, pri_feature, feature, _ = model(src, trg[:, :-1], src_char=src_char,
                                                                                                data_type=data_type, model_type=model_type)

                # output = [batch size, trg len - 1, output dim]
                # trg = [batch size, trg len]
                # share_feature = [batch size, trg len, hid dim]
                # pri_feature = [batch size, trg len, hid dim]

                output_dim = output.shape[-1]
                batch_size = output.shape[0]
                # trg_len = output.shape[1]

                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)

                # output = [batch size * trg len - 1, output dim]
                # trg = [batch size * trg len - 1]

                pred_label.extend(torch.max(output, dim=1)[1].view(batch_size, -1).cpu().numpy().tolist())
                gold_label.extend(trg.squeeze().view(batch_size, -1).cpu().numpy().tolist())
                src_label.extend(src[:, 1:].squeeze().view(batch_size, -1).cpu().numpy().tolist())
                share_attns.extend(share_attn.cpu().numpy().tolist())
                pri_attns.extend(pri_attn.cpu().numpy().tolist())

                loss = norm_criterion(output, trg)

                if is_adv_loss:
                    # adversarial loss
                    target = torch.ones(src.size(0), dtype=torch.long)
                    if device.type == 'cuda':
                        target = target.cuda()
                    adv_loss = adv_criterion(feature, target)
                    # normalization different loss
                    batch_size = share_feature.size(0)
                    share_feature = share_feature.contiguous().view(batch_size, -1)
                    pri_feature = pri_feature.contiguous().view(batch_size, -1)
                    share_features = share_feature - torch.mean(share_feature, dim=0)
                    pri_feature = pri_feature - torch.mean(pri_feature, dim=0)

                    correlation_matrix = torch.mm(share_features, torch.transpose(pri_feature, 0, 1))
                    diff_loss = torch.mean(torch.pow(correlation_matrix, 2))

                    loss = loss + adv_loss * adv_weight + diff_loss * diff_weight
            elif data_type == 'class':
                output, share_attn, pri_attn, share_feature, pri_feature, feature, _ = model(src, trg, src_char=src_char,
                                                                                             data_type=data_type, model_type=model_type)
                pred_label.extend(torch.max(output, dim=1)[1].cpu().numpy().tolist())
                if type(trg.squeeze().cpu().numpy().tolist()) == int:
                    gold_label.append(trg.squeeze().cpu().numpy().tolist())
                else:
                    gold_label.extend(trg.squeeze().cpu().numpy().tolist())
                share_attns.extend(share_attn.cpu().numpy().tolist())
                pri_attns.extend(pri_attn.cpu().numpy().tolist())
                loss = class_criterion(output, trg.squeeze())

                if is_adv_loss:
                    # adversarial loss
                    target = torch.zeros(src.size(0), dtype=torch.long)
                    if device.type == 'cuda':
                        target = target.cuda()
                    adv_loss = adv_criterion(feature, target)

                    # classification different loss
                    batch_size = share_feature.size(0)
                    share_feature = share_feature.contiguous().view(batch_size, -1)
                    pri_feature = pri_feature.contiguous().view(batch_size, -1)
                    share_features = share_feature - torch.mean(share_feature, dim=0)
                    pri_feature = pri_feature - torch.mean(pri_feature, dim=0)

                    correlation_matrix = torch.mm(share_features, torch.transpose(pri_feature, 0, 1))
                    diff_loss = torch.mean(torch.pow(correlation_matrix, 2))

                    loss = loss + adv_loss * adv_weight + diff_loss * diff_weight

            epoch_loss += loss.item()
            if idx % 200 == 0:
                logger.info('valid loss {}/{} : {}'.format(idx, len(iterator), loss.item()))

    return epoch_loss / len(iterator), pred_label, gold_label, src_label, share_attns, pri_attns


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def translate_sentence(sentence, src_vocab, tgt_vocab, model, device, max_len=200):
    model.eval()

    if isinstance(sentence, str):
        tokens = [token.lower for token in tokenize(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = ['<sos>'] + tokens + ['<eos>']

    elmo_id = batch_to_ids([tokens])
    elmo_id = elmo_id.view(-1, 50)

    src_indexes = [src_vocab.word_to_id(token) for token in tokens]
    src_tensor = torch.tensor(src_indexes, dtype=torch.long).unsqueeze(0).to(device)
    char_inputs = elmo_id.unsqueeze(0).to(device)

    with torch.no_grad():
        src_embedded, src_mask = model.src_embedding_layer(src_tensor, char_inputs)
        share_enc_src, share_att = model.share_encoder(src_embedded, src_mask)
        pri_enc_src, norm_att = model.norm_encoder(src_embedded, src_mask)
        _, private_att = model.class_encoder(src_embedded, src_mask)
        enc_src = torch.cat([share_enc_src, pri_enc_src], dim=2)
        output = model.classification(enc_src)

    trg_indexes = [tgt_vocab.word2id['<sos>']]
    if len(tokens) < max_len:
        max_len = len(tokens)
    for i in range(max_len):

        trg_tensor = torch.tensor(trg_indexes, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            trg_embedded, trg_mask = model.trg_embedding_layer(trg_tensor)
            output, attention = model.decoder(trg_embedded, enc_src, trg_mask, src_mask)

        pred_token = output.cpu().argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == tgt_vocab.word2id['<eos>']:
            break

    trg_tokens = [tgt_vocab.id2word[i] for i in trg_indexes]
    return trg_tokens[1:], attention.cpu()


def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2, outpath='atten.pdf'):
    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(20, 25))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='ocean_r')
        # fig.colorbar(cax)

        ax.tick_params(labelsize=12)
        xlabels = [''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>']
        ylabels = [''] + ['<sos>'] + translation + ['<eos>']
        ax.set_xticklabels(xlabels, rotation=75, fontsize=22)
        ax.set_yticklabels(ylabels, fontsize=22)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    fig.tight_layout()
    plt.savefig(outpath)
    plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-options_file', type=str, default='./data/elmo_2x1024_128_2048cnn_1xhighway_options.json')
    parser.add_argument('-weights_file', type=str, default='./data/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
    parser.add_argument('-log-level', dest='log_level', default='info', help='Logging level.')
    parser.add_argument('-src_vocab_path', type=str, default='./data/src_vocab.pt')
    parser.add_argument('-tgt_vocab_path', type=str, default='./data/tgt_vocab.pt')
    parser.add_argument('-src_embedding_path', type=str, default='./data/src_embedding.pt')
    parser.add_argument('-tgt_embedding_path', type=str, default='./data/tgt_embedding.pt')
    parser.add_argument('-pretrain_file', type=str, default='/../../glove.840B.300d.txt')
    parser.add_argument('-expt_dir', action='store', dest='expt_dir', default='./experiment/')
    parser.add_argument('-w_embedding_size', type=int, default=300)
    parser.add_argument('-emb_dropout', type=int, default=0.2)
    parser.add_argument('-c_embedding_size', type=int, default=50)
    parser.add_argument('-enc_layers', type=int, default=4)
    parser.add_argument('-dec_layers', type=int, default=4)
    parser.add_argument('-enc_heads', type=int, default=5)
    parser.add_argument('-dec_heads', type=int, default=5)
    parser.add_argument('-enc_pf_dim', type=int, default=512)
    parser.add_argument('-dec_pf_dim', type=int, default=512)
    parser.add_argument('-enc_dropout', type=float, default=0.4)
    parser.add_argument('-dec_dropout', type=float, default=0.4)
    parser.add_argument('-hidden_dim', type=int, default=150)
    parser.add_argument('-norm_batch_size', type=int, default=32)
    parser.add_argument('-class_batch_size', type=int, default=16)
    parser.add_argument('-lr', type=float, default=0.0005)
    parser.add_argument('-round1', type=int, default=1)
    parser.add_argument('-round2', type=int, default=15)
    parser.add_argument('-norm_epochs', type=int, default=20)
    parser.add_argument('-class_epochs', type=int, default=1)
    parser.add_argument('-clip', type=int, default=1)
    parser.add_argument('-max_sent', type=int, default=6)
    parser.add_argument('-max_word', type=int, default=200)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-model_type', type=str, default='elmo')
    parser.add_argument('-adv_weight', type=float, default=0.05)
    parser.add_argument('-diff_weight', type=float, default=0.01)
    parser.add_argument('-embedding_output', type=int, default=556)
    parser.add_argument('-num_class', type=int, default=3)

    parser.add_argument('-is_adv_loss', type=bool, default=True)  # Whether to use adv_loss during training
    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    DIC = './data/dict.pt'
    NORM_TRAIN = './data/dict.pt'
    NORM_DEV = './data/test_lexnorm2015.pt'
    CLASS_TRAIN = './data/agr_en_train.pt'
    CLASS_DEV = './data/agr_en_dev.pt'
    CLASS_TEST_FB = './data/agr_en_fb_test.pt'
    CLASS_TEST_TW = './data/agr_en_tw_test.pt'

    logger = get_logger(opt.expt_dir + "Train_{}.txt".format(time.strftime("%m-%d_%H-%M-%S")))
    logger.info(opt)
    logger.info('start loading vocabulary......')
    if os.path.exists(opt.src_vocab_path) and os.path.exists(opt.tgt_vocab_path):
        input_vocab = load_from_pickle(opt.src_vocab_path)
        output_vocab = load_from_pickle(opt.tgt_vocab_path)
    else:
        input_vocab, output_vocab = build_src_tgt_vocab(DIC, opt.src_vocab_path, opt.tgt_vocab_path)

    opt.src_vocab_size = input_vocab.n_words
    opt.alphabet_size = input_vocab.n_chars
    opt.trg_vocab_size = output_vocab.n_words

    logger.info('number of words in input vocabulary is {}, number of words in output vocabulary is {}'.format(
        input_vocab.n_words, output_vocab.n_words))
    logger.info('loading vocabulary done.')

    logger.info('start loading dataset ......')
    norm_train = load_data_with_diff_vocab(NORM_TRAIN, input_vocab, output_vocab, max_word=opt.max_word)
    norm_dev = load_data_with_diff_vocab(NORM_DEV, input_vocab, output_vocab, max_word=opt.max_word)
    class_train = load_data_with_diff_vocab(CLASS_TRAIN, input_vocab, output_vocab, max_word=opt.max_word, data_type='class')
    class_dev = load_data_with_diff_vocab(CLASS_DEV, input_vocab, output_vocab, max_word=opt.max_word, data_type='class')
    class_fb_test = load_data_with_diff_vocab(CLASS_TEST_FB, input_vocab, output_vocab, max_word=opt.max_word, data_type='class')
    class_tw_test = load_data_with_diff_vocab(CLASS_TEST_TW, input_vocab, output_vocab, max_word=opt.max_word, data_type='class')

    norm_train_loader = DataLoader(norm_train, batch_size=opt.norm_batch_size, shuffle=True, collate_fn=norm_train.collate_fn)
    norm_dev_loader = DataLoader(norm_dev, batch_size=opt.norm_batch_size, shuffle=True, collate_fn=norm_dev.collate_fn)
    class_train_loader = DataLoader(class_train, batch_size=opt.class_batch_size, shuffle=True, collate_fn=class_train.collate_fn)
    class_dev_loader = DataLoader(class_dev, batch_size=opt.class_batch_size, shuffle=True, collate_fn=class_dev.collate_fn)
    class_fb_loader = DataLoader(class_fb_test, batch_size=opt.class_batch_size, shuffle=False, collate_fn=class_fb_test.collate_fn)
    class_tw_loader = DataLoader(class_tw_test, batch_size=opt.class_batch_size, shuffle=False, collate_fn=class_tw_test.collate_fn)
    logger.info('loading dataset done.')

    logger.info('start loading embedding ......')
    if os.path.exists(opt.src_embedding_path) and os.path.exists(opt.tgt_embedding_path):
        src_embedding = load_from_pickle(opt.src_embedding_path)
        trg_embedding = load_from_pickle(opt.tgt_embedding_path)
        src_embedding = torch.FloatTensor(src_embedding)
        trg_embedding = torch.FloatTensor(trg_embedding)
    else:
        src_embedding = load_embedding(opt.pretrain_file, opt.w_embedding_size, input_vocab)
        trg_embedding = load_embedding(opt.pretrain_file, opt.w_embedding_size, output_vocab)
        save_to_pickle(opt.src_embedding_path, src_embedding)
        save_to_pickle(opt.tgt_embedding_path, trg_embedding)
        src_embedding = torch.FloatTensor(src_embedding)
        trg_embedding = torch.FloatTensor(trg_embedding)
    logger.info('loading embedding done.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SRC_PAD_IDX = input_vocab.word2id['<pad>']
    TRG_PAD_IDX = output_vocab.word2id['<pad>']

    src_embedding_layer = SrcEmbeddingLayer(opt.src_vocab_size, opt.alphabet_size, opt.w_embedding_size,
                                            opt.c_embedding_size, src_embedding,
                                            options_file=opt.options_file, weights_file=opt.weights_file,
                                            model_type=opt.model_type, src_pad_idx=SRC_PAD_IDX,
                                            input_dropout_p=opt.emb_dropout)
    trg_embedding_layer = TrgEmbeddingLayer(opt.trg_vocab_size, opt.alphabet_size, opt.w_embedding_size,
                                            opt.c_embedding_size, trg_embedding,
                                            options_file=opt.options_file, weights_file=opt.weights_file,
                                            model_type='word', trg_pad_idx=TRG_PAD_IDX,
                                            device=device, input_dropout_p=opt.emb_dropout)
    share_enc = Encoder(opt.embedding_output, opt.hidden_dim, opt.enc_layers,
                        opt.enc_heads, opt.enc_pf_dim, opt.enc_dropout, device)
    norm_enc = Encoder(opt.embedding_output, opt.hidden_dim, opt.enc_layers,
                       opt.enc_heads, opt.enc_pf_dim, opt.enc_dropout, device)
    class_enc = Encoder(opt.embedding_output, opt.hidden_dim, opt.enc_layers,
                        opt.enc_heads, opt.enc_pf_dim, opt.enc_dropout, device)

    dec = Decoder(300, opt.trg_vocab_size, opt.hidden_dim*2, opt.dec_layers,
                  opt.dec_heads, opt.dec_pf_dim, opt.dec_dropout, device)

    classification = Classification(opt.hidden_dim*2, opt.num_class)

    SRC_PAD_IDX = input_vocab.word2id['<pad>']
    TGT_PAD_IDX = output_vocab.word2id['<pad>']

    model = MultiTask(src_embedding_layer, trg_embedding_layer,
                      share_enc, norm_enc, class_enc, dec, classification,
                      SRC_PAD_IDX, TGT_PAD_IDX, device, hid_dim=opt.hidden_dim, src_len=opt.max_word)
    # logger.info('model parameters: ', count_parameters(model))
    initialize_weights(model)
    if device.type == 'cuda':
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    norm_criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)
    class_criterion = nn.CrossEntropyLoss()
    adv_criterion = nn.CrossEntropyLoss()

    best_valid_loss = float('inf')
    best_class_loss = float('inf')
    best_fb_f1 = 0.0
    best_tw_f1 = 0.0

    # train the model
    for round in range(opt.round1):
        for epoch in range(opt.norm_epochs):
            start_time = time.time()

            norm_train_loss = train(model, norm_train_loader, optimizer, norm_criterion, adv_criterion, class_criterion,
                                    opt.clip, device, opt.adv_weight, opt.diff_weight, 'norm', opt.model_type, opt.is_adv_loss)
            norm_valid_loss, pred_label, gold_label, src_label, _, _ = evaluate(model, norm_dev_loader, optimizer, norm_criterion, adv_criterion, class_criterion, device,
                                                               opt.adv_weight, opt.diff_weight, 'norm', opt.model_type, opt.is_adv_loss)

            if norm_valid_loss < best_valid_loss:
                best_valid_loss = norm_valid_loss
                torch.save(model.state_dict(), 'model.pt')
            acc = accuracy(gold_label, pred_label, output_vocab.word_to_id('<eos>'), opt.max_word)
            p, r, f1 = evaluate_metrics(src_label, gold_label, pred_label, output_vocab.word_to_id('<eos>'),
                                        input_vocab, output_vocab)

            logger.info(f'\tNorm  Val. Loss: {norm_valid_loss:.3f} |  Val. PPL: {math.exp(norm_valid_loss):7.3f}')
            logger.info(f'\t Val. Loss: {norm_valid_loss:.3f} |  Val. acc: {acc:.4f}')
            logger.info(f'\tNorm Precise score: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}')
        for epoch in range(opt.class_epochs):
            end_time = time.time()
            class_train_loss = train(model, class_train_loader, optimizer, norm_criterion, adv_criterion,
                                     class_criterion, opt.clip, device, opt.adv_weight, opt.diff_weight,
                                     'class', opt.model_type, opt.is_adv_loss)
            class_valid_loss, pred_label, gold_label, _, _, _ = evaluate(model, class_dev_loader, optimizer, norm_criterion,
                                                                adv_criterion, class_criterion,
                                                                device, opt.adv_weight, opt.diff_weight, 'class',
                                                                opt.model_type, opt.is_adv_loss)
            logger.info(
                f'\tClass  Val. Loss: {class_valid_loss:.3f} |  Val. accuracy: {accuracy_score(gold_label, pred_label):.4f}')
            if class_valid_loss < best_class_loss:
                best_class_loss = class_valid_loss
                torch.save(model.state_dict(), 'model.pt')

    logger.info(f'Round 2 is starting ......')
    for round in range(opt.round2):

        norm_train_loss = train(model, norm_train_loader, optimizer, norm_criterion, adv_criterion, class_criterion,
                                opt.clip, device, opt.adv_weight, opt.diff_weight, 'norm', opt.model_type, opt.is_adv_loss)
        norm_valid_loss, pred_label, gold_label, src_label, _, _ = evaluate(model, norm_dev_loader, optimizer, norm_criterion, adv_criterion,
                                                           class_criterion, device, opt.adv_weight,
                                                           opt.diff_weight, 'norm', opt.model_type, opt.is_adv_loss)
        if norm_valid_loss < best_valid_loss:
            best_valid_loss = norm_valid_loss
            torch.save(model.state_dict(), 'model.pt')
        acc = accuracy(gold_label, pred_label, output_vocab.word_to_id('<eos>'), opt.max_word)
        p, r, f1 = evaluate_metrics(src_label, gold_label, pred_label,
                                    output_vocab.word_to_id('<eos>'), input_vocab, output_vocab)
        logger.info(f'\tNorm  Val. Loss: {norm_valid_loss:.3f} |  Val. PPL: {math.exp(norm_valid_loss):7.3f}')
        logger.info(f'\t Val. Loss: {norm_valid_loss:.3f} |  Val. acc: {acc:.4f}')
        logger.info(f'\tNorm Precise score: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}')

        class_train_loss = train(model, class_train_loader, optimizer, norm_criterion,
                                 adv_criterion, class_criterion, opt.clip, device, opt.adv_weight, opt.diff_weight,
                                 'class', opt.model_type, opt.is_adv_loss)
        class_valid_loss, pred_label, gold_label, _, _, _ = evaluate(model, class_dev_loader, optimizer, norm_criterion,
                                                            adv_criterion, class_criterion, device, opt.adv_weight,
                                                            opt.diff_weight, 'class', opt.model_type, opt.is_adv_loss)
        logger.info(
            f'\tClass  Val. Loss: {class_valid_loss:.3f} |  Val. accuracy: {accuracy_score(gold_label, pred_label):.4f}')

        if class_valid_loss < best_class_loss:
            best_class_loss = class_valid_loss
            torch.save(model.state_dict(), 'model.pt')

    # """ Test """

    # load model
    model.load_state_dict(torch.load('model.pt'))

    # test data
    norm_valid_loss, pred_label, gold_label, src_label, _, _ = evaluate(model, norm_dev_loader, optimizer,
                                                                        norm_criterion, adv_criterion, class_criterion,
                                                                        device,
                                                                        opt.adv_weight, opt.diff_weight, 'norm',
                                                                        opt.model_type, opt.is_adv_loss)
    acc = accuracy(gold_label, pred_label, output_vocab.word_to_id('<eos>'), opt.max_word)
    p, r, f1 = evaluate_metrics(src_label, gold_label, pred_label, output_vocab.word_to_id('<eos>'),
                                input_vocab, output_vocab)

    logger.info(f'\tNorm  Val. Loss: {norm_valid_loss:.3f} |  Val. PPL: {math.exp(norm_valid_loss):7.3f}')
    logger.info(f'\t Val. Loss: {norm_valid_loss:.3f} |  Val. acc: {acc:.4f}')
    logger.info(f'\tNorm Precise score: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}')

    class_fb_loss, pred_label, gold_label, _, share_attn, pri_attn = evaluate(model, class_fb_loader, optimizer,
                                                                              norm_criterion,
                                                                              adv_criterion,
                                                                              class_criterion, device, opt.adv_weight,
                                                                              opt.diff_weight, 'class', opt.model_type,
                                                                              opt.is_adv_loss)
    logger.info(f'Facebook result:\n, {classification_report(gold_label, pred_label, digits=4)}')
    class_tw_loss, pred_label, gold_label, _, share_attn, pri_attn = evaluate(model, class_tw_loader, optimizer,
                                                                              norm_criterion,
                                                                              adv_criterion,
                                                                              class_criterion, device, opt.adv_weight,
                                                                              opt.diff_weight, 'class', opt.model_type,
                                                                              opt.is_adv_loss)

    epoch_mins, epoch_secs = epoch_time(end_time, time.time())
    logger.info(f'Twitter result:\n, {classification_report(gold_label, pred_label, digits=4)}')
