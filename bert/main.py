# coding=utf-8
import random
import numpy as np
import time
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam

from .Utils.utils import get_device, accuracy, evaluate_metrics, save_to_pickle, load_from_pickle, load_embedding, build_src_tgt_vocab
from .Utils.load_datasets import load_data

from .train_evaluate import train, evaluate, evaluate_save
from .model.BertLayer import BertLayer
from .model.Decoder import Decoder
from .model.classification import Classification
from .model.multitask import MultiTask


def main(config, model_times, label_list):
    if not os.path.exists(config.output_dir + model_times):
        os.makedirs(config.output_dir + model_times)

    if not os.path.exists(config.cache_dir + model_times):
        os.makedirs(config.cache_dir + model_times)

    # Bert 模型输出文件
    output_model_file = os.path.join(config.output_dir, model_times, WEIGHTS_NAME)
    output_config_file = os.path.join(config.output_dir, model_times, CONFIG_NAME)

    # 设备准备
    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]
    device, n_gpu = get_device(gpu_ids[0])
    if n_gpu > 1:
        n_gpu = len(gpu_ids)

    config.train_batch_size = config.train_batch_size // config.gradient_accumulation_steps

    # 设定随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)

    # 数据准备
    tokenizer = BertTokenizer.from_pretrained(
        config.bert_vocab_file, do_lower_case=config.do_lower_case)  # 分词器选择

    num_labels = len(label_list)

    print('start loading vocabulary......')
    DIC = './data/agr-en/dict.pt'
    if os.path.exists(config.tgt_vocab_path):
        output_vocab = load_from_pickle(config.tgt_vocab_path)
        # input_vocab = load_from_pickle(config.src_vocab_path)
    else:
        input_vocab, output_vocab = build_src_tgt_vocab(DIC, config.src_vocab_path, config.tgt_vocab_path, label_list)

    if os.path.exists(config.tgt_embedding_path):
        trg_embedding = load_from_pickle(config.tgt_embedding_path)
        trg_embedding = torch.FloatTensor(trg_embedding)
    else:
        trg_embedding = load_embedding(config.pretrain_file, config.w_embedding_size, output_vocab)
        save_to_pickle(config.tgt_embedding_path, trg_embedding)
        trg_embedding = torch.FloatTensor(trg_embedding)
    TRG_PAD_IDX = output_vocab.word2id['<pad>']

    norm_train, norm_train_examples_len = load_data(
        config.data_dir, tokenizer, config.max_seq_length, config.train_batch_size, "train", label_list,
        filename = 'dict.csv', trg_vocab = output_vocab)
    norm_dev, _ = load_data(
        config.data_dir, tokenizer, config.max_seq_length, config.dev_batch_size, "dev", label_list,
        filename = 'test_lexnorm2015.csv', trg_vocab = output_vocab)

    class_train, class_train_examples_len = load_data(
        config.data_dir, tokenizer, config.max_seq_length, config.class_batch_size, "train", label_list,
        filename = 'agr_en_train.csv')
    class_dev, _ = load_data(
        config.data_dir, tokenizer, config.max_seq_length, config.class_batch_size, "dev", label_list,
        filename = 'agr_en_dev.csv')
    class_fb_test, _ = load_data(
        config.data_dir, tokenizer, config.max_seq_length, config.class_batch_size, "test", label_list,
        filename = 'agr_en_fb_test.csv')
    class_tw_test, _ = load_data(
        config.data_dir, tokenizer, config.max_seq_length, config.class_batch_size, "test", label_list,
        filename = 'agr_en_tw_test.csv')

    num_train_optimization_steps = int(
        norm_train_examples_len / config.train_batch_size / config.gradient_accumulation_steps) * config.num_train_epochs + int(
        class_train_examples_len / config.class_batch_size / config.gradient_accumulation_steps) * config.class_epochs

    # Train and dev
    if config.do_train:

        # 模型准备
        print("model name is {}".format(config.model_name))

        dec = Decoder(300, output_vocab.n_words, config.hidden_dim, config.dec_layers,
                      config.dec_heads, config.dec_pf_dim, config.dec_dropout, device,
                      embedding=trg_embedding, trg_pad_idx=TRG_PAD_IDX, enc_out_dim=config.bert_hid_dim*2)
        share_enc = BertLayer.from_pretrained(config.bert_model_dir, cache_dir=config.cache_dir)
        norm_enc = BertLayer.from_pretrained(config.bert_model_dir, cache_dir=config.cache_dir)
        class_enc = BertLayer.from_pretrained(config.bert_model_dir, cache_dir=config.cache_dir)
        classification = Classification(config.bert_hid_dim*2, config.num_class)

        model = MultiTask(share_enc, norm_enc, class_enc, dec, classification,
                          device, config.bert_hid_dim)

        model.to(device)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

        """ 优化器准备 """
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=config.learning_rate,
                             warmup=config.warmup_proportion,
                             t_total=num_train_optimization_steps)

        """ 损失函数准备 """
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        best_norm_f1 = 0.0
        best_class_acc = 0.0
        for round in range(config.round1):
            for i in range(10):
                train(1, n_gpu, model, norm_train, optimizer,
                      criterion, config.gradient_accumulation_steps, device, label_list,
                      data_type='norm', adv_weight=0.05, diff_weight=0.01)
                norm_valid_loss, pred_label, gold_label, src_label = evaluate(model, norm_dev, criterion, device,
                                                                              label_list, data_type='norm',
                                                                              adv_weight=0.05,
                                                                              diff_weight=0.01)
                # acc = accuracy(gold_label, pred_label, tokenizer.convert_tokens_to_ids['<eos>'])
                p, r, f1 = evaluate_metrics(src_label, gold_label, pred_label, output_vocab.word2id['<pad>'], output_vocab, tokenizer)

                print(f'\tNorm Precise score: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}')
                if best_norm_f1 < f1:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), output_model_file)
                    with open(output_config_file, 'w') as f:
                        f.write(model_to_save.config.to_json_string())
            for i in range(7):
                train(1, n_gpu, model, class_train, optimizer,
                      criterion, config.gradient_accumulation_steps, device, label_list,
                      data_type='class', adv_weight=0.05, diff_weight=0.01)
                class_valid_loss, pred_label, gold_label, _ = evaluate(model, class_dev, criterion, device,
                                                                       label_list, data_type='class', adv_weight=0.05,
                                                                       diff_weight=0.01)
                class_valid_acc = accuracy_score(gold_label, pred_label)
                print(f'\tClass  Val. Loss: {class_valid_loss:.3f} |  Val. accuracy: {class_valid_acc:.4f}')

                if best_class_acc < class_valid_acc:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), output_model_file)
                    with open(output_config_file, 'w') as f:
                        f.write(model_to_save.config.to_json_string())

                class_valid_loss, pred_label, gold_label, _ = evaluate(model, class_fb_test, criterion, device,
                                                                       label_list, data_type='class', adv_weight=0.05,
                                                                       diff_weight=0.01)
                print(f'Facebook result:\n, {classification_report(gold_label, pred_label, digits=4)}')

                class_valid_loss, pred_label, gold_label, _ = evaluate(model, class_tw_test, criterion, device,
                                                                       label_list, data_type='class', adv_weight=0.05,
                                                                       diff_weight=0.01)
                print(f'Twitter result:\n, {classification_report(gold_label, pred_label, digits=4)}')

    # """ Test """

    # test 数据
    test_dataloader, _ = load_data(
        config.data_dir, tokenizer, config.max_seq_length, config.test_batch_size, "test", label_list)

    # 加载模型
    dec = Decoder(300, output_vocab.n_words, config.hidden_dim, config.dec_layers,
                  config.dec_heads, config.dec_pf_dim, config.dec_dropout, device,
                  embedding = trg_embedding, trg_pad_idx = TRG_PAD_IDX, enc_out_dim = config.bert_hid_dim * 2)
    share_enc = BertLayer.from_pretrained(config.bert_model_dir, cache_dir = config.cache_dir)
    norm_enc = BertLayer.from_pretrained(config.bert_model_dir, cache_dir = config.cache_dir)
    class_enc = BertLayer.from_pretrained(config.bert_model_dir, cache_dir = config.cache_dir)
    classification = Classification(config.bert_hid_dim * 2, config.num_class)

    model = MultiTask(share_enc, norm_enc, class_enc, dec, classification,
                      device, config.bert_hid_dim)

    model.load_state_dict(torch.load(output_model_file))
    model.to(device)

    # 损失函数准备
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # test the model
    class_valid_loss, pred_label, gold_label, _ = evaluate(model, class_fb_test, criterion, device,
                                                           label_list, data_type = 'class', adv_weight = 0.05,
                                                           diff_weight = 0.01)
    print(f'Facebook result:\n, {classification_report(gold_label, pred_label, digits=4)}')

    class_valid_loss, pred_label, gold_label, _ = evaluate(model, class_tw_test, criterion, device,
                                                           label_list, data_type = 'class', adv_weight = 0.05,
                                                           diff_weight = 0.01)
    print(f'Twitter result:\n, {classification_report(gold_label, pred_label, digits=4)}')
