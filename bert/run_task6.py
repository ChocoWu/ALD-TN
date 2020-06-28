# coding=utf-8
from .main import main
from .args import get_args


if __name__ == "__main__":

    model_name = "BertLSTM"
    label_list = ['CAG', 'OAG', 'NAG']
    # label_list = ['HATE', 'OFF', 'NEITHER']
    data_dir = "./data"
    output_dir = "./experiment/sst_output/"
    cache_dir = "./experiment/data/sst_cache/"
    log_dir = "./experiment/data/sst_log/"

    # bert-base
    bert_vocab_file = "/home/wsq/backup/Analysis-Social-Media-Language-v10/bert_pretrain/bert-base-uncased-vocab.txt"
    bert_model_dir = "/home/wsq/backup/Analysis-Social-Media-Language-v10/bert_pretrain"

    # # bert-large
    # bert_vocab_file = "./bert_pretrain/bert-large-uncased-vocab.txt"
    # bert_model_dir = "./bert_pretrain/"

    config = get_args(data_dir, output_dir, cache_dir,
                      bert_vocab_file, bert_model_dir, log_dir)

    main(config, config.save_name, label_list)
