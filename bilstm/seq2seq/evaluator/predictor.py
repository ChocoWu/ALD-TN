#!/user/bin/env python3 
# -*- utf-8 -*-
# author shengqiong.wu

import torch


class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()

        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def get_decoder_features(self, word_input, char_input):
        # src_id_seq = torch.LongTensor([self.src_vocab.word_to_id(tok) for tok in src_seq]).view(1, -1)
        # if torch.cuda.is_available():
        #     src_id_seq = src_id_seq.cuda()
        # print('src_id_seq', src_id_seq.device)
        with torch.no_grad():
            result, _, _, _ = self.model(word_input, char_input, train_type='pre_train')
            softmax_list, _, other = result

        return other

    def predict(self, src_seq):
        """ Make prediction given `src_seq` as input.
        Args:
            src_seq (list): list of tokens in source language
        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        other = self.get_decoder_features(src_seq)

        length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        print(tgt_id_seq)
        tgt_seq = [self.tgt_vocab.id_to_word(tok) for tok in tgt_id_seq]
        print(tgt_seq)
        return tgt_seq

    def predict_n(self, src_seq, n=1):
        """ Make 'n' predictions given `src_seq` as input.
        Args:
            src_seq (list): list of tokens in source language
            n (int): number of predicted seqs to return. If None,
                     it will return just one seq.
        Returns:
            tgt_seq (list): list of tokens in target language as predicted
                            by the pre-trained model
        """
        # other = self.get_decoder_features(src_seq)
        #
        # result = []
        # for x in range(0, int(n)):
        #     length = other['topk_length'][0][x]
        #     tgt_id_seq = [other['topk_sequence'][di][0, x, 0].data[0] for di in range(length)]
        #     tgt_seq = [self.tgt_vocab.id_to_word(tok) for tok in tgt_id_seq]
        #     result.append(tgt_seq)
        #
        # return result
        result = []
        for input_variables, c_inputs, target_variables in src_seq:

            if torch.cuda.is_available():
                input_variables = input_variables.cuda()
                c_inputs = c_inputs.cuda()
                target_variables = target_variables.cuda()
                other = self.get_decoder_features(input_variables, c_inputs)

                length = other['length'][0]

                tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
                print(tgt_id_seq)
                tgt_seq = [self.tgt_vocab.id_to_word(tok.item()) for tok in tgt_id_seq]
                print(tgt_seq)

                result.append(' '.join(tgt_seq))


        return result
