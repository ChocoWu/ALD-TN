import torch
import torch.nn as nn


class MultiTask(nn.Module):
    def __init__(self, share_encoder, norm_encoder,
                 class_encoder, decoder, classification, device,
                 hid_dim=768):
        super(MultiTask, self).__init__()

        self.share_encoder = share_encoder
        self.norm_encoder = norm_encoder
        self.class_encoder = class_encoder
        self.decoder = decoder
        self.classification = classification
        self.device = device

        self.linear = nn.Linear(hid_dim, 2)
        self.share_linear = nn.Linear(hid_dim, 5)
        self.pri_linear = nn.Linear(hid_dim, 5)

    def get_share_feature(self, share_enc):
        batch_size = share_enc.size(0)
        # feature = share_enc.view(batch_size, -1)
        feature = torch.max(share_enc, dim=1)[0]
        feature = self.linear(feature)
        return feature

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, data_type='norm'):
        """

        :param src: list, [input_ids, token_type_ids, attention_mask]
        :param trg: [batch size, trg len]
        :param src_mask: [batch size, 1, 1, src len]
        :param data_type:
        :return:
        """
        src_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        share_enc_src = self.share_encoder(input_ids, token_type_ids, attention_mask)
        # share_enc_src = [batch_size, seq_len, bert_dim=768]
        share_feature = self.get_share_feature(share_enc_src)
        # share_feature = [batch_size, 5]

        if data_type == 'norm':
            # tgt_embedded, trg_mask = self.trg_embedding_layer(labels)
            # tgt_embedded = [batch size, tgt len, embedding_size]
            # trg_mask = [batch size, 1, trg len, trg len]

            pri_enc_src = self.norm_encoder(input_ids, token_type_ids, attention_mask)
            # pri_enc_src = [batch size, src len, bert_dim]

            enc_src = torch.cat([share_enc_src, pri_enc_src], dim=2)
            # enc_src = [batch size, src len, bert_dim*2]

            output, attention = self.decoder(labels, enc_src, src_mask)
            # output = [batch size, trg len, output dim]
            # attention = [batch size, n heads, trg len, src len]

            share_enc_src = self.share_linear(share_enc_src)
            pri_enc_src = self.pri_linear(pri_enc_src)
            # share_enc_src = [batch size, trg len, 5]
            # pri_enc_src = [batch size, trg len, 5]

            return output, share_enc_src, pri_enc_src, share_feature
        elif data_type == 'class':

            pri_enc_src = self.class_encoder(input_ids, token_type_ids, attention_mask)
            # share/pri_enc_src = [batch size, src len, bert_dim]

            enc_src = torch.cat([share_enc_src, pri_enc_src], dim=2)
            # enc_src = [batch size, src len, bert_dim*2]

            output = self.classification(enc_src)
            # output = [batch size, num class]

            share_enc_src = self.share_linear(share_enc_src)
            pri_enc_src = self.pri_linear(pri_enc_src)
            # share_enc_src = [batch size, trg len, 5]
            # pri_enc_src = [batch size, trg len, 5]

            return output, share_enc_src, pri_enc_src, share_feature
