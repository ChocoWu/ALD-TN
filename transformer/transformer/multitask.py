import torch
import torch.nn as nn


class MultiTask(nn.Module):
    def __init__(self, src_embedding_layer, trg_embedding_layer, share_encoder, norm_encoder,
                 class_encoder, decoder, classification, src_pad_idx, trg_pad_idx, device,
                 hid_dim=300, src_len=200):
        super(MultiTask, self).__init__()

        self.src_embedding_layer = src_embedding_layer
        self.trg_embedding_layer = trg_embedding_layer
        self.share_encoder = share_encoder
        self.norm_encoder = norm_encoder
        self.class_encoder = class_encoder
        self.decoder = decoder
        self.classification = classification
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
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

    def forward(self, src_word, trg, src_char=None, trg_char=None, data_type='norm', model_type='word'):
        # src_word = [batch size, src len]
        # trg = [batch size, trg len]
        # src_char = [batch size, src len, max char]
        # tgt_char = [batch size, src len, max char]

        # src_mask = self.make_src_mask(src)
        # trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]
        src_embedded, src_mask = self.src_embedding_layer(src_word, src_char)
        share_enc_src, share_attention = self.share_encoder(src_embedded, src_mask)
        share_feature = self.get_share_feature(share_enc_src)

        if data_type == 'norm':
            # _, src_embedded, src_mask = self.src_embedding_layer(src_word, src_char)
            tgt_embedded, trg_mask = self.trg_embedding_layer(trg, trg_char)
            # src_embedded = [batch size, src len, embedding_size]
            # src_mask = [batch size, 1, 1, src len]
            # tgt_embedded = [batch size, tgt len, embedding_size]
            # trg_mask = [batch size, 1, trg len, trg len]

            # share_enc_src = self.share_encoder(src_embedded, src_mask)
            pri_enc_src, norm_attention = self.norm_encoder(src_embedded, src_mask)
            # share/pri_enc_src = [batch size, src len, hid dim]

            enc_src = torch.cat([share_enc_src, pri_enc_src], dim=2)
            # enc_src = [batch size, src len, hid dim*2]

            output, attention = self.decoder(tgt_embedded, enc_src, trg_mask, src_mask)
            # output = [batch size, trg len, output dim]
            # attention = [batch size, n heads, trg len, src len]

            share_enc_src = self.share_linear(share_enc_src)
            pri_enc_src = self.pri_linear(pri_enc_src)
            # share_enc_src = [batch size, trg len, 5]
            # pri_enc_src = [batch size, trg len, 5]

            return output, attention, share_attention, norm_attention, share_enc_src, pri_enc_src, share_feature, enc_src
        elif data_type == 'class':
            # _, src_embedded, src_mask = self.src_embedding_layer(src_word, src_char)
            # src_embedded = [batch size, src len, embedding_size]
            # src_mask = [batch size, 1, 1, src len]

            pri_enc_src, class_attention = self.class_encoder(src_embedded, src_mask)
            # share/pri_enc_src = [batch size, src len, hid dim]

            enc_src = torch.cat([share_enc_src, pri_enc_src], dim=2)
            # enc_src = [batch size, src len, hid dim*2]

            output = self.classification(enc_src)
            # output = [batch size, trg len, num class]

            share_enc_src = self.share_linear(share_enc_src)
            pri_enc_src = self.pri_linear(pri_enc_src)
            # share_enc_src = [batch size, trg len, 5]
            # pri_enc_src = [batch size, trg len, 5]

            return output, share_attention, class_attention, share_enc_src, pri_enc_src, share_feature, enc_src

