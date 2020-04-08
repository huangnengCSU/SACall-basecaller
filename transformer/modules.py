#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Filename: 
# @Date: 2018-12-05 16:30
# @author: huangneng
# @contact: huangneng@csu.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import transformer.constants as Constants


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=Constants.PAD):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1).to(output.device)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        # model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class LossComputeBase(nn.Module):
    def __init__(self):
        super(LossComputeBase, self).__init__()

    def forward(self, x, target):
        loss = F.cross_entropy(x, target, ignore_index=Constants.PAD, reduction='sum')
        return loss


class Criterion(nn.Module):
    def __init__(self, label_size, label_smoothing, label_smoothing_value):
        super(Criterion, self).__init__()
        if label_smoothing:
            self.crit = LabelSmoothingLoss(label_smoothing=label_smoothing_value, tgt_vocab_size=label_size)
        else:
            self.crit = LossComputeBase()

    def cal_performance(self, pred, gold):
        ''' Apply label smoothing if needed '''
        gold = gold.contiguous().view(-1)
        loss = self.crit(pred, gold)

        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(Constants.PAD)
        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

        return loss, n_correct


# def get_subsequent_mask(seq):
#     assert seq.dim() == 2
#     sz_b, len_s = seq.size()
#     subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
#     subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)
#     return subsequent_mask

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & nn.Parameter(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data), requires_grad=False)
    return tgt_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -float('inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class multiheadattention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(multiheadattention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        # nn.init.xavier_normal_(self.w_q.weight)
        # nn.init.xavier_normal_(self.w_k.weight)
        # nn.init.xavier_normal_(self.w_v.weight)
        # nn.init.xavier_normal_(self.fc.weight)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        query = self.w_q(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        output, self.attn = attention(query=query, key=key, value=value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        output = output.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        output = self.fc(output)

        return output, self.attn


class FFN(nn.Module):
    # feedforward layer
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.w_2 = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.w_2(inter)
        return output + x


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):
        x = x + nn.Parameter(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = multiheadattention(h=n_head, d_model=d_model, dropout=dropout)
        self.ffn = FFN(input_size=d_model, hidden_size=d_ff, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_dropout = nn.Dropout(dropout)

    def forward(self, signal_emb, src_mask):
        input = signal_emb
        input_norm = self.layer_norm(signal_emb)
        enc_out, enc_self_attn = self.slf_attn(input_norm, input_norm, input_norm, src_mask)
        enc_out = input + self.norm_dropout(enc_out)

        enc_out = self.ffn(enc_out)

        return enc_out, enc_self_attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = multiheadattention(h=n_head, d_model=d_model, dropout=dropout)
        self.enc_dec_attn = multiheadattention(h=n_head, d_model=d_model, dropout=dropout)
        self.ffn = FFN(input_size=d_model, hidden_size=d_ff, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_dropout = nn.Dropout(dropout)

    def forward(self, label_emb, enc_out, tgt_mask=None, src_mask=None):
        input = label_emb
        input_norm = self.layer_norm1(label_emb)
        slf_attn_out, dec_slf_attn = self.slf_attn(input_norm, input_norm, input_norm, tgt_mask)
        slf_attn_out = input + self.norm_dropout(slf_attn_out)

        input = slf_attn_out
        input_norm = self.layer_norm2(slf_attn_out)
        enc_dec_attn_out, enc_dec_attn = self.enc_dec_attn(input_norm, enc_out, enc_out, src_mask)
        enc_dec_attn_out = input + self.norm_dropout(enc_dec_attn_out)

        ffn_out = self.ffn(enc_dec_attn_out)

        return ffn_out, dec_slf_attn, enc_dec_attn


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, n_head, num_encoder_layers, dropout=0.1):
        super(Encoder, self).__init__()
        # self.src_embed = nn.Sequential(nn.Conv1d(in_channels=1,
        #                                          out_channels=d_model,
        #                                          kernel_size=1,
        #                                          stride=1,
        #                                          padding=0,
        #                                          bias=False),
        #                                nn.BatchNorm1d(num_features=d_model),
        #                                nn.ReLU(inplace=True))
        self.src_embed = nn.Sequential(nn.Conv1d(in_channels=1,
                                                 out_channels=d_model//2,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=d_model//2),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
                                       nn.Conv1d(in_channels=d_model//2,
                                                 out_channels=d_model,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=d_model),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
                                       nn.Conv1d(in_channels=d_model,
                                                 out_channels=d_model,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=d_model),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool1d(kernel_size=3,stride=2,padding=1))
        # TODO: why padding_idx=0
        self.position_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)

        self.stack_layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, d_ff=d_ff, n_head=n_head, dropout=dropout) for _ in range(
                num_encoder_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, signal):
        # param signal: shape (batch, signal_len, feature_num)
        #src_mask = signal.squeeze(2).eq(Constants.SIG_PAD).unsqueeze(-2)
        #print(src_mask.size())
        signal = signal.transpose(-1, -2)  # (N,C,L)
        embed_out = self.src_embed(signal)  # (N,C,L)
        embed_out = embed_out.transpose(-1, -2)  # (N,L,C)
        enc_output = self.position_encoding(embed_out)
        #print(enc_output.size())
        src_mask = torch.zeros(enc_output.size(0),1,enc_output.size(1),dtype=torch.uint8).to(enc_output.device)
        #print(src_mask.size())
        for layer in self.stack_layers:
            enc_output, enc_slf_attn = layer(enc_output, src_mask)
        enc_output = self.layer_norm(enc_output)
        return enc_output, src_mask


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, n_head, label_vocab_size, d_word_vec, num_decoder_layers,
                 dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.label_word_embed = nn.Embedding(label_vocab_size, d_word_vec, padding_idx=Constants.PAD)
        # self.label_word_embed = Embeddings(label_vocab_size, d_word_vec)
        self.position_encoding = PositionalEncoding(d_model, dropout)
        self.stack_layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, d_ff=d_ff, n_head=n_head, dropout=dropout) for _ in
             range(num_decoder_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, label_seq, enc_out, src_mask):
        dec_output = self.position_encoding(self.label_word_embed(label_seq))

        tgt_mask = make_std_mask(label_seq, Constants.PAD)

        for layer in self.stack_layers:
            dec_output, dec_slf_attn, enc_dec_attn = layer(dec_output, enc_out, tgt_mask, src_mask)
        dec_output = self.layer_norm(dec_output)
        return dec_output


class Transformer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, num_encoder_layers,
                 num_decoder_layers, label_vocab_size, d_word_vec, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, d_ff, n_head, num_encoder_layers, dropout)
        self.decoder = Decoder(d_model, d_ff, n_head, label_vocab_size, d_word_vec, num_decoder_layers, dropout)
        self.final_proj = nn.Linear(d_model, label_vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        # # share the decoder embedding weight and final project generate weight.
        # self.final_proj.weight = self.decoder.label_word_embed.weight

        assert d_word_vec == d_model

    def forward(self, signal, label):
        label = label[:, :-1]
        enc_out, src_mask = self.encoder(signal)
        dec_out = self.decoder(label, enc_out, src_mask)
        seq_logit = self.log_softmax(self.final_proj(dec_out))
        return seq_logit.view(-1, seq_logit.size(2))
