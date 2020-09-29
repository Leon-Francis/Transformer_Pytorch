#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: Leon-Francis
Contact: 15290552788@163.com
Date: 2020-09-29 23:20:59
LastEditTime: 2020-09-29 23:21:17
LastEditors: Leon-Francis
Description: build&train
FilePath: /Transformer_Pytorch/Build&Train.py
(C)Copyright 2019-2020, Leon-Francis
'''
# Here put the import lib
import copy
import torch.nn as nn
from transformer.Coders import MultiHeadedAttention, PositionwiseFeedForward
from transformer.Coders import EncoderDecoder, Encoder, Decoder, EncoderLayer, DecoderLayer
from transformer.Models import PositionalEncoding
from transformer.Models import Embeddings
from transformer.Models import Generator
from transformer.Utils import subsequent_mask
import torch.autograd as Variable
import time


def make_model(src_vocab,
               tgt_vocab,
               N=6,
               d_model=512,
               d_ff=2048,
               h=8,
               dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # 随机初始化参数，这非常重要
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "创建Mask，使得我们不能attend to未来的词"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask,
                            batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens
