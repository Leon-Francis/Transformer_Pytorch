#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: Leon-Francis
Contact: 15290552788@163.com
Date: 2020-09-28 20:37:32
LastEditTime: 2020-09-28 20:38:17
LastEditors: Leon-Francis
Description:
FilePath: /Transformer_Pytorch/transformer/Models.py
(C)Copyright 2019-2020, Leon-Francis
'''
# here put the import lib
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class EncoderDecoder(nn.Module):
    """
    标准的Encoder-Decoder架构。这是很多模型的基础
    具体的Encoder、Decoder、src_embed、target_embed和generator都是构造函数传入的参数
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        Args:
            encoder (class): encoder_layer
            decoder (class): decoder_layer
            src_embed (class): 源语言embedding
            tgt_embed (class): 目标语言embedding
            generator (class):
        """
        super(EncoderDecoder, self).__init__()
        # encoder和decoder都是构造的时候传入的，这样会非常灵活
        self.encoder = encoder
        self.decoder = decoder
        # 源语言和目标语言的embedding
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        # generator后面会讲到，就是根据Decoder的隐状态输出当前时刻的词
        # 基本的实现就是隐状态输入一个全连接层，全连接层的输出大小是词的个数
        # 然后接一个softmax变成概率。
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """前向计算

        Args:
            src ([type]): [description]
            tgt ([type]): [description]
            src_mask ([type]): [description]
            tgt_mask ([type]): [description]

        Returns:
            [type]: [description]
        """
        # 首先调用encode方法对输入进行编码，然后调用decode方法解码
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """编码计算

        Args:
            src ([type]): [description]
            src_mask ([type]): [description]

        Returns:
            [type]: [description]
        """
        # 调用encoder来进行编码，传入的参数embedding的src和src_mask
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """解码计算

        Args:
            memory ([type]): [description]
            src_mask ([type]): [description]
            tgt ([type]): [description]
            tgt_mask ([type]): [description]

        Returns:
            [type]: [description]
        """
        # 调用decoder
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    根据Decoder的隐状态输出一个词
    实质就是一个全连接加上一个softmax
    """
    def __init__(self, d_model, vocab):
        """
        Args:
            d_model ([type]): Decoder输出的大小
            vocab (int): 词典大小
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
