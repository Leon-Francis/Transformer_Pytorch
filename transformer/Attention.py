#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: Leon-Francis
Contact: 15290552788@163.com
Date: 2020-09-29 12:05:18
LastEditTime: 2020-09-29 12:05:22
LastEditors: Leon-Francis
Description: Attention
FilePath: /Transformer_Pytorch/transformer/Attention.py
(C)Copyright 2019-2020, Leon-Francis
'''
# Here put the import lib
import torch
import math
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    """一个attention运算

    Args:
        query (tensor): Q
        key (tensor): K
        value (tensor): V
        mask (tenosr, optional): mask. Defaults to None.
        dropout (double, optional): dropout. Defaults to None.

    Returns:
        tensor
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
