#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: Leon-Francis
Contact: 15290552788@163.com
Date: 2020-09-28 21:21:29
LastEditTime: 2020-09-28 21:31:33
LastEditors: Leon-Francis
Description: Utils
FilePath: /Transformer_Pytorch/transformer/utils.py
(C)Copyright 2019-2020, Leon-Francis
'''
# Here put the import lib
import torch
import torch.nn as nn
import copy
import numpy as np


@staticmethod
def clones(module, N):
    """克隆N个完全相同的Sublayer，使用了copy.deepcopy

    Args:
        module (nn.module): Sublayer
        N (int): 克隆个数

    Returns:
        nn.module: 深复制的nn.module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


@staticmethod
def subsequent_mask(size):
    """实质上就是生成一个下三角矩阵
    目标是为了令decoder在第t时刻只能使用1...t时刻的输入
    Args:
        size (int): 下三角矩阵大小
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
