# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 5:35 下午
# @Author  : lizhen
# @FileName: loss.py
# @Description:
import torch.nn.functional as F
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import numpy as np


def nll_loss(output, target):
    return F.nll_loss(output, target)


def crossentropy_loss(output, target):
    output = output.transpose(1,2)
    weight_ = [1.0] * 435
    weight_[0] = 0.001
    weight_ = torch.from_numpy(np.array(weight_)).cuda().float()
    return F.cross_entropy(output, target,weight=weight_)


def binary_crossentropy_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)
