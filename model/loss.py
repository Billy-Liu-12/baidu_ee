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


def crossentropy_loss(output, target, device,weight_):
    output = output.transpose(1, 2)

    weight_ = torch.from_numpy(np.array(weight_)).cuda(device).float()
    return F.cross_entropy(output, target, weight=weight_)


def cut_crossentropy_loss(output, target, device, cut_side,weight_):
    output = output.transpose(1, 2)

    weight_ = torch.from_numpy(np.array(weight_)).cuda().float()
    loss = F.cross_entropy(output, target,weight=weight_, reduction='none')

    fm = torch.ones_like(loss, dtype=loss.dtype)
    fm_0 = 0.01 * torch.ones_like(loss, dtype=loss.dtype)
    fm_b = 1.5 * torch.ones_like(loss, dtype=loss.dtype)
    fm = torch.where(target == 0, fm_0, fm)
    fm = torch.where(target%2 == 1, fm_b,fm)
    fm = fm.sum()
    # target 里面每个位置被乘的不同的weight 现在 要对第i个位置cut——side也乘这个权重。
    cut = cut_side * torch.ones_like(loss, dtype=loss.dtype)
    cut = torch.where(target == 0, cut * weight_[0], cut)
    cut = torch.where(target%2 == 1, cut * 1.5, cut)
    return_loss = torch.where(loss < cut, cut, loss).sum()/fm
    return return_loss


def binary_crossentropy_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)
