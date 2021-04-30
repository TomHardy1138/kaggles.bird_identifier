import math

import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from torch.nn import Parameter


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class AngleSimpleLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine.clamp(-1, 1)

    
class ArcfaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.5, pr_product=False):
        super().__init__()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.gamma = 1.0
        self.pr_product = pr_product

    def forward(self, logits, labels, iteration):
        if self.pr_product:
            pr_alpha = torch.sqrt(1.0 - logits.pow(2.0))
            logits = pr_alpha.detach() * logits + logits.detach() * (1.0 - pr_alpha)
        
        logits = logits.float()
        cosine = logits

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        index = torch.zeros_like(logits, dtype=torch.uint8)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        output = torch.where(index, phi, logits)
        
        output *= self.s

        return focal_loss(F.cross_entropy(output, labels, reduction='none'), self.gamma)
