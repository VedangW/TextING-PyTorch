import torch
import torch.nn as nn

def uniform(shape, scale=0.05):
    return nn.init.uniform_(torch.empty(shape), a=-scale, b=scale)

def glorot(shape):
    return nn.init.xavier_uniform_(torch.empty(shape), gain=1.)

def zeros(shape):
    return nn.init.zeros_(torch.empty(shape))

def ones(shape):
    return nn.init.ones_(torch.empty(shape))