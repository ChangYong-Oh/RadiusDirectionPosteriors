import math
from numbers import Number
import collections
from itertools import repeat
import torch

EXP_UPPER_BOUND = 10.0


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def softplus_inv(x):
    if isinstance(x, Number):
        if x > 1e2:
            return x
        return math.log(math.exp(x) - 1.0)
    y = torch.log(1 - torch.exp(x))
    y[x > EXP_UPPER_BOUND] = x[x > EXP_UPPER_BOUND]
    return y


def softplus_derivative(x):
    if isinstance(x, Number):
        if x < -1e2:
            return 0
        return 1.0 / (1.0 + math.exp(-x))
    return 1.0 / (1.0 + torch.exp(-x))


def softplus_inv_derivative(x):
    if isinstance(x, Number):
        if x < -1e2:
            return 0
        return 1.0 / (1.0 - math.exp(-x))
    return 1.0 / (1.0 - torch.exp(-x))


def ml_kappa(dim, eps):
    return (max(4, dim) - 3.0) / (1.0 - (1.0 - eps) ** 2) * (1.0 - eps)


def kaiming_transpose(tensor):
    return torch.nn.init.kaiming_normal_(tensor, mode='fan_out')
