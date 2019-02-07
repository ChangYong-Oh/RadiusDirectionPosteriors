import collections
from itertools import repeat
from operator import mul

import torch
from torch.nn import functional as F
from BayesianNeuralNetwork.torch_user.nn.bayesian_modules.radial_module import RadialModule
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.von_mises_fisher import VonMisesFisherReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.gamma import GammaReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.weibull import WeibullReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.lognormal import LognormalReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.normal import NormalReparametrizedSample


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


class _RadialConvNd(RadialModule):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias, prior=None, with_global=True):
        super(_RadialConvNd, self).__init__()
        self.with_global = with_global
        self.in_channels = in_channels
        self.out_channels = out_channels
        # each filter is group
        self.batch_shape = torch.Size([out_channels])
        self.event_shape = torch.Size([in_channels * reduce(mul, kernel_size)])
        # all filter to a out channel is group
        self.weight_shape = torch.Size([out_channels, in_channels]) + kernel_size
        self.register_buffer('batch_ones', torch.ones(self.batch_shape))
        if bias:
            self.register_buffer('bias_ones', torch.ones((out_channels,)))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.group = group

        assert prior['direction'][0] == 'vMF'
        self.direction_prior_param = prior['direction'][1]
        self.radius_prior_type = prior['radius'][0]
        self.radius_prior_param = prior['radius'][1]


class RadialConv2d(_RadialConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, prior=None, with_global=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(RadialConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, prior, with_global)

        assert prior['direction'][0] == 'vMF'
        self.direction_prior_param = prior['direction'][1]
        self.radius_prior_type = prior['radius'][0]
        self.radius_prior_param = prior['radius'][1]

        self.direction_rsampler = VonMisesFisherReparametrizedSample(batch_shape=self.batch_shape, event_shape=self.event_shape)
        if self.with_global:
            self.global_scale_rsampler = LognormalReparametrizedSample(batch_shape=torch.Size([1]))
            self.global_scale_rsampler1 = LognormalReparametrizedSample(batch_shape=torch.Size([1]))
        self.radius_rsampler = LognormalReparametrizedSample(batch_shape=self.batch_shape)
        self.radius_rsampler1 = LognormalReparametrizedSample(batch_shape=self.batch_shape)
        self.bias_rsampler = NormalReparametrizedSample(batch_shape=torch.Size([out_channels])) if bias else None

    def forward(self, input):
        direction_sample = self.direction_rsampler(1)[0]
        radius_sample = self.radius_rsampler(1)[0]
        radius_sample = (radius_sample * self.radius_rsampler1(1)[0]) ** 0.5
        weight = (direction_sample * radius_sample.unsqueeze(-1)).reshape(self.weight_shape)
        bias = self.bias_rsampler(1)[0] if self.bias_rsampler is not None else None
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.group)
