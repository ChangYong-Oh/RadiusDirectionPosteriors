from operator import mul

import torch
from torch.nn import functional as F
from BayesianNeuralNetwork.torch_user.nn.bayesian_modules.double_radial_module import DoubleRadialModule
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.von_mises_fisher import VonMisesFisherReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.gamma import GammaReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.weibull import WeibullReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.lognormal import LognormalReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.normal import NormalReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.utils import _pair


class DoubleRadialConv2dBundle(DoubleRadialModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, prior=None, with_global=True):
        super(DoubleRadialConv2dBundle, self).__init__()
        self.with_global = with_global
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight_size = torch.Size([out_channels, in_channels // groups]) + self.kernel_size
        self.row_batch_shape = torch.Size([self.weight_size[0]])
        self.row_event_shape = torch.Size([self.weight_size[1]])
        self.col_batch_shape = torch.Size([self.weight_size[1]])
        self.col_event_shape = torch.Size([self.weight_size[0]])

        filter_rsampler_batch_shape = torch.Size([self.weight_size[0] * self.weight_size[1]])
        filter_rsampler_event_shape = torch.Size([reduce(mul, self.weight_size[2:], 1)])

        self.register_buffer('filter_batch_ones', torch.ones(filter_rsampler_batch_shape))
        self.register_buffer('row_batch_ones', torch.ones(self.row_batch_shape))
        self.register_buffer('col_batch_ones', torch.ones(self.col_batch_shape))
        if bias:
            self.register_buffer('bias_shape_ones', torch.ones([out_channels]))

        self.filter_rsampler = VonMisesFisherReparametrizedSample(batch_shape=filter_rsampler_batch_shape, event_shape=filter_rsampler_event_shape)

        assert prior['direction'][0] == 'vMF'
        self.filter_direction_prior_param = prior['direction'][1]
        self.row_direction_prior_param = prior['direction'][1]
        self.row_radius_prior_type = prior['radius'][0]
        self.row_radius_prior_param = prior['radius'][1]
        self.col_direction_prior_param = prior['direction'][1]
        self.col_radius_prior_type = prior['radius'][0]
        self.col_radius_prior_param = prior['radius'][1]

        if in_channels > 1 and out_channels > 1:
            self.row_direction_rsampler = VonMisesFisherReparametrizedSample(batch_shape=self.row_batch_shape, event_shape=self.row_event_shape)
            if self.with_global:
                self.row_global_scale_rsampler = LognormalReparametrizedSample(batch_shape=torch.Size([1]))
                self.row_global_scale_rsampler1 = LognormalReparametrizedSample(batch_shape=torch.Size([1]))
            self.row_radius_rsampler = LognormalReparametrizedSample(batch_shape=self.row_batch_shape)
            self.row_radius_rsampler1 = LognormalReparametrizedSample(batch_shape=self.row_batch_shape)

            self.col_direction_rsampler = VonMisesFisherReparametrizedSample(batch_shape=self.col_batch_shape,
                                                                             event_shape=self.col_event_shape)
            if self.with_global:
                self.col_global_scale_rsampler = LognormalReparametrizedSample(batch_shape=torch.Size([1]))
                self.col_global_scale_rsampler1 = LognormalReparametrizedSample(batch_shape=torch.Size([1]))
            self.col_radius_rsampler = LognormalReparametrizedSample(batch_shape=self.col_batch_shape)
            self.col_radius_rsampler1 = LognormalReparametrizedSample(batch_shape=self.col_batch_shape)
        elif in_channels == 1 and out_channels > 1:
            self.row_rsampler = NormalReparametrizedSample(batch_shape=self.row_batch_shape)
        elif out_channels == 1 and in_channels > 1:
            self.col_rsampler = NormalReparametrizedSample(batch_shape=self.col_batch_shape)
        self.bias_rsampler = NormalReparametrizedSample(batch_shape=torch.Size([out_channels])) if bias else None

    def forward(self, input):
        filter_sample = self.filter_rsampler(1)[0].reshape(self.weight_size)

        if self.in_channels > 1 and self.out_channels > 1:
            row_direction_sample = self.row_direction_rsampler(1)[0]
            row_radius_sample = self.row_radius_rsampler(1)[0]
            row_radius_sample = (row_radius_sample * self.row_radius_rsampler1(1)[0]) ** 0.5
            if self.with_global:
                row_global_scale_sample = (self.row_global_scale_rsampler(1)[0] *
                                           self.row_global_scale_rsampler1(1)[0]) ** 0.5
                row_radius_sample = row_radius_sample * row_global_scale_sample
            row_weight = row_direction_sample * row_radius_sample.unsqueeze(-1)

            col_direction_sample = self.col_direction_rsampler(1)[0]
            col_radius_sample = self.col_radius_rsampler(1)[0]
            col_radius_sample = (col_radius_sample * self.col_radius_rsampler1(1)[0]) ** 0.5
            if self.with_global:
                col_global_scale_sample = (self.col_global_scale_rsampler(1)[0] *
                                           self.col_global_scale_rsampler1(1)[0]) ** 0.5
                col_radius_sample = col_radius_sample * col_global_scale_sample
            col_weight = col_direction_sample * col_radius_sample.unsqueeze(-1)

            weight = row_weight * col_weight.t()
        elif self.in_channels == 1 and self.out_channels > 1:
            weight = self.row_rsampler(1)[0].unsqueeze(1)
        elif self.out_channels == 1 and self.in_channels > 1:
            weight = self.col_rsampler(1)[0].unsqueeze(0)

        weight_expand_size = torch.Size(self.weight_size[:2]) + torch.Size([1] * (len(self.weight_size) - 2))
        filter_bank = weight.view(weight_expand_size) * filter_sample
        bias = self.bias_rsampler(1)[0] if self.bias_rsampler is not None else None
        return F.conv2d(input, filter_bank, bias, self.stride, self.padding, self.dilation, self.groups)
