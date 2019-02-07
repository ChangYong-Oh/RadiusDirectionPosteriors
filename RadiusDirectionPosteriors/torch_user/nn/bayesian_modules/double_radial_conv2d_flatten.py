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


class DoubleRadialConv2dFlatten(DoubleRadialModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, prior=None, with_global=True):
        super(DoubleRadialConv2dFlatten, self).__init__()
        self.with_global = with_global
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight_size = torch.Size([out_channels, in_channels // groups]) + self.kernel_size
        self.weight_size_col = torch.Size([in_channels // groups, out_channels]) + self.kernel_size
        kernel_size_cumprod = reduce(mul, self.kernel_size, 1)
        self.row_batch_shape = torch.Size([self.weight_size[0]])
        self.row_event_shape = torch.Size([self.weight_size[1] * kernel_size_cumprod])
        self.col_batch_shape = torch.Size([self.weight_size[1]])
        self.col_event_shape = torch.Size([self.weight_size[0] * kernel_size_cumprod])

        self.register_buffer('row_batch_ones', torch.ones(self.row_batch_shape))
        self.register_buffer('col_batch_ones', torch.ones(self.col_batch_shape))
        if bias:
            self.register_buffer('bias_shape_ones', torch.ones([out_channels]))

        assert prior['direction'][0] == 'vMF'
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

            self.col_direction_rsampler = VonMisesFisherReparametrizedSample(batch_shape=self.col_batch_shape, event_shape=self.col_event_shape)
            if self.with_global:
                self.col_global_scale_rsampler = LognormalReparametrizedSample(batch_shape=torch.Size([1]))
                self.col_global_scale_rsampler1 = LognormalReparametrizedSample(batch_shape=torch.Size([1]))
            self.col_radius_rsampler = LognormalReparametrizedSample(batch_shape=self.col_batch_shape)
            self.col_radius_rsampler1 = LognormalReparametrizedSample(batch_shape=self.col_batch_shape)
        elif in_channels == 1 and out_channels > 1:
            self.col_rsampler = NormalReparametrizedSample(batch_shape=self.col_event_shape)
        elif out_channels == 1 and in_channels > 1:
            self.row_rsampler = NormalReparametrizedSample(batch_shape=self.row_event_shape)
        self.bias_rsampler = NormalReparametrizedSample(batch_shape=torch.Size([out_channels])) if bias else None

    # TODO : ELBO optimization initialization should be given

    def forward(self, input):
        # TODO : check reshaping works correctly

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

            weight = row_weight.view(self.weight_size) * col_weight.view(self.weight_size_col).permute([1, 0, 2, 3])# * float(row_weight.size(0)) ** 0.5
        elif self.in_channels == 1 and self.out_channels > 1:
            weight = self.col_rsampler(1)[0].unsqueeze(0).view(self.weight_size_col).permute([1, 0, 2, 3])
        elif self.out_channels == 1 and self.in_channels > 1:
            weight = self.row_rsampler(1)[0].unsqueeze(1).view(self.weight_size)

        bias = self.bias_rsampler(1)[0] if self.bias_rsampler is not None else None
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
