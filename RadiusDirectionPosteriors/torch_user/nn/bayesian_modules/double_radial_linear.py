import torch
from torch.nn import functional as F
from BayesianNeuralNetwork.torch_user.nn.bayesian_modules.double_radial_module import DoubleRadialModule
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.von_mises_fisher import VonMisesFisherReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.gamma import GammaReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.weibull import WeibullReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.lognormal import LognormalReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.normal import NormalReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.utils import softplus_inv
from torch.nn.functional import softplus


class DoubleRadialLinear(DoubleRadialModule):
    def __init__(self, in_features, out_features, bias, prior=None, with_global=True):
        super(DoubleRadialLinear, self).__init__()
        self.with_global = with_global
        self.in_features = in_features
        self.out_features = out_features
        self.row_batch_shape = torch.Size([out_features])
        self.row_event_shape = torch.Size([in_features])
        self.col_batch_shape = torch.Size([in_features])
        self.col_event_shape = torch.Size([out_features])

        self.register_buffer('row_batch_ones', torch.ones(self.row_batch_shape))
        self.register_buffer('col_batch_ones', torch.ones(self.col_batch_shape))
        if bias:
            self.register_buffer('bias_shape_ones', torch.ones([out_features]))

        assert prior['direction'][0] == 'vMF'
        self.row_direction_prior_param = prior['direction'][1]
        self.row_radius_prior_type = prior['radius'][0]
        self.row_radius_prior_param = prior['radius'][1].copy()
        self.col_direction_prior_param = prior['direction'][1]
        self.col_radius_prior_type = prior['radius'][0]
        self.col_radius_prior_param = prior['radius'][1].copy()
        # self.col_radius_prior_param['halfcauchy_tau'] /= 10.0

        if in_features > 1 and out_features > 1:
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
        elif in_features == 1 and out_features > 1:
            self.row_rsampler = NormalReparametrizedSample(batch_shape=self.row_batch_shape)
        elif out_features == 1 and in_features > 1:
            self.col_rsampler = NormalReparametrizedSample(batch_shape=self.col_batch_shape)
        self.bias_rsampler = NormalReparametrizedSample(batch_shape=torch.Size([out_features])) if bias else None

    # TODO : ELBO optimization initialization should be given

    def forward(self, input):
        if self.in_features > 1 and self.out_features > 1:
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
        elif self.in_features == 1 and self.out_features > 1:
            weight = self.row_rsampler(1)[0].unsqueeze(1)
        elif self.out_features == 1 and self.in_features > 1:
            weight = self.col_rsampler(1)[0].unsqueeze(0)

        bias = self.bias_rsampler(1)[0] if self.bias_rsampler is not None else None
        return F.linear(input, weight, bias)
