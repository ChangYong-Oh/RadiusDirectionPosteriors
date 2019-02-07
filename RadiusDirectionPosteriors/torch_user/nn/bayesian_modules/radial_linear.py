import torch
from torch.nn import functional as F
from BayesianNeuralNetwork.torch_user.nn.bayesian_modules.radial_module import RadialModule
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.von_mises_fisher import VonMisesFisherReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.lognormal import LognormalReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.normal import NormalReparametrizedSample


class RadialLinear(RadialModule):
    def __init__(self, in_features, out_features, bias, prior=None, with_global=True):
        super(RadialLinear, self).__init__()
        self.with_global = with_global
        self.in_features = in_features
        self.out_features = out_features
        batch_shape = torch.Size([in_features])
        event_shape = torch.Size([out_features])
        self.batch_shape = batch_shape

        self.register_buffer('batch_ones', torch.ones(batch_shape))
        if bias:
            self.register_buffer('bias_ones', torch.ones([out_features]))

        assert prior['direction'][0] == 'vMF'
        self.direction_prior_param = prior['direction'][1]
        self.radius_prior_type = prior['radius'][0]
        self.radius_prior_param = prior['radius'][1]

        self.direction_rsampler = VonMisesFisherReparametrizedSample(batch_shape=batch_shape, event_shape=event_shape)
        if self.with_global:
            self.global_scale_rsampler = LognormalReparametrizedSample(batch_shape=torch.Size([1]))
            self.global_scale_rsampler1 = LognormalReparametrizedSample(batch_shape=torch.Size([1]))
        self.radius_rsampler = LognormalReparametrizedSample(batch_shape=batch_shape)
        self.radius_rsampler1 = LognormalReparametrizedSample(batch_shape=batch_shape)
        self.bias_rsampler = NormalReparametrizedSample(batch_shape=torch.Size([out_features])) if bias else None

    def forward(self, input):
        direction_sample = self.direction_rsampler(1)[0]
        radius_sample = self.radius_rsampler(1)[0]
        radius_sample = (radius_sample * self.radius_rsampler1(1)[0]) ** 0.5
        weight = direction_sample * radius_sample.unsqueeze(-1) ** 0.5
        bias = self.bias_rsampler(1)[0] if self.bias_rsampler is not None else None
        return F.linear(input, weight.t(), bias)
