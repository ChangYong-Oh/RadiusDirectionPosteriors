import math
from numbers import Number

import torch
from torch.nn import Parameter
from torch.nn.functional import softplus
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.reparametrized_sample import ReparametrizedSample


class WeibullReparametrizedSample(ReparametrizedSample):

    def __init__(self, batch_shape):
        super(WeibullReparametrizedSample, self).__init__()
        self.batch_shape = batch_shape
        self.softplus_inv_shape = Parameter(torch.Tensor(batch_shape))
        self.softplus_inv_scale = Parameter(torch.Tensor(batch_shape))
        self.softplus_inv_shape_normal_mean = math.log(0.5)
        self.softplus_inv_shape_normal_std = 0.0001
        self.softplus_inv_scale_normal_mean = 0
        self.softplus_inv_scale_normal_std = 0.0001

    # TODO : sometimes gradient is nan
    def forward(self, sample_shape):
        if isinstance(sample_shape, Number):
            sample_shape = torch.Size([sample_shape])
        if self.deterministic:
            assert sample_shape == torch.Size([1])
            return self.mean().unsqueeze(0)
        shape = softplus(self.softplus_inv_shape)
        scale = softplus(self.softplus_inv_scale)

        uniform_sample = scale.new(torch.Size(sample_shape + self.batch_shape)).uniform_()
        weibull_sample = (-torch.log(1.0 - uniform_sample)) ** (1.0 / shape) * scale

        return weibull_sample

    def mean(self):
        shape = softplus(self.softplus_inv_shape)
        scale = softplus(self.softplus_inv_scale)
        return scale * torch.exp(torch.lgamma(1.0 + 1.0 / shape))

    def mode(self):
        shape = softplus(self.softplus_inv_shape)
        scale = softplus(self.softplus_inv_scale)
        return scale * ((shape - 1).clamp(min=0) / shape) ** (1.0 / shape)

    def variance(self):
        shape = softplus(self.softplus_inv_shape)
        scale = softplus(self.softplus_inv_scale)
        return scale ** 2 * (torch.exp(torch.lgamma(1.0 + 2.0 / shape)) - torch.exp(2.0 * torch.lgamma(1.0 + 1.0 / shape)))

    def signal_to_noise_ratio(self):
        shape = softplus(self.softplus_inv_shape)
        scale = softplus(self.softplus_inv_scale)
        return torch.exp(self.log_scale + torch.lgamma(1.0 + 1.0 / torch.exp(self.log_shape))) / scale / (torch.exp(torch.lgamma(1.0 + 2.0 / shape)) - torch.exp(2.0 * torch.lgamma(1.0 + 1.0 / shape))) ** 0.5

    def reset_parameters(self, hyperparams={}):
        if 'Weibull' in hyperparams.keys():
            self.softplus_inv_shape_normal_mean = hyperparams['Weibull']['softplus_inv_shape_normal_mean']
            self.softplus_inv_shape_normal_std = hyperparams['Weibull']['softplus_inv_shape_normal_std']
            self.softplus_inv_scale_normal_mean = hyperparams['Weibull']['softplus_inv_scale_normal_mean']
            self.softplus_inv_scale_normal_std = hyperparams['Weibull']['softplus_inv_scale_normal_std']
        torch.nn.init.normal_(self.softplus_inv_shape, self.softplus_inv_shape_normal_mean, self.softplus_inv_shape_normal_std)
        torch.nn.init.normal_(self.softplus_inv_scale, self.softplus_inv_scale_normal_mean, self.softplus_inv_scale_normal_std)

    def init_hyperparams_repr(self):
        return 'softplus_inv(shape)~Normal(%.2E, %.2E), softplus_inv(scale)~Normal(%.2E, %.2E)' % (self.softplus_inv_shape_normal_mean, self.softplus_inv_shape_normal_std, self.softplus_inv_scale_normal_mean, self.softplus_inv_scale_normal_std)

    def extra_repr(self):
        return 'batch_shape={}'.format(
            self.batch_shape
        )


if __name__ == '__main__':
    pass
