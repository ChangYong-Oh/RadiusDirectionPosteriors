import math
from numbers import Number

import torch
from torch.nn import Parameter
from torch.nn.functional import softplus
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.reparametrized_sample import ReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.utils import softplus_inv


class LognormalReparametrizedSample(ReparametrizedSample):

    def __init__(self, batch_shape):
        super(LognormalReparametrizedSample, self).__init__()
        self.batch_shape = batch_shape
        self.mu = Parameter(torch.Tensor(batch_shape))
        self.softplus_inv_std = Parameter(torch.Tensor(batch_shape))
        self.mu_normal_mean = 0.0
        self.mu_normal_std = 0.0001
        self.softplus_inv_std_normal_mean = softplus_inv(1e-4)
        self.softplus_inv_std_normal_std = 0.0001
        self.mu_init_type = 'random'
        self.std_init_type = 'random'

    # TODO : sometimes gradient is nan
    def forward(self, sample_shape):
        if self.deterministic:
            return self.mean().unsqueeze(0)
        if isinstance(sample_shape, Number):
            sample_shape = torch.Size([sample_shape])
        std = softplus(self.softplus_inv_std)

        std_lognormal_sample = self.mu.new(torch.Size(sample_shape + self.batch_shape)).log_normal_()
        lognormal_sample = std_lognormal_sample ** std * torch.exp(self.mu)

        return lognormal_sample

    def mean(self):
        return torch.exp(self.mu + softplus(self.softplus_inv_std) ** 2 / 2.0)

    def mode(self):
        return torch.exp(self.mu - softplus(self.softplus_inv_std) ** 2)

    def variance(self):
        var = softplus(self.softplus_inv_std) ** 2
        return (torch.exp(var) - 1.0) * torch.exp(2.0 * self.mu + var)

    def signal_to_noise_ratio(self):
        return (torch.exp(softplus(self.softplus_inv_std)) - 1.0) ** -0.5

    def reset_parameters(self, hyperparams={}):
        if 'LogNormal' in hyperparams.keys():
            if 'mu_normal_mean' in hyperparams['LogNormal'].keys():
                self.mu_normal_mean = hyperparams['LogNormal']['mu_normal_mean']
            if 'mu_normal_std' in hyperparams['LogNormal'].keys():
                self.mu_normal_std = hyperparams['LogNormal']['mu_normal_std']
            if 'softplus_inv_std_normal_mean' in hyperparams['LogNormal'].keys():
                self.softplus_inv_std_normal_mean = hyperparams['LogNormal']['softplus_inv_std_normal_mean']
            if 'softplus_inv_std_normal_std' in hyperparams['LogNormal'].keys():
                self.softplus_inv_std_normal_std = hyperparams['LogNormal']['softplus_inv_std_normal_std']
            torch.nn.init.normal_(self.mu, self.mu_normal_mean, self.mu_normal_std)
            torch.nn.init.normal_(self.softplus_inv_std, self.softplus_inv_std_normal_mean, self.softplus_inv_std_normal_std)
            if 'mu' in hyperparams['LogNormal'].keys():
                self.mu.data.copy_(hyperparams['LogNormal']['mu'])
                self.mu_init_type = 'fixed'
            if 'std' in hyperparams['LogNormal'].keys():
                self.softplus_inv_std.data.copy_(softplus_inv(hyperparams['LogNormal']['std']))
                self.std_init_type = 'fixed'

    def init_hyperparams_repr(self):
        if self.mu_init_type == 'random':
            mu_init_str = 'mu=Normal(%.2E,%.2E)' % (self.mu_normal_mean, self.mu_normal_std)
        elif self.mu_init_type == 'fixed':
            mu_init_str = 'mu=Fixed value'
        else:
            raise NotImplementedError
        if self.std_init_type == 'random':
            std_init_str = 'std=Normal(%.2E,%.2E)' % (float(softplus(self.softplus_inv_std_normal_mean * torch.ones(1))), self.mu_normal_std)
        elif self.std_init_type == 'fixed':
            std_init_str = 'std=Fixed value'
        else:
            raise NotImplementedError
        return '%s, %s' % (mu_init_str, std_init_str)

if __name__ == '__main__':
    pass
