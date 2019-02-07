from numbers import Number

import torch
from torch.nn import Parameter
from torch.nn.functional import softplus
from BayesianNeuralNetwork.torch_user.nn.utils import softplus_inv
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.reparametrized_sample import ReparametrizedSample


class NormalReparametrizedSample(ReparametrizedSample):

    def __init__(self, batch_shape):
        super(NormalReparametrizedSample, self).__init__()
        self.batch_shape = batch_shape
        self.mu = Parameter(torch.Tensor(batch_shape))
        self.softplus_inv_std = Parameter(torch.Tensor(batch_shape))
        self.mu_normal_mean = 0.0
        self.mu_normal_std = 0.0001
        self.softplus_inv_std_normal_mean = softplus_inv(0.0001)
        self.softplus_inv_std_normal_std = 0.0001
        self.normal_sample = None
        self.mu_init_type = 'random'
        self.std_init_type = 'random'

    # TODO : sometimes gradient is nan
    def forward(self, sample_shape):
        if isinstance(sample_shape, Number):
            sample_shape = torch.Size([sample_shape])
        if self.deterministic:
            assert sample_shape == torch.Size([1])
            return self.mu.unsqueeze(0)
        std = softplus(self.softplus_inv_std)

        std_normal_sample = self.mu.new(torch.Size(sample_shape + self.batch_shape)).normal_()
        normal_sample = std_normal_sample * std + self.mu
        self.normal_sample = normal_sample
        return normal_sample

    def sample_kld(self):
        std = softplus(self.softplus_inv_std)
        return (-0.5 * ((self.normal_sample - self.mu) / std) ** 2 - torch.log(std)) - (-0.5 * self.normal_sample ** 2)

    def mean(self):
        return self.mu

    def mode(self):
        return self.mu

    def variance(self):
        return softplus(self.softplus_inv_std) ** 2

    def signal_to_noise_ratio(self):
        return self.mu / softplus(self.softplus_inv_std)

    def reset_parameters(self, hyperparams={}):
        if 'Normal' in hyperparams.keys():
            if 'mu_normal_mean' in hyperparams['Normal'].keys():
                self.mu_normal_mean = hyperparams['Normal']['mu_normal_mean']
            if 'mu_normal_std' in hyperparams['Normal'].keys():
                self.mu_normal_std = hyperparams['Normal']['mu_normal_std']
            if 'log_var_normal_mean' in hyperparams['Normal'].keys():
                self.softplus_inv_std_normal_mean = hyperparams['Normal']['softplus_inv_std_normal_mean']
            if 'log_var_normal_std' in hyperparams['Normal'].keys():
                self.softplus_inv_std_normal_std = hyperparams['Normal']['softplus_inv_std_normal_std']
            torch.nn.init.normal_(self.mu, self.mu_normal_mean, self.mu_normal_std)
            torch.nn.init.normal_(self.softplus_inv_std, self.softplus_inv_std_normal_mean, self.softplus_inv_std_normal_std)
            if 'mu' in hyperparams['Normal'].keys():
                self.mu.data.copy_(hyperparams['Normal']['mu'])
                self.mu_init_type = 'fixed'
            if 'std' in hyperparams['Normal'].keys():
                self.softplus_inv_std.data.copy_(softplus_inv(hyperparams['Normal']['std']))
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
