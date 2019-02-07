from numbers import Number

import torch
from torch.nn import Parameter
from torch.distributions import Gamma
from torch.nn.functional import softplus
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.reparametrized_sample import ReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.utils import softplus_inv_derivative


class GammaReparametrizedSample(ReparametrizedSample):

    def __init__(self, batch_shape):
        super(GammaReparametrizedSample, self).__init__()
        self.batch_shape = batch_shape
        self.shape_aug = 3
        self.softplus_inv_shape = Parameter(torch.Tensor(batch_shape))
        self.softplus_inv_rate = Parameter(torch.Tensor(batch_shape))
        self.c_aug_detach = None
        self.gamma_aug_sample = None
        self.normal_aug_sample = None
        self.concentration_aug_detach = None
        self.gradient_correction_required = True
        self.softplus_inv_shape_normal_mean = 0
        self.softplus_inv_shape_normal_std = 0.0001
        self.softplus_inv_rate_normal_mean = 0
        self.softplus_inv_rate_normal_std = 0.0001

    # TODO : sometimes gradient is nan
    def forward(self, sample_shape):
        if isinstance(sample_shape, Number):
            sample_shape = torch.Size([sample_shape])
        if self.deterministic:
            assert sample_shape == torch.Size([1])
            return self.mean().unsqueeze(0)
        concentration = softplus(self.softplus_inv_shape)
        concentration_aug = concentration + self.shape_aug

        concentration_aug_detach = concentration_aug.detach()
        gamma_aug_sample = Gamma(concentration_aug_detach, torch.ones_like(concentration_aug_detach)).sample(sample_shape)
        self.concentration_aug_detach = concentration_aug_detach
        self.gamma_aug_sample = gamma_aug_sample
        self.c_aug_detach = (9.0 * concentration_aug_detach - 3.0) ** 0.5
        self.normal_aug_sample = ((gamma_aug_sample / (concentration_aug_detach - 1.0 / 3.0)) ** (1.0 / 3.0) - 1.0) * self.c_aug_detach

        reparametrized_gamma_aug_sample = (self.normal_aug_sample / (9.0 * concentration_aug - 3.0) ** 0.5 + 1.0) ** 3 * (concentration_aug - 1.0 / 3.0)

        aug_exp_denom = concentration.unsqueeze(-1) + torch.arange(0, self.shape_aug, device=concentration.device, dtype=concentration.dtype)
        #pytorch uniform random generator uniform_() usually does not generate nonzero number smaller than 1e-8, but does return zero.
        uniform_aug = torch.exp((torch.log(aug_exp_denom.new(sample_shape + aug_exp_denom.size()).uniform_().clamp(min=1e-8)) / aug_exp_denom).sum(dim=-1))
        return reparametrized_gamma_aug_sample * uniform_aug / softplus(self.softplus_inv_rate)

    def mean(self):
        shape = softplus(self.softplus_inv_shape)
        rate = softplus(self.softplus_inv_rate)
        return shape / rate

    def mode(self):
        shape = softplus(self.softplus_inv_shape)
        rate = softplus(self.softplus_inv_rate)
        return ((shape - 1) / rate).clamp(min=0)

    def variance(self):
        shape = softplus(self.softplus_inv_shape)
        rate = softplus(self.softplus_inv_rate)
        return shape / rate ** 2

    def signal_to_noise_ratio(self):
        shape = softplus(self.softplus_inv_shape)
        return shape ** 0.5

    def reset_parameters(self, hyperparams={}):
        if 'Gamma' in hyperparams.keys():
            self.softplus_inv_shape_normal_mean = hyperparams['Gamma']['softplus_inv_shape_normal_mean']
            self.softplus_inv_shape_normal_std = hyperparams['Gamma']['softplus_inv_shape_normal_std']
            self.softplus_inv_rate_normal_mean = hyperparams['Gamma']['softplus_inv_rate_normal_mean']
            self.softplus_inv_rate_normal_std = hyperparams['Gamma']['softplus_inv_rate_normal_std']
        torch.nn.init.normal_(self.softplus_inv_shape, self.softplus_inv_shape_normal_mean, self.softplus_inv_shape_normal_std)
        torch.nn.init.normal_(self.softplus_inv_rate, self.softplus_inv_rate_normal_mean, self.softplus_inv_rate_normal_std)

    def init_hyperparams_repr(self):
        return 'Softplus inverse(shape)~Normal(%.2E, %.2E), Softplus inverse(scale)~Normal(%.2E, %.2E)' % (self.softplus_inv_shape_normal_mean, self.softplus_inv_shape_normal_std, self.softplus_inv_rate_normal_mean, self.softplus_inv_rate_normal_std)

    # TODO : it seems correction is a little bit large, usually non-corrected and corrected one captures true value in between
    def gradient_correction(self, integrand):
        gamma_aug_sample = self.gamma_aug_sample
        normal_aug_sample = self.normal_aug_sample
        c_aug_detach = self.c_aug_detach
        concentration_aug_detach = self.concentration_aug_detach #broadcasting will be used in dlogq and dlogr due to size of gamma sample and normal sample
        dlogq = torch.log(gamma_aug_sample) - torch.digamma(concentration_aug_detach)\
                + ((concentration_aug_detach - 1.0) / gamma_aug_sample - 1.0) * self._reparametrization_grad_concentration(normal_aug_sample, c_aug_detach)
        dlogr = (normal_aug_sample / (c_aug_detach + normal_aug_sample) - 0.5) / (concentration_aug_detach - 1.0 / 3.0)
        correction = ((dlogq - dlogr) * integrand.detach() * softplus_inv_derivative(concentration_aug_detach)).sum(dim=range(gamma_aug_sample.dim() - len(self.batch_shape)))
        self.softplus_inv_shape.grad.data += correction

    @staticmethod
    def _reparametrization_grad_concentration(normal_sample, c):
        return (1 + normal_sample / c) ** 2 * (1 - 0.5 * normal_sample / c)


if __name__ == '__main__':
    pass
    ## Implementation check code by comparing histogram GammaRejectionReparametrizationSample.forward
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from torch import distributions
    # n_sample = 10000
    # concentration = (torch.exp(torch.randn(1) + 0.5))
    # rate = torch.exp(torch.randn(1))
    # reparam_module = GammaRejectionReparametrizedSample(torch.Size([1]))
    # reparam_module.log_shape.data = torch.log(concentration)
    # reparam_module.log_rate.data = torch.log(rate)
    # print('alpha', concentration[0])
    # print('beta', rate[0])
    # gamma_sample = reparam_module(n_sample).detach()
    # rv = distributions.Gamma(concentration[:1], rate[:1])
    # x = torch.linspace(0, float(torch.max(gamma_sample)), 1000)
    # y = rv.log_prob(x).exp()
    # plt.plot(x.numpy(), y.numpy())
    # bw = torch.max(gamma_sample) / 200
    # print(min(gamma_sample), max(gamma_sample))
    # plt.hist(gamma_sample.numpy(), normed=True, bins=np.arange(min(gamma_sample), max(gamma_sample) + bw, bw))
    # plt.title('alpha :%6.4f, beta :%6.4f' % (concentration[0], rate[0]))
    # plt.show()

    ## Gradient Correction checking code
    n_sample = 100000
    n_batch = 20
    eps = 1e-4
    concentration0 = (torch.exp(torch.randn(n_batch))) * 0 + torch.rand(1) * 2
    concentration1 = concentration0 + eps
    rate = torch.exp(torch.randn(n_batch)) * 0 + torch.rand(1) * 2
    func = lambda x: torch.log(x)
    print('For logarithm gradient should be polygamma(1) x concentraion')
    print('Exact gradient      : ' + ' '.join(['%+6.4f' % (torch.polygamma(1, concentration0) * concentration0)[i] for i in range(n_batch)]))
    # func = lambda x: x
    # print('For identity gradient should be 1.0 / rate x concentraion')
    # print('Exact gradient      : ' + ' '.join(['%+6.4f' % (1.0 / rate * concentration0)[i] for i in range(n_batch)]))
    # func = lambda x: x ** 2
    # print('For square gradient should be (1.0 + 2.0 concentration) / rate ** 2 x concentraion')
    # print('Exact gradient      : ' + ' '.join(['%+6.4f' % ((1.0 + 2.0 * concentration0) / rate ** 2 * concentration0)[i] for i in range(n_batch)]))
    reparam_module0 = GammaReparametrizedSample(torch.Size([n_batch]))
    reparam_module0.log_shape.data = torch.log(concentration0)
    reparam_module0.log_rate.data = torch.log(rate)
    gamma_sample0 = reparam_module0(n_sample)
    func_val0 = func(gamma_sample0)
    reparam_module1 = GammaReparametrizedSample(torch.Size([n_batch]))
    reparam_module1.log_shape.data = torch.log(concentration1)
    reparam_module1.log_rate.data = torch.log(rate)
    gamma_sample1 = reparam_module1(n_sample)
    func_val1 = func(gamma_sample1)
    numerical_grad_sample = (func_val1 - func_val0) / eps * concentration0
    numerical_grad_mean = numerical_grad_sample.sum(dim=0) / n_sample
    numerical_grad_std = numerical_grad_sample.std(dim=0)
    print('%d samles' % n_sample)
    print('Concentration       : ' + ' '.join(['%+6.4f' % concentration0[i] for i in range(n_batch)]))
    print('Rate                : ' + ' '.join(['%+6.4f' % rate[i] for i in range(n_batch)]))
    # print('Sample   Mean       : ' + ' '.join(['%+6.4f' % gamma_sample0.mean(dim=0)[i] for i in range(n_batch)]))
    # print('Sample    Var       : ' + ' '.join(['%+6.4f' % gamma_sample0.var(dim=0)[i] for i in range(n_batch)]))
    # print('Function Mean       : ' + ' '.join(['%+6.4f' % func_val0.mean(dim=0)[i] for i in range(n_batch)]))
    # print('Function  Var       : ' + ' '.join(['%+6.4f' % func_val0.var(dim=0)[i] for i in range(n_batch)]))
    print('-' * 50)
    print('Numerical Grad')
    print('                sum : ' + ' '.join(['%+6.4f' % numerical_grad_mean[i] for i in range(n_batch)]))
    print('               Mean : %+6.4f   Std : %6.4f' % (torch.mean(numerical_grad_mean), torch.std(numerical_grad_mean)))
    func_val0.backward(torch.ones_like(func_val0))
    print('Analytic  Grad')
    print(' w/o correction sum : ' + ' '.join(['%+6.4f' % (reparam_module0.log_shape.grad.data[i] / n_sample) for i in range(n_batch)]))
    nonnan_grad = reparam_module0.log_shape.grad.data[reparam_module0.log_shape.grad.data == reparam_module0.log_shape.grad.data]
    print('               Mean : %+6.4f   Std : %12.10f' % (torch.mean(nonnan_grad / n_sample), torch.std(nonnan_grad / n_sample)))
    reparam_module0.gradient_correction(func_val0)
    print('     correction sum : ' + ' '.join(['%+6.4f' % (reparam_module0.log_shape.grad.data[i] / n_sample) for i in range(n_batch)]))
    nonnan_grad = reparam_module0.log_shape.grad.data[reparam_module0.log_shape.grad.data == reparam_module0.log_shape.grad.data]
    print('               Mean : %+6.4f   Std : %12.10f' % (torch.mean(nonnan_grad / n_sample), torch.std(nonnan_grad / n_sample)))
    print('If acceptance rate is high in rejection sampling, can we expect correction is small?')