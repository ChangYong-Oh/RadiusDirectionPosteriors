import math
from numbers import Number

import torch
from torch.distributions import Beta
from torch.nn import Parameter
from torch.nn.functional import softplus
from torch.autograd import grad
from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.reparametrized_sample import ReparametrizedSample
from BayesianNeuralNetwork.torch_user.nn.utils import ml_kappa, softplus_inv, softplus_inv_derivative, kaiming_transpose

EPSILON = 0.005


class VonMisesFisherReparametrizedSample(ReparametrizedSample):

    def __init__(self, batch_shape, event_shape):
        self.batch_shape = batch_shape
        if isinstance(event_shape, Number):
            event_shape = torch.Size([event_shape])
        self.event_shape = event_shape
        assert len(event_shape) == 1
        self.dim = int(event_shape[0])
        super(VonMisesFisherReparametrizedSample, self).__init__()
        self.loc = Parameter(torch.Tensor(batch_shape + event_shape))
        self.softplus_inv_concentration = Parameter(torch.Tensor(torch.Size([1])))
        # Too large kappa slow down rejection sampling, so we set upper bound, which is called in forward pass
        self.softplus_inv_concentration_upper_bound = softplus_inv(ml_kappa(dim=float(event_shape[0]), eps=2e-3))
        self.beta_sample = None
        self.concentration = None
        self.gradient_correction_required = True
        self.softplus_inv_concentration_normal_mean = softplus_inv(ml_kappa(dim=float(event_shape[0]), eps=EPSILON))
        self.softplus_inv_concentration_normal_std = 0.001
        self.direction_init_method = None
        self.rsample = None
        self.loc_init_type = 'random'

    def forward(self, sample_shape):
        if isinstance(sample_shape, Number):
            sample_shape = torch.Size([sample_shape])
        if self.deterministic:
            assert sample_shape == torch.Size([1])
            return self.loc.unsqueeze(0)
        w_sample = self._rejection_sampling(sample_shape).unsqueeze(-1)
        assert (torch.abs(w_sample) < 1).all()
        spherical_section_sample = self._spherical_section_uniform_sampling(sample_shape)
        rsample_concentration = torch.cat([w_sample, (1 - w_sample ** 2) ** 0.5 * spherical_section_sample], dim=-1)
        rsample = self._householder_transformation(rsample_concentration, sample_shape)
        return rsample

    def sample_kld(self):
        pass

    def mode(self):
        return self.loc / torch.sum(self.loc ** 2, dim=-1, keepdim=True) ** 0.5

    def parameter_adjustment(self):
        self.loc.data /= torch.sum(self.loc.data ** 2, dim=-1, keepdim=True) ** 0.5
        self.softplus_inv_concentration.data = self.softplus_inv_concentration.data.clamp(max=self.softplus_inv_concentration_upper_bound)

    def reset_parameters(self, hyperparams={}):
        if 'vMF' in hyperparams.keys():
            if 'direction' in hyperparams['vMF'].keys():
                if type(hyperparams['vMF']['direction']) == str:
                    if hyperparams['vMF']['direction'] == 'kaiming':
                        self.direction_init_method = torch.nn.init.kaiming_normal_
                    elif hyperparams['vMF']['direction'] == 'kaiming_transpose':
                        self.direction_init_method = kaiming_transpose
                    elif hyperparams['vMF']['direction'] == 'orthogonal':
                        self.direction_init_method = torch.nn.init.orthogonal_
                    self.direction_init_method(self.loc)
                elif type(hyperparams['vMF']['direction']) == torch.Tensor:
                    self.loc.data.copy_(hyperparams['vMF']['direction'])
                    self.loc_init_type = 'fixed'
                else:
                    raise NotImplementedError

                self.loc.data /= torch.sum(self.loc.data ** 2, dim=-1, keepdim=True) ** 0.5

            if 'softplus_inv_concentration_normal_mean' in hyperparams['vMF'].keys():
                self.softplus_inv_concentration_normal_mean = hyperparams['vMF']['softplus_inv_concentration_normal_mean']
            if 'softplus_inv_concentration_normal_mean_via_epsilon' in hyperparams['vMF'].keys():
                epsilon = hyperparams['vMF']['softplus_inv_concentration_normal_mean_via_epsilon']
                self.softplus_inv_concentration_normal_mean = softplus_inv(ml_kappa(dim=float(self.event_shape[0]), eps=epsilon))
            if 'softplus_inv_concentration_normal_std' in hyperparams['vMF'].keys():
                self.softplus_inv_concentration_normal_std = hyperparams['vMF']['softplus_inv_concentration_normal_std']
        torch.nn.init.normal_(self.softplus_inv_concentration, self.softplus_inv_concentration_normal_mean, self.softplus_inv_concentration_normal_std)

    def init_hyperparams_repr(self):
        if self.loc_init_type == 'random':
            loc_init_str = 'location ' + self.direction_init_method.func_name
        elif self.loc_init_type == 'fixed':
            loc_init_str = 'location fixed'
        else:
            raise NotImplementedError
        return '%s, Softplus inverse(scale)~Normal(%.2E, %.2E)' % (loc_init_str, self.softplus_inv_concentration_normal_mean, self.softplus_inv_concentration_normal_std)

    def _rejection_sampling(self, sample_shape):
        """
        :param concentration: tensor
        :param dim: scalar
        :return:
        """
        # concentration = softplus(self.softplus_inv_concentration)
        concentration = softplus(self.softplus_inv_concentration).repeat(self.batch_shape)
        beta_param = 0.5 * (self.dim - 1)
        flattened_concentration = concentration.repeat(sample_shape + torch.Size([1] * concentration.dim())).view(-1)
        sqrt = (4 * flattened_concentration ** 2 + (self.dim - 1) ** 2) ** 0.5

        b = (-2 * flattened_concentration + sqrt) / (self.dim - 1)
        # when concentration is too large compared to dim then b is zero due to underflow. so in this case, we use taylor approximation, bad_b means b == 0 or b is inf
        bad_b_ind = torch.isinf(b.detach()) + (b.detach() == 0)
        b[bad_b_ind] = sqrt_taylor_approximation(((self.dim - 1) / (2.0 * flattened_concentration[bad_b_ind])) ** 0.2) * 2.0 * flattened_concentration[bad_b_ind] / (self.dim - 1)

        a = (self.dim - 1 + 2 * flattened_concentration + sqrt) / 4.0
        # in calculation of sqrt square of concentration may give inf
        inf_a_ind = torch.isinf(a.detach())
        a[inf_a_ind] = ((sqrt_taylor_approximation(((self.dim - 1) / (2.0 * flattened_concentration[inf_a_ind])) ** 0.2) + 2) * 2 * flattened_concentration[inf_a_ind] + self.dim - 1) / 4.0

        d = 4 * a * b / (1 + b) - (self.dim - 1) * math.log(self.dim - 1)
        rejected = torch.ones_like(flattened_concentration).byte()
        beta_sample = flattened_concentration.new(flattened_concentration.size())
        if torch.isinf(d).any():
            raise RuntimeError('w sampling in vMF generates infinite d')
        if (d != d).any():
            print(flattened_concentration[d != d])
            print(sqrt[d != d])
            print(b[d != d])
            raise RuntimeError('w sampling in vMF generates nan d')
        while rejected.any():
            rejected_ind = rejected.nonzero().squeeze(1)
            beta_param_tensor = concentration.new_full((int(torch.sum(rejected)),), beta_param)
            eps_beta = Beta(beta_param_tensor, beta_param_tensor).rsample()
            eps_unif = torch.rand_like(eps_beta)
            b_sub = b[rejected]
            a_sub = a[rejected]
            d_sub = d[rejected]
            t_sub = 2 * a_sub * b_sub / (1 - (1 - b_sub) * eps_beta)
            criteria_sub = (self.dim - 1) * torch.log(t_sub) - t_sub + d_sub >= torch.log(eps_unif)
            beta_sample[rejected_ind[criteria_sub]] = eps_beta[criteria_sub]
            rejected[rejected_ind[criteria_sub]] = 0
            # aug = torch.exp((self.dim - 1) * torch.log(t_sub) - t_sub + d_sub)
            # print('%3d %.4E~%.4E' % (rejected_ind.numel(), float(torch.min(aug)), float(torch.max(aug))))
        self.beta_sample = beta_sample.reshape(torch.Size(sample_shape + concentration.size()))
        self.concentration = flattened_concentration.reshape(torch.Size(sample_shape + concentration.size()))
        vmf_sample = (1 - (1 + b.reshape(torch.Size(sample_shape + concentration.size()))) * self.beta_sample) \
                     / (1 - (1 - b.reshape(torch.Size(sample_shape + concentration.size()))) * self.beta_sample)
        return vmf_sample.clamp(min=-1.0 + 1e-7)

    def _spherical_section_uniform_sampling(self, sample_shape):
        shape = torch.Size(sample_shape + self.batch_shape + torch.Size([self.dim - 1]))
        sample = self.loc.new(shape).normal_()
        return sample / (sample ** 2).sum(dim=-1, keepdim=True) ** 0.5

    def _householder_transformation(self, rsample_concentration, sample_shape):
        loc = self.loc / torch.sum(self.loc ** 2, dim=-1, keepdim=True) ** 0.5
        u_prime = torch.zeros_like(loc).index_fill_(dim=-1, index=loc.new_zeros((1,)).long(), value=1) - loc
        u = (u_prime / (u_prime ** 2).sum(dim=-1, keepdim=True) ** 0.5).repeat(torch.Size(sample_shape + torch.Size([1] * loc.dim())))
        return rsample_concentration - 2 * (rsample_concentration * u).sum(dim=-1, keepdim=True) * u

    def gradient_correction(self, integrand):
        dim = self.dim
        nu = float(dim) / 2.0
        beta_sample = self.beta_sample.detach()
        concentration = self.concentration.detach()
        concentration.requires_grad_()
        correction_derivative = self._correction_derivative(concentration, beta_sample, dim)
        correction_deriv_grad = grad(correction_derivative, concentration, grad_outputs=torch.ones_like(correction_derivative))[0]
        concentration.requires_grad_(False)

        ## Bound from Thm4 (A new type of sharp bounds for ratios of modified Bessel functions), a bit loose upper bound for small nu, for larger nu, this works better??
        # bessel_ratio_upper = concentration / ((nu - 1.0) + ((nu + 1.0) ** 2 + concentration ** 2) ** 0.5)
        # bessel_ratio_lower = concentration / ((nu - 0.5) + ((nu + 0.5) ** 2 + concentration ** 2) ** 0.5)
        ## Bound from Thm5 (A new type of sharp bounds for ratios of modified Bessel functions), This is better with larger kappa (less uncertainty case)
        concentration_sq = concentration ** 2
        lambda0 = nu - 0.5
        delta0 = (nu - 0.5) + lambda0 / (lambda0 ** 2 + concentration_sq) ** 0.5 / 2.0
        bessel_ratio_upper = concentration / (delta0 + (delta0 ** 2 + concentration_sq) ** 0.5)
        lambda2 = nu + 0.5
        delta2 = (nu - 0.5) + lambda2 / (lambda2 ** 2 + concentration_sq) ** 0.5 / 2.0
        bessel_ratio_lower = concentration / (delta2 + (delta2 ** 2 + concentration_sq) ** 0.5)

        correction_bessel_ratio = -(bessel_ratio_upper + bessel_ratio_lower) / 2.0
        correction = (integrand.detach() * (correction_bessel_ratio + correction_deriv_grad) * softplus_inv_derivative(concentration)).sum(dim=range(concentration.dim() - len(self.batch_shape)))
        if torch.isinf(correction).any():
            raise RuntimeError('vMF gradient correction is infinite. Check the argument of gradient_correction.')
        if (correction != correction).any():
            if (concentration != concentration).any():
                raise RuntimeError('vMF gradient correction is nan due to concentration.')
            elif (correction_bessel_ratio != correction_bessel_ratio).any():
                raise RuntimeError('vMF gradient correction is nan due to correction bessel ratio.')
            elif (correction_deriv_grad != correction_deriv_grad).any():
                raise RuntimeError('vMF gradient correction is nan due to correction derivative gradient.')
            elif (integrand != integrand).any():
                raise RuntimeError('vMF gradient correction is nan due to integrand.')
        if (self.softplus_inv_concentration.grad.data != self.softplus_inv_concentration.grad.data).any():
            raise RuntimeError('vMF backward is nan.')

        # self.softplus_inv_concentration.grad.data += correction
        self.softplus_inv_concentration.grad.data += correction.sum()

        # This is for updating concentration sampler parameters
        # self.log_concentration_rsample.backward(self.log_concentration.grad.data)

    @staticmethod
    def _correction_derivative(concentration, beta_sample, dim):
        b = (-2 * concentration + (4 * concentration ** 2 + (dim - 1) ** 2) ** 0.5) / (dim - 1)
        # when concentration is too large compared to dim then b is zero due to underflow. so in this case, we use taylor approximation
        bad_b_ind = torch.isinf(b.detach()) + (b.detach() == 0)
        b[bad_b_ind] = sqrt_taylor_approximation(((dim - 1) / (2.0 * concentration[bad_b_ind])) ** 0.2) * 2.0 * concentration[bad_b_ind] / (dim - 1)
        w = (1 - (1 + b) * beta_sample) / (1 - (1 - b) * beta_sample)
        one_w_ind = torch.abs(w.detach()) == 1
        b_reciprocal = 1.0 / b[one_w_ind]
        w[one_w_ind] = -(1 + 2.0 * b_reciprocal + 2.0 * b_reciprocal ** 2)
        w = w.clamp(min=-1 + 1e-7)
        if (torch.abs(w) == 1).any():
            raise RuntimeError('vMF gradient derivative is nan due to w.')
        return w * concentration + (dim - 3.0) / 2.0 * torch.log(1 - w ** 2) + torch.log(torch.abs(2 * b / ((b - 1) * beta_sample + 1) ** 2))


def sqrt_taylor_approximation(z):
    series = z / 2.0
    series += -z ** 2 / 8.0
    series += z ** 3 / 16.0
    series += -z ** 4 / 128.0 * 5.0
    return series


def ive(v, z):
    import scipy.special
    np_output = scipy.special.ive(v, z.cpu() if z.is_cuda else z).numpy()
    torch_output = torch.from_numpy(np_output).type(torch.float)
    if z.is_cuda:
        return torch_output.cuda()
    else:
        return torch_output


if __name__ == '__main__':
    pass
    ## Implementation check code by comparing histogram (projection to location vector mu)
    # import matplotlib.pyplot as plt
    # import numpy as np
    # dim = 5
    # n_in = dim
    # n_out = 1
    # n_sample = 100000
    # kappa = torch.exp(torch.randn(n_out))
    # mu_aug = torch.randn(n_out, n_in)
    # mu = mu_aug / (mu_aug ** 2).sum(dim=-1, keepdim=True) ** 0.5
    # reparam_module = VonMisesFisherReparametrizedSample(batch_shape=torch.Size([n_out]), event_shape=n_in)
    # reparam_module.loc.data = mu
    # reparam_module.log_concentration.data = torch.log(kappa)
    # vmf_sample = reparam_module(n_sample).detach()
    # print(kappa)
    # print(mu)
    # print(vmf_sample.mean(dim=0))
    # print(torch.sum(((vmf_sample ** 2).sum(dim=-1) - 1.0) **2))
    # projection = (vmf_sample * mu).sum(dim=[1, 2]).numpy()
    # x = np.linspace(-1, 1, 1000)
    # y = np.exp(kappa.numpy() * x) * (1 - x ** 2) ** ((dim - 3.0) / 2.0)
    # y /= np.sum(y) * (x[1] - x[0])
    # plt.plot(x, y)
    # plt.hist(projection, normed=True, bins=np.arange(-1, 1 + 0.05, 0.05))
    # plt.show()

    ## Gradient Correction checking code
    # n_sample = 10000
    # n_batch = 20
    # eps = 1e-4
    #
    # # Large dim(n_in) less rejection thus smaller correction
    # n_in = 25
    # n_out = n_batch
    # n_data = 3
    # input_data = torch.randn(n_in, n_data)
    # output_data = torch.randn(n_out, n_data)
    #
    # kappa0 = torch.exp(torch.randn(n_batch)) * 0 + torch.rand(1) * 20
    # kappa1 = kappa0 + eps
    # mu_aug = torch.randn(1, n_in).repeat(n_batch, 1)
    # mu = mu_aug / (mu_aug ** 2).sum(dim=-1, keepdim=True) ** 0.5
    # dim = n_in
    # # For this function derivative should be derivative of C_p(k) multiplied by sphere area
    # func = lambda x: torch.exp((x * mu).sum(dim=-1) * (-kappa0))
    # sphere_area = 2 * math.pi ** (dim / 2.0) / math.gamma(dim / 2.0)
    # normalization_gradient = (dim / 2.0 - 1.0) - kappa0 / ive(dim / 2.0 - 1.0, kappa0) * (ive(dim / 2.0 - 2.0, kappa0) + ive(dim / 2.0, kappa0)) / 2.0
    # normalization_gradient *= kappa0 ** (dim / 2.0 - 2.0) / (2 * math.pi) ** (dim / 2.0) / (ive(dim / 2.0 - 1.0, kappa0) * torch.exp(kappa0))
    # print('For exp(-k mu z), gradient is given in closed form')
    # print('Exact gradient      : ' + ' '.join(['%+6.4f' % (normalization_gradient * sphere_area * kappa0)[i] for i in range(n_batch)]))
    # reparam_module0 = VonMisesFisherReparametrizedSample(torch.Size([n_batch]), event_shape=n_in)
    # reparam_module0.loc.data = mu
    # reparam_module0.log_concentration.data = torch.log(kappa0)
    # vmf_sample0 = reparam_module0(n_sample)
    # func_val0 = func(vmf_sample0)
    # reparam_module1 = VonMisesFisherReparametrizedSample(torch.Size([n_batch]), event_shape=n_in)
    # reparam_module1.loc.data = mu
    # reparam_module1.log_concentration.data = torch.log(kappa1)
    # vmf_sample1 = reparam_module1(n_sample)
    # func_val1 = func(vmf_sample1)
    # numerical_grad_sample = (func_val1 - func_val0) / eps * kappa0
    # numerical_grad_mean = numerical_grad_sample.sum(dim=0) / n_sample
    # numerical_grad_std = numerical_grad_sample.std(dim=0)
    # print('%d samles' % n_sample)
    # print('Concentration       : ' + ' '.join(['%+6.4f' % kappa0[i] for i in range(n_batch)]))
    # print('-' * 50)
    # print('Numerical Grad')
    # print('                sum : ' + ' '.join(['%+6.4f' % numerical_grad_mean[i] for i in range(n_batch)]))
    # print('               Mean : %+6.4f   Std : %6.4f' % (torch.mean(numerical_grad_mean), torch.std(numerical_grad_mean)))
    # # print('Numerical Grad  Std : ' + ' '.join(['%6.4f' % numerical_grad_std[i] for i in range(n_batch)]))
    # func_val0.backward(torch.ones_like(func_val0))
    # print('Analytic  Grad')
    # print(' w/o correction sum : ' + ' '.join(['%+6.4f' % (reparam_module0.log_concentration.grad.data[i] / n_sample) for i in range(n_batch)]))
    # print('               Mean : %+6.4f   Std : %12.10f' % (torch.mean(reparam_module0.log_concentration.grad.data / n_sample), torch.std(reparam_module0.log_concentration.grad.data / n_sample)))
    # reparam_module0.gradient_correction(func_val0)
    # print('     correction sum : ' + ' '.join(['%+6.4f' % (reparam_module0.log_concentration.grad.data[i] / n_sample) for i in range(n_batch)]))
    # print('               Mean : %+6.4f   Std : %12.10f' % (torch.mean(reparam_module0.log_concentration.grad.data / n_sample), torch.std(reparam_module0.log_concentration.grad.data / n_sample)))
    # print('If acceptance rate is high in rejection sampling, can we expect correction is small?')

    # m = VonMisesFisherReparametrizedSample(batch_shape=torch.Size([5]), event_shape=3)
    # m.reset_parameters()
    # output = torch.sum(m(1)[0] ** 2)
    # output.backward()
    # print(m.loc.grad.data)
    # m.gradient_correction(output)