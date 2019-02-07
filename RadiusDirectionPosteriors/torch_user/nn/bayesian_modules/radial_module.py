import torch
from torch.nn import Module
from torch.nn.functional import softplus
from BayesianNeuralNetwork.torch_user.kl_divergence import KL_vMF_kappa, KL_Gamma, KL_Weibull, KL_LogNormal_Gamma, KL_LogNormal_invGamma, KL_Normal


class RadialModule(Module):
    def __init__(self):
        super(RadialModule, self).__init__()

    def kl_divergence(self):
        q_concentration = softplus(self.direction_rsampler.softplus_inv_concentration) * self.batch_ones
        p_concentration = self.direction_prior_param['concentration'] * self.batch_ones
        kld_direction = KL_vMF_kappa(self.direction_rsampler.loc, q_concentration, p_concentration)
        kld = kld_direction.sum()
        kld_radius = 0
        if self.with_global:
            one_tensor = self.batch_ones.new_ones([1])
            halfcauchy_tau_sq_global = one_tensor * self.radius_prior_param['tau_global'] ** 2
            global_q_var = softplus(self.global_scale_rsampler.softplus_inv_std) ** 2
            global_q_var1 = softplus(self.global_scale_rsampler1.softplus_inv_std) ** 2
            kld_radius += KL_LogNormal_Gamma(self.global_scale_rsampler.mu, global_q_var, 0.5 * one_tensor, halfcauchy_tau_sq_global).sum()
            kld_radius += KL_LogNormal_invGamma(self.global_scale_rsampler1.mu, global_q_var1, 0.5 * one_tensor, one_tensor).sum()
            halfcauchy_tau_sq_local = self.batch_ones * self.radius_prior_param['tau_local'] ** 2
        else:
            halfcauchy_tau_sq_local = self.batch_ones * self.radius_prior_param['tau_local'] ** 2
        q_var = softplus(self.radius_rsampler.softplus_inv_std) ** 2
        q_var1 = softplus(self.radius_rsampler1.softplus_inv_std) ** 2
        kld_radius += KL_LogNormal_Gamma(self.radius_rsampler.mu, q_var, 0.5 * self.batch_ones, halfcauchy_tau_sq_local).sum()
        kld_radius += KL_LogNormal_invGamma(self.radius_rsampler1.mu, q_var1, 0.5 * self.batch_ones, self.batch_ones).sum()
        kld += kld_radius.sum()
        if self.bias_rsampler is not None:
            q_var = softplus(self.bias_rsampler.softplus_inv_std) ** 2
            kld_bias = KL_Normal(self.bias_rsampler.mu, q_var, self.bias_ones * 0, self.bias_ones * 1)
            kld += kld_bias.sum()
        return kld