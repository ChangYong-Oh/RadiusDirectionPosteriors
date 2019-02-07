import torch
from torch.nn import Module
from torch.nn.functional import softplus
from BayesianNeuralNetwork.torch_user.kl_divergence import KL_vMF_kappa, KL_Gamma, KL_Weibull, KL_LogNormal_Normal, KL_LogNormal_Gamma, KL_LogNormal_invGamma, KL_Normal


class DoubleRadialModule(Module):
    def __init__(self):
        super(DoubleRadialModule, self).__init__()

    def kl_divergence(self):
        kld = 0

        if hasattr(self, 'filter_rsampler'):
            filter_q_concentration = softplus(self.filter_rsampler.softplus_inv_concentration) * self.filter_batch_ones
            filter_p_concentration = self.filter_direction_prior_param['filter_concentration'] * self.filter_batch_ones
            kld_filter = KL_vMF_kappa(self.filter_rsampler.loc, filter_q_concentration, filter_p_concentration)
            kld += kld_filter.sum()

        if hasattr(self, 'row_direction_rsampler'):
            row_q_concentration = softplus(self.row_direction_rsampler.softplus_inv_concentration) * self.row_batch_ones
            row_p_concentration = self.row_direction_prior_param['row_concentration'] * self.row_batch_ones
            kld_row_direction = KL_vMF_kappa(self.row_direction_rsampler.loc, row_q_concentration, row_p_concentration).sum()
            kld += kld_row_direction.sum()
        elif hasattr(self, 'row_rsampler'):
            row_q_mu = self.row_rsampler.mu
            row_q_var = softplus(self.row_rsampler.softplus_inv_std) ** 2
            kld_row = KL_Normal(row_q_mu, row_q_var, self.row_batch_ones * 0, self.row_batch_ones * 1)
            kld += kld_row.sum()

        if hasattr(self, 'row_radius_rsampler'):
            kld_row_radius = 0
            if self.with_global:
                one_tensor = self.row_batch_ones.new_ones([1])
                row_halfcauchy_tau_sq_global = one_tensor * self.row_radius_prior_param['tau_global'] ** 2
                row_global_q_var = softplus(self.row_global_scale_rsampler.softplus_inv_std) ** 2
                row_global_q_var1 = softplus(self.row_global_scale_rsampler1.softplus_inv_std) ** 2
                kld_row_radius += KL_LogNormal_Gamma(self.row_global_scale_rsampler.mu, row_global_q_var, 0.5 * one_tensor, row_halfcauchy_tau_sq_global).sum()
                kld_row_radius += KL_LogNormal_invGamma(self.row_global_scale_rsampler1.mu, row_global_q_var1, 0.5 * one_tensor, one_tensor).sum()
                row_halfcauchy_tau_sq_local = self.row_batch_ones * self.row_radius_prior_param['tau_local'] ** 2
            else:
                row_halfcauchy_tau_sq_local = self.row_batch_ones * self.row_radius_prior_param['tau_local'] ** 2
            row_q_var = softplus(self.row_radius_rsampler.softplus_inv_std) ** 2
            row_q_var1 = softplus(self.row_radius_rsampler1.softplus_inv_std) ** 2
            kld_row_radius += KL_LogNormal_Gamma(self.row_radius_rsampler.mu, row_q_var, 0.5 * self.row_batch_ones, row_halfcauchy_tau_sq_local).sum()
            kld_row_radius += KL_LogNormal_invGamma(self.row_radius_rsampler1.mu, row_q_var1, 0.5 * self.row_batch_ones, self.row_batch_ones).sum()
            kld += kld_row_radius

        if hasattr(self, 'col_direction_rsampler'):
            col_q_concentration = softplus(self.col_direction_rsampler.softplus_inv_concentration) * self.col_batch_ones
            col_p_concentration = self.col_direction_prior_param['col_concentration'] * self.col_batch_ones
            kld_col_direction = KL_vMF_kappa(self.col_direction_rsampler.loc, col_q_concentration, col_p_concentration).sum()
            kld += kld_col_direction.sum()
        elif hasattr(self, 'col_rsampler'):
            col_q_mu = self.col_rsampler.mu
            col_q_var = softplus(self.col_rsampler.softplus_inv_std) ** 2
            kld_col = KL_Normal(col_q_mu, col_q_var, self.col_batch_ones * 0, self.col_batch_ones * 1)
            kld += kld_col.sum()

        if hasattr(self, 'col_radius_rsampler'):
            kld_col_radius = 0
            if self.with_global:
                one_tensor = self.col_batch_ones.new_ones([1])
                col_halfcauchy_tau_sq_global = one_tensor * self.col_radius_prior_param['tau_global'] ** 2
                col_global_q_var = softplus(self.col_global_scale_rsampler.softplus_inv_std) ** 2
                col_global_q_var1 = softplus(self.col_global_scale_rsampler1.softplus_inv_std) ** 2
                kld_col_radius += KL_LogNormal_Gamma(self.col_global_scale_rsampler.mu, col_global_q_var, 0.5 * one_tensor, col_halfcauchy_tau_sq_global).sum()
                kld_col_radius += KL_LogNormal_invGamma(self.col_global_scale_rsampler1.mu, col_global_q_var1, 0.5 * one_tensor, one_tensor).sum()
                col_halfcauchy_tau_sq_local = self.col_batch_ones * self.col_radius_prior_param['tau_local'] ** 2
            else:
                col_halfcauchy_tau_sq_local = self.col_batch_ones * self.col_radius_prior_param['tau_local'] ** 2
            col_q_var = softplus(self.col_radius_rsampler.softplus_inv_std) ** 2
            col_q_var1 = softplus(self.col_radius_rsampler1.softplus_inv_std) ** 2
            kld_col_radius += KL_LogNormal_Gamma(self.col_radius_rsampler.mu, col_q_var, 0.5 * self.col_batch_ones, col_halfcauchy_tau_sq_local).sum()
            kld_col_radius += KL_LogNormal_invGamma(self.col_radius_rsampler1.mu, col_q_var1, 0.5 * self.col_batch_ones, self.col_batch_ones).sum()
            kld += kld_col_radius

        if self.bias_rsampler is not None:
            q_var = softplus(self.bias_rsampler.softplus_inv_std) ** 2
            kld_bias = KL_Normal(self.bias_rsampler.mu, q_var, self.bias_shape_ones * 0, self.bias_shape_ones * 1)
            kld += kld_bias.sum()
        return kld