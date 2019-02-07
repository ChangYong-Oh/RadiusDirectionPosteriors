import math
import warnings

import scipy.special

import torch
from torch.autograd import Function


LOG_2 = math.log(2.0)
LOG_PI = math.log(math.pi)
LOG_2PI = math.log(2.0 * math.pi)
PYTORCH_LOG_LOWER_BOUND = 1e-45
SCIPY_IVE_UPPER_BOUND = 1e9
STABLE_THRESHOLD = (math.exp(6) + math.exp(10)) / 2.0


def KL_Normal(q_mean, q_var, p_mean, p_var):
    kld = (q_mean - p_mean) ** 2 / (2 * p_var) + (q_var / p_var - 1 - torch.log(q_var) + torch.log(p_var)) / 2.0
    if (kld < 0).any():
        warnings.warn('KLD Normal has negative value %f' % float(torch.min(kld)))
    return kld


def KL_Gamma(q_shape, q_rate, p_shape, p_rate):
    kld = (q_shape - p_shape) * torch.digamma(q_shape)
    kld += torch.lgamma(p_shape) - torch.lgamma(q_shape)
    kld += p_shape * (torch.log(q_rate) - torch.log(p_rate)) + q_shape * (p_rate - q_rate) / q_rate
    if (kld < 0).any():
        warnings.warn('KLD Gamma has negative value %f' % float(torch.min(kld)))
    return kld


def KL_Weibull(q_shape, q_scale, p_shape, p_scale):
    EM_const = 0.5772
    q_log_shape = torch.log(q_shape)
    p_log_shape = torch.log(p_shape)
    kld = (q_log_shape - p_log_shape) - (q_shape * torch.log(q_scale) - p_shape * torch.log(p_scale))
    kld += (q_shape - p_shape) * (torch.log(q_scale) - EM_const / q_shape)
    kld += (q_scale / p_scale) ** p_shape * torch.exp(torch.lgamma(p_shape / q_shape + 1)) - 1
    if (kld < 0).any():
        warnings.warn('KLD Weibull has negative value %f' % float(torch.min(kld)))
    return kld


def KL_LogNormal_Normal(q_mean, q_var, p_mean, p_var):
    kld = -(0.5 * LOG_2PI + 0.5 + 0.5 * torch.log(q_var) + q_mean)
    kld += (0.5 * LOG_2PI + 0.5 * torch.log(p_var))
    kld += (0.5 / p_var * torch.exp(q_mean + 0.5 * q_var) * (torch.exp(q_mean + 1.5 * q_var) - 2.0 * p_mean))
    kld += (0.5 * p_mean ** 2 / p_var)
    if (kld < 0).any():
        warnings.warn('KLD LogNormal_Normal has negative value %f' % float(torch.min(kld)))
    return kld


def KL_LogNormal_Gamma(q_mean, q_var, p_shape, p_rate):
    kld = p_shape * torch.log(p_rate) + torch.lgamma(p_shape) - p_shape * q_mean + torch.exp(q_mean + q_var / 2.0 - torch.log(p_rate)) - (torch.log(q_var) + LOG_2 + 1) / 2.0
    if (kld < 0).any():
        warnings.warn('KLD LogNormal_Gamma has negative value %f' % float(torch.min(kld)))
    return kld


def KL_LogNormal_invGamma(q_mean, q_var, p_shape, p_rate):
    kld = -p_shape * torch.log(p_rate) + torch.lgamma(p_shape) + p_shape * q_mean + torch.exp(-q_mean + q_var / 2.0 + torch.log(p_rate)) - (torch.log(q_var) + LOG_2 + 1) / 2.0
    if (kld < 0).any():
        warnings.warn('KLD LogNormal_invGamma has negative value %f' % float(torch.min(kld)))
    return kld


def KL_LogNormal_Laplace(q_mean, q_var, p_scale):
    kld = 0
    if (kld < 0).any():
        warnings.warn('KLD LogNormal_Laplace has negative value %f' % float(torch.min(kld)))
    return kld


class KL_vMF_Unif_Function(Function):

    @staticmethod
    def forward(ctx, loc, concentration):
        assert loc.size()[:-1] == concentration.size()
        dim = loc.size(-1)
        nu = float(dim / 2.0)
        ctx.intermediate = (loc, concentration, loc.new_full((1,), dim))

        bessel_ratio_lower, bessel_ratio_upper = _bessel_ratio_bound(nu, concentration)
        bessel_ratio = (bessel_ratio_lower + bessel_ratio_upper) / 2.0 * concentration
        # Below approximation is far superior than using scipy.special.ive when nu is large, scipy only returns nan (log(0))
        # sometimes even this return negative KLD which does not make sense, but much smaller than the error with scipy.special.ive
        # normalization = (nu - 1.0) * log_concentration - _approximate_log_iv_series(nu - 1.0, concentration)
        # Exact calculation is not needed because this value is not needed in backward pass so we use cheaper approximation for monitoring purpose.
        normalization = (nu - 1.0) * torch.log(concentration) - _approximate_log_iv_bound(nu - 1.0, concentration)
        constant = (1.0 - nu) * LOG_2 - math.lgamma(dim / 2.0)
        kld = bessel_ratio + normalization + constant
        if (kld != kld).any():
            raise RuntimeError('KL divergence of vMF : %d nan values out of %d from %.6E' % ((kld != kld).sum(), kld.numel(), torch.min(kld)))
        # if (kld < 0).any():
        #     print('KL divergence of vMF : %d negative values out of %d from %.6E' % ((kld < 0).sum(), kld.numel(), torch.min(kld)))
            # Due to numerical instabiliy of Bessel function some negative values are inevitable
            # raise RuntimeError('KL divergence of vMF is negative.')
        return kld

    @staticmethod
    def backward(ctx, grad_output):
        loc, concentration, dim = ctx.intermediate
        nu = float(dim / 2.0)
        grad_loc = grad_concentration = None

        if ctx.needs_input_grad[0]:
            grad_loc = torch.zeros_like(loc)
        if ctx.needs_input_grad[1]:
            ratio1_lower_bound, ratio1_upper_bound = _bessel_ratio_bound(nu + 1.0, concentration)
            ratio0_lower_bound, ratio0_upper_bound = _bessel_ratio_bound(nu, concentration)

            upper_bound = 0.5 * (ratio0_upper_bound * (concentration * (ratio1_upper_bound - 2 * ratio0_lower_bound) - 2 * nu + 2) + concentration)
            lower_bound = 0.5 * (ratio0_lower_bound * (concentration * (ratio1_lower_bound - 2 * ratio0_upper_bound) - 2 * nu + 2) + concentration)
            estimate = (upper_bound + lower_bound) / 2.0

            grad_concentration = estimate * grad_output

        return grad_loc, grad_concentration


class KL_vMF_kappa_Function(Function):

    @staticmethod
    def forward(ctx, loc, q_concentration, p_concentration):
        assert loc.size()[:-1] == q_concentration.size()
        assert q_concentration.size() == p_concentration.size()
        dim = loc.size(-1)
        nu = float(dim / 2.0)
        
        bessel_ratio_lower, bessel_ratio_upper = _bessel_ratio_bound(nu, q_concentration)
        bessel_ratio = (bessel_ratio_lower + bessel_ratio_upper) / 2.0
        term_bessel_ratio = (q_concentration - p_concentration) * bessel_ratio
        term_log_kappa = (nu - 1.0) * (torch.log(q_concentration) - torch.log(p_concentration))
        log_bessel = _approximate_log_iv_series(nu - 1.0, torch.stack([q_concentration, p_concentration], dim=0))
        term_log_bessel = log_bessel[0] - log_bessel[1]
        kld = term_bessel_ratio + term_log_kappa - term_log_bessel
        if (kld != kld).any():
            raise RuntimeError('KL divergence of vMF : %d nan values out of %d from %.6E' % ((kld != kld).sum(), kld.numel(), torch.min(kld)))
        if (kld < -1e-1).any():
            print('KL divergence of vMF : %d negative values out of %d from %.6E' % ((kld < 0).sum(), kld.numel(), torch.min(kld)))
            print(q_concentration)
            print(p_concentration)
            print(kld)
            raise RuntimeError('KL divergence of vMF is negative.')
        ctx.intermediate = (loc, q_concentration, p_concentration, bessel_ratio_lower, bessel_ratio_upper, loc.new_full((1,), dim))
        return kld

    @staticmethod
    def backward(ctx, grad_output):
        loc, q_concentration, p_concentration, bessel_ratio_lower, bessel_ratio_upper, dim = ctx.intermediate
        nu = float(dim / 2.0)
        grad_loc = grad_q_concentration = grad_p_concentration = None

        if ctx.needs_input_grad[0]:
            grad_loc = torch.zeros_like(loc)
        if ctx.needs_input_grad[1]:
            ratio1_lower_bound, ratio1_upper_bound = _bessel_ratio_bound(nu + 1.0, q_concentration)
            ratio0_lower_bound, ratio0_upper_bound = bessel_ratio_lower, bessel_ratio_upper

            kappa_const = (q_concentration - p_concentration) / 2.0
            upper_bound = kappa_const * (ratio0_upper_bound * (ratio1_upper_bound - 2 * ratio0_lower_bound - (2 * nu - 2) / q_concentration) + 1.0)
            lower_bound = kappa_const * (ratio0_lower_bound * (ratio1_lower_bound - 2 * ratio0_upper_bound - (2 * nu - 2) / q_concentration) + 1.0)
            grad_estimate = (upper_bound + lower_bound) / 2.0

            # For large kappa_q even above approximation becomes unstable
            unstable_grad_ind = q_concentration > STABLE_THRESHOLD
            taylor_lower_bound, taylor_upper_bound = _approximate_derivative_taylor(nu, q_concentration[unstable_grad_ind])
            grad_estimate[unstable_grad_ind] = kappa_const[unstable_grad_ind] * (taylor_lower_bound + taylor_upper_bound) / 2.0

            grad_q_concentration = grad_estimate * grad_output

        return grad_loc, grad_q_concentration, grad_p_concentration


def _approximate_log_iv_bound(nu, z):
    iv = torch.zeros_like(z)
    ind = z < 2 * nu
    small_z = z[ind]
    iv[ind] = nu * (torch.log(small_z) - LOG_2) - math.lgamma(nu + 1.0) + (nu + 1.5) / (nu + 1.0) / 2.0 * (small_z - LOG_2)
    large_x = z[~ind]
    iv[~ind] = large_x - 0.5 * (torch.log(large_x) + LOG_2 + LOG_PI)
    return iv


def _approximate_log_iv_series(nu, z):
    fractional_order = nu % 1
    assert fractional_order in [0, 0.5]
    init_ive_value = torch.log(scipy.special.ive(fractional_order, z.clamp(max=SCIPY_IVE_UPPER_BOUND)))
    if z.is_cuda:
        init_ive_value = init_ive_value.cuda()
    init_value = init_ive_value + z
    log_lower_bound = init_value
    log_upper_bound = init_value
    nu_list = torch.FloatTensor([n + fractional_order for n in range(1, int(nu), 2)])
    if z.is_cuda:
        nu_list = nu_list.cuda()
    if nu_list.numel() > 0:
        log_lower_bound_mid, log_upper_bound_mid = _log_bessel_ratio_2step_bound(nu_list, z)
        log_lower_bound += log_lower_bound_mid
        log_upper_bound += log_upper_bound_mid
    if int(nu) % 2 == 1:
        lower_bound_last, upper_bound_last = _bessel_ratio_bound(nu, z)
        log_lower_bound += torch.log(lower_bound_last)
        log_upper_bound += torch.log(upper_bound_last)
    return (log_lower_bound + log_upper_bound) / 2.0


def _log_bessel_ratio_2step_bound(nu_list, z):
    """
    :param nu_list: tensor
    :param z: tensor
    :return: I(2*n)/I(0) or I(2*n+0.5)/I(0.5) bounds
    """
    nu = nu_list.view(torch.Size([-1] + [1] * z.dim()))
    z_sq = z ** 2

    lambda0 = nu - 0.5
    delta0 = (nu - 0.5) + lambda0 / (lambda0 ** 2 + z_sq) ** 0.5 / 2.0
    log_lower_bound = torch.log((1 - 2 * nu / (delta0 + (delta0 ** 2 + z_sq) ** 0.5)).clamp(min=PYTORCH_LOG_LOWER_BOUND))
    lambda2 = nu + 0.5
    delta2 = (nu - 0.5) + lambda2 / (lambda2 ** 2 + z_sq) ** 0.5 / 2.0
    log_upper_bound = torch.log((1 - 2 * nu / (delta2 + (delta2 ** 2 + z_sq) ** 0.5)).clamp(min=PYTORCH_LOG_LOWER_BOUND))

    return log_lower_bound.sum(dim=0), log_upper_bound.sum(dim=0)


def _bessel_ratio_bound(nu, z):
    """
    :param nu:
    :param z:
    :return: I(nu)/I(nu-1) bounds
    """
    z_sq = z ** 2

    lambda0 = nu - 0.5
    delta0 = (nu - 0.5) + lambda0 / (lambda0 ** 2 + z_sq) ** 0.5 / 2.0
    upper_bound = z / (delta0 + (delta0 ** 2 + z_sq) ** 0.5)
    lambda2 = nu + 0.5
    delta2 = (nu - 0.5) + lambda2 / (lambda2 ** 2 + z_sq) ** 0.5 / 2.0
    lower_bound = z / (delta2 + (delta2 ** 2 + z_sq) ** 0.5)

    return lower_bound, upper_bound


def _bessel_ratio_bound_denom(nu, z):
    """
    :param nu:
    :param z:
    :return: I(nu)/I(nu-1) bounds
    """
    z_sq = z ** 2

    lambda0 = nu - 0.5
    delta0 = (nu - 0.5) + lambda0 / (lambda0 ** 2 + z_sq) ** 0.5 / 2.0
    upper_bound = 1.0 / (delta0 + (delta0 ** 2 + z_sq) ** 0.5)
    lambda2 = nu + 0.5
    delta2 = (nu - 0.5) + lambda2 / (lambda2 ** 2 + z_sq) ** 0.5 / 2.0
    lower_bound = 1.0 / (delta2 + (delta2 ** 2 + z_sq) ** 0.5)

    return lower_bound, upper_bound


def _approximate_derivative_taylor(nu, z):
    ratio1_lower_bound_denom, ratio1_upper_bound_denom = _bessel_ratio_bound_denom(nu + 1.0, z)
    ratio0_lower_bound_denom, ratio0_upper_bound_denom = _bessel_ratio_bound_denom(nu, z)

    lower_bound = -(nu - 0.5) * (ratio1_lower_bound_denom - ratio0_upper_bound_denom)
    upper_bound = -(nu - 0.5) * (ratio1_lower_bound_denom - ratio0_upper_bound_denom)

    return lower_bound, upper_bound


def _ive_scipy(v, z):
    np_output = scipy.special.ive(v, z.cpu() if z.is_cuda else z).numpy()
    torch_output = torch.from_numpy(np_output).type(torch.float)
    if z.is_cuda:
        return torch_output.cuda()
    else:
        return torch_output


KL_vMF_Unif = KL_vMF_Unif_Function.apply
KL_vMF_kappa = KL_vMF_kappa_Function.apply


if __name__ == '__main__':
    pass
    # import matplotlib.pyplot as plt
    # n_batch = 1000
    # n_in = 50
    # nu = float(n_in) / 2.0
    #
    # q_kappa = torch.linspace(math.exp(6), math.exp(10), n_batch)
    # p_kappa = torch.ones_like(q_kappa) * float(torch.mean(q_kappa))
    # q_log_kappa = torch.log(q_kappa)
    # p_log_kappa = torch.log(p_kappa)
    # loc_aug = torch.randn(n_batch, n_in)
    # loc = loc_aug / (loc_aug ** 2).sum(dim=-1, keepdim=True) ** 0.5
    # q_log_kappa.requires_grad_()
    # loc.requires_grad_()
    # #
    # y = KL_vMF_kappa(loc, q_log_kappa, p_log_kappa)
    # # print(y)
    # # exit(0)
    # # y = KL_vMF_Unif(loc, q_log_kappa)
    # y.backward(torch.ones_like(q_log_kappa))
    # kappa_grad = q_log_kappa.grad.data
    #
    # ratio0_lower_bound, ratio0_upper_bound = _bessel_ratio_bound(nu, q_kappa)
    # gradient_lower_bound, gradient_upper_bound = _approximate_derivative_taylor(nu, q_kappa)
    #
    # kappa_const = (q_kappa - p_kappa) / 2.0
    # upper_bound = kappa_const * gradient_upper_bound
    # lower_bound = kappa_const * gradient_lower_bound
    # avg_bound = (upper_bound + lower_bound) / 2.0
    # # avg_bound = kappa_const * (ratio0 * (ratio1 - 2 * ratio0) + 1.0) - (nu - 1.0) * ratio0 * (1 - p_kappa / q_kappa)
    # grad_estimate = avg_bound * q_kappa
    #
    # fig, axes = plt.subplots(2, 1, sharex=True)
    #
    # p_kappa_value = float(p_kappa[0])
    # p_log_kappa_value = math.log(p_kappa_value)
    # x = q_kappa.detach().numpy()
    # vline_value = p_kappa_value
    # axes[0].plot(x, y.detach().numpy())
    # axes[0].axvline(x=vline_value, ls='--', color='m')
    # axes[1].plot(x, kappa_grad.detach().numpy(), alpha=0.5)
    # axes[1].axvline(x=vline_value, ls='--', color='m')
    # axes[1].axhline(y=0, ls=':')
    # axes[1].plot(x, grad_estimate.detach().numpy(), ls='-', color='g')
    #
    # plt.suptitle('(True:%f)' % vline_value)
    # plt.show()

    ## For vMF backward check
    # from torch.autograd import gradcheck
    #
    # n_batch = 5
    # n_in = 25
    #
    # kappa = torch.exp(torch.randn(n_batch))
    # log_kappa = kappa.log()
    # mu_aug = torch.randn(n_batch, n_in)
    # mu = mu_aug / (mu_aug ** 2).sum(dim=-1, keepdim=True) ** 0.5
    # log_kappa.requires_grad_()
    # mu.requires_grad_()
    #
    # test = gradcheck(KL_vMF_Unif, (mu, log_kappa), eps=1e-3, atol=1e-4)
    # print(test)

    ## Running time check
    # import time
    # n_repeat = 1000
    # n_batch = 1000
    # n_in = 1500
    # accum_time_forward = 0
    # accum_time_backward = 0
    # for i in range(n_repeat):
    #     kappa = torch.exp(torch.randn(n_batch) * 5 + 1)
    #     log_kappa = kappa.log()
    #     mu_aug = torch.randn(n_batch, n_in)
    #     mu = mu_aug / (mu_aug ** 2).sum(dim=-1, keepdim=True) ** 0.5
    #     log_kappa.requires_grad_()
    #     mu.requires_grad_()
    #     gradient = torch.ones_like(log_kappa)
    #
    #     start_time = time.time()
    #     y = KL_vMF_Unif(mu, log_kappa)
    #     accum_time_forward += time.time() - start_time
    #     start_time = time.time()
    #     y.backward(gradient)
    #     accum_time_backward += time.time() - start_time
    # print('Forward  pass : %f %f' % (accum_time_forward, accum_time_forward / n_repeat))
    # print('Backward pass : %f %f' % (accum_time_backward, accum_time_backward / n_repeat))

    ## For vMF backward check
    n_batch = 5
    n_in = 500

    q_kappa = torch.rand(n_batch) * 50000
    p_kappa = torch.rand(n_batch) * 50000
    mu_aug = torch.randn(n_batch, n_in)
    mu = mu_aug / (mu_aug ** 2).sum(dim=-1, keepdim=True) ** 0.5
    print(q_kappa)
    print(p_kappa)
    kld = KL_vMF_kappa(mu, q_kappa, p_kappa)

    # q_log_shape = torch.ones(1, 1).normal_() * 0 + 0.2636
    # q_log_shape.requires_grad_()
    # q_log_rate = torch.ones(1, 1).normal_() * 0 + 0.0853
    # p_log_shape = torch.ones(1, 1) * 0
    # p_log_rate = torch.ones(1, 1) * 0
    #
    # q_shape = torch.exp(q_log_shape)
    # q_rate = torch.exp(q_log_rate)
    # p_shape = torch.exp(p_log_shape)
    # p_rate = torch.exp(p_log_rate)
    #
    # kld = (q_shape - p_shape) * torch.digamma(q_shape)
    # kld += torch.lgamma(p_shape) - torch.lgamma(q_shape)
    # kld += p_shape * (q_log_rate - p_log_rate) + q_shape * (p_rate - q_rate) / q_rate
    # kld.backward(torch.ones(1, 1))
    # print(q_log_shape)
    # print(q_log_rate)
    # print(q_log_shape.grad.data)

