import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import softplus

from BayesianNeuralNetwork.torch_user.nn.reparametrized_sampler.normal import NormalReparametrizedSample


from BayesianNeuralNetwork.torch_user.kl_divergence import KL_Normal


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_rsampler = NormalReparametrizedSample(batch_shape=torch.Size([out_features, in_features]))
        self.bias_rsampler = NormalReparametrizedSample(batch_shape=torch.Size([out_features])) if bias else None

    # TODO : ELBO optimization initialization should be given

    def forward(self, input):
        weight = self.weight_rsampler(1)[0]
        bias = self.bias_rsampler(1)[0] if self.bias_rsampler is not None else None
        return F.linear(input, weight, bias)

    def kl_divergence(self):
        weight_batch_one = torch.ones_like(self.weight_rsampler.mu)
        kld = KL_Normal(self.weight_rsampler.mu, softplus(self.weight_rsampler.softplus_inv_std) ** 2, weight_batch_one * 0, weight_batch_one * 1).sum()
        if self.bias_rsampler is not None:
            bias_batch_one = torch.ones_like(self.bias_rsampler.mu)
            kld += KL_Normal(self.bias_rsampler.mu, softplus(self.bias_rsampler.softplus_inv_std) ** 2, bias_batch_one * 0, bias_batch_one * 1).sum()
        return kld

    def sample_kld(self):
        return self.weight_rsampler.sample_kld().sum() + self.bias_rsampler.sample_kld().sum()