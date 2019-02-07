import math
import torch.nn as nn


class BayesianModel(nn.Module):
    def __init__(self):
        super(BayesianModel, self).__init__()

    def kl_divergence(self):
        kld = 0
        for name, m in self.named_modules():
            if m != self and hasattr(m, 'kl_divergence'):
                kld += m.kl_divergence()
        return kld

    def reset_parameters(self, hyperparams={}):
        for name, m in self.named_modules():
            if m != self and hasattr(m, 'reset_parameters') and 'ReparametrizedSample' in m._get_name():
                m.reset_parameters(hyperparams)