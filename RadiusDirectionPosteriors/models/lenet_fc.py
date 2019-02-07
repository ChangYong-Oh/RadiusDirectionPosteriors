import torch
import torch.nn as nn

from BayesianNeuralNetwork.models.model import BayesianModel
from BayesianNeuralNetwork.torch_user.nn.bayesian_modules.radial_linear import RadialLinear
from BayesianNeuralNetwork.models.lenet_prior import LeNetFCPrior


class LeNetFC(BayesianModel):
    def __init__(self, prior_info):
        super(LeNetFC, self).__init__()
        self.prior = LeNetFCPrior(prior_info)
        fc1_prior, fc2_prior, fc3_prior = self.prior()

        self.fc1 = RadialLinear(in_features=784, out_features=300, bias=True, prior=fc1_prior, with_global=True)
        self.nonlinear1 = nn.Sigmoid()
        self.fc2 = RadialLinear(in_features=300, out_features=100, bias=True, prior=fc2_prior, with_global=True)
        self.nonlinear2 = nn.Sigmoid()
        self.fc3 = RadialLinear(in_features=100, out_features=10, bias=True, prior=fc3_prior, with_global=True)

    def forward(self, input):
        x = input.view(input.size(0), -1)
        x = self.nonlinear1(self.fc1(x))
        x = self.nonlinear2(self.fc2(x))
        x = self.fc3(x)
        return x

    def kl_divergence(self):
        kld = self.fc1.kl_divergence()
        kld += self.fc2.kl_divergence()
        kld += self.fc3.kl_divergence()
        return kld

    def deterministic_forward(self, set_deterministic):
        for m in self.modules():
            if hasattr(m, 'deterministic_forward') and m != self:
                m.deterministic_forward(set_deterministic)

    def init_hyperparam_value(model):
        hyperparam_info_list = []
        for name, m in model.named_modules():
            if hasattr(m, 'init_hyperparams_repr'):
                hyperparam_info_list.append('%s(%s) : %s' % (name, m._get_name(), m.init_hyperparams_repr()))
        return '\n'.join(hyperparam_info_list)