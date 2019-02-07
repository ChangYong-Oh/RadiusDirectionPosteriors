import math

import torch
import torch.nn as nn
from torch.nn.functional import softplus

from BayesianNeuralNetwork.models.model import BayesianModel
from BayesianNeuralNetwork.torch_user.nn.bayesian_modules.double_radial_linear import DoubleRadialLinear
from BayesianNeuralNetwork.models.utils import double_prior_summary
from BayesianNeuralNetwork.torch_user.nn.utils import ml_kappa, softplus_inv


PRIOR_EPSILON = 0.5
GAMMA_SHAPE = 6.0
GAMMA_RATE = 6.0
WEIBULL_SHAPE = 1.0
WEIBULL_SCALE = 1.0
HALFCAUCHY_TAU = 1.0


class MNISTDOUBLEFC3(BayesianModel):
    def __init__(self, prior_type, n_hidden):
        super(MNISTDOUBLEFC3, self).__init__()
        self.prior = MNISTDOUBLEFC3Prior(prior_type, 784, n_hidden, n_hidden, n_hidden)
        fc1_prior, fc2_prior, fc3_prior, fc4_prior = self.prior()

        self.fc1 = DoubleRadialLinear(in_features=784, out_features=n_hidden, bias=True, prior=fc1_prior)
        self.nonlinear1 = nn.ReLU()
        self.fc2 = DoubleRadialLinear(in_features=n_hidden, out_features=n_hidden, bias=True, prior=fc2_prior)
        self.nonlinear2 = nn.ReLU()
        self.fc3 = DoubleRadialLinear(in_features=n_hidden, out_features=n_hidden, bias=True, prior=fc3_prior)
        self.nonlinear3 = nn.ReLU()
        self.fc4 = DoubleRadialLinear(in_features=n_hidden, out_features=10, bias=True, prior=fc4_prior)

    def forward(self, input):
        x = input.view(input.size(0), -1)
        x = self.nonlinear1(self.fc1(x))
        x = self.nonlinear2(self.fc2(x))
        x = self.nonlinear3(self.fc3(x))
        x = self.fc4(x)
        return x

    def kl_divergence(self):
        kld = self.fc1.kl_divergence()
        kld += self.fc2.kl_divergence()
        kld += self.fc3.kl_divergence()
        kld += self.fc4.kl_divergence()
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


class MNISTDOUBLEFC3Prior(object):

    def __init__(self, prior_type, n1, n2, n3, n4):
        self.fc1_prior = None
        self.fc2_prior = None
        self.fc3_prior = None
        self.fc4_prior = None
        if prior_type == 'Gamma':
            self._prior_gamma()
        elif prior_type == 'Weibull':
            self._prior_weibull()
        elif prior_type == 'HalfCauchy':
            self._prior_halfcauchy()
        else:
            raise NotImplementedError
        self.fc1_prior['direction'] = ('vMF', {'row_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n1, eps=PRIOR_EPSILON)),
                                               'col_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n1, eps=PRIOR_EPSILON))})
        self.fc2_prior['direction'] = ('vMF', {'row_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n2, eps=PRIOR_EPSILON)),
                                               'col_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n1, eps=PRIOR_EPSILON))})
        self.fc3_prior['direction'] = ('vMF', {'row_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n3, eps=PRIOR_EPSILON)),
                                               'col_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n1, eps=PRIOR_EPSILON))})
        self.fc4_prior['direction'] = ('vMF', {'row_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n4, eps=PRIOR_EPSILON)),
                                               'col_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n1, eps=PRIOR_EPSILON))})

    def __call__(self):
        return self.fc1_prior, self.fc2_prior, self.fc3_prior, self.fc4_prior

    def __repr__(self):
        prior_info_str_list = ['***PRIORS***']
        prior_info_str_list.append('FC1 : ' + double_prior_summary(self.fc1_prior))
        prior_info_str_list.append('FC2 : ' + double_prior_summary(self.fc2_prior))
        prior_info_str_list.append('FC3 : ' + double_prior_summary(self.fc3_prior))
        prior_info_str_list.append('FC4 : ' + double_prior_summary(self.fc4_prior))
        return '\n'.join(prior_info_str_list)

    def _prior_gamma(self):
        self.fc1_prior = {'radius': ('Gamma', {'softplus_inv_shape': softplus_inv(GAMMA_SHAPE), 'softplus_inv_rate': softplus_inv(GAMMA_RATE)})}
        self.fc2_prior = {'radius': ('Gamma', {'softplus_inv_shape': softplus_inv(GAMMA_SHAPE), 'softplus_inv_rate': softplus_inv(GAMMA_RATE)})}
        self.fc3_prior = {'radius': ('Gamma', {'softplus_inv_shape': softplus_inv(GAMMA_SHAPE), 'softplus_inv_rate': softplus_inv(GAMMA_RATE)})}
        self.fc4_prior = {'radius': ('Gamma', {'softplus_inv_shape': softplus_inv(GAMMA_SHAPE), 'softplus_inv_rate': softplus_inv(GAMMA_RATE)})}

    def _prior_weibull(self):
        self.fc1_prior = {'radius': ('Weibull', {'softplus_inv_shape': softplus_inv(WEIBULL_SHAPE), 'softplus_inv_scale': softplus_inv(WEIBULL_SCALE)})}
        self.fc2_prior = {'radius': ('Weibull', {'softplus_inv_shape': softplus_inv(WEIBULL_SHAPE), 'softplus_inv_scale': softplus_inv(WEIBULL_SCALE)})}
        self.fc3_prior = {'radius': ('Weibull', {'softplus_inv_shape': softplus_inv(WEIBULL_SHAPE), 'softplus_inv_scale': softplus_inv(WEIBULL_SCALE)})}
        self.fc4_prior = {'radius': ('Weibull', {'softplus_inv_shape': softplus_inv(WEIBULL_SHAPE), 'softplus_inv_scale': softplus_inv(WEIBULL_SCALE)})}

    def _prior_halfcauchy(self):
        self.fc1_prior = {'radius': ('HalfCauchy', {'softplus_inv_shape': softplus_inv(0.5), 'softplus_inv_rate': 2 * softplus_inv(HALFCAUCHY_TAU), 'softplus_inv_shape1': softplus_inv(0.5), 'softplus_inv_rate1': softplus_inv(1)})}
        self.fc2_prior = {'radius': ('HalfCauchy', {'softplus_inv_shape': softplus_inv(0.5), 'softplus_inv_rate': 2 * softplus_inv(HALFCAUCHY_TAU), 'softplus_inv_shape1': softplus_inv(0.5), 'softplus_inv_rate1': softplus_inv(1)})}
        self.fc3_prior = {'radius': ('HalfCauchy', {'softplus_inv_shape': softplus_inv(0.5), 'softplus_inv_rate': 2 * softplus_inv(HALFCAUCHY_TAU), 'softplus_inv_shape1': softplus_inv(0.5), 'softplus_inv_rate1': softplus_inv(1)})}
        self.fc4_prior = {'radius': ('HalfCauchy', {'softplus_inv_shape': softplus_inv(0.5), 'softplus_inv_rate': 2 * softplus_inv(HALFCAUCHY_TAU), 'softplus_inv_shape1': softplus_inv(0.5), 'softplus_inv_rate1': softplus_inv(1)})}


class MNISTDOUBLEFC2(BayesianModel):
    def __init__(self, prior_type, n_hidden):
        super(MNISTDOUBLEFC2, self).__init__()
        self.prior = MNISTDOUBLEFC2Prior(prior_type, 784, n_hidden, n_hidden)
        fc1_prior, fc2_prior, fc3_prior = self.prior()

        self.fc1 = DoubleRadialLinear(in_features=784, out_features=n_hidden, bias=True, prior=fc1_prior)
        self.nonlinear1 = nn.ReLU()
        self.fc2 = DoubleRadialLinear(in_features=n_hidden, out_features=n_hidden, bias=True, prior=fc2_prior)
        self.nonlinear2 = nn.ReLU()
        self.fc3 = DoubleRadialLinear(in_features=n_hidden, out_features=10, bias=True, prior=fc3_prior)

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

    def init_hyperparam_value(self):
        hyperparam_info_list = []
        for name, m in self.named_modules():
            if hasattr(m, 'init_hyperparams_repr'):
                hyperparam_info_list.append('%s(%s) : %s' % (name, m._get_name(), m.init_hyperparams_repr()))
        return '\n'.join(hyperparam_info_list)


class MNISTDOUBLEFC2Prior(object):

    def __init__(self, prior_type, n1, n2, n3):
        self.fc1_prior = None
        self.fc2_prior = None
        self.fc3_prior = None
        if prior_type == 'Gamma':
            self._prior_gamma()
        elif prior_type == 'Weibull':
            self._prior_weibull()
        elif prior_type == 'HalfCauchy':
            self._prior_halfcauchy()
        else:
            raise NotImplementedError
        self.fc1_prior['direction'] = ('vMF', {'row_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n1, eps=PRIOR_EPSILON)),
                                               'col_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n1, eps=PRIOR_EPSILON))})
        self.fc2_prior['direction'] = ('vMF', {'row_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n2, eps=PRIOR_EPSILON)),
                                               'col_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n1, eps=PRIOR_EPSILON))})
        self.fc3_prior['direction'] = ('vMF', {'row_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n3, eps=PRIOR_EPSILON)),
                                               'col_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n1, eps=PRIOR_EPSILON))})

    def __call__(self):
        return self.fc1_prior, self.fc2_prior, self.fc3_prior

    def __repr__(self):
        prior_info_str_list = ['***PRIORS***']
        prior_info_str_list.append('FC1 : ' + prior_summary(self.fc1_prior))
        prior_info_str_list.append('FC2 : ' + prior_summary(self.fc2_prior))
        prior_info_str_list.append('FC3 : ' + prior_summary(self.fc3_prior))
        return '\n'.join(prior_info_str_list)

    def _prior_gamma(self):
        self.fc1_prior = {'radius': ('Gamma', {'softplus_inv_shape': softplus_inv(GAMMA_SHAPE), 'softplus_inv_rate': softplus_inv(GAMMA_RATE)})}
        self.fc2_prior = {'radius': ('Gamma', {'softplus_inv_shape': softplus_inv(GAMMA_SHAPE), 'softplus_inv_rate': softplus_inv(GAMMA_RATE)})}
        self.fc3_prior = {'radius': ('Gamma', {'softplus_inv_shape': softplus_inv(GAMMA_SHAPE), 'softplus_inv_rate': softplus_inv(GAMMA_RATE)})}

    def _prior_weibull(self):
        self.fc1_prior = {'radius': ('Weibull', {'softplus_inv_shape': softplus_inv(WEIBULL_SHAPE), 'softplus_inv_scale': softplus_inv(WEIBULL_SCALE)})}
        self.fc2_prior = {'radius': ('Weibull', {'softplus_inv_shape': softplus_inv(WEIBULL_SHAPE), 'softplus_inv_scale': softplus_inv(WEIBULL_SCALE)})}
        self.fc3_prior = {'radius': ('Weibull', {'softplus_inv_shape': softplus_inv(WEIBULL_SHAPE), 'softplus_inv_scale': softplus_inv(WEIBULL_SCALE)})}

    def _prior_halfcauchy(self):
        self.fc1_prior = {'radius': ('HalfCauchy', {'softplus_inv_shape': softplus_inv(0.5), 'softplus_inv_rate': 2 * softplus_inv(HALFCAUCHY_TAU), 'softplus_inv_shape1': softplus_inv(0.5), 'softplus_inv_rate1': softplus_inv(1)})}
        self.fc2_prior = {'radius': ('HalfCauchy', {'softplus_inv_shape': softplus_inv(0.5), 'softplus_inv_rate': 2 * softplus_inv(HALFCAUCHY_TAU), 'softplus_inv_shape1': softplus_inv(0.5), 'softplus_inv_rate1': softplus_inv(1)})}
        self.fc3_prior = {'radius': ('HalfCauchy', {'softplus_inv_shape': softplus_inv(0.5), 'softplus_inv_rate': 2 * softplus_inv(HALFCAUCHY_TAU), 'softplus_inv_shape1': softplus_inv(0.5), 'softplus_inv_rate1': softplus_inv(1)})}





if __name__ == '__main__':
    model = MNISTDOUBLEFC3(prior_type='HalfCauchy')
    model.reset_parameters()
    n_batch = 32
    input_data = torch.randn(n_batch, 784)
    output_data = torch.empty(n_batch, dtype=torch.long).random_(10)
    pred = model(input_data)
    loss_module = nn.CrossEntropyLoss()
    loss = loss_module(pred, output_data)
    loss.backward()
    for name, p in model.named_parameters():
        if torch.isinf(p.grad.data).any():
            print('Infinity in grad of %s' % name)
        elif (p.grad.data != p.grad.data).any():
            print('Nan in grad of %s' % name)
        else:
            print('%s : %.4E ~ %.4E' % (name, float(p.grad.data.min()), float(p.grad.data.max())))
    kld = model.kl_divergence()
    print(pred)
    print(kld)
