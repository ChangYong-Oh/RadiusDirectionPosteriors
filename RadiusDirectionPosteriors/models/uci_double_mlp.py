import torch
import torch.nn as nn
from torch.nn.functional import softplus

from BayesianNeuralNetwork.models.model import BayesianModel
from BayesianNeuralNetwork.models.utils import double_prior_summary
from BayesianNeuralNetwork.torch_user.nn.bayesian_modules.double_radial_linear import DoubleRadialLinear
from BayesianNeuralNetwork.torch_user.kl_divergence import KL_Gamma
from BayesianNeuralNetwork.torch_user.nn.utils import ml_kappa, softplus_inv


PRIOR_EPSILON = 0.5
GAMMA_SHAPE = 1.0
GAMMA_RATE = 1.0
WEIBULL_SHAPE = 1.0
WEIBULL_SCALE = 1.0
HALFCAUCHY_TAU = 1.0
NOISE_GAMMA_PRIOR_SHAPE = 1.0
NOISE_GAMMA_PRIOR_RATE = 1.0
NOISE_GAMMA_PRIOR_SHAPE_INIT = 1.0
NOISE_GAMMA_PRIOR_RATE_INIT = 1.0


class UCIFCRadial(BayesianModel):
    def __init__(self, prior_type, n_in, n_hidden):
        super(UCIFCRadial, self).__init__()
        self.prior = UCIFCRadialPrior(prior_type, n_in, n_hidden)
        fc1_prior, fc2_prior = self.prior()
        self.obs_precision_softplus_inv_shape = nn.Parameter(torch.Tensor([softplus_inv(NOISE_GAMMA_PRIOR_SHAPE_INIT)]))
        self.obs_precision_softplus_inv_rate = nn.Parameter(torch.Tensor([softplus_inv(NOISE_GAMMA_PRIOR_RATE_INIT)]))

        self.fc1 = DoubleRadialLinear(in_features=n_in, out_features=n_hidden, bias=True, prior=fc1_prior)
        self.nonlinear1 = nn.ReLU()
        self.fc2 = DoubleRadialLinear(in_features=n_hidden, out_features=1, bias=True, prior=fc2_prior)

    def forward(self, input):
        x = self.nonlinear1(self.fc1(input))
        x = self.fc2(x)
        return x

    def kl_divergence(self):
        q_shape = softplus(self.obs_precision_softplus_inv_shape)
        q_rate = softplus(self.obs_precision_softplus_inv_rate)
        p_shape = torch.ones_like(self.obs_precision_softplus_inv_shape) * NOISE_GAMMA_PRIOR_SHAPE
        p_rate = torch.ones_like(self.obs_precision_softplus_inv_rate) * NOISE_GAMMA_PRIOR_RATE
        kld = KL_Gamma(q_shape, q_rate, p_shape, p_rate)
        kld += self.fc1.kl_divergence()
        kld += self.fc2.kl_divergence()
        return kld

    def init_hyperparam_value(self):
        hyperparam_info_list = ['Observation Noise Variance ~ Gamma(%.2E, %.2E)' % (float(softplus(self.obs_precision_softplus_inv_shape)), float(softplus(self.obs_precision_softplus_inv_rate)))]
        for name, m in self.named_modules():
            if hasattr(m, 'init_hyperparams_repr'):
                hyperparam_info_list.append('%s(%s) : %s' % (name, m._get_name(), m.init_hyperparams_repr()))
        return '\n'.join(hyperparam_info_list)


class UCIFCRadialPrior(object):

    def __init__(self, prior_type, n1, n2):
        self.fc1_prior = None
        self.fc2_prior = None
        if prior_type == 'Gamma':
            self._prior_gamma()
        elif prior_type == 'Weibull':
            self._prior_weibull()
        elif prior_type == 'HalfCauchy':
            self._prior_halfcauchy()
        else:
            raise NotImplementedError
        self.fc1_prior['direction'] = ('vMF', {'row_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n1, eps=PRIOR_EPSILON)),
                                               'col_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n2, eps=PRIOR_EPSILON))})
        self.fc2_prior['direction'] = ('vMF', {'row_softplus_inv_concentration': softplus_inv(ml_kappa(dim=n2, eps=PRIOR_EPSILON)),
                                               'col_softplus_inv_concentration': softplus_inv(ml_kappa(dim=1, eps=PRIOR_EPSILON))})

    def __call__(self):
        return self.fc1_prior, self.fc2_prior

    def __repr__(self):
        prior_info_str_list = ['***PRIORS***']
        prior_info_str_list.append('FC1 : ' + double_prior_summary(self.fc1_prior))
        prior_info_str_list.append('FC2 : ' + double_prior_summary(self.fc2_prior))
        return '\n'.join(prior_info_str_list)

    def _prior_gamma(self):
        self.fc1_prior = {'radius': ('Gamma', {'softplus_inv_shape': softplus_inv(GAMMA_SHAPE), 'softplus_inv_rate': softplus_inv(GAMMA_RATE)})}
        self.fc2_prior = {'radius': ('Gamma', {'softplus_inv_shape': softplus_inv(GAMMA_SHAPE), 'softplus_inv_rate': softplus_inv(GAMMA_RATE)})}

    def _prior_weibull(self):
        self.fc1_prior = {'radius': ('Weibull', {'softplus_inv_shape': softplus_inv(WEIBULL_SHAPE), 'softplus_inv_scale': softplus_inv(WEIBULL_SCALE)})}
        self.fc2_prior = {'radius': ('Weibull', {'softplus_inv_shape': softplus_inv(WEIBULL_SHAPE), 'softplus_inv_scale': softplus_inv(WEIBULL_SCALE)})}

    def _prior_halfcauchy(self):
        self.fc1_prior = {'radius': ('HalfCauchy', {'softplus_inv_shape': softplus_inv(0.5), 'softplus_inv_rate': 2 * softplus_inv(HALFCAUCHY_TAU), 'softplus_inv_shape1': softplus_inv(0.5), 'softplus_inv_rate1': softplus_inv(1)})}
        self.fc2_prior = {'radius': ('HalfCauchy', {'softplus_inv_shape': softplus_inv(0.5), 'softplus_inv_rate': 2 * softplus_inv(HALFCAUCHY_TAU), 'softplus_inv_shape1': softplus_inv(0.5), 'softplus_inv_rate1': softplus_inv(1)})}


if __name__ == '__main__':
    model = UCIFCRadial(prior_type='HalfCauchy')
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
