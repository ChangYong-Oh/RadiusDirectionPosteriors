import torch
import torch.nn as nn

from BayesianNeuralNetwork.models.model import BayesianModel
from BayesianNeuralNetwork.torch_user.nn.bayesian_modules.double_radial_linear import DoubleRadialLinear
from BayesianNeuralNetwork.models.lenet_prior import LeNetFCDoublePrior


class LeNetFCDouble(BayesianModel):
    def __init__(self, prior_info):
        super(LeNetFCDouble, self).__init__()
        self.prior = LeNetFCDoublePrior(prior_info)
        fc1_prior, fc2_prior, fc3_prior = self.prior()

        self.fc1 = DoubleRadialLinear(in_features=784, out_features=300, bias=True, prior=fc1_prior, with_global=True)
        self.nonlinear1 = nn.Sigmoid()
        self.fc2 = DoubleRadialLinear(in_features=300, out_features=100, bias=True, prior=fc2_prior, with_global=True)
        self.nonlinear2 = nn.Sigmoid()
        self.fc3 = DoubleRadialLinear(in_features=100, out_features=10, bias=True, prior=fc3_prior, with_global=True)

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


if __name__ == '__main__':
    model = LeNetFCDouble(prior_info='HalfCauchy')
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
