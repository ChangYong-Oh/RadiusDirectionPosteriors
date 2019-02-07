import torch
import torch.nn as nn

from BayesianNeuralNetwork.models.model import BayesianModel
from BayesianNeuralNetwork.torch_user.nn.bayesian_modules.radial_conv import RadialConv2d
from BayesianNeuralNetwork.torch_user.nn.bayesian_modules.radial_linear import RadialLinear
from BayesianNeuralNetwork.models.lenet_prior import LeNet5Prior


class LeNet5(BayesianModel):
    def __init__(self, prior_info):
        super(LeNet5, self).__init__()
        self.prior = LeNet5Prior(prior_info)
        conv1_prior, conv2_prior, fc1_prior, fc2_prior = self.prior()

        self.conv1 = RadialConv2d(in_channels=1, out_channels=20, kernel_size=5, bias=True, prior=conv1_prior, with_global=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = RadialConv2d(in_channels=20, out_channels=50, kernel_size=5, bias=True, prior=conv2_prior, with_global=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = RadialLinear(in_features=800, out_features=500, bias=True, prior=fc1_prior, with_global=True)
        self.fc1_nonlinear = nn.ReLU()
        self.fc2 = RadialLinear(in_features=500, out_features=10, bias=True, prior=fc2_prior, with_global=True)

    def forward(self, input):
        x = input
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1_nonlinear(self.fc1(x))
        x = self.fc2(x)
        return x

    def kl_divergence(self):
        kld = self.conv1.kl_divergence()
        kld += self.conv2.kl_divergence()
        kld += self.fc1.kl_divergence()
        kld += self.fc2.kl_divergence()
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
    model = LeNet5(prior_info='HalfCauchy')
    model.reset_parameters()
    n_batch = 32
    input_data = torch.randn(n_batch, 1, 28, 28)
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
