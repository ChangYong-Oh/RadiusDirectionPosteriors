from BayesianNeuralNetwork.models.utils import double_prior_summary, prior_summary
from BayesianNeuralNetwork.torch_user.nn.utils import ml_kappa


PRIOR_EPSILON = 0.1


class LeNet5DoublePrior(object):

    def __init__(self, prior_info):
        prior_type, prior_hyper = prior_info
        self.conv1_prior = None
        self.conv2_prior = None
        self.fc1_prior = None
        self.fc2_prior = None
        self._prior_halfcauchy(prior_type, prior_hyper)

        self.conv1_prior['direction'] = ('vMF', {'row_concentration': ml_kappa(dim=1 * 25, eps=PRIOR_EPSILON),
                                                 'col_concentration': ml_kappa(dim=20 * 25, eps=PRIOR_EPSILON)})
        self.conv2_prior['direction'] = ('vMF', {'row_concentration': ml_kappa(dim=20 * 25, eps=PRIOR_EPSILON),
                                                 'col_concentration': ml_kappa(dim=50 * 25, eps=PRIOR_EPSILON)})
        self.fc1_prior['direction'] = ('vMF', {'row_concentration': ml_kappa(dim=800, eps=PRIOR_EPSILON),
                                               'col_concentration': ml_kappa(dim=500, eps=PRIOR_EPSILON)})
        self.fc2_prior['direction'] = ('vMF', {'row_concentration': ml_kappa(dim=500, eps=PRIOR_EPSILON),
                                               'col_concentration': ml_kappa(dim=10, eps=PRIOR_EPSILON)})

    def __call__(self):
        return self.conv1_prior, self.conv2_prior, self.fc1_prior, self.fc2_prior

    def __repr__(self):
        prior_info_str_list = ['***PRIORS***']
        prior_info_str_list.append('Conv1 : ' + double_prior_summary(self.conv1_prior))
        prior_info_str_list.append('Conv2 : ' + double_prior_summary(self.conv2_prior))
        prior_info_str_list.append('FC1 : ' + double_prior_summary(self.fc1_prior))
        prior_info_str_list.append('FC2 : ' + double_prior_summary(self.fc2_prior))
        return '\n'.join(prior_info_str_list)

    def _prior_halfcauchy(self, prior_type, prior_hyper):
        tau_fc_local = prior_hyper['tau_fc_local']
        tau_fc_global = prior_hyper['tau_fc_global']
        tau_conv_local = prior_hyper['tau_conv_local']
        tau_conv_global = prior_hyper['tau_conv_global']
        self.conv1_prior = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.conv2_prior = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.fc1_prior = {'radius': (prior_type, {'tau_global': tau_fc_global, 'tau_local': tau_fc_local})}
        self.fc2_prior = {'radius': (prior_type, {'tau_global': tau_fc_global, 'tau_local': tau_fc_local})}


class LeNet5Prior(object):

    def __init__(self, prior_info):
        prior_type, prior_hyper = prior_info
        self.conv1_prior = None
        self.conv2_prior = None
        self.fc1_prior = None
        self.fc2_prior = None
        self._prior_halfcauchy(prior_type, prior_hyper)

        self.conv1_prior['direction'] = ('vMF', {'concentration': ml_kappa(dim=1 * 25, eps=PRIOR_EPSILON)})
        self.conv2_prior['direction'] = ('vMF', {'concentration': ml_kappa(dim=20 * 25, eps=PRIOR_EPSILON)})
        self.fc1_prior['direction'] = ('vMF', {'concentration': ml_kappa(dim=500, eps=PRIOR_EPSILON)})
        self.fc2_prior['direction'] = ('vMF', {'concentration': ml_kappa(dim=10, eps=PRIOR_EPSILON)})

    def __call__(self):
        return self.conv1_prior, self.conv2_prior, self.fc1_prior, self.fc2_prior

    def __repr__(self):
        prior_info_str_list = ['***PRIORS***']
        prior_info_str_list.append('Conv1 : ' + prior_summary(self.conv1_prior))
        prior_info_str_list.append('Conv2 : ' + prior_summary(self.conv2_prior))
        prior_info_str_list.append('FC1 : ' + prior_summary(self.fc1_prior))
        prior_info_str_list.append('FC2 : ' + prior_summary(self.fc2_prior))
        return '\n'.join(prior_info_str_list)

    def _prior_halfcauchy(self, prior_type, prior_hyper):
        tau_fc_local = prior_hyper['tau_fc_local']
        tau_fc_global = prior_hyper['tau_fc_global']
        tau_conv_local = prior_hyper['tau_conv_local']
        tau_conv_global = prior_hyper['tau_conv_global']
        self.conv1_prior = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.conv2_prior = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.fc1_prior = {'radius': (prior_type, {'tau_global': tau_fc_global, 'tau_local': tau_fc_local})}
        self.fc2_prior = {'radius': (prior_type, {'tau_global': tau_fc_global, 'tau_local': tau_fc_local})}


class LeNetFCDoublePrior(object):

    def __init__(self, prior_info):
        prior_type, prior_hyper = prior_info
        self.fc1_prior = None
        self.fc2_prior = None
        self.fc3_prior = None
        if prior_type == 'Gamma':
            self._prior_gamma(prior_type, prior_hyper)
        elif prior_type == 'Weibull':
            self._prior_weibull(prior_type, prior_hyper)
        elif 'HalfCauchy' in prior_type:
            self._prior_halfcauchy(prior_type, prior_hyper)
        else:
            raise NotImplementedError

        self.fc1_prior['direction'] = ('vMF', {'row_concentration': ml_kappa(dim=784, eps=PRIOR_EPSILON),
                                               'col_concentration': ml_kappa(dim=300, eps=PRIOR_EPSILON)})
        self.fc2_prior['direction'] = ('vMF', {'row_concentration': ml_kappa(dim=300, eps=PRIOR_EPSILON),
                                               'col_concentration': ml_kappa(dim=100, eps=PRIOR_EPSILON)})
        self.fc3_prior['direction'] = ('vMF', {'row_concentration': ml_kappa(dim=100, eps=PRIOR_EPSILON),
                                               'col_concentration': ml_kappa(dim=10, eps=PRIOR_EPSILON)})

    def __call__(self):
        return self.fc1_prior, self.fc2_prior, self.fc3_prior

    def __repr__(self):
        prior_info_str_list = ['***PRIORS***']
        prior_info_str_list.append('FC1 : ' + double_prior_summary(self.fc1_prior))
        prior_info_str_list.append('FC2 : ' + double_prior_summary(self.fc2_prior))
        prior_info_str_list.append('FC3 : ' + double_prior_summary(self.fc3_prior))
        return '\n'.join(prior_info_str_list)

    def _prior_halfcauchy(self, prior_type, prior_hyper):
        tau_fc_local = prior_hyper['tau_fc_local']
        tau_fc_global = prior_hyper['tau_fc_global']
        self.fc1_prior = {'radius': (prior_type, {'tau_global': tau_fc_global, 'tau_fc_local': tau_fc_local})}
        self.fc2_prior = {'radius': (prior_type, {'tau_global': tau_fc_global, 'tau_fc_local': tau_fc_local})}
        self.fc3_prior = {'radius': (prior_type, {'tau_global': tau_fc_global, 'tau_fc_local': tau_fc_local})}


class LeNetFCPrior(object):

    def __init__(self, prior_info):
        prior_type, prior_hyper = prior_info
        self.fc1_prior = None
        self.fc2_prior = None
        self.fc3_prior = None
        if prior_type == 'Gamma':
            self._prior_gamma(prior_type, prior_hyper)
        elif prior_type == 'Weibull':
            self._prior_weibull(prior_type, prior_hyper)
        elif 'HalfCauchy' in prior_type:
            self._prior_halfcauchy(prior_type, prior_hyper)
        else:
            raise NotImplementedError

        self.fc1_prior['direction'] = ('vMF', {'concentration': ml_kappa(dim=300, eps=PRIOR_EPSILON)})
        self.fc2_prior['direction'] = ('vMF', {'concentration': ml_kappa(dim=100, eps=PRIOR_EPSILON)})
        self.fc3_prior['direction'] = ('vMF', {'concentration': ml_kappa(dim=10, eps=PRIOR_EPSILON)})

    def __call__(self):
        return self.fc1_prior, self.fc2_prior, self.fc3_prior

    def __repr__(self):
        prior_info_str_list = ['***PRIORS***']
        prior_info_str_list.append('FC1 : ' + prior_summary(self.fc1_prior))
        prior_info_str_list.append('FC2 : ' + prior_summary(self.fc2_prior))
        prior_info_str_list.append('FC3 : ' + prior_summary(self.fc3_prior))
        return '\n'.join(prior_info_str_list)

    def _prior_halfcauchy(self, prior_type, prior_hyper):
        tau_fc_local = prior_hyper['tau_fc_local']
        tau_fc_global = prior_hyper['tau_fc_global']
        self.fc1_prior = {'radius': (prior_type, {'tau_global': tau_fc_global, 'tau_fc_local': tau_fc_local})}
        self.fc2_prior = {'radius': (prior_type, {'tau_global': tau_fc_global, 'tau_fc_local': tau_fc_local})}
        self.fc3_prior = {'radius': (prior_type, {'tau_global': tau_fc_global, 'tau_fc_local': tau_fc_local})}