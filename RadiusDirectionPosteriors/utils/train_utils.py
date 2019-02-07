import os
import socket
import time
from datetime import datetime

import numpy as np

import torch
import torch.cuda

from BayesianNeuralNetwork.utils.dir_utils import EXP_DIR

EXP_INFO = ['data_type', 'gan_type', 'n_latent', 'in_h', 'in_w', 'n_ch_img', 'n_ch_base', 'max_gen_update', 'n_dis', 'lr', 'beta2', 'beta1', 'lambda_reg', 'spectrum_deviance']
_STR_FORMAT_DICT = {int: '%d', float: '%8.6f', str: '%s'}
if socket.gethostname() == 'hekla':
    LOGGING_DIR = '/is/ei/com/Experiments/BayesianNeuralNetwork/Initialization'
else:
    LOGGING_DIR = '/home/com/Experiments/BayesianNeuralNetwork/Initialization'


def model_init(model, init_type):
    if init_type == 'simple':
        model_init_simple(model)
    elif init_type == 'xavier':
        model_init_xavier(model)
    elif init_type == 'kaiming':
        model_init_kaiming(model)
    elif init_type == 'orthogonal':
        model_init_orthogonal(model)
    elif init_type == 'trace':
        model_init_trace(model)
    elif init_type == 'mixed':
        model_init_mixed(model)
    else:
        raise NotImplementedError


def model_init_simple(model):
    for m in model.modules():
        m_type = m._get_name()
        if 'Linear' in m_type:
            torch.nn.init.normal_(m.weight, std=0.05)
            torch.nn.init.constant_(m.bias, val=0)
        if 'Conv' in m_type:
            torch.nn.init.normal_(m.weight, std=0.05)
            torch.nn.init.constant_(m.bias, val=0)
        elif 'Norm' in m_type:
            torch.nn.init.constant_(m.weight, val=1)
            torch.nn.init.constant_(m.bias, val=0)


def model_init_xavier(model):
    for m in model.modules():
        m_type = m._get_name()
        if 'Linear' in m_type:
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.constant_(m.bias, val=0)
        if 'Conv' in m_type:
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.constant_(m.bias, val=0)
        elif 'Norm' in m_type:
            torch.nn.init.constant_(m.weight, val=1)
            torch.nn.init.constant_(m.bias, val=0)


def model_init_kaiming(model):
    for m in model.modules():
        m_type = m._get_name()
        if 'Linear' in m_type:
            if hasattr(model, 'lReLU_slope'):
                a = model.lReLU_slope
            else:
                a = 0
            torch.nn.init.kaiming_normal_(m.weight, a=a)
            torch.nn.init.constant_(m.bias, val=0)
        if 'Conv' in m_type:
            if model._get_name() == 'VGG':
                a = 0.0
            elif hasattr(model, 'lReLU_slope'):
                a = model.lReLU_slope
            torch.nn.init.kaiming_normal_(m.weight, a=a)
            torch.nn.init.constant_(m.bias, val=0)
        elif 'Norm' in m_type:
            torch.nn.init.constant_(m.weight, val=1)
            torch.nn.init.constant_(m.bias, val=0)


def model_init_orthogonal(model):
    for m in model.modules():
        m_type = m._get_name()
        if 'Linear' in m_type:
            torch.nn.init.orthogonal_(m.weight, gain=1.0/0.0012755 ** 0.5)
            torch.nn.init.constant_(m.bias, val=0)
        if 'Conv' in m_type:
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.constant_(m.bias, val=0)
        elif 'Norm' in m_type:
            torch.nn.init.constant_(m.weight, val=1)
            torch.nn.init.constant_(m.bias, val=0)


def model_init_trace(model):
    for m in model.modules():
        m_type = m._get_name()
        if 'Linear' in m_type:
            torch.nn.init.normal_(m.weight, mean=0, std=1.0/max(m.weight.size()) ** 0.5)
            torch.nn.init.constant_(m.bias, val=0)
        if 'Conv' in m_type:
            std_factor = m.out_channels if m.out_channels > m.in_channels * np.prod(m.stride) else m.in_channels
            torch.nn.init.normal_(m.weight, mean=0, std=(1.0/(std_factor * np.prod(m.kernel_size))) ** 0.5)
            torch.nn.init.constant_(m.bias, val=0)
        elif 'Norm' in m_type:
            torch.nn.init.constant_(m.weight, val=1)
            torch.nn.init.constant_(m.bias, val=0)


def model_init_mixed(model):
    for m in model.modules():
        m_type = m._get_name()
        if 'Linear' in m_type:
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.constant_(m.bias, val=0)
        if 'Conv' in m_type:
            std_factor = m.out_channels if m.out_channels > m.in_channels * np.prod(m.stride) else m.in_channels
            torch.nn.init.normal_(m.weight, mean=0, std=(1.0/(std_factor * np.prod(m.kernel_size))) ** 0.5)
            torch.nn.init.constant_(m.bias, val=0)
        elif 'Norm' in m_type:
            torch.nn.init.constant_(m.weight, val=1)
            torch.nn.init.constant_(m.bias, val=0)


def dataloader_and_info(data_type, batch_size, num_workers):
    if data_type == 'CIFAR10':
        from BayesianNeuralNetwork.data_loaders.cifar10 import data_loader
        n_ch_img, in_h, in_w, n_category = 3, 32, 32, 10
    elif data_type == 'CIFAR100':
        from BayesianNeuralNetwork.data_loaders.cifar100 import data_loader
        n_ch_img, in_h, in_w, n_category = 3, 32, 32, 100
    elif data_type == 'STL10':
        from BayesianNeuralNetwork.data_loaders.stl10 import data_loader
        n_ch_img, in_h, in_w, n_category = 3, 32, 32, 10 # Resized, originally 96, 96
    elif data_type == 'MNIST':
        from BayesianNeuralNetwork.data_loaders.mnist import data_loader
        n_ch_img, in_h, in_w, n_category = 1, 28, 28, 10
    elif data_type == 'FashionMNIST':
        from BayesianNeuralNetwork.data_loaders.fashionmnist import data_loader
        n_ch_img, in_h, in_w, n_category = 1, 28, 28, 10
    else:
        raise NotImplementedError
    train_loader, test_loader = data_loader(batch_size, num_workers)
    return train_loader, test_loader, n_ch_img, in_h, in_w, n_category


def log_file_naming(method, data_type, architecture, init_type, lr, beta1, beta2):
    filename = '%s_%s_(lr:%.1E_beta1:%6.4f_beta2:%6.4f)_%s_%s_%s.txt' \
               % (data_type, architecture, lr, beta1, beta2, method, init_type, datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f"))
    fullpath = os.path.join(LOGGING_DIR, filename)
    print('Logfile is at ' + fullpath)
    return fullpath


def train_log(experiment_info):
    training_begin_time = datetime.now()
    training_begin_time_str = training_begin_time.strftime("%Y-%m-%d_%H:%M:%S:%f")

    experiment_info = experiment_info.copy()
    experiment_info_str = '\n'.join([(elm + ' : ' + _STR_FORMAT_DICT[type(experiment_info[elm])]) % experiment_info[elm] for elm in EXP_INFO if elm in experiment_info.keys()])

    exp_path = os.path.join(EXP_DIR, experiment_info['data_type'])
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    file_exp_info = open(os.path.join(exp_path, training_begin_time_str + '.txt'), 'wt')
    file_exp_info.write(experiment_info_str)
    file_exp_info.close()

    trained_model_filename = os.path.join(exp_path, training_begin_time_str)

    return trained_model_filename


def train_progress(step, max_step, prev_time=None, time_spent=None):
    curr_time = time.time()
    n_digit = str(len(str(max_step)))
    progress_str = ('%' + n_digit + 'd / %' + n_digit + 'd %s ') % (step, max_step, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(curr_time)))
    if prev_time is not None:
        progress_str += '(%s) ' % time.strftime("%H:%M:%S", time.gmtime(curr_time - prev_time))
        time_spent = (2 * (curr_time - prev_time) if time_spent is None else time_spent + (curr_time - prev_time))
        progress_str += 'Estimated remaining : %s' % dhms(time_spent * float(max_step - step) / float(step))
    print(progress_str)
    return curr_time, time_spent


def dhms(milliseconds):
    remainder = int(milliseconds)
    days = remainder // 86400
    remainder = remainder % 86400
    hours = remainder // 3600
    remainder = remainder % 3600
    minutes = remainder // 60
    seconds = remainder % 60
    return '%d-%02d:%02d:%02d' % (days, hours, minutes, seconds)


def train_machine_info():
    machine_name = socket.gethostname()
    machine_ip = socket.gethostbyname(machine_name)
    n_gpu = 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    return '*** Models are trained on the machine (NAME : %s (%d)GPU @ %s) ***' % (machine_name, n_gpu, machine_ip)
