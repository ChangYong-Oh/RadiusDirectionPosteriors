import os
import sys
import math
import socket
import dill
import pickle
import argparse
from datetime import datetime

import torch
import torch.optim as optim
import torch.cuda as cuda
from torch.nn.functional import softplus

from BayesianNeuralNetwork.data_loaders.uci_regression import data_loader
from BayesianNeuralNetwork.models.uci_double_mlp import UCIFCRadial
from BayesianNeuralNetwork.data_loaders.uci_regression import architecture_info

from BayesianNeuralNetwork.torch_user.nn.utils import softplus_inv


HOSTNAME = socket.gethostname()
if HOSTNAME == 'hekla':
    SAVE_DIR = '/is/ei/com/Experiments/BayesianNeuralNetwork'
else:  # Any other machine in MPI Tuebingen
    SAVE_DIR = '/home/com/Experiments/BayesianNeuralNetwork'
assert os.path.exists(SAVE_DIR)
UCI_DATA = ['BOSTON', 'CONCRETE', 'ENERGY', 'KIN8NM', 'NAVAL', 'POWERPLANT', 'PROTEIN', 'WINE', 'YACHT', 'YEAR']
MODEL_FILENAME = lambda filename_prefix: os.path.join(SAVE_DIR, filename_prefix + '.pkl')
OPTIM_FILENAME = lambda filename_prefix: os.path.join(SAVE_DIR, filename_prefix + '_optim.pkl')
EXP_INFO_FILENAME = lambda filename_prefix: os.path.join(SAVE_DIR, filename_prefix + '_exp_info.pkl')
LOG_FILENAME = lambda filename_prefix: os.path.join(SAVE_DIR, filename_prefix + '.log')


def train_initiate(prior_type, data_type, split_id, output_normalize, n_pred_samples,
                   n_epoch, lr, batch_size=32, num_workers=4, use_gpu=True):
    exp_info_dict = {'prior_type': prior_type, 'data_type': data_type,
                     'data_id': split_id, 'output_normalize': output_normalize,
                     'n_epoch': n_epoch, 'lr': lr, 'batch_size': batch_size}
    time_tag = datetime.now().strftime("%H:%M:%S:%f")
    exp_filename_prefix = '_'.join(['Radial-double', data_type, prior_type, 'E' + str(n_epoch).zfill(4), str(split_id).zfill(2), time_tag])
    use_gpu = use_gpu and cuda.is_available()
    print(exp_filename_prefix)

    train_loader, test_loader, train_loader_eval, normalization_info = data_loader(data_type, split_id=split_id,
                                                                                   batch_size=batch_size, num_workers=num_workers,
                                                                                   output_normalize=output_normalize)
    model = load_model(prior_type, data_type, use_gpu)
    initialization_hyperparams = {'vMF': {'direction': 'kaiming',
                                          'log_concentration_normal_mean_via_epsilon': 0.05,
                                          'log_concentration_normal_std': 0.01},
                                  'LogNormal': {'mu_normal_mean': math.log(1.0),
                                                'mu_normal_std': 0.0001,
                                                'softplus_inv_var_normal_mean': softplus_inv(1e-4),
                                                'softplus_inv_var_normal_std': 0.0001},
                                  'Gamma': {'softplus_inv_shape_normal_mean': softplus_inv(2.0 ** 0.5),
                                            'softplus_inv_shape_normal_std': 0.1,
                                            'softplus_inv_rate_normal_mean': softplus_inv(1.0),
                                            'softplus_inv_rate_normal_std': 0.1}
                                  }
    model.reset_parameters(initialization_hyperparams)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    annealing_steps = float(50 * math.ceil(len(train_loader.dataset) / batch_size))
    beta_func = lambda s: min(s, annealing_steps) / (annealing_steps)

    train_log, n_steps = train(model=model, optimizer=optimizer, train_loader=train_loader,
                               begin_step=0, epoch_begin=0, epoch_end=n_epoch,
                               beta_func=beta_func, use_gpu=use_gpu)
    eval_log = evaluate(model=model, train_loader_eval=train_loader_eval, test_loader=test_loader,
                        normalization_info=normalization_info, n_pred_samples=n_pred_samples)
    exp_info_dict['n_steps'] = n_steps
    exp_info_dict['beta_func'] = beta_func
    exp_filename = save_experiment(model=model, optimizer=optimizer, log_text='\n'.join([exp_filename_prefix, train_log, eval_log]),
                                   exp_info_dict=exp_info_dict, filename_prefix=exp_filename_prefix)
    return exp_filename


def train_continue(model_filename, n_epoch, n_pred_samples, num_workers=4, use_gpu=False):
    dirname, filename = os.path.split(model_filename)
    filename_prefix = filename.split('.')[0]
    exp_info_file = open(os.path.join(SAVE_DIR, EXP_INFO_FILENAME(filename_prefix)), 'rb')
    exp_info_dict = pickle.load(exp_info_file)
    exp_info_file.close()
    time_tag = datetime.now().strftime("%H:%M:%S:%f")
    exp_filename_prefix = '_'.join(filename_prefix.split('_')[:-2] + ['E' + str(exp_info_dict['n_epoch']+n_epoch).zfill(4), time_tag])

    model = load_model(prior_type=exp_info_dict['prior_type'], data_type=exp_info_dict['data_type'], use_gpu=use_gpu)
    optimizer = optim.Adam(model.parameters(), lr=exp_info_dict['lr'])
    model.load_state_dict(torch.load(model_filename))
    optimizer.load_state_dict(torch.load(OPTIM_FILENAME(filename_prefix)))
    train_loader, test_loader, train_loader_eval, normalization_info = data_loader(exp_info_dict['data_type'], split_id=exp_info_dict['data_id'],
                                                                                   batch_size=exp_info_dict['batch_size'], num_workers=num_workers, output_normalize=exp_info_dict['output_normalize'])
    prev_n_epoch = exp_info_dict['n_epoch']
    train_log, n_steps = train(model=model, optimizer=optimizer, train_loader=train_loader,
                               begin_step=exp_info_dict['n_steps'], epoch_begin=prev_n_epoch, epoch_end=prev_n_epoch+n_epoch,
                               beta_func=exp_info_dict['beta_func'], use_gpu=use_gpu)

    eval_log = evaluate(model=model, train_loader_eval=train_loader_eval, test_loader=test_loader,
                        normalization_info=normalization_info, n_pred_samples=n_pred_samples)
    exp_info_dict['n_epoch'] += n_epoch
    exp_info_dict['n_steps'] += n_steps
    exp_filename = save_experiment(model=model, optimizer=optimizer, log_text='\n'.join([exp_filename_prefix, train_log, eval_log]),
                                   exp_info_dict=exp_info_dict, filename_prefix=exp_filename_prefix)
    return exp_filename


def train(model, optimizer, train_loader, begin_step, epoch_begin, epoch_end, beta_func, use_gpu=False):
    assert 'Random' in train_loader.sampler.__class__.__name__
    train_info = 'epoch:%d ~ %d' % (epoch_begin, epoch_end)
    model_hyperparam_info = model.init_hyperparam_value()
    model_prior_info = model.prior.__repr__()
    print(train_info)
    print(model_prior_info)
    print(model_hyperparam_info)
    logging = train_info + '\n' + model_prior_info + '\n' + model_hyperparam_info + '\n'

    n_data = len(train_loader.sampler)
    n_step = begin_step
    running_reconstruction = 0.0
    running_kld = 0.0
    running_loss = 0.0
    for e in range(epoch_begin, epoch_end):
        for data in train_loader:
            # get the inputs
            inputs, outputs = data
            if use_gpu:
                inputs = inputs.cuda()
                outputs = outputs.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + gradient_correction + optimize
            # TODO : annealing kl-divergence term is better
            pred = model(inputs)
            beta = beta_func(n_step)
            kld = model.kl_divergence() / float(n_data)
            if torch.isinf(kld):
                raise RuntimeError("KL divergence is infinite. It is likely that ive is zero and is passed to log.")
            # kld = 0
            reconstruction = -gaussian_log_likelihood(y=outputs, mu=pred,
                                                      precision_noise_shape=softplus(model.obs_precision_softplus_inv_shape),
                                                      precision_noise_rate=softplus(model.obs_precision_softplus_inv_rate)).mean()
            loss = reconstruction + kld * beta
            loss.backward()
            for m in model.modules():
                if hasattr(m, 'gradient_correction'):
                    m.gradient_correction(reconstruction)
            optimizer.step()
            for m in model.modules():
                if hasattr(m, 'parameter_adjustment'):
                    m.parameter_adjustment()
            n_step += 1

            # print statistics
            running_reconstruction += float(reconstruction)
            running_kld += float(kld)
            running_loss += float(loss)
            display_steps = 1000
            if n_step % display_steps == display_steps - 1:
                training_progress_str = '%s [%6d steps in (%4d epochs) ] loss: %.4E, reconstruction: %.4E, regularizer: %.4E, beta:%.3E, %s' % \
                                        (datetime.now().strftime("%H:%M:%S.%f"), n_step + 1, e + 1,
                                         running_loss / display_steps, running_reconstruction / display_steps,
                                         running_kld / display_steps, beta, train_info)
                print(training_progress_str)
                logging += training_progress_str + '\n'
                running_reconstruction = 0.0
                running_kld = 0.0
                running_loss = 0.0
    logging += train_info + '\n' + model_prior_info + '\n' + model_hyperparam_info
    return logging, n_step


def evaluate(model, train_loader_eval, test_loader, normalization_info, n_pred_samples):
    train_outputs, train_pred_mean, train_pred_var, train_pred_samples = sample_prediction(model, train_loader_eval,normalization_info, n_pred_samples)
    train_rmse = torch.mean((train_outputs - train_pred_mean) ** 2) ** 0.5
    train_ll = log_likelihood(outputs=train_outputs, pred_samples=train_pred_samples,
                              precision_noise_shape=softplus(model.obs_precision_softplus_inv_shape),
                              precision_noise_rate=softplus(model.obs_precision_softplus_inv_rate))
    test_outputs, test_pred_mean, test_pred_var, test_pred_samples = sample_prediction(model, test_loader, normalization_info, n_pred_samples)
    test_rmse = torch.mean((test_outputs - test_pred_mean) ** 2) ** 0.5
    test_ll = log_likelihood(outputs=test_outputs, pred_samples=test_pred_samples,
                             precision_noise_shape=softplus(model.obs_precision_softplus_inv_shape),
                             precision_noise_rate=softplus(model.obs_precision_softplus_inv_rate))
    logging = '\n------------------'
    logging += '\n' + model.init_hyperparam_value()
    logging += '\n' + model.prior.__repr__()
    logging += ('\nTrain output mean prediction %6.4f' % float( torch.mean((test_outputs - torch.mean(train_outputs)) ** 2) ** 0.5 * normalization_info['output std']))
    logging += ('\nTrain(%d samples) RMSE : %6.4f / LL : %+6.4f' % (n_pred_samples, train_rmse, train_ll))
    logging += ('\nTest (%d samples) RMSE : %6.4f / LL : %+6.4f' % (n_pred_samples, test_rmse, test_ll))
    print(logging)
    return logging


def save_experiment(model, optimizer, log_text, exp_info_dict, filename_prefix):
    torch.save(model.state_dict(), MODEL_FILENAME(filename_prefix))
    torch.save(optimizer.state_dict(), OPTIM_FILENAME(filename_prefix))
    exp_info_file = open(os.path.join(SAVE_DIR, EXP_INFO_FILENAME(filename_prefix)), 'wb')
    pickle.dump(exp_info_dict, exp_info_file, pickle.HIGHEST_PROTOCOL)
    exp_info_file.close()
    log_file = open(LOG_FILENAME(filename_prefix), 'wt')
    log_file.write(log_text)
    log_file.close()
    return os.path.join(SAVE_DIR, filename_prefix + '.pkl')


def load_trained_model(model_filename):
    dirname, filename = os.path.split(model_filename)
    data_type, data_index, prior_type = filename.split('_')[:3]
    data_index = int(data_index)
    model = load_model(prior_type, data_type, False)
    model.load_state_dict(torch.load(model_filename))
    return model


def load_model(prior_type, data_type, use_gpu):
    n_in, n_hidden = architecture_info(data_type)
    model = UCIFCRadial(prior_type=prior_type, n_in=n_in, n_hidden=n_hidden)
    if use_gpu:
        model.cuda()
    return model


def sample_prediction(model, data_loader, normalization_info, n_pred_samples):
    assert 'Random' not in data_loader.sampler.__class__.__name__
    unnormalized_pred_sum = 0
    unnormalized_pred_ssq = 0
    output_mean = normalization_info['output mean']
    output_std = normalization_info['output std']
    output_data = torch.empty((0,))
    for _, output_batch in data_loader:
        output_data = torch.cat([output_data, output_batch])
    print('Data Size : %d' % output_data.size(0))
    unnormalized_outputs = output_data * output_std + output_mean
    unnormalized_pred_samples = unnormalized_outputs.new_zeros(unnormalized_outputs.size()[:1] + torch.Size([n_pred_samples]))
    for i in range(n_pred_samples):
        pred = torch.empty((0,))
        for inputs, outputs in data_loader:
            pred = torch.cat([pred, model(inputs).detach()])
        unnormalized_pred = pred * output_std + output_mean
        unnormalized_pred_samples[:, i:i + 1] = unnormalized_pred
        unnormalized_pred_sum += unnormalized_pred
        unnormalized_pred_ssq += unnormalized_pred ** 2
    unnormalized_pred_mean = unnormalized_pred_sum / float(n_pred_samples)
    unnormalized_pred_var = unnormalized_pred_ssq / float(n_pred_samples) - unnormalized_pred_mean ** 2
    return unnormalized_outputs, unnormalized_pred_mean, unnormalized_pred_var, unnormalized_pred_samples


# TODO : log likelihood check
def gaussian_log_likelihood(y, mu, precision_noise_shape, precision_noise_rate):
    etau = precision_noise_shape / precision_noise_rate
    elogtau = torch.digamma(precision_noise_shape) - torch.log(precision_noise_rate)
    return 0.5 * elogtau - 0.5 * math.log(2 * math.pi) - (0.5 * etau * (y - mu) ** 2)


def log_likelihood(outputs, pred_samples, precision_noise_shape, precision_noise_rate):
    """
    :param outputs: (n_data by 1) tensor
    :param pred_samples: (n_data by n_pred_samples) tensor
    :param precision_noise_shape_log: 
    :param precision_noise_rate_log:
    :return:
    """
    etau = precision_noise_shape / precision_noise_rate
    elogtau = torch.digamma(precision_noise_shape) - torch.log(precision_noise_rate)
    return torch.mean(torch.log(torch.exp(-0.5 * etau * (outputs - pred_samples) ** 2).sum(dim=0)) - math.log(pred_samples.size(-1)) - 0.5 * math.log(2 * math.pi) + 0.5 * elogtau)


if __name__ == '__main__':
    # 'BOSTON', 'CONCRETE', 'ENERGY', 'KIN8NM', 'NAVAL', 'POWERPLANT', 'PROTEIN', 'WINE', 'YACHT', 'YEAR'

    if HOSTNAME == 'hekla' and len(sys.argv) == 1:
        for ind in [0]:
            exp_filename = train_initiate(prior_type='HalfCauchy', data_type='PROTEIN',
                                          split_id=ind, output_normalize=True, n_pred_samples=100,
                                          n_epoch=20, lr=0.001, batch_size=25, num_workers=1, use_gpu=False)
            print(exp_filename)
        model_filename = ''
        train_continue(model_filename, n_epoch=100, n_pred_samples=100, num_workers=4, use_gpu=False)
        # model_file = '/is/ei/com/Experiments/BayesianNeuralNetwork/BOSTON_09_UCIFCRadial_HalfCauchy_17:56:01:673425.pkl'
        # model = load_trained_model(model_filename=model_file)
        exit(0)

    parser = argparse.ArgumentParser(description='UCI Train script')
    parser.add_argument('--data', dest='data', type=str, help='\n'.join(['BOSTON', 'CONCRETE', 'ENERGY', 'KIN8NM', 'NAVAL', 'POWERPLANT', 'PROTEIN', 'WINE', 'YACHT', 'YEAR']))
    parser.add_argument('--prior', dest='prior', type=str, help='\n'.join(['HalfCauchy', 'Gamma']))
    parser.add_argument('--index', dest='index', type=int, help='0~19')
    parser.add_argument('--all_index', dest='all_index', type=int, help='Run for all indices')
    parser.add_argument('--epochs', dest='epochs', type=int, default=400)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=5)

    args = parser.parse_args()

    indices = range(args.all_index, 20) if args.all_index is not None else range(args.index, args.index+1)
    for ind in indices:
        exp_filename = train_initiate(prior_type=args.prior, data_type=args.data,
                                      split_id=ind, output_normalize=True, n_pred_samples=100,
                                      n_epoch=args.epochs, lr=0.001, batch_size=args.batch_size, num_workers=1, use_gpu=False)
        print(exp_filename)


