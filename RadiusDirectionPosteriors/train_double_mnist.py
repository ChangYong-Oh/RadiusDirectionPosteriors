import os
import sys
import math
import socket
import dill
import pickle
import progressbar
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda

from BayesianNeuralNetwork.torch_user.nn.utils import softplus_inv

HOSTNAME = socket.gethostname()
if HOSTNAME == 'hekla':
    SAVE_DIR = '/is/ei/com/Experiments/BayesianNeuralNetwork'
else:  # Any other machine in MPI Tuebingen
    SAVE_DIR = '/home/com/Experiments/BayesianNeuralNetwork'
assert os.path.exists(SAVE_DIR)
MODEL_FILENAME = lambda filename_prefix: os.path.join(SAVE_DIR, filename_prefix + '.pkl')
OPTIM_FILENAME = lambda filename_prefix: os.path.join(SAVE_DIR, filename_prefix + '_optim.pkl')
EXP_INFO_FILENAME = lambda filename_prefix: os.path.join(SAVE_DIR, filename_prefix + '_exp_info.pkl')
LOG_FILENAME = lambda filename_prefix: os.path.join(SAVE_DIR, filename_prefix + '_log.txt')


def train_initiate(model_type, prior_type, data_type, n_pred_samples,
                   n_epoch, lr, batch_size=32, num_workers=4, use_gpu=False):
    exp_info_dict = {'model_type': model_type, 'prior_type': prior_type, 'data_type': data_type,
                     'n_epoch': n_epoch, 'lr': lr, 'batch_size': batch_size}
    time_tag = datetime.now().strftime("%H:%M:%S:%f")
    exp_filename_prefix = '_'.join([data_type, model_type, prior_type, 'E' + str(n_epoch).zfill(4), time_tag])
    use_gpu = use_gpu and cuda.is_available()
    print(exp_filename_prefix)

    initialization_hyperparams = {'vMF': {'direction': 'kaiming',
                                          'softplus_inv_concentration_normal_mean_via_epsilon': 0.1,
                                          'softplus_inv_concentration_normal_std': 0.1},
                                  'LogNormal': {'mu_normal_mean': math.log(1.0),
                                                'mu_normal_std': 0.1,
                                                'softplus_inv_std_normal_mean': math.log(1e-2),
                                                'softplus_inv_std_normal_std': 0.1},
                                  'Gamma': {'softplus_inv_shape_normal_mean': softplus_inv(2.0 ** 0.5),
                                            'softplus_inv_shape_normal_std': 0.1,
                                            'softplus_inv_rate_normal_mean': softplus_inv(1.0),
                                            'softplus_inv_rate_normal_std': 0.1}
                                  }
    model = load_model(model_type=model_type, prior_type=prior_type, use_gpu=use_gpu)
    model.reset_parameters(initialization_hyperparams)
    for c in model.children():
        if c._get_name() == 'DoubleRadialLinear':
            if c.in_features > 1:
                c.row_direction_rsampler.reset_parameters({'vMF': {'direction': 'kaiming',
                                                                   'log_concentration_normal_mean_via_epsilon': 0.05,
                                                                   'log_concentration_normal_std': 0.5}})
            if c.out_features > 1:
                c.col_direction_rsampler.reset_parameters({'vMF': {'direction': 'kaiming_transpose',
                                                                   'log_concentration_normal_mean_via_epsilon': 0.05,
                                                                   'log_concentration_normal_std': 0.5}})

    train_loader, valid_loader, test_loader, train_loader_eval = load_data(data_type=data_type, batch_size=batch_size, num_workers=num_workers, use_gpu=use_gpu)
    eval_loaders = [train_loader_eval, valid_loader, test_loader]
    annealing_steps = float(50.0 * math.ceil(len(train_loader.dataset) / batch_size))
    beta_func = lambda s: min(s, annealing_steps) / annealing_steps
    # beta_func = lambda s: 1.0
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_log, n_steps = train(model=model, optimizer=optimizer, train_loader=train_loader, begin_step=0, epoch_begin=0, epoch_end=n_epoch,
                               beta_func=beta_func, filename_prefix=exp_filename_prefix, eval_loaders=eval_loaders, use_gpu=use_gpu)
    eval_log = evaluate(model=model,
                        train_loader_eval=train_loader_eval, valid_loader=valid_loader, test_loader=test_loader,
                        n_pred_samples=n_pred_samples)

    exp_info_dict['n_steps'] = n_steps
    exp_info_dict['beta_func'] = beta_func
    exp_filename = save_experiment(model=model, optimizer=optimizer, log_text='\n'.join([exp_filename_prefix, train_log, eval_log]), exp_info_dict=exp_info_dict, filename_prefix=exp_filename_prefix)
    return exp_filename


# if n_epoch == 0, then this is just evaluation
def train_continue(model_filename, n_epoch, n_pred_samples, num_workers=4, use_gpu=False):
    dirname, filename = os.path.split(model_filename)
    filename_prefix = filename.split('.')[0]
    exp_info_file = open(os.path.join(SAVE_DIR, EXP_INFO_FILENAME(filename_prefix)), 'rb')
    exp_info_dict = pickle.load(exp_info_file)
    exp_info_file.close()
    time_tag = datetime.now().strftime("%H:%M:%S:%f")
    exp_filename_prefix = '_'.join(filename_prefix.split('_')[:-2] + ['E' + str(exp_info_dict['n_epoch'] + n_epoch).zfill(4), time_tag])

    model = load_model(model_type=exp_info_dict['model_type'], prior_type=exp_info_dict['prior_type'], use_gpu=use_gpu)
    optimizer = optim.Adam(model.parameters(), lr=exp_info_dict['lr'])
    model.load_state_dict(torch.load(model_filename))
    optimizer.load_state_dict(torch.load(OPTIM_FILENAME(filename_prefix)))
    train_loader, valid_loader, test_loader, train_loader_eval = load_data(data_type=exp_info_dict['data_type'], batch_size=exp_info_dict['batch_size'], num_workers=num_workers, use_gpu=use_gpu)
    eval_loaders = [train_loader_eval, valid_loader, test_loader]
    prev_n_epoch = exp_info_dict['n_epoch']
    train_log, n_steps = train(model=model, optimizer=optimizer, train_loader=train_loader,
                               begin_step=exp_info_dict['n_steps'], epoch_begin=prev_n_epoch, epoch_end=prev_n_epoch+n_epoch,
                               beta_func=exp_info_dict['beta_func'],
                               filename_prefix=exp_filename_prefix, eval_loaders=eval_loaders, use_gpu=use_gpu)

    exp_info_dict['n_epoch'] += n_epoch
    exp_info_dict['n_steps'] += n_steps
    eval_log = evaluate(model=model, train_loader_eval=train_loader_eval, valid_loader=valid_loader, test_loader=test_loader , n_pred_samples=n_pred_samples)
    exp_filename = save_experiment(model=model, optimizer=optimizer, log_text='\n'.join([exp_filename_prefix, train_log, eval_log]), exp_info_dict=exp_info_dict, filename_prefix=exp_filename_prefix)
    return exp_filename


def load_data(data_type, batch_size, num_workers, use_gpu):
    if data_type == 'MNIST':
        from BayesianNeuralNetwork.data_loaders.mnist import data_loader
        train_loader, valid_loader, test_loader, train_loader_eval = data_loader(batch_size=batch_size, num_workers=num_workers, use_gpu=use_gpu)
        return train_loader, valid_loader, test_loader, train_loader_eval
    elif data_type == 'FashionMNIST':
        from BayesianNeuralNetwork.data_loaders.fashionmnist import data_loader
        train_loader, valid_loader, test_loader, train_loader_eval = data_loader(batch_size=batch_size, num_workers=num_workers, use_gpu=use_gpu)
        return train_loader, valid_loader, test_loader, train_loader_eval
    else:
        raise NotImplementedError


def load_model(model_type, prior_type, use_gpu):
    if model_type in ['MNISTDOUBLEFC3_150', 'MNISTDOUBLEFC3_250', 'MNISTDOUBLEFC3_500', 'MNISTDOUBLEFC3_750']:
        from BayesianNeuralNetwork.models.mnist_double_mlp import MNISTDOUBLEFC3
        model = MNISTDOUBLEFC3(prior_type=prior_type, n_hidden=int(model_type[-3:]))
    elif model_type in ['MNISTDOUBLEFC2_150', 'MNISTDOUBLEFC2_400']:
        from BayesianNeuralNetwork.models.mnist_double_mlp import MNISTDOUBLEFC2
        model = MNISTDOUBLEFC2(prior_type=prior_type, n_hidden=int(model_type[-3:]))
    if use_gpu:
        model.cuda()
    return model


def train(model, optimizer, train_loader, begin_step, epoch_begin, epoch_end, beta_func, filename_prefix, eval_loaders=[], use_gpu=False):
    assert 'Random' in train_loader.sampler.__class__.__name__
    train_info = 'epoch:%d ~ %d' % (epoch_begin, epoch_end)
    model_hyperparam_info = model.init_hyperparam_value()
    model_prior_info = model.prior.__repr__()
    print(train_info)
    print(model_prior_info)
    print(model_hyperparam_info)
    logging = train_info + '\n' + model_prior_info + '\n' + model_hyperparam_info + '\n'

    criterion = nn.CrossEntropyLoss(size_average=True)

    n_data = len(train_loader.sampler)
    n_step = begin_step
    running_reconstruction = 0.0
    running_kld = 0.0
    running_loss = 0.0
    for e in range(epoch_begin, epoch_end):
        for b, data in enumerate(train_loader):
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
            reconstruction = criterion(pred, outputs)
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
            display_steps = 250
            if n_step % display_steps == display_steps - 1:
                log_str = '%s [%6d steps in (%4d epochs) ] loss: %.6f, reconstruction: %.6f, regularizer: %.6f, beta:%.3E, %s' % \
                          (datetime.now().strftime("%H:%M:%S.%f"), n_step + 1, e + 1,
                           running_loss / display_steps, running_reconstruction / display_steps, running_kld / display_steps,
                           beta, train_info)
                print(log_str)
                logging += log_str + '\n'
                running_reconstruction = 0.0
                running_kld = 0.0
                running_loss = 0.0
        if e % 5 == 4 and len(eval_loaders) == 3:
            evaluate(model, eval_loaders[0], eval_loaders[1], eval_loaders[2], 0)
        if e % 5 == 4:
            print(os.path.join(SAVE_DIR, MODEL_FILENAME(filename_prefix + '_e' + str(e + 1).zfill(4))))
            torch.save(model.state_dict(), MODEL_FILENAME(filename_prefix + '_e' + str(e + 1).zfill(4)))
            torch.save(optimizer.state_dict(), OPTIM_FILENAME(filename_prefix + '_e' + str(e + 1).zfill(4)))
    logging += train_info + '\n' + model_prior_info + '\n' + model_hyperparam_info

    return logging, n_step


def evaluate(model, train_loader_eval, valid_loader, test_loader, n_pred_samples):
    if n_pred_samples <= 0:
        train_pred, train_output = deterministic_prediction(model, train_loader_eval)
        valid_pred, valid_output = deterministic_prediction(model, valid_loader)
        test_pred, test_output = deterministic_prediction(model, test_loader)
    else:
        train_pred_samples, train_output = sample_prediction(model, train_loader_eval, n_pred_samples)
        train_pred_statistics = multinomial_statistics(train_pred_samples)
        train_pred = torch.argmax(train_pred_statistics, dim=1)
        print('\nTrain Digit count' + ' '.join(['%5d' % int(torch.sum(train_output == d)) for d in range(10)]))
        print(' Prediction count' + ' '.join(['%5d' % int(torch.sum(train_pred == d)) for d in range(10)]))

        valid_pred_samples, valid_output = sample_prediction(model, valid_loader, n_pred_samples)
        valid_pred_statistics = multinomial_statistics(valid_pred_samples)
        valid_pred = torch.argmax(valid_pred_statistics, dim=1)
        print('\nValid Digit count' + ' '.join(['%5d' % int(torch.sum(valid_output == d)) for d in range(10)]))
        print(' Prediction count' + ' '.join(['%5d' % int(torch.sum(valid_pred == d)) for d in range(10)]))

        test_pred_samples, test_output = sample_prediction(model, test_loader, n_pred_samples)
        test_pred_statistics = multinomial_statistics(test_pred_samples)
        test_pred = torch.argmax(test_pred_statistics, dim=1)
        print('\nTest  Digit count' + ' '.join(['%5d' % int(torch.sum(test_output == d)) for d in range(10)]))
        print(' Prediction count' + ' '.join(['%5d' % int(torch.sum(test_pred == d)) for d in range(10)]))

    train_acc = float((train_pred == train_output).sum()) / float(train_output.size(0))
    logging = 'Train accuracy(%d samples) : %4.2f%%' % (n_pred_samples, train_acc * 100.0) + '\n'
    valid_acc = float((valid_pred == valid_output).sum()) / float(valid_output.size(0))
    logging += 'Valid accuracy(%d samples) : %4.2f%%' % (n_pred_samples, valid_acc * 100.0) + '\n'
    test_acc = float((test_pred == test_output).sum()) / float(test_output.size(0))
    logging += 'Test  accuracy(%d samples) : %4.2f%%' % (n_pred_samples, test_acc * 100.0)

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


def deterministic_prediction(model, data_loader):
    model.deterministic_forward(True)

    assert 'Random' not in data_loader.sampler.__class__.__name__
    output_data = torch.empty((0,)).long()
    pred = torch.empty((0,))
    ind = 0
    for input_batch, output_batch in data_loader:
        output_data = torch.cat([output_data, output_batch])
        pred_batch = model(input_batch)
        pred = torch.cat([pred, pred_batch])
        batch_size = input_batch.size(0)
        ind += batch_size

    model.deterministic_forward(False)
    return pred.argmax(dim=1), output_data


def sample_prediction(model, data_loader, n_pred_samples):
    n_category = 10
    n_data = len(data_loader.dataset)
    assert 'Random' not in data_loader.sampler.__class__.__name__
    output_data = torch.empty((0,)).long()

    if HOSTNAME == 'hekla':
        bar = progressbar.ProgressBar(max_value=n_data)
    pred_samples = torch.zeros(n_pred_samples, n_data, n_category)
    ind = 0
    for inputs, outputs in data_loader:
        output_data = torch.cat([output_data, outputs])
        batch_size = inputs.size(0)
        for i in range(n_pred_samples):
            pred_samples[i, ind:ind+batch_size, :] = model(inputs).detach()
        ind += batch_size
        if HOSTNAME == 'hekla':
            bar.update(ind)
        else:
            if ind // 1000 != (ind - batch_size) // 1000:
                print(ind)
    return pred_samples, output_data


def multinomial_statistics(pred_samples):
    n_pred_samples, n_data, n_category = pred_samples.size()
    sample_argmax = torch.argmax(pred_samples, dim=2)
    argmax_count = sample_argmax.new_zeros(n_data, n_category)
    for c in range(n_category):
        argmax_count[:, c] = (sample_argmax == c).sum(dim=0)
    return argmax_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST Train script')
    parser.add_argument('--architecture', dest='architecture', type=str, default='MNISTDOUBLEFC2_400',
                        help='  / \n'.join(['MNISTDOUBLEFC2_400', 'MNISTDOUBLEFC3_150', 'MNISTDOUBLEFC3_250', 'MNISTDOUBLEFC3_500', 'MNISTDOUBLEFC3_750']))
    parser.add_argument('--prior', dest='prior', type=str, default='HalfCauchy',
                        help='  / \n'.join(['HalfCauchy', 'Gamma', 'Weibull']))
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--batch_size', dest='batch_size', default=100, type=int)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001)
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=False, help='Use gpu if available')
    parser.add_argument('--model_file', dest='model_file', help='Continue training')
    parser.add_argument('--eval', dest='eval', action='store_true', default=False, help='Evaluating model')

    args = parser.parse_args()
    arg_dict = vars(args)
    print(arg_dict)

    if HOSTNAME == 'hekla' and len(sys.argv) == 1:
        model_file = '/is/ei/com/Experiments/BayesianNeuralNetwork/MNIST_MNISTDOUBLEFC2_400_Gamma_E0200_20:00:51:198637_e0060.pkl'
        dirname, filename = os.path.split(model_file)
        model_type = '_'.join(filename.split('_')[1:3])
        prior_type = filename.split('_')[3]
        model = load_model(model_type=model_type, prior_type=prior_type, use_gpu=False)
        # exp_filename = train_initiate(model_type=args.architecture, prior_type=args.prior, data_type='MNIST',
        #                               n_pred_samples=0, n_epoch=1, lr=args.lr, batch_size=args.batch_size,
        #                               num_workers=1, use_gpu=args.gpu)
        model.load_state_dict(torch.load(model_file))
        exit(0)

    if args.eval:
        exp_filename = train_continue(model_filename=args.model_file, n_epoch=0, n_pred_samples=0, num_workers=1, use_gpu=args.gpu)
        print(exp_filename)
    elif args.model_file is not None:
        exp_filename = train_continue(model_filename=args.model_file, n_epoch=args.epochs, n_pred_samples=0, num_workers=1, use_gpu=args.gpu)
        print(exp_filename)
    else:
        exp_filename = train_initiate(model_type=args.architecture, prior_type=args.prior, data_type='MNIST',
                                      n_pred_samples=0, n_epoch=args.epochs, lr=args.lr, batch_size=args.batch_size,
                                      num_workers=1, use_gpu=args.gpu)
        print(exp_filename)


# TODO : how about scaling only vMF KL-term?? Check value of each KLD
# TODO : scaling weight according to input dimension for MNIST experiment?