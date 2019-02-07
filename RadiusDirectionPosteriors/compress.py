import os
import importlib
import dill
import pickle
import socket
import math
import json
import progressbar
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
from BayesianNeuralNetwork.utils.dir_utils import exp_dir
from vgg16_bn_init import load_vgg16_bn_to_double, load_vgg16_bn_to_radial


HOSTNAME = socket.gethostname()
SAVE_DIR = exp_dir()
FIG_SAVE_DIR = exp_dir()
assert os.path.exists(SAVE_DIR)

MODEL_FILENAME = lambda filename_prefix: os.path.join(SAVE_DIR, filename_prefix + '.pkl')
OPTIM_FILENAME = lambda filename_prefix: os.path.join(SAVE_DIR, filename_prefix + '_optim.pkl')
EXP_INFO_FILENAME = lambda filename_prefix: os.path.join(SAVE_DIR, filename_prefix + '_exp_info.pkl')
LOG_FILENAME = lambda filename_prefix: os.path.join(SAVE_DIR, filename_prefix + '_log.txt')


def train_initiate(data_type, model_type, init_hyper, prior_info, n_epoch, lr, batch_size=32, num_workers=4, use_gpu=False, load_pretrain=None):
	prior_type, prior_hyper = prior_info
	prior_info_str = prior_type + '-'
	for k in sorted(prior_hyper.keys()):
		prior_info_str += k + ('(%.2E)' % prior_hyper[k])
	exp_info_dict = {'prior_info': prior_info, 'n_epoch': n_epoch, 'lr': lr, 'batch_size': batch_size}
	time_tag = datetime.now().strftime("%H:%M:%S:%f")
	use_gpu = use_gpu and cuda.is_available()

	model = load_model(model_type=model_type, prior_info=prior_info, use_gpu=use_gpu)
	model.reset_parameters(init_hyper)
	if load_pretrain is not None:
		if model_type in ['VGG16-Flatten', 'VGG16-Bundle', 'VGG16-BN-Flatten', 'VGG16-BN-Bundle']:
			load_vgg16_bn_to_double(model, load_pretrain)
		elif model_type in ['VGG16-BN']:
			load_vgg16_bn_to_radial(model, load_pretrain)
	exp_filename_prefix = '_'.join([data_type, model._get_name(), prior_info_str, 'E' + str(n_epoch).zfill(4), time_tag])
	print(exp_filename_prefix)
	train_loader, valid_loader, test_loader, train_loader_eval = load_data(data_type=data_type, batch_size=batch_size, num_workers=num_workers, use_gpu=use_gpu)
	eval_loaders = [train_loader_eval, valid_loader, test_loader]
	if 'VGG' in model_type:
		annealing_steps = float(100.0 * math.ceil(len(train_loader.dataset) / batch_size))
		beta_func = lambda s: min(s, annealing_steps) / annealing_steps * 0.01
	else:
		annealing_steps = float(100.0 * math.ceil(len(train_loader.dataset) / batch_size))
		beta_func = lambda s: min(s, annealing_steps) / annealing_steps
	optimizer = optim.Adam(model.parameters(), lr=lr)

	model_hyperparam_info = model.init_hyperparam_value()
	print(model_hyperparam_info)

	train_log, n_steps = train(model=model, optimizer=optimizer, train_loader=train_loader, begin_step=0, epoch_begin=0, epoch_end=n_epoch,
	                           beta_func=beta_func, filename_prefix=exp_filename_prefix, eval_loaders=eval_loaders, use_gpu=use_gpu)

	exp_info_dict['n_steps'] = n_steps
	exp_info_dict['beta_func'] = beta_func
	exp_filename = save_log_exp_info(log_text='\n'.join([exp_filename_prefix, model_hyperparam_info, train_log]), exp_info_dict=exp_info_dict, filename_prefix=exp_filename_prefix)
	return exp_filename


def train_continue(data_type, model_type, model_filename, n_epoch, lr, num_workers=4, use_gpu=False):
	dirname, filename = os.path.split(model_filename)
	filename_prefix = filename[:-4]
	exp_info_file = open(os.path.join(SAVE_DIR, EXP_INFO_FILENAME(filename_prefix)), 'rb')
	exp_info_dict = pickle.load(exp_info_file)
	exp_info_file.close()
	time_tag = datetime.now().strftime("%H:%M:%S:%f")
	exp_filename_prefix = '_'.join(filename_prefix.split('_')[:-2] + ['E' + str(exp_info_dict['n_epoch'] + n_epoch).zfill(4), time_tag])

	model = load_model(model_type=model_type, prior_info=exp_info_dict['prior_info'], use_gpu=use_gpu)
	optimizer = optim.Adam(model.parameters(), lr=exp_info_dict['lr'])
	model.load_state_dict(torch.load(os.path.join(SAVE_DIR, model_filename)))
	optimizer.load_state_dict(torch.load(OPTIM_FILENAME(filename_prefix)))
	if lr is not None:
		exp_info_dict['lr'] = lr
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
	train_loader, valid_loader, test_loader, train_loader_eval = load_data(data_type=data_type, batch_size=exp_info_dict['batch_size'], num_workers=num_workers, use_gpu=use_gpu)
	eval_loaders = [train_loader_eval, valid_loader, test_loader]
	prev_n_epoch = exp_info_dict['n_epoch']
	train_log, n_steps = train(model=model, optimizer=optimizer, train_loader=train_loader,
	                           begin_step=exp_info_dict['n_steps'], epoch_begin=prev_n_epoch, epoch_end=prev_n_epoch+n_epoch,
	                           beta_func=exp_info_dict['beta_func'], filename_prefix=exp_filename_prefix, eval_loaders=eval_loaders, use_gpu=use_gpu)

	exp_info_dict['n_epoch'] += n_epoch
	exp_info_dict['n_steps'] += n_steps
	exp_filename = save_log_exp_info(log_text='\n'.join([exp_filename_prefix, train_log]), exp_info_dict=exp_info_dict, filename_prefix=exp_filename_prefix)
	return exp_filename


def train(model, optimizer, train_loader, begin_step, epoch_begin, epoch_end, beta_func, filename_prefix, eval_loaders=[], use_gpu=False):
	assert 'Random' in train_loader.sampler.__class__.__name__
	train_info = 'epoch:%d ~ %d learning rate:%8.6f' % (epoch_begin, epoch_end, optimizer.state_dict()['param_groups'][0]['lr'])
	model_prior_info = model.prior.__repr__()
	print(train_info)
	print(model_prior_info)
	logging = train_info + '\n' + model_prior_info + '\n'

	criterion = nn.CrossEntropyLoss(size_average=True)

	n_data = len(train_loader.sampler)
	n_step = begin_step
	running_rate = 0.01
	running_xent = 0.0
	running_kld = 0.0
	running_loss = 0.0
	best_valid_acc = -float('inf')
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
			beta = beta_func(n_step)
			pred = model(inputs)
			xent = criterion(pred, outputs)
			kld = model.kl_divergence() / float(n_data)
			if torch.isinf(kld):
				raise RuntimeError("KL divergence is infinite. It is likely that ive is zero and is passed to log.")
			if torch.isnan(kld):
				model.kl_divergence()
				raise RuntimeError("KL divergence is Nan.")
			# kld = 0
			loss = xent + kld * beta
			loss.backward()
			for m in model.modules():
				if hasattr(m, 'gradient_correction'):
					m.gradient_correction(xent)
			optimizer.step()
			for m in model.modules():
				if hasattr(m, 'parameter_adjustment'):
					m.parameter_adjustment()
			n_step += 1

			# log_str = '%s [%6d steps in (%4d epochs) ] loss: %.6f, xentropy: %.6f, regularizer: %.6f, beta:%.3E, %s' % \
			# 		  (datetime.now().strftime("%H:%M:%S.%f"), n_step, e + 1, float(loss), float(xent),
			# 		   float(kld), beta, train_info)
			# print(log_str)

			# print statistics
			running_xent = running_rate * float(xent) + (1 - running_rate) * running_xent
			running_kld = running_rate * float(kld) + (1 - running_rate) * running_kld
			running_loss = running_rate * float(loss) + (1 - running_rate) * running_loss

		log_str = '%s [%6d steps in (%4d epochs) ] loss: %.6f, xentropy: %.6f, regularizer: %.6f, beta:%.3E, %s' % \
		          (datetime.now().strftime("%H:%M:%S.%f"), n_step, e + 1, running_loss, running_xent, running_kld, beta, train_info)
		print(log_str)
		logging += log_str + '\n'
		running_xent = 0.0
		running_kld = 0.0
		running_loss = 0.0
		train_acc, valid_acc, test_acc, eval_str = evaluate(model, eval_loaders[0], eval_loaders[1], eval_loaders[2])
		if valid_acc > best_valid_acc or epoch_end - e <= 20:
			print('Best validation accuracy has been updated at %4d epoch.' % (e + 1))
			torch.save(model.state_dict(), MODEL_FILENAME(filename_prefix + '_e' + str(e + 1).zfill(4)))
			torch.save(optimizer.state_dict(), OPTIM_FILENAME(filename_prefix + '_e' + str(e + 1).zfill(4)))
			best_valid_acc = valid_acc
		logging += eval_str + '\n'
	print('Last update is stored.')
	print(os.path.join(SAVE_DIR, MODEL_FILENAME(filename_prefix)))
	torch.save(model.state_dict(), MODEL_FILENAME(filename_prefix))
	torch.save(optimizer.state_dict(), OPTIM_FILENAME(filename_prefix))

	logging += train_info + '\n' + model_prior_info

	return logging, n_step


def load_model(model_type, prior_info, use_gpu):
	if model_type[:3] == 'VGG':
		vgg_type = model_type.split('-')[0][3:]
		bn_type = 'BN' if 'BN' in model_type else ''
		layer_type = model_type.split('-')[-1]
		is_double_layer = layer_type in ['Flatten', 'Bundle']
		if is_double_layer:
			layer_type = 'Double' + layer_type
		else:
			layer_type = ''
		class_name = 'VGG' + vgg_type + bn_type + layer_type
		module = importlib.import_module('BayesianNeuralNetwork.models.' + ('vgg_double' if is_double_layer else 'vgg'))
		model = getattr(module, class_name)(prior_info=prior_info)
	elif model_type == 'LeNet5-Flatten':
		from BayesianNeuralNetwork.models.lenet_conv_double_flatten import LeNet5DoubleFlatten
		model = LeNet5DoubleFlatten(prior_info=prior_info)
	elif model_type == 'LeNet5-Bundle':
		from BayesianNeuralNetwork.models.lenet_conv_double_bundle import LeNet5DoubleBundle
		model = LeNet5DoubleBundle(prior_info=prior_info)
	elif model_type == 'LeNet5':
		from BayesianNeuralNetwork.models.lenet_conv import LeNet5
		model = LeNet5(prior_info=prior_info)
	elif model_type == 'LeNetFC-Double':
		from BayesianNeuralNetwork.models.lenet_fc_double import LeNetFCDouble
		model = LeNetFCDouble(prior_info=prior_info)
	elif model_type == 'LeNetFC':
		from BayesianNeuralNetwork.models.lenet_fc import LeNetFC
		model = LeNetFC(prior_info=prior_info)
	else:
		raise NotImplementedError
	if use_gpu:
		model.cuda()
	return model


def load_data(data_type, batch_size, num_workers, use_gpu):
	if data_type == 'MNIST':
		from BayesianNeuralNetwork.data_loaders.mnist import data_loader
		train_loader, valid_loader, test_loader, train_loader_eval = data_loader(batch_size=batch_size, num_workers=num_workers, use_gpu=use_gpu, validation=True)
	elif data_type == 'CIFAR10':
		from BayesianNeuralNetwork.data_loaders.cifar10 import data_loader
		train_loader, valid_loader, test_loader, train_loader_eval = data_loader(batch_size=batch_size, num_workers=num_workers, use_gpu=use_gpu, validation=True)
	return train_loader, valid_loader, test_loader, train_loader_eval


def evaluate(model, train_loader_eval, valid_loader, test_loader):
	model_status = model.training
	model.eval()
	train_pred, train_output = deterministic_prediction(model, train_loader_eval)
	if valid_loader is not None:
		valid_pred, valid_output = deterministic_prediction(model, valid_loader)
	test_pred, test_output = deterministic_prediction(model, test_loader)
	train_acc = float((train_pred == train_output).sum()) / float(train_output.size(0))
	if valid_loader is not None:
		valid_acc = float((valid_pred == valid_output).sum()) / float(valid_output.size(0))
	else:
		valid_acc = None
	test_acc = float((test_pred == test_output).sum()) / float(test_output.size(0))
	model.train(model_status)

	if valid_loader is not None:
		accuracy_info = 'Train : %4.2f%% / Valid : %4.2f%% / Test : %4.2f%%' % (train_acc * 100.0, valid_acc * 100.0, test_acc * 100.0)
	else:
		accuracy_info = 'Train : %4.2f%% / Test : %4.2f%%' % (train_acc * 100.0, test_acc * 100.0)

	print(accuracy_info)
	return train_acc, valid_acc, test_acc, accuracy_info


def double_layer_logmode_mask(layer, row_threshold, col_threshold):
	# Assuming 2 log-normal posterior for radius component
	row_mode = (layer.row_radius_rsampler.mode() * layer.row_radius_rsampler.mode()) ** 0.5
	log_row_mode = torch.log(row_mode)
	row_mask = (log_row_mode > row_threshold).float()

	col_mode = (layer.col_radius_rsampler.mode() * layer.col_radius_rsampler.mode()) ** 0.5
	log_col_mode = torch.log(col_mode)
	col_mask = (log_col_mode > col_threshold).float()

	print('-' * 50)
	print('    ROW %.2E~%.2E' % (float(torch.min(row_mode)), float(torch.max(row_mode))))
	print('    COL %.2E~%.2E' % (float(torch.min(col_mode)), float(torch.max(col_mode))))
	print('log ROW %.2E~%.2E' % (float(torch.min(log_row_mode)), float(torch.max(log_row_mode))))
	print('log COL %.2E~%.2E' % (float(torch.min(log_col_mode)), float(torch.max(log_col_mode))))

	return log_row_mode, row_mask, log_col_mode, col_mask


def radial_layer_logmode_mask(layer, threshold):
	mode = (layer.radius_rsampler.mode() * layer.radius_rsampler.mode()) ** 0.5
	log_mode = torch.log(mode)
	mask = (log_mode > threshold).float()

	print('-' * 50)
	print('    %.2E~%.2E' % (float(torch.min(mode)), float(torch.max(mode))))
	print('log %.2E~%.2E' % (float(torch.min(log_mode)), float(torch.max(log_mode))))

	return log_mode, mask


def save_log_exp_info(log_text, exp_info_dict, filename_prefix):
	exp_info_file = open(os.path.join(SAVE_DIR, EXP_INFO_FILENAME(filename_prefix)), 'wb')
	pickle.dump(exp_info_dict, exp_info_file, pickle.HIGHEST_PROTOCOL)
	exp_info_file.close()
	log_file = open(LOG_FILENAME(filename_prefix), 'wt')
	log_file.write(log_text)
	log_file.close()
	return os.path.join(SAVE_DIR, filename_prefix + '.pkl')


def deterministic_prediction(model, data_loader):
	model.deterministic_forward(True)
	is_cuda = next(model.parameters()).is_cuda

	assert 'Random' not in data_loader.sampler.__class__.__name__
	output_data = torch.empty((0,)).long()
	pred = torch.empty((0,))
	if is_cuda:
		output_data = output_data.cuda()
		pred = pred.cuda()
	ind = 0
	for input_batch, output_batch in data_loader:
		if is_cuda:
			input_batch = input_batch.cuda()
			output_batch = output_batch.cuda()
		output_data = torch.cat([output_data, output_batch])
		pred_batch = model(input_batch).detach()
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


def prior_info_from_json(prior_filepath):
	prior_filepath = os.path.join('/'.join(os.path.split(__file__)[0].split('/')[:-1] + ['prior_json']), os.path.split(prior_filepath)[1])
	json_file = open(prior_filepath)
	prior_dict = json.load(json_file)
	json_file.close()
	prior_type = prior_dict['type']
	prior_hyper = prior_dict.copy()
	del prior_hyper['type']
	return (prior_type, prior_hyper)


def nonzero_ratio_lenetfc(fc1_col_mask, fc1_row_mask, fc2_col_mask, fc2_row_mask, fc3_col_mask, fc3_row_mask):
	original_model = 784 * 300 + 300 * 100 + 100 * 10
	fc1 = int(torch.sum(fc1_col_mask))
	fc2 = int(torch.sum(fc1_row_mask * fc2_col_mask))
	fc3 = int(torch.sum(fc2_row_mask * fc3_col_mask))
	prunned_model = fc1 * fc2 + fc2 * fc3 + fc3 * 10
	return float(prunned_model) / float(original_model)


def nonzero_ratio_lenet5(conv1_col_mask, conv1_row_mask, conv2_col_mask, conv2_row_mask, fc1_col_mask, fc1_row_mask, fc2_col_mask, fc2_row_mask):
	original_model = 1 * 20 * 25 + 20 * 50 * 25 + 800 * 500 + 500 * 10
	conv1 = int(torch.sum(conv1_col_mask))
	conv2 = int(torch.sum(conv1_row_mask * conv2_col_mask))
	fc1 = int(torch.sum(conv2_row_mask.view(-1, 1, 1).repeat(1, 4, 4).view(-1) * fc1_col_mask))
	fc2 = int(torch.sum(fc1_row_mask * fc2_col_mask))
	prunned_model = 1 * conv1 * 25 + conv1 * conv2 * 25 + fc1 * fc2 + fc2 * 10
	return float(prunned_model) / float(original_model)