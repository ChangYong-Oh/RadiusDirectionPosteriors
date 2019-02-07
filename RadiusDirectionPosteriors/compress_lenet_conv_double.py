import os
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch

from BayesianNeuralNetwork.compress import double_layer_logmode_mask, FIG_SAVE_DIR, prior_info_from_json
from BayesianNeuralNetwork.compress import load_data, load_model, nonzero_ratio_lenet5
from BayesianNeuralNetwork.utils.dir_utils import exp_dir


def evaluate_with_prunning(model, train_loader_eval, valid_loader, test_loader, row_threshold, col_threshold, tag):
	model.deterministic_forward(True)

	if model.conv1.in_channels > 1 and model.conv1.out_channels > 1:
		log_conv1_row_mode, conv1_row_mask, log_conv1_col_mode, conv1_col_mask = double_layer_logmode_mask(model.conv1, row_threshold[0], col_threshold[0])
	elif model.conv1.in_channels == 1:
		log_conv1_col_mode = torch.ones([1])
		conv1_col_mask = torch.ones([1])
		log_conv1_row_mode = torch.log((model.conv1.col_rsampler.mode().reshape(20, -1) ** 2).sum(dim=1))
		conv1_row_mask = (log_conv1_row_mode > row_threshold[0]).float()

	log_conv2_row_mode, conv2_row_mask, log_conv2_col_mode, conv2_col_mask = double_layer_logmode_mask(model.conv2, row_threshold[1], col_threshold[1])
	log_fc1_row_mode, fc1_row_mask, log_fc1_col_mode, fc1_col_mask = double_layer_logmode_mask(model.fc1, row_threshold[2], col_threshold[2])
	log_fc2_row_mode, fc2_row_mask, log_fc2_col_mode, fc2_col_mask = double_layer_logmode_mask(model.fc2, row_threshold[3], col_threshold[3])

	data_loader = train_loader_eval
	output_data = torch.empty((0,)).long()
	pred = torch.empty((0,))
	for input_batch, output_batch in data_loader:
		output_data = torch.cat([output_data, output_batch])

		x = input_batch
		x = model.pool1(model.conv1(x * conv1_col_mask.view(1, -1, 1, 1)) * conv1_row_mask.view(1, -1, 1, 1))
		x = model.pool2(model.conv2(x * conv2_col_mask.view(1, -1, 1, 1)) * conv2_row_mask.view(1, -1, 1, 1))
		x = x.view(x.size(0), -1)
		x = model.fc1_nonlinear(model.fc1(x * fc1_col_mask) * fc1_row_mask)
		pred_batch = model.fc2(x * fc2_col_mask) * fc2_row_mask

		pred = torch.cat([pred, pred_batch.detach()])
	train_pred = pred.argmax(dim=1)
	train_output = output_data
	train_acc = float((train_pred == train_output).sum()) / float(train_output.size(0))

	if valid_loader is not None:
		data_loader = valid_loader
		output_data = torch.empty((0,)).long()
		pred = torch.empty((0,))
		for input_batch, output_batch in data_loader:
			output_data = torch.cat([output_data, output_batch])

			x = input_batch
			x = model.pool1(model.conv1(x * conv1_col_mask.view(1, -1, 1, 1)) * conv1_row_mask.view(1, -1, 1, 1))
			x = model.pool2(model.conv2(x * conv2_col_mask.view(1, -1, 1, 1)) * conv2_row_mask.view(1, -1, 1, 1))
			x = x.view(x.size(0), -1)
			x = model.fc1_nonlinear(model.fc1(x * fc1_col_mask) * fc1_row_mask)
			pred_batch = model.fc2(x * fc2_col_mask) * fc2_row_mask

			pred = torch.cat([pred, pred_batch.detach()])
		valid_pred = pred.argmax(dim=1)
		valid_output = output_data
		valid_acc = float((valid_pred == valid_output).sum()) / float(valid_output.size(0))

	data_loader = test_loader
	output_data = torch.empty((0,)).long()
	pred = torch.empty((0,))
	for input_batch, output_batch in data_loader:
		output_data = torch.cat([output_data, output_batch])

		x = input_batch
		x = model.pool1(model.conv1(x * conv1_col_mask.view(1, -1, 1, 1)) * conv1_row_mask.view(1, -1, 1, 1))
		x = model.pool2(model.conv2(x * conv2_col_mask.view(1, -1, 1, 1)) * conv2_row_mask.view(1, -1, 1, 1))
		x = x.view(x.size(0), -1)
		x = model.fc1_nonlinear(model.fc1(x * fc1_col_mask) * fc1_row_mask)
		pred_batch = model.fc2(x * fc2_col_mask) * fc2_row_mask

		pred = torch.cat([pred, pred_batch.detach()])
	test_pred = pred.argmax(dim=1)
	test_output = output_data
	test_acc = float((test_pred == test_output).sum()) / float(test_output.size(0))

	if valid_loader is not None:
		accuracy_info = 'Train : %4.2f%% / Valid : %4.2f%% / Test : %4.2f%%' % (train_acc * 100.0, valid_acc * 100.0, test_acc * 100.0)
	else:
		accuracy_info = 'Train : %4.2f%% / Test : %4.2f%%' % (train_acc * 100.0, test_acc * 100.0)
	print(accuracy_info)

	print('Nonzero row')
	prunning_info = 'CONV 1 : row %3d/%3d col %3d/%3d\n' % (int(torch.sum(conv1_row_mask)), conv1_row_mask.numel(), int(torch.sum(conv1_col_mask)), conv1_col_mask.numel())
	prunning_info += 'CONV 2 : row %3d/%3d col %3d/%3d\n' % (int(torch.sum(conv2_row_mask)), conv2_row_mask.numel(), int(torch.sum(conv2_col_mask)), conv2_col_mask.numel())
	prunning_info += 'FC 1 : row %3d/%3d col %3d/%3d\n' % (int(torch.sum(fc1_row_mask)), fc1_row_mask.numel(), int(torch.sum(fc1_col_mask)), fc1_col_mask.numel())
	prunning_info += 'FC 2 : row %3d/%3d col %3d/%3d\n' % (int(torch.sum(fc2_row_mask)), fc2_row_mask.numel(), int(torch.sum(fc2_col_mask)), fc2_col_mask.numel())
	print(prunning_info)

	row_mode_list = [log_conv1_row_mode, log_conv2_row_mode, log_fc1_row_mode, log_fc2_row_mode]
	row_mask_list = [conv1_row_mask, conv2_row_mask, fc1_row_mask, fc2_row_mask]
	col_mode_list = [log_conv1_col_mode, log_conv2_col_mode, log_fc1_col_mode, log_fc2_col_mode]
	col_mask_list = [conv1_col_mask, conv2_col_mask, fc1_col_mask, fc2_col_mask]
	layername_list = ['CONV1', 'CONV2', 'FC1', 'FC2']
	fig, axes = plt.subplots(2, 4, figsize=(12, 9))
	for i in range(len(layername_list)):
		row_mode = row_mode_list[i]
		row_mask = row_mask_list[i]
		row_thhl = row_threshold[i]
		col_mode = col_mode_list[i]
		col_mask = col_mask_list[i]
		col_thhl = col_threshold[i]
		layername = layername_list[i]
		row_ax, col_ax = axes[:, i]
		row_ax.hist(row_mode[~torch.isinf(row_mode)].detach().numpy())
		row_ax.axvline(row_thhl, ls=':', color='r')
		row_ax_min, row_ax_max = float(torch.min(row_mode)), float(torch.max(row_mode))
		row_ax.set_xlim([row_ax_min - 0.05 * (row_ax_max - row_ax_min), row_ax_max + 0.05 * (row_ax_max - row_ax_min)])
		row_ax.set_title(layername + ' Row %3d/%3d' % (int(torch.sum(row_mask)), row_mask.numel()), fontsize=8)
		col_ax.hist(col_mode[~torch.isinf(col_mode)].detach().numpy())
		col_ax.axvline(col_thhl, ls=':', color='r')
		col_ax_min, col_ax_max = float(torch.min(col_mode)), float(torch.max(col_mode))
		if col_ax_min == col_ax_max:
			col_ax.set_xlim([col_ax_min - 0.05, col_ax_max + 0.05])
		else:
			col_ax.set_xlim([col_ax_min - 0.05 * (col_ax_max - col_ax_min), col_ax_max + 0.05 * (col_ax_max - col_ax_min)])
		col_ax.set_title(layername + ' Col %3d/%3d' % (int(torch.sum(col_mask)), col_mask.numel()), fontsize=8)
		row_ax.tick_params(labelsize=6)
		col_ax.tick_params(labelsize=6)
		row_ax.title.set_position([0.5, 1.0])
		col_ax.title.set_position([0.5, 1.0])
		row_ax.tick_params(axis='both', which='major', pad=0)
		col_ax.tick_params(axis='both', which='major', pad=0)
	threshold_str = 'ROW:' + ','.join(['%+3.2f' % elm for elm in row_threshold]) + ' COL:' + ','.join(['%+3.2f' % elm for elm in col_threshold])
	nonzero_weight_ratio = nonzero_ratio_lenet5(conv1_col_mask, conv1_row_mask, conv2_col_mask, conv2_row_mask, fc1_col_mask, fc1_row_mask, fc2_col_mask, fc2_row_mask) * 100
	print('Nonzero ratio %6.4f%%' % nonzero_weight_ratio)
	plt.suptitle(tag + '\n' + accuracy_info + ' ' + threshold_str + (' %6.4f%%' % nonzero_weight_ratio), fontsize=8)
	plt.tight_layout(rect=[0, 0, 1, 0.94])
	plt.subplots_adjust(hspace=0.15, wspace=0.15)
	plt.show()
	while True:
		save_figure = raw_input('Do you want to save this figure? (YES/NO)')
		if save_figure in ['YES', 'NO']:
			break
	if save_figure == 'YES':
		fig_filename = tag + '_' + accuracy_info.replace('/', '_') + '.pdf'
		fig.savefig(os.path.join(FIG_SAVE_DIR, fig_filename), bbox_inches='tight')

	model.deterministic_forward(False)


if __name__ == '__main__':
	data_type = 'MNIST'
	model_type = 'LeNet5-Flatten'
	if len(sys.argv) < 2:
		# Choose the best one by the lowest training reconstruction error in last 10 epochs
		# filename = 'MNIST_LeNet5DoubleFlatten_HalfCauchy-LogNormal-tau_conv_global(1.00E-04)tau_conv_local(3.00E-03)tau_fc_global(1.00E-04)tau_fc_local(3.00E-03)_E0200_01:20:10:600426_e0197.pkl'
		# filename = 'MNIST_LeNet5DoubleFlatten_HalfCauchy-LogNormal-tau_conv_global(1.00E-04)tau_conv_local(1.00E-03)tau_fc_global(1.00E-04)tau_fc_local(1.00E-03)_E0200_01:20:13:088725_e0196.pkl'
		# filename = 'MNIST_LeNet5DoubleFlatten_HalfCauchy-LogNormal-tau_conv_global(1.00E-04)tau_conv_local(5.00E-04)tau_fc_global(1.00E-04)tau_fc_local(5.00E-04)_E0200_01:20:18:234706_e0191.pkl'
		# filename = 'MNIST_LeNet5DoubleFlatten_HalfCauchy-LogNormal-tau_conv_global(1.00E-04)tau_conv_local(1.00E-04)tau_fc_global(1.00E-04)tau_fc_local(1.00E-04)_E0200_01:20:21:680255_e0194.pkl'
		# filename = 'MNIST_LeNet5DoubleFlatten_HalfCauchy-LogNormal-tau_conv_global(1.00E-04)tau_conv_local(5.00E-03)tau_fc_global(1.00E-04)tau_fc_local(5.00E-03)_E0200_01:20:27:171604_e0199.pkl'
		# filename = 'MNIST_LeNet5DoubleFlatten_HalfCauchy-LogNormal-tau_conv_global(1.00E-04)tau_conv_local(1.00E-02)tau_fc_global(1.00E-04)tau_fc_local(1.00E-02)_E0200_12:08:27:167291_e0197.pkl'
		# filename = 'MNIST_LeNet5DoubleFlatten_HalfCauchy-LogNormal-tau_conv_global(1.00E-04)tau_conv_local(5.00E-02)tau_fc_global(1.00E-04)tau_fc_local(5.00E-02)_E0200_12:08:12:007496_e0191.pkl'
		# filename = 'MNIST_LeNet5DoubleFlatten_HalfCauchy-LogNormal-tau_conv_global(1.00E-04)tau_conv_local(1.00E-01)tau_fc_global(1.00E-04)tau_fc_local(1.00E-01)_E0200_16:30:05:309278_e0193.pkl'
		filename = 'MNIST_LeNet5DoubleFlatten_E0200_e0002.pkl'
	else:
		filename = sys.argv[1]
	row_threshold = [2.0, -8.0, -10.0, -9.0]
	col_threshold = [1.0, -8.0, -11.0, -9.6]
	model_file = os.path.join(exp_dir(), os.path.split(filename)[1])
	dirname, filename = os.path.split(model_file)
	model = load_model(model_type=model_type, prior_info=prior_info_from_json('HalfCauchy.json'), use_gpu=False)
	model.load_state_dict(torch.load(model_file))
	train_loader, valid_loader, test_loader, train_loader_eval = load_data(data_type=data_type, batch_size=100, num_workers=0, use_gpu=False)
	evaluate_with_prunning(model, train_loader_eval, valid_loader, test_loader, row_threshold=row_threshold, col_threshold=col_threshold, tag=filename)
