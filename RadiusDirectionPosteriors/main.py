import math
import argparse

from BayesianNeuralNetwork.torch_user.nn.utils import softplus_inv
from BayesianNeuralNetwork.compress import train_continue, train_initiate, prior_info_from_json


INIT_HYPER = {'vMF': {'direction': 'kaiming',
                      'softplus_inv_concentration_normal_mean_via_epsilon': 0.1,
                      'softplus_inv_concentration_normal_std': 0.0001},
              'LogNormal': {'mu_normal_mean': None,
                            'mu_normal_std': 0.0001,
                            'softplus_inv_std_normal_mean': softplus_inv(0.0001),
                            'softplus_inv_std_normal_std': 0.0001},
              'Normal': {'mu_normal_mean': 0.0,
                         'mu_normal_std': 0.0001,
                         'softplus_inv_std_normal_mean': softplus_inv(0.0001),
                         'softplus_inv_std_normal_std': 0.0001}
              }


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='MNIST Train script')
	parser.add_argument('--model_type', dest='model_type', type=str, help='  / \n'.join(['LeNetFC', 'LeNet5-Flatten', 'LeNet5-Bundle', 'VGG16-Flatten', 'VGG16-Bundle', 'VGG16-BN-Flatten', 'VGG16-BN-Bundle']))
	parser.add_argument('--prior_file', dest='prior_file', type=str)
	parser.add_argument('--epochs', dest='epochs', type=int)
	parser.add_argument('--batch_size', dest='batch_size', default=100, type=int, help='Default(100)')
	parser.add_argument('--lr', dest='lr', type=float, default=0.001)
	parser.add_argument('--gpu', dest='gpu', action='store_true', default=False,
						help='Use gpu if available. Default(False)')
	parser.add_argument('--model_file', dest='model_file', help='Continue training')
	parser.add_argument('--load_pretrain', dest='load_pretrain', default=None,
	                    help='Load a pretrained model')

	args = parser.parse_args()
	arg_dict = vars(args)
	print(arg_dict)
	if 'LeNet' in args.model_type:
		assert not args.load_pretrain
		data_type = 'MNIST'
		INIT_HYPER['LogNormal']['mu_normal_mean'] = math.log(2.0 ** (1.0 / 1.0))
	elif 'VGG' in args.model_type:
		data_type = 'CIFAR10'
		if 'BN' is args.model_type:
			INIT_HYPER['LogNormal']['mu_normal_mean'] = math.log(2.0 ** (1.0 / 1.0))
		else:
			INIT_HYPER['LogNormal']['mu_normal_mean'] = math.log(2.0 ** (1.0 / 2.0))
		INIT_HYPER['vMF']['softplus_inv_concentration_normal_mean_via_epsilon'] = 0.025
	else:
		raise NotImplementedError

	if args.model_file is not None:
		exp_filename = train_continue(data_type=data_type, model_type=args.model_type, model_filename=args.model_file, n_epoch=args.epochs, lr=args.lr, num_workers=1, use_gpu=args.gpu)
		print(exp_filename)
	else:
		exp_filename = train_initiate(data_type=data_type, model_type=args.model_type, init_hyper=INIT_HYPER, prior_info=prior_info_from_json(args.prior_file), n_epoch=args.epochs, lr=args.lr, batch_size=args.batch_size, num_workers=1, use_gpu=args.gpu, load_pretrain=args.load_pretrain)
		print(exp_filename)
