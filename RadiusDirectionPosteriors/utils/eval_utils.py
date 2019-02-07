import inspect

import torch


def evaluation(model, data_loader, eval_func, use_gpu):
    """

    :param model:
    :param data_loader:
    :param use_gpu:
    :param eval_func: value averaged over batch
    :return:
    """
    model_status = model.training
    model.eval()
    eval_score = 0
    n_eval_data = 0
    for i, data in enumerate(data_loader, 0):
        batch_input, batch_output = data
        if use_gpu:
            batch_input = batch_input.cuda().reshape(batch_input.size(0), -1)
            batch_output = batch_output.cuda()
        batch_pred = model(batch_input)
        n_eval_data += batch_input.size(0)
        eval_score += float(eval_func(batch_pred, batch_output)) * batch_input.size(0)

    if model_status:
        model.train()
    return eval_score / n_eval_data


def accuracy(softmax_vec, output):
    return torch.mean((torch.argmax(softmax_vec, dim=1) == output).float())


def classification_result_str(model, train_loader, test_loader, loss_module, use_gpu):
    model.eval()
    train_loss = evaluation(model=model, data_loader=train_loader, eval_func=loss_module, use_gpu=use_gpu)
    test_loss = evaluation(model=model, data_loader=test_loader, eval_func=loss_module, use_gpu=use_gpu)
    train_acc = evaluation(model=model, data_loader=train_loader, eval_func=accuracy, use_gpu=use_gpu)
    test_acc = evaluation(model=model, data_loader=test_loader, eval_func=accuracy, use_gpu=use_gpu)
    info_str = '         Loss     | Accuracy\n'
    info_str += 'Train :  %8.6f | %8.6f\n' % (train_loss, train_acc)
    info_str += 'Test  :  %8.6f | %8.6f' % (test_loss, test_acc)
    return info_str


def generator_config(txt_file):
    info_file = open(txt_file, 'rt')
    info = {}
    for elm in info_file.read().split('\n'):
        key, value = elm.strip().split(' : ')
        try:#check it is int?
            value = int(value)
        except ValueError:
            try:#check it is float?
                value = float(value)
            except ValueError:#it is string
                value = value
        info[key] = value
    return info


def generator_from_file(filename):
    generator_architecture_info = generator_config(filename + '.txt')
    generator_parameters = torch.load(filename + '.pkl')
    needed_args = inspect.getargspec(model_and_objective).args
    args_dict = {key: value for key, value in generator_architecture_info.iteritems() if key in needed_args}
    args_dict['n_gpu'] = 1
    generator = model_and_objective(**args_dict)[1]
    generator.load_state_dict(generator_parameters)
    return generator, args_dict['n_latent']