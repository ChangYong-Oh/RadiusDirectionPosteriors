import os
import numpy as np

import torch
import torch.utils.data

DATA_DIR = '/is/ei/com/git_repositories/DropoutUncertaintyExps/UCI_Datasets'
# dirname, n_in, n_hidden
BOSTON = os.path.join(DATA_DIR, 'bostonHousing'), 13, 50
CONCRETE = os.path.join(DATA_DIR, 'concrete'), 8, 50
ENERGY = os.path.join(DATA_DIR, 'energy'), 8, 50
KIN8NM = os.path.join(DATA_DIR, 'kin8nm'), 8, 50
NAVAL = os.path.join(DATA_DIR, 'naval-propulsion-plant'), 16, 50
POWERPLANT = os.path.join(DATA_DIR, 'power-plant'), 4, 50
PROTEIN = os.path.join(DATA_DIR, 'protein-tertiary-structure'), 9, 100
WINE = os.path.join(DATA_DIR, 'wine-quality-red'), 11, 50
YACHT = os.path.join(DATA_DIR, 'yacht'), 6, 50
YEAR = os.path.join(DATA_DIR, 'year-prediction-msd'), 90, 100


def read_from_dir(data_info, split_id):
    data_dirname = os.path.join(data_info[0], 'data')
    index_features = np.genfromtxt(os.path.join(data_dirname, 'index_features.txt')).astype(np.int)
    index_target = np.genfromtxt(os.path.join(data_dirname, 'index_target.txt')).astype(np.int)
    assert np.intersect1d(index_features, index_target).size == 0
    data = np.genfromtxt(os.path.join(data_dirname, 'data.txt'))
    index_train = np.genfromtxt(os.path.join(data_dirname, 'index_train_' + str(split_id) + '.txt')).astype(np.int)
    index_test = np.genfromtxt(os.path.join(data_dirname, 'index_test_' + str(split_id) + '.txt')).astype(np.int)
    assert np.intersect1d(index_train, index_test).size == 0
    train_input = torch.from_numpy(data[index_train][:, index_features]).type(torch.float32)
    train_output = torch.from_numpy(data[index_train][:, index_target]).type(torch.float32)
    test_input = torch.from_numpy(data[index_test][:, index_features]).type(torch.float32)
    test_output = torch.from_numpy(data[index_test][:, index_target]).type(torch.float32)
    return train_input, train_output, test_input, test_output


def architecture_info(data_type):
    data_type = globals()[data_type]
    return data_type[1:]


def data_loader(data_type, split_id, batch_size=32, num_workers=4, output_normalize=True):
    data_type = globals()[data_type]
    train_input, train_output, test_input, test_output = read_from_dir(data_type, split_id)
    input_mean = torch.mean(train_input, dim=0, keepdim=True)
    input_std = torch.std(train_input, dim=0, keepdim=True)
    input_std[input_std == 0] = 1
    train_input = (train_input - input_mean) / input_std
    test_input = (test_input - input_mean) / input_std
    if output_normalize:
        output_mean = torch.mean(train_output, dim=0, keepdim=True)
        output_std = torch.std(train_output, dim=0, keepdim=True)
        output_std[output_std == 1] = 1
        train_output = (train_output - output_mean) / output_std
        test_output = (test_output - output_mean) / output_std
    else:
        output_mean = 0
        output_std = 1
    train_data = torch.utils.data.TensorDataset(train_input, train_output.view(-1, 1))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_data = torch.utils.data.TensorDataset(test_input, test_output.view(-1, 1))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False, num_workers=num_workers)
    train_loader_eval = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=False, num_workers=num_workers)

    normalization_info = {'input mean': input_mean, 'input std': input_std, 'output mean': output_mean, 'output std': output_std}

    return train_loader, test_loader, train_loader_eval, normalization_info


if __name__ == '__main__':
    train_loader, test_loader, normalization_info = data_loader('KIN8NM', split_id=1, batch_size=32, num_workers=4, output_normalize=True)
    print(normalization_info['output mean'])
    print(normalization_info['output std'])