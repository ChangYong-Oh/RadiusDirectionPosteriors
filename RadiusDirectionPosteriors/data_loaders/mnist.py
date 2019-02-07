import os

from BayesianNeuralNetwork.data_loaders import DATA_DIR

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


def data_loader(batch_size=32, num_workers=1, use_gpu=False, validation=True):
    # normalization_coef = ((0.0,), (126.0 / 255.0,)) # for some old MNIST compression experiments
    normalization_coef = ((0.5,), (0.5,))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*normalization_coef)])

    if validation:
        train_valid_set = torchvision.datasets.MNIST(root=os.path.join(DATA_DIR, 'MNIST'), train=True, download=True, transform=transform)
        train_set = torch.utils.data.Subset(train_valid_set, xrange(50000))
        valid_set = torch.utils.data.Subset(train_valid_set, xrange(50000, 60000))
        test_set = torchvision.datasets.MNIST(root=os.path.join(DATA_DIR, 'MNIST'), train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_gpu)
        train_loader_eval = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)
    else:
        train_set = torchvision.datasets.MNIST(root=os.path.join(DATA_DIR, 'MNIST'), train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=os.path.join(DATA_DIR, 'MNIST'), train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_gpu)
        train_loader_eval = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)

    return train_loader, valid_loader, test_loader, train_loader_eval


if __name__ == '__main__':
    train_loader, valid_loader, test_loader, train_loader_eval = data_loader(32, 0)
    input_max = -100
    input_min = 100
    for inputs, outputs in test_loader:
        input_batch_max = torch.max(inputs)
        input_batch_min = torch.min(inputs)
        if input_batch_max > input_max:
            input_max = input_batch_max
        if input_batch_min < input_min:
            input_min = input_batch_min
    print(input_min)
    print(input_max)
    print('%.2E ~ %.2E' % (input_min, input_max))