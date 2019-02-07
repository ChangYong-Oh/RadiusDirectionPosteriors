import os
from BayesianNeuralNetwork.data_loaders import DATA_DIR

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


def data_loader(batch_size=32, num_workers=4, use_gpu=False):
    normalization_coef = ((0.1307,), (0.3081,)) # given in pytorch example
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*normalization_coef)])

    train_valid_set = torchvision.datasets.FashionMNIST(root=os.path.join(DATA_DIR, 'FashionMNIST'), train=True, download=True, transform=transform)
    train_set = torch.utils.data.Subset(train_valid_set, xrange(50000))
    valid_set = torch.utils.data.Subset(train_valid_set, xrange(50000, 60000))
    test_set = torchvision.datasets.FashionMNIST(root=os.path.join(DATA_DIR, 'FashionMNIST'), train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_gpu)
    train_loader_eval = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)

    return train_loader, valid_loader, test_loader, train_loader_eval