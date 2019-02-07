from BayesianNeuralNetwork.data_loaders import DATA_DIR

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


def data_loader(batch_size=32, num_workers=4):
    normalization_coef = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # given in pytorch example
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*normalization_coef)])

    trainset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # For unsupervised learning, it would be better use train+test, for now we only use trainset
    testset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader