import os

from BayesianNeuralNetwork.data_loaders import DATA_DIR

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


def data_loader(batch_size=32, num_workers=1, use_gpu=False, validation=True):
    # normalization_coef = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    normalization_coef = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.Pad(padding=4, padding_mode='reflect'),
                                          transforms.RandomCrop(size=(32, 32)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(*normalization_coef)])
    transform_eval = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(*normalization_coef)])

    if validation:
        n_train = 45000
        train_set = torch.utils.data.Subset(torchvision.datasets.CIFAR10(root=os.path.join(DATA_DIR, 'CIFAR10'), train=True, download=True, transform=transform_train), xrange(n_train))
        train_eval_set = torch.utils.data.Subset(torchvision.datasets.CIFAR10(root=os.path.join(DATA_DIR, 'CIFAR10'), train=True, download=True, transform=transform_eval), xrange(n_train))
        valid_set = torch.utils.data.Subset(torchvision.datasets.CIFAR10(root=os.path.join(DATA_DIR, 'CIFAR10'), train=True, download=True, transform=transform_eval), xrange(n_train, 50000))
        test_set = torchvision.datasets.CIFAR10(root=os.path.join(DATA_DIR, 'CIFAR10'), train=False, download=True, transform=transform_eval)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_gpu)
        train_loader_eval = torch.utils.data.DataLoader(train_eval_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)
    else:
        train_set = torchvision.datasets.CIFAR10(root=os.path.join(DATA_DIR, 'CIFAR10'), train=True, download=True, transform=transform_train)
        train_eval_set = torchvision.datasets.CIFAR10(root=os.path.join(DATA_DIR, 'CIFAR10'), train=True, download=True, transform=transform_eval)
        test_set = torchvision.datasets.CIFAR10(root=os.path.join(DATA_DIR, 'CIFAR10'), train=False, download=True, transform=transform_eval)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_gpu)
        train_loader_eval = torch.utils.data.DataLoader(train_eval_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)
        valid_loader = None
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)

    return train_loader, valid_loader, test_loader, train_loader_eval


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    train_loader, valid_loader, test_loader, train_loader_eval = data_loader(100)
    for input_data, output_data in test_loader:
        n_data = input_data.size(0)
        for i in range(n_data):
            img = input_data[i].permute([1, 2, 0]).numpy()
            plt.figure(figsize=(1.0, 1.2))
            plt.imshow(img)
            plt.title(cifar10_classes[output_data[i]])
            plt.show()
