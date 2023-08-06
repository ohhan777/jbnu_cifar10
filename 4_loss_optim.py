import os
import argparse
import torch
import torchvision
from models.vgg import vgg11_bn
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

def train(opt):
    epochs = opt.epochs
    batch_size = opt.batch_size
    name = opt.name
    # Train dataset
    transforms = torchvision.transforms.ToTensor()
    train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, 
                                                    transform=transforms)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Train dataloader
    num_workers = min([os.cpu_count(), batch_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers, drop_last=True)

    # Network model
    model = vgg11_bn()

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    pass # debug checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='target epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--name', default='ohhan', help='name for the run')

    opt = parser.parse_args()

    train(opt)
