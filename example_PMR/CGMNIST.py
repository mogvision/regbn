import copy
import csv
import os
import tarfile
import shutil
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets



class CGMNISTDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])
            self.gray_dataset = datasets.MNIST(root='data/Mnist/mnist/', train=True, download=True,
                                               transform=transform)

        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])
            self.gray_dataset = datasets.MNIST(root='data/Mnist/mnist/', train=False, download=True,
                                               transform=transform)

        # download color minist data from https://drive.google.com/file/d/11K-GmFD5cg3_KTtyBRkj9VBEnHl-hx_Q/view
        # and then locate it at `data/Mnist/colored_mnist`
        data_dic = np.load('data/Mnist/colored_mnist/mnist_10color_jitter_var_%.03f.npy' % 0.020,
                           encoding='latin1', allow_pickle=True).item()

        if self.mode == 'train':
            self.colored_image = data_dic['train_image']
            self.colored_label = data_dic['train_label']
        elif self.mode == 'test':
            self.colored_image = data_dic['test_image']
            self.colored_label = data_dic['test_label']

        self.T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        self.ToPIL = transforms.Compose([
            transforms.ToPILImage(),
        ])


    def __len__(self):
        return len(self.gray_dataset)

    def __getitem__(self, idx):
        gray_image, gray_label = self.gray_dataset[idx]

        colored_label = self.colored_label[idx]
        colored_image = self.colored_image[idx]

        colored_image = self.ToPIL(colored_image)

        return gray_image, self.T(colored_image), gray_label

