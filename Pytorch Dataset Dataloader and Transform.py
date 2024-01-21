# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:49:51 2023

@author: Maksim Bannikov
"""

import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader

#Dataset by hand with transform function
class MNISTDataset(Dataset):
    def __init__(self, transform=None):
        path = r"D:\Training_code\mnist_train.csv"
        xy = np.loadtxt(path, delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:,1:]
        self.y = xy[:,[0]]
        self.n_samples = xy.shape[0]
        self.transform = transform
        
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
        
    def __len__(self):
        return self.n_samples
    
#Transform by hand
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
class MulTransform:
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets
        
#Composed transform example
#composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])

dataset = MNISTDataset(transform=ToTensor())

#Dataset from preinstalled MNIST with transform function. ToTensor for example get Tensor from Numpy array.
dataset = torchvision.datasets.MNIST(root='./data', transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset=dataset, batch_size=1000, shuffle=True)
        