# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:10:53 2023

@author: Maksim Bannikov
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784
hidden_size = 100

