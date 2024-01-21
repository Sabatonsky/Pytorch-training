# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 18:03:03 2023

@author: Maksim Bannikov
"""

import torch
import torch.nn as nn
import numpy as np

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)

loss = nn.CrossEntropyLoss()
Y = torch.tensor([0])
# N x D array
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l_good = loss(Y_pred_good, Y)
l_bad = loss(Y_pred_bad, Y)

print("good prediction loss:", l_good.item())
print("bad prediction loss:", l_bad.item())

_, prediction_good = torch.max(Y_pred_good, 1)
_, prediction_bad = torch.max(Y_pred_bad, 1)

print(prediction_good.item())
print(prediction_bad.item())
