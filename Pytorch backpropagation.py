# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:48:45 2023

@author: Maksim Bannikov
"""

import torch
 
x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

y_hat = w * x
loss = (y_hat - y)**2

print(loss.item())

loss.backward()
print(w.grad.item())


