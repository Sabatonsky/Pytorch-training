# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:48:45 2023

@author: Maksim Bannikov
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

X_np, Y_np = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_np.astype(np.float32))
Y = torch.from_numpy(Y_np.astype(np.float32))
Y = Y.view(Y.shape[0], 1)
N, D = X.shape

model = nn.Linear(D, 1)

lr = 0.01
n_iters = 200

#Loss function 
loss = nn.MSELoss()

#Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_iters):
    #Forward
    Y_hat = model(X)
    #Loss
    l = loss(Y_hat, Y)
    #Backprop
    l.backward()
    #Update W
    optimizer.step()
    #Clear grad
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
    
predicted = model(X).detach().numpy()
plt.plot(X_np, Y_np, 'ro')
plt.plot(X_np, predicted, 'b')
plt.show()