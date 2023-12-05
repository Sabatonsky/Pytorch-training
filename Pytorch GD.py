# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:48:45 2023

@author: Maksim Bannikov
"""

import torch
from torch import nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
#model = nn.Linear(n_features, n_features)

class LinearRegression(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.lin(x)
    
model = LinearRegression(n_features, 1)

lr = 0.01
n_iters = 100

#Loss function 
loss = nn.MSELoss()

#Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_iters):
    #Prediction = forward pass
    Y_hat = model(X)
    
    #Loss 
    l = loss(Y, Y_hat)
    
    #Gradients = backward pass
    l.backward()
    
    #Update weights
    optimizer.step()
    
    #Zero gradients
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
    
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')