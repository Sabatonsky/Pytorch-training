# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:18:58 2023

@author: Maksim Bannikov
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

bc = datasets.load_breast_cancer()
X, Y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

Y_test = Y_test.view(Y_test.shape[0], 1)
Y_train = Y_train.view(Y_train.shape[0], 1)

class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    
model = LogisticRegression(n_features)

lr = 0.01
n_iters = 1000

loss = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
for epoch in range(n_iters):
    P_train = model(X_train)
    l = loss(P_train, Y_train)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.3f}')

with torch.no_grad():
    P_train = model(X_train)
    P_test = model(X_test)
    P_train_cls = P_train.round()
    P_test_cls = P_test.round()    
    acc_train = P_train_cls.eq(Y_train).sum() / float(Y_train.shape[0])
    acc_test = P_test_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    
print(f"Loss on train set: {loss(P_train, Y_train).item():.3f}")
print(f"Loss on test set: {loss(P_test, Y_test).item():.3f}")
print(f"Accuracy on train set: {acc_train:.2f}")
print(f"Accuracy on test set: {acc_test:.2f}")

