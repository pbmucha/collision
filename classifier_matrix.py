import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F

def symmetric(X):
    return X.triu() + X.triu(1).transpose(-1, -2)

#a simple NN controller
class Net(nn.Module):
    name = "Net"
    def __init__(self):
        super().__init__()
        self.dim = 22
        self.N = 10
        self.fc = nn.ModuleList([nn.Linear(self.dim, self.dim, bias = False) for i in range(self.N)])

    def forward(self, x):
        xs = [torch.sum(self.fc[i](x) * x, dim = -1) for i in range(self.N)]
        r = torch.vstack(xs)
        r = torch.transpose(r, 0, 1)
        return r

class LinearSymmetric(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(n_features, n_features))

    def forward(self, x):
        A = symmetric(self.weight)
        return x @ A

#a simple NN controller
class NetSym(nn.Module):
    name = "Net"
    def __init__(self):
        super().__init__()
        self.dim = 22
        self.N = 10
        self.fc = nn.ModuleList([LinearSymmetric(self.dim ) for i in range(self.N)])

    def forward(self, x):
        xs = [torch.sum(self.fc[i](x) * x, dim = -1) for i in range(self.N)]
        r = torch.vstack(xs)
        r = torch.transpose(r, 0, 1)
        return r



#loads data
X = np.loadtxt('data/10/XXX-10.csv', delimiter = ',')
N = X.shape[0]
x = np.reshape(X, (N,10, 2))
mean = np.mean(x, axis = 1)
newX = np.concatenate((X, mean), axis = 1)

D = np.zeros((N, 9))
for c in range(N):
    d = [ np.linalg.norm(x[c, 0, :] - x[c, i, :]) for i in range(1,10)]
    D[c, :] = d

print(D[0])

Y_logits = np.loadtxt('data/10/CCC-10.csv', delimiter = ',')[:,0].astype(int)

print(X[0])
print(Y_logits[0])
Class = 10

print(f"total #exampes={N}")

Y_onehot = np.eye(Class)[Y_logits]
print(Y_onehot[0])

#test some classifiers using the 0-th particle communication vector

X_train, X_test, y_train, y_test = train_test_split(newX, Y_onehot, stratify=Y_onehot, random_state=1)

net = NetSym()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()

for i in range(3000):
    optimizer.zero_grad()
    out = net(torch.FloatTensor(X_train))
    l = loss(out, torch.FloatTensor(y_train))
    l.backward()
    optimizer.step()
    print(l.item())

net.eval()
out_test = net( torch.FloatTensor(X_test) )
print(out_test.shape)
y_labels = torch.argmax(F.softmax(out_test, dim=1).int(), dim=1)

correct = (y_labels == torch.argmax(torch.IntTensor(y_test), dim=1).int()).sum().item()
print( correct, y_test.shape[0], correct / y_test.shape[0])

for i in range(net.N):
    np.savetxt(f'sym_matrix{i}.csv', symmetric(net.fc[i].weight).detach().numpy(), delimiter=',' )