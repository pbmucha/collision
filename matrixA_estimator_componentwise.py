from termios import TAB0
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F

#a simple NN controller
class ComponentwiseANet(nn.Module):
    name = "Net"
    def __init__(self):
        super().__init__()
        self.dim = 10
        self.fc1 = nn.Linear(self.dim, self.dim, bias = False)
        self.fc2 = nn.Linear(self.dim, self.dim, bias = False)

    def forward(self, x0, x1):
        r0 = torch.sum(self.fc1(x0) * x0, dim=-1)
        r1 = torch.sum(self.fc2(x1) * x1, dim=-1)
        return r0 + r1

def L(x):
    return np.sum([ np.sum((c - x) ** 2) for c in x ])



filesn0 = ['X0-24paz-10-100k.csv']
filesn1 = ['X1-24paz-10-100k.csv']
dirn = 'data/10/'
x0s = []
x1s = []
N = 10
T = 1
fixedI = 0
component = 2

for fn in filesn0:
    #loads data
    X0 = np.loadtxt(dirn + fn, delimiter = ',')
    x0s.append( np.reshape(X0, (X0.shape[0], N, 1)) )
for fn in filesn1:    
    X1 = np.loadtxt(dirn + fn, delimiter = ',')
    x1s.append( np.reshape(X1, (X1.shape[0], N, 1)) )

newX0 = np.concatenate(x0s, axis = 0)
newX1 = np.concatenate(x1s, axis = 0)
print(newX0.shape, newX1.shape)

#labels for the first particle
ys = []
fileCn = [ 'CCC-24paz-10-100k.csv' ]
for fn in fileCn:
    Y = np.loadtxt(dirn + fn, delimiter = ',').astype(int)
    ys.append( Y )

Y_logits = np.concatenate(ys, axis = 0)

#load labels
fileLn = ['LLL-24paz-10-100k.cvs']
ls = []
for fn in fileLn:
    L = np.loadtxt(dirn + fn, delimiter = ',')
    ls.append(L)

Labels = np.concatenate(ls, axis = 0)

print(newX0[0])
print(newX1[0])
print(Labels[0])

#group rows by c particle
inds = []
X0_group = []
X1_group = []
increments_group = []
Labels_group = []
tildeX_group = []
for i in range(N):
    curind = np.where(Y_logits == i)[0]
    print(f"group particle {i} len {len(curind)}")
    inds.append( curind )
    X0_group.append( newX0[ curind ].reshape((len(curind), N)) )
    X1_group.append( newX1[ curind ].reshape((len(curind), N)) )
    Labels_group.append( Labels[ curind ] )

print(X0_group[0].shape, X1_group[0].shape, Labels_group[0].shape)

currentg = 2

print("groups")
print(X0_group[currentg][0])
print(X1_group[currentg][0])
print(Labels_group[currentg][0])

N = X0_group[currentg].shape[0]

Ntrain = int(0.9 * N)

X0_train = torch.FloatTensor(X0_group[currentg])[:Ntrain]
X1_train = torch.FloatTensor(X1_group[currentg])[:Ntrain]
Y_train = Labels_group[currentg][:Ntrain]

X0_test = torch.FloatTensor(X0_group[currentg])[Ntrain:]
X1_test = torch.FloatTensor(X1_group[currentg])[Ntrain:]
Y_test = Labels_group[currentg][Ntrain:]


net = ComponentwiseANet()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss = nn.MSELoss()

for i in range(20000):
    optimizer.zero_grad()
    out = net( X0_train, X1_train )
    l = loss(out, torch.squeeze(torch.FloatTensor(Y_train)))
    l.backward()
    optimizer.step()
    print(l.item())

net.eval()

r = net(X0_test, X1_test)

print(r[:10], Y_test[:10])

print( np.sum( (r.detach().numpy() - Y_test) ** 2 ) / (N-Ntrain) )

np.savetxt(f'A_0_{fixedI}{currentg}.csv', net.fc1.weight.detach().numpy() + net.fc1.weight.detach().numpy().T, delimiter=',' )
np.savetxt(f'A_1_{fixedI}{currentg}.csv', net.fc2.weight.detach().numpy() + net.fc2.weight.detach().numpy().T, delimiter=',' )