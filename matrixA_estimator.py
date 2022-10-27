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
class Net(nn.Module):
    name = "Net"
    def __init__(self):
        super().__init__()
        self.dim = 20
        self.fc = nn.Linear(self.dim, self.dim, bias = False)

    def forward(self, x):
        x = torch.sum(self.fc(x) * x, dim=-1)
        return x

def L(x):
    return np.sum([ np.sum((c - x) ** 2) for c in x ])




filesn = ['XXX-10-7paz-A.csv', 'XXX-10-7paz-B.csv', 'XXX-10-7paz-c.csv', 'XXX-10-7paz-d.csv' ]

dirn = 'data/10/'
xs = []
N = 10
T = 40
fixedI = 0

for fn in filesn:
    #loads data
    X = np.loadtxt(dirn + fn, delimiter = ',')
    xs.append( np.reshape(X, (X.shape[0], 10, 2)) )

newX = np.concatenate(xs, axis = 0)
print(newX.shape)
trajs = int(newX.shape[0] / T)
print(trajs)




#labels for the first particle
ys = []
fileCn = ['CCC-10-7paz-A.csv', 'CCC-10-7paz-B.csv', 'CCC-10-7paz-c.csv', 'CCC-10-7paz-d.csv' ]
for fn in fileCn:
    Y = np.loadtxt(dirn + fn, delimiter = ',')[:,fixedI].astype(int)
    ys.append( Y )

Y_logits = np.concatenate(ys, axis = 0)

#convention here is that i-th row is the i-th particle communication
increments = np.zeros_like(newX)
for i in range(trajs):    
    increments[i*T : (i+1)*T - 1] = newX[i*T + 1 : (i+1)*T] - newX[i*T : (i+1)*T - 1]

tildeX = newX.copy()

for i in range(trajs):    
    tildeX[i*T : (i+1)*T - 1, fixedI, : ] = newX[i*T + 1 : (i+1)*T, fixedI, : ]

todelete = []
for i in range(trajs):    
    todelete.append((i+1)*TAB0 - 1)

increments = np.delete(increments, todelete, axis = 0)
newX = np.delete(newX, todelete, axis = 0)
Y_logits = np.delete(Y_logits, todelete, axis = 0)
tildeX = np.delete(tildeX, todelete, axis = 0)

Labels = np.vstack( [ L(newX[i]) - L(tildeX[i]) for i in range(newX.shape[0]) ] )
print(newX[0])
print(L(newX[0]))
print(tildeX[0])
print(L(tildeX[0]))

#group rows by c particle
inds = []
X_group = []
increments_group = []
Labels_group = []
tildeX_group = []
for i in range(N):
    curind = np.where(Y_logits == i)[0]
    print(f"group particle {i} len {len(curind)}")
    inds.append( curind )
    X_group.append( newX[ curind ].reshape((len(curind), 2*N)) )
    tildeX_group.append( tildeX[ curind ].reshape((len(curind), 2*N)) )
    increments_group.append( increments[ curind ].reshape((len(curind), 2*N)) )
    Labels_group.append( Labels[ curind ] )


"""
print(X_group[1][0])
print(tildeX_group[1][0])
print(L(tildeX_group[1][0]) - L (X_group[1][0]))

def pochodna(X):
    print([(X[0] - x) for x in X])
    print([(X[0] - x)*(X[0] - X[1]) for x in X])
    return np.sum( [(X[0] - x)*(X[0] - X[1]) for x in X] )

print(X_group[1][0].reshape((10, 2)))
print(pochodna(X_group[1][0].reshape((10, 2))))
dt = 0.01
print((-4)*dt/N * pochodna(X_group[1][0].reshape((10, 2))))

exit(1)
"""

#test if there are nonzero increments where shouldn't
print(np.where(increments_group[0][:, 0][0] != 0.) )
print(np.where(increments_group[0][:, 0][1] != 0.) )


currentg = 4
print("groups")
print(X_group[currentg][0])
print(tildeX_group[currentg][0])
print(Labels_group[currentg][0])

#do not take into account the last position (marking end of trajectory)
X_train, X_test, y_train, y_test = train_test_split(X_group[currentg], Labels_group[currentg], random_state=1)

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss = nn.MSELoss()

for i in range(110000):
    optimizer.zero_grad()
    out = net(torch.FloatTensor(X_train))
    l = loss(out, torch.squeeze(torch.FloatTensor(y_train)))
    l.backward()
    optimizer.step()
    print(l.item())

net.eval()
out_test = net( torch.FloatTensor(X_test) )

sym = net.fc.weight.detach().numpy() + net.fc.weight.detach().numpy().T
np.savetxt(f'A{fixedI}{currentg}.csv', sym, delimiter=',' ,  fmt="%.2e")