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

filesn = ['XXX-10.csv', 'XXX-10-23wrz-A.csv', 'XXX-10-23wrz-B.csv']
dirn = 'data/10/'
xs = []
N = 10
T = 20
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
fileCn = ['CCC-10.csv', 'CCC-10-23wrz-A.csv', 'CCC-10-23wrz-B.csv']
for fn in fileCn:
    Y = np.loadtxt(dirn + fn, delimiter = ',')[:,fixedI].astype(int)
    ys.append( Y )

Y_logits = np.concatenate(ys, axis = 0)

#convention here is that i-th row is the i-th particle communication
increments = np.zeros_like(newX)
for i in range(trajs):    
    increments[i*20 : (i+1)*20 - 1] = newX[i*20 + 1 : (i+1)*20] - newX[i*20 : (i+1)*20 - 1]

tildeX = newX.copy()

for i in range(trajs):    
    tildeX[i*20 : (i+1)*20 - 1, fixedI, : ] = newX[i*20 + 1 : (i+1)*20, fixedI, : ]

todelete = []
for i in range(trajs):    
    todelete.append((i+1)*20 - 1)

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

#test if there are nonzero increments where shouldn't
print(np.where(increments_group[0][:, 0][0] != 0.) )
print(np.where(increments_group[0][:, 0][1] != 0.) )


currentg = 0
print("groups")
print(X_group[currentg][0])
print(tildeX_group[currentg][0])
print(Labels_group[currentg][0])

#do not take into account the last position (marking end of trajectory)
X_train, X_test, y_train, y_test = train_test_split(X_group[currentg], Labels_group[currentg], random_state=1)

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss = nn.MSELoss()

for i in range(10000):
    optimizer.zero_grad()
    out = net(torch.FloatTensor(X_train))
    l = loss(out, torch.squeeze(torch.FloatTensor(y_train)))
    l.backward()
    optimizer.step()
    print(l.item())

net.eval()
out_test = net( torch.FloatTensor(X_test) )

np.savetxt(f'sym_matrix{currentg}.csv', net.fc.weight.detach().numpy(), delimiter=',' )