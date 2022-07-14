import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

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


clf = MLPClassifier(hidden_layer_sizes=(25,), max_iter=1000, random_state=1, verbose = 1).fit(X_train, y_train)

print(clf.predict_proba(X_test[:1]))
print(clf.predict(X_test[:5, :]))
print(clf.score(X_test, y_test))

np.savetxt('coefs_[0].csv', clf.coefs_[0], delimiter=',')
np.savetxt('intercepts_[0].csv', clf.intercepts_[0], delimiter=',')
np.savetxt('coefs_[1].csv', clf.coefs_[1], delimiter=',')
np.savetxt('intercepts_[1].csv', clf.intercepts_[1], delimiter=',')

"""
feature_names = []
for i in range(int(D.shape[1])):
    feature_names.append(f"d{i+1}")
print(feature_names)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print(tree.score(X_test, y_test))


forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)
print(forest.score(X_test, y_test))

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()
"""
