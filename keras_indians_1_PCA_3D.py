# Create first network with Keras

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import math
import time

from sklearn import decomposition

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

import os
home = os.path.expanduser("~")
dir_path = "/ML_DATA/MLM/pima-indians/"
filename = "pima-indians-diabetes.csv"
datafile = home + dir_path + filename

print("loading file : ", datafile),
dataset = np.loadtxt(datafile, delimiter=",")
print("DONE")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

X_rows, X_cols = X.shape
print(X)

result_log = [] # empty list to log processing results

start = time.time()
pca = decomposition.PCA()
pca.fit(X)
X_pca = pca.transform(X)
time_pca = (time.time() - start)
print("time PCA = ", time_pca)


# draw first 3 components in 3D


y = Y
y = Y.astype(np.int)
# print("y = ", y)

centers = [[1, 1], [-1, -1], [1, -1]]

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

for name, label in [('PCA_0', 0), ('PCA_1', 1)]: #, ('PCA_2', 2)]:
    print(X_pca[y == label, 0].mean())

for name, label in [('value_0', 0), ('value_1', 1)]: # number of classes
    ax.text3D(X_pca[y == label, 0].mean(),
              X_pca[y == label, 1].mean() + 1.5,
              X_pca[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

# Reorder the labels to have colors matching the cluster results
# y = np.choose(y, [1, 2, 0]).astype(np.float)
y = np.choose(y, [1, 0]).astype(np.float)
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap=plt.cm.spectral)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()
