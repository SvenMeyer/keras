# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import math
import time

from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn import datasets

# Fix for bug in keras 0.12.1
# http://stackoverflow.com/questions/41796618/python-keras-cross-val-score-error/41841066#41841066
from keras.wrappers.scikit_learn import BaseWrapper
import copy
def custom_get_params(self, **params):
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res
BaseWrapper.get_params = custom_get_params

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

import os
home = os.path.expanduser("~")
dir = "/MLdata/test/"
filename = "pima-indians-diabetes.csv"
datafile = home + dir + filename

print("loading file : ", datafile),
dataset = np.loadtxt(datafile, delimiter=",")
print("DONE")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

X_rows, X_cols = X.shape
print(X)

result_log = [] # empty list to log processing results

start = time.clock()
pca = decomposition.PCA()
pca.fit(X)
X_pca = pca.transform(X)
time_pca = (time.clock() - start)
print("time PCA = ", time_pca)

for n_components in range(1, X_cols+1):
    print("********** n_components = ", n_components)

    # create model 2*n_components
    model = Sequential()
    model.add(Dense(256, input_dim=n_components, init='uniform', activation='relu'))
    model.add(Dense(256, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # Compile model
    print("start model.compile ..."),
    start = time.clock()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    time_compile = (time.clock() - start)
    print("DONE in ", time_compile, "sec")

    # Fit the model
    print("start model.fit ..."),
    start = time.clock()
    model.fit(X_pca[:,:n_components], Y, nb_epoch=150, batch_size=10, verbose=0)
    time_fit = (time.clock() - start)
    print("DONE in ", time_fit, "sec")

    # evaluate the model
    scores = model.evaluate(X_pca[:,:n_components], Y)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # calculate predictions
    predictions = model.predict(X_pca[:,:n_components])
    # round predictions
    rounded = [np.around(x) for x in predictions]
    np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
    print()
    print(rounded)

    result_log.append((n_components, time_compile, time_fit, scores[1]*100))

# print("-----------")
# for r in result_log:
#    print(r)

print("-----------")

result_df = pd.DataFrame(result_log, columns=['n_components', 'time_compile', 'time_fit', 'score'])
print("result_df :")
print(result_df)

# tf cpu ~ 23 s
# th cpu ~  3.6 s (~1 core)

# 256 | 256
# tf cpu ~   23 s
# tf GPU ~   61 s
# th cpu ~  137 - 95 s (~ all core)
# th GPU ~   20 - 14 s (~ all core)