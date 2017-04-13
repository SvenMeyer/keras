# MLP for Pima Indians Dataset with grid search via sklearn
import numpy

# processor = 'cpu'
processor = 'gpu'

# Theano stuff here
'''
import theano.sandbox.cuda
theano.sandbox.cuda.use(processor)

import theano
print("theano version     = ", theano.__version__)

from theano import function, config, shared
import theano.tensor as T
vlen = 10 * 30 * 768
rng = numpy.random.RandomState(22)
x = theano.shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = theano.function([], T.exp(x))
print(f.maker.fgraph.toposort())
'''

import tensorflow as tf
print("tensorflow.__version__ = ", tf.__version__)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

import time

import keras
print("keras version      = ", keras.__version__)

# print("tensorflow version = ", tensorflow.__version__)

# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
	model.add(Dense(8, kernel_initializer=init, activation='relu'))
	model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# grid search epochs, batch size and optimizer
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs  = [50, 100, 150]
batches = [10, 20, 30]
param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)

n_jobs = 8 if processor == 'cpu' else 1
print("processor = ", processor)
print("n_jobs    = ", n_jobs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=n_jobs)

print("start model.fit ..."),
start = time.time()
grid_result = grid.fit(X, Y)
time_fit = (time.time() - start)
print("DONE in ", time_fit, "sec")


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))

# Theano CPU

# epochs = [50, 100, 150]
# Best: 0.751302 using {'nb_epoch': 150, 'init': 'normal', 'batch_size': 5, 'optimizer': 'rmsprop'}
# Best: 0.748698 using {'nb_epoch': 150, 'optimizer': 'adam', 'init': 'normal', 'batch_size': 5}
# DONE in  90.60715198516846 sec - CPU - n_jobs=8
# Best: 0.752604 using {'batch_size': 5, 'optimizer': 'rmsprop', 'nb_epoch': 150, 'init': 'normal'}


# epochs = [50, 100, 150, 200, 250, 300]
# DONE in  246.56513023376465 sec
# Best: 0.759115 using {'nb_epoch': 300, 'batch_size': 20, 'init': 'normal', 'optimizer': 'adam'}

# epochs = [400, 500, 600]
# batches = [10, 20, 30]
# DONE in  181.38080048561096 sec
# Best: 0.761719 using {'nb_epoch': 600, 'batch_size': 10, 'optimizer': 'adam', 'init': 'uniform'}
# DONE in  183.35850596427917 sec
# Best: 0.765625 using {'optimizer': 'adam', 'batch_size': 30, 'nb_epoch': 600, 'init': 'uniform'}

# keras 1.2.2
# DONE in  182.50592589378357 sec
# Best: 0.761719 using {'batch_size': 10, 'init': 'uniform', 'nb_epoch': 600, 'optimizer': 'adam'}

# GPU Tensorflow 0.12.1
# epochs = [50]
# batches = [10]
# epochs = [100]
# batches = [5]
# DONE in  506.75493478775024 sec
# Best: 0.739583 using {'batch_size': 5, 'nb_epoch': 100, 'optimizer': 'adam', 'init': 'normal'}

# GPU Tensorflow 1.0.0rc1
# DONE in  500.74833512306213 sec
# Best: 0.739583 using {'optimizer': 'adam', 'nb_epoch': 100, 'init': 'normal', 'batch_size': 5}

# GPU Tensorflow 1.0.0
# DONE in  493.63156175613403 sec
# Best: 0.739583 using {'init': 'normal', 'optimizer': 'adam', 'nb_epoch': 100, 'batch_size': 5}