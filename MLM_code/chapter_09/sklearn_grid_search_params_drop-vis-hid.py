# MLP for Pima Indians Dataset with grid search via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy
import time

# Fix for bug in keras 0.12.1
# http://stackoverflow.com/questions/41796618/python-keras-cross-val-score-error/41841066#41841066
from keras.wrappers.scikit_learn import BaseWrapper
import copy
def custom_get_params(self, **params):
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res
BaseWrapper.get_params = custom_get_params

# Function to create model, required for KerasClassifier
# def create_model(optimizer='rmsprop', init='glorot_uniform'):
    # dropout in hidden layers with weight constraint
def create_model():
    # create model
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(60,)))
    model.add(Dense(60, input_dim=60, init='normal', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(30, init='normal', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
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
epochs = [50, 100, 150]
batches = [5, 10, 20]
param_grid = dict(optimizer=optimizers, nb_epoch=epochs) #, batch_size=batches) # , init=init) # ValueError: init is not a legal parameter
grid = GridSearchCV(estimator=model, param_grid=param_grid)

print("start model.fit ..."),
start = time.clock()

grid_result = grid.fit(X, Y)

time_fit = (time.clock() - start)
print("DONE in ", time_fit, "sec")


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))

# 8 - 12
# th GPU : DONE in  1327.5 sec
# Best: 0.751302 using {'nb_epoch': 150, 'init': 'normal', 'optimizer': 'rmsprop', 'batch_size': 5}

# 256 - 256
# DONE in  1257.4560079999999 sec
# Best: 0.726563 using {'init': 'uniform', 'nb_epoch': 100, 'batch_size': 5, 'optimizer': 'adam'}
