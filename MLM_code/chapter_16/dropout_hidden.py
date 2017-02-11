# Example of Dropout on the Sonar Dataset: Hidden Layer
import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = read_csv("sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

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

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, nb_epoch=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

print("start model.fit ..."),
start = time.clock()
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
time_fit = (time.clock() - start)
print("DONE in ", time_fit, "sec")

print("Hidden: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# GPU Theano
# DONE in  76.34 sec
# Hidden: 85.07% (7.05%)

# GPU Theano - CUDA 8.0.6? (10. Feb 2017 update)
# DONE in  79.9 sec
# Hidden: 82.64% (7.28%)
#
# DONE in  80.056522 sec
# Hidden: 82.64% (7.28%)

# GPU tensorflow
# DONE in  265.87 sec
# Hidden: 84.04% (6.75%)

# GPU tensorflow - CUDA 8.0.6? (10. Feb 2017 update)
# DONE in  407.981268 sec
# results.mean: 84.09% , results.std: 6.66%