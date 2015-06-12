from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l1l2
import cPickle as pickle
from misc import load_data, preprocess_labels, preprocess_data
import os
import paths


np.random.seed(1337)  # for reproducibility


def model_factory(n_classes, n_dims):
    print("Building model...")

    lmbd1 = 0
    lmbd2 = 0

    model = Sequential()
    model.add(Dense(n_dims, 1024, init='glorot_uniform',
                    W_regularizer=l1l2(lmbd1, lmbd2)))
    model.add(PReLU((1024,)))
    model.add(BatchNormalization((1024,)))
    model.add(Dropout(0.5))

    model.add(Dense(1024, 512, init='glorot_uniform',
                    W_regularizer=l1l2(lmbd1, lmbd2)))
    model.add(PReLU((512,)))
    model.add(BatchNormalization((512,)))
    model.add(Dropout(0.5))

    model.add(Dense(512, 256, init='glorot_uniform',
                    W_regularizer=l1l2(lmbd1, lmbd2)))
    model.add(PReLU((256,)))
    model.add(BatchNormalization((256,)))
    model.add(Dropout(0.5))

    model.add(Dense(256, n_classes, init='glorot_uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam")

    return model


model_name = 'dump_keras_ensemble.0.5.0.5.0.5'
n_models = 20

print("Loading data...")
X, labels = load_data(paths.train_file, train=True)
y = preprocess_labels(labels)
X_test, ids = load_data(paths.test_file, train=False)
X = preprocess_data(X)
X_test = preprocess_data(X_test)

n_classes = y.shape[1]
n_dims = X.shape[1]

print("Training %d models..." % n_models)

proba = 0
models = range(1, n_models+1)
for i in models:
    print("\n-------------- Model %d --------------\n" % i)
    model = model_factory(n_classes, n_dims)
    model.fit(X, y, nb_epoch=120, batch_size=128, validation_split=0.0,
              show_accuracy=True, verbose=2)
    proba += model.predict_proba(X_test, verbose=2)
    with open(os.path.join(paths.model_path, model_name+'.pkl'), 'wb') as f:
        pickle.dump((proba, i), f, protocol=pickle.HIGHEST_PROTOCOL)
