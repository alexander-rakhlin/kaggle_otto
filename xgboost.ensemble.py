from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import xgboost as xgb
import cPickle as pickle
from misc import load_data, preprocess_labels, preprocess_data
import paths
import os

np.random.seed(1337)  # for reproducibility

mode = 'submission'  # crossvalidate|submission

model_name = 'dump_xgboost_ensemble'

print("Loading data...")
X_train, labels = load_data(paths.train_file, train=True)
y_train = preprocess_labels(labels, categorical=False)
X_train = preprocess_data(X_train)

dtrain = xgb.DMatrix(X_train, label=y_train)

n_classes = max(y_train) + 1
n_round = 332

param = {'objective': 'multi:softprob',
         'eval_metric': 'mlogloss',
         'eval.metric': 'merror',
         'num_class': n_classes,
         'max_depth': 16,
         'eta': 0.05,
         'sub_sample': 0.9,
         'colsample_bytree': 0.8,
         'min_child_weight': 4}

if mode == 'crossvalidate':
# cross validate
    print("Cross validating...")
    xgb.cv(param, dtrain, n_round, nfold=3, seed=0,
           metrics={'merror', 'mlogloss'}, show_stdv=False)
else:
# full training and submission
    print('Full training...')
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(param, dtrain, n_round, watchlist)

    # predict
    print('Predicting...')
    X_test, ids = load_data(paths.test_file, train=False)
    X_test = preprocess_data(X_test)
    dtest = xgb.DMatrix(X_test)
    probs = bst.predict(dtest)
    with open(os.path.join(paths.model_path, model_name+'.pkl'), 'wb') as f:
                pickle.dump(probs, f, protocol=pickle.HIGHEST_PROTOCOL)
