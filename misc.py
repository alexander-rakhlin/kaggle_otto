from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np
import pandas as pd
import zipfile
import os


def make_submission(probs, path='my_submission.csv'):
    classes = ['Class_'+str(i+1) for i in range(9)]
    with open(path, 'w') as f:
        f.write('id,' + ','.join(classes) + '\n')
        for id, p in enumerate(probs):
            probas = ','.join([str(id+1)] + map(str, p)) + '\n'
            f.write(probas)
    with zipfile.ZipFile(path+'.zip', 'w', zipfile.ZIP_DEFLATED) as f:
        f.write(path, arcname=os.path.basename(path))
    os.remove(path)
    print("Wrote submission to file {}".format(path+'.zip'))


def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X)
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids


# http://stats.stackexchange.com/questions/46418/why-is-the-square-root-transformation-recommended-for-count-data
# http://en.wikipedia.org/wiki/Anscombe_transform
def preprocess_data(X):
    return np.sqrt(X+3.0/8.0)


def preprocess_labels(labels, categorical=True):
    encoder = LabelEncoder()
    encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y