import cPickle as pickle
from misc import make_submission
import os
import paths


probs = 0
weights = 0
for e, w in paths.ensemble:
    with open(os.path.join(paths.model_path, e+'.pkl'), 'rb') as f:
        probs += w * pickle.load(f)
    weights += w

probs /= weights

print("Generating submission...")
make_submission(probs, paths.submission)
