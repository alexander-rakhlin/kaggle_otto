import os


root = 'E:/kaggle/Otto'

# DATA
data_path = os.path.join(root, 'Data')
train_file = os.path.join(data_path, 'train.csv')
test_file = os.path.join(data_path, 'test.csv')

# MODELS
model_path = os.path.join(root, 'results')
ensemble = (('dump_keras_ensemble.0.5.0.5.0.5', 4),
            ('dump_xgboost_ensemble', 4))

# SUBMISSION
submission = os.path.join(model_path, 'ensemble_submission.4-4.csv')
