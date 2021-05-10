import os
from os import path
import sys
import torch
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, '..')

from chr.black_boxes import QNet, QRF
from chr.methods import CHR
from chr.others import CQR, CQR2, DistSplit, DCP
from chr.utils import evaluate_predictions

from dataset import GetDataset

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import pickle
import pdb

# Default arguments
alpha = 0.1
test_size = 2000
random_state = 2020
n_cal = 2000
base_dataset_path = '../../data/'
n_jobs = 1
verbose = False
out_dir = './results_real'
tmp_dir = './tmp_real'


# Input arguments
dataset_name = str(sys.argv[1])
bbox_method = str(sys.argv[2])
experiment = int(sys.argv[3])

print(dataset_name)
print(bbox_method)
print(experiment)


# Set random seed
np.random.seed(random_state)
# Random state for this experiment
random_state = 2020 + experiment

X, Y = GetDataset(dataset_name, base_dataset_path)

# Add noise to response
Y += 1e-6*np.random.normal(size=Y.shape)

# if transform_y:
    # Y = np.log(1 + Y - min(Y))

y_min = min(Y)
y_max = max(Y)

if X.shape[0] <= 2*n_cal + test_size:
    raise

# Temporary file to store black-box model
tmp_file = tmp_dir + "/dataset_" + dataset_name + "_bbox_" + bbox_method + "_exp_" + str(int(experiment)) + ".pk"

# Determine output file
out_file = out_dir + "/dataset_" + dataset_name + "_bbox_" + bbox_method + "_exp_" + str(int(experiment)) + ".txt"

print(out_file)

# Load output file if it exists
if path.exists(out_file):
    results = pd.read_csv(out_file)
else:
    results = pd.DataFrame()


# Set random seed
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_state)


# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
                                                       random_state=random_state)

X_train, X_calib, Y_train, Y_calib = train_test_split(X_train, Y_train, test_size=n_cal,
                                                               random_state=random_state)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_calib = scaler.transform(X_calib)
X_test = scaler.transform(X_test)


n_train = X_train.shape[0]
assert(n_cal == X_calib.shape[0])
n_test = X_test.shape[0]

if len(X.shape) == 1:
    n_features = 1
else:
    n_features = X.shape[1]

# Initialize the black-box and the conformalizer
if bbox_method == 'NNet':

    epochs = 2000
    lr = 0.0005
    batch_size = n_train
    dropout = 0.1

    grid_quantiles = np.arange(0.01,1.0,0.01)
    bbox = QNet(grid_quantiles, n_features, no_crossing=True, batch_size=batch_size,
                            dropout=dropout, num_epochs=epochs, learning_rate=lr, calibrate=1,
                            verbose=verbose)
elif bbox_method == 'RF':
    n_estimators = 100
    min_samples_leaf=50
#        min_samples_leaf = 20
    #max_features = x_train.shape[1]
    grid_quantiles = np.arange(0.01,1.0,0.01)
    bbox = QRF(grid_quantiles, n_estimators=n_estimators,
                 min_samples_leaf=min_samples_leaf, random_state=2020, n_jobs=n_jobs, verbose=verbose)


# Train the black-box model
#if os.path.exists(tmp_file):
#    print("Loading black box model...")
#    filehandler = open(tmp_file, 'rb')
#    bbox = pickle.load(filehandler)
#else:
print("Training black box model...")
bbox.fit(X_train, Y_train)
filehandler = open(tmp_file, 'wb')
pickle.dump(bbox, filehandler)


# Define list of methods to use in experiments
methods = {
   'CHR'         : CHR(bbox, ymin=y_min, ymax=y_max, y_steps=1000, randomize=True),
   'DistSplit'   : DistSplit(bbox, ymin=y_min, ymax=y_max),
   'DCP'         : DCP(bbox, ymin=y_min, ymax=y_max),
   'CQR'         : CQR(bbox),
   'CQR2'        : CQR2(bbox)
  }

for method_name in methods:
    # Skip methods that we have already run
    if method_name in np.array(results['Method']):
        print("Found results for {}. Skipping method.".format(method_name))
        continue

    # Apply the conformalization method
    method = methods[method_name]
    method.calibrate(X_calib, Y_calib, alpha)
    # Compute prediction on test data
    pred = method.predict(X_test)

    # Evaluate results
    res = evaluate_predictions(pred, Y_test, X=X_test)
    # Add information about this experiment
    res['Box'] = bbox_method
    res['Dataset'] = dataset_name
    res['Method'] = method_name
    res['Experiment'] = experiment
    res['Nominal'] = 1-alpha
    res['n_train'] = n_train
    res['n_cal'] = n_cal
    res['n_test'] = n_test

    # Add results to the list
    results = results.append(res)

    # Write results on output files
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    results.to_csv(out_file, index=False, float_format="%.4f")
    print("Updated summary of results on\n {}".format(out_file))
    sys.stdout.flush()

#pdb.set_trace()
