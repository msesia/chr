

import os
import sys
import torch
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, '.')

from chr.black_boxes import QNet, QRF
from chr.methods import CHR
from chr.others import CQR, CQR2, DistSplit, DCP
from chr.utils import evaluate_predictions

from dataset import GetDataset

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# transform_y = False

def run_experiment(base_dataset_path,
                   dataset_name,
                   test_size,
                   n_cal,
                   alpha=0.1,
                   experiment=0,
                   bbox_method='NNet',
                   out_dir = './results',
                   random_state=2020,
                   n_jobs=1,
                   verbose = False):
    
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
        return
    
    # Determine output file
    out_file = out_dir + "/summary.csv"
    print(out_file)
    
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
    
    results = pd.DataFrame()
    results_full = pd.DataFrame()
    
    if len(X.shape) == 1:
        n_features = 1
    else:
        n_features = X.shape[1]
        
    y_step = 1000 # maybe larger than 1000
    
    # Initialize the black-box and the conformalizer
    if bbox_method == 'NNet':

        epochs =  2000
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
    bbox.fit(X_train, Y_train)
    
    # Define list of methods to use in experiments
    methods = {
       'CHR'         : CHR(bbox, ymin=y_min, ymax=y_max, y_steps=1000, randomize=True),
       'DistSplit'   : DistSplit(bbox, ymin=y_min, ymax=y_max),
       'DCP'         : DCP(bbox, ymin=y_min, ymax=y_max),
       'CQR'         : CQR(bbox),
       'CQR2'        : CQR2(bbox)
      }

    for method_name in methods:
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

    return results

