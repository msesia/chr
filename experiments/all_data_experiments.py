
from data_experiment import run_experiment

import os

if os.path.isdir('/scratch'):
    local_machine = 0
else:
    local_machine = 1


if local_machine:
    base_dataset_path = '/Users/romano/mydata/regression_data/'
else:
    base_dataset_path = '/scratch/users/yromano/data/regression_data/'



alpha = 0.1
test_size = 2000

random_state = 2020

N_CALIB_LIST   = [2000]

DATASET_LIST =  ['meps_19',
                 'meps_20',
                 'meps_21',
                 'facebook_1',
                 'facebook_2',
                 'bio',
                 'blog_data']

BBOX_LIST      = ["NNet", "RF"]

# Where to write results
out_dir = "./results"

for EXP_id in range(1):
    for N_CALIB_LIST_id in range(1):
       for DATASET_LIST_id in range(1):
           for BBOX_LIST_id in range(1):
               
               dataset_name = DATASET_LIST[DATASET_LIST_id]
               n_calib = N_CALIB_LIST[N_CALIB_LIST_id]
               bbox_method = BBOX_LIST[BBOX_LIST_id]
               run_experiment(base_dataset_path,
                              dataset_name,
                              test_size,
                              n_calib,
                              alpha=alpha,
                              experiment=EXP_id,
                              bbox_method=bbox_method,
                              out_dir = out_dir,
                              random_state=random_state)
  

               