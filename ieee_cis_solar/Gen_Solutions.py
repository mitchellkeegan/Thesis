#!/home/mitch/anaconda3/envs/Thesis/bin/python

import sys
import os

file_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(file_dir))

from Opt_Model_Defs import column_gen
import yaml

with open(os.path.join(file_dir,'config.yaml'),"r") as f:
    opt_params = yaml.safe_load(f)

if 'base directory' not in opt_params:
    opt_params['base directory'] = file_dir

model = column_gen(opt_params)
model.solve_all_instances()