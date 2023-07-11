#!/home/mitch/anaconda3/envs/Thesis/bin/python

import sys
import os

cwd = os.getcwd()

# Allows the lib package to be imported when running this file from the command line
if os.path.split(cwd)[1] != 'Thesis':
    sys.path.append(os.path.dirname(cwd))

print(sys.path)

from Opt_Model_Defs import vanilla_IP
import yaml

with open('config.yaml',"r") as f:
    opt_params = yaml.safe_load(f)

# Threads parameter only applied for large instances
# opt_params = {'instance index': 0,
#               'threads': 1,
#               'MIPGap': 0,
#               'TimeLimit': 600,
#               'problem': '0-1 Knapsack',
#               'instance folder': 'Weights_and_Values_and_Capacity'}

print(opt_params)

# model = vanilla_IP(opt_params)
# model.solve_all_instances()
# for instance in range(model.available_instances):
#     model.setup_and_optimize(instance)
#     model.save_model()
# model.save_model()
# model.plot_results()