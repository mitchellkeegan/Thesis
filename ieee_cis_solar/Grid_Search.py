import os
import sys

from ML_Model_Defs import NN_model
from ML_Models.M2 import classScheduleNN
import yaml

file_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(file_dir))

with open(os.path.join(file_dir, 'ml_configs/model_config.yaml'), "r") as f: model_params = yaml.safe_load(f)
with open(os.path.join(file_dir, 'ml_configs/training_config.yaml'), "r") as f: training_params = yaml.safe_load(f)
with open(os.path.join(file_dir, 'ml_configs/directories.yaml'), "r") as f: directories = yaml.safe_load(f)

if 'Base Directory' not in directories:
    directories['Base Directory'] = file_dir

if 'Grid Search' not in training_params:
    training_params['Grid Search'] = True

directory_number = 1

# for lr in [1e-3,1e-4]:
#     training_params['lr'] = lr
#     directories['Hyperparameters'] = f'Trial {directory_number} - LR: {lr}'
#     directory_number += 1
#
#     model = NN_model(model_params, training_params, directories, classScheduleNN)
#     model.load_data()
#     model.fit()

for lr in [1e-3,1e-4]:
    training_params['lr'] = lr
    for gn in [1,10]:
        training_params['Max Grad Norm'] = gn
        for ls in [1e-2,1e-3,1e-4]:
            training_params['Lagrange Step'] = ls
            directories['Hyperparameters'] = f'Trial {directory_number} - LR: {lr}, GN: {gn}, LS: {ls}'
            directory_number += 1

            model = NN_model(model_params, training_params, directories, classScheduleNN)
            model.load_data()
            model.fit('base_model.pt')