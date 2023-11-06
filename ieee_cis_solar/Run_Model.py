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
    training_params['Grid Search'] = False

model = NN_model(model_params, training_params, directories, classScheduleNN)
model.load_data()
# model.load_model('base_model.pt')

# model.load_model('model_params_best_loss.pt')
# model.load_model('model_params_final.pt')
# model.load_model('model_params_best_AR.pt')
model.load_model('model_params_best_1-normed_loss.pt')
# model.load_model('model_params_best_10-normed_loss.pt')
pred, pred_on = model.predict('test')
model.eval_prediction(pred,pred_on)