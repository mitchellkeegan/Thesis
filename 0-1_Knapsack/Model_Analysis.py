import numpy as np
import matplotlib.pyplot as plt

from ML_Model_Defs import RF_model, NN_model, ANN
from Opt_Model_Defs import constraint_metrics, objective_metrics

directories = {'Instance Type': 'Weights_and_Values_and_Capacity',
               'problem': '0-1 Knapsack',
               'Opt Model': 'Vanilla'}

ml_params = {'lr': 1e-3,
             'Epochs': 1500,
             'Training Batch Size': 256,
             'Lagrange Step': 1,
             'k Round': 25,
             'Constraints': 'LDF',
             'Batch Norm': True,
             'Dropout': False,
             'Grid Search': True,
             'LM Step Scheduler': None}

model = NN_model(ml_params, directories, ANN)
model.load_data('Vanilla')
# model.fit(preload=False)
model.load_model()
# model.predict('Test')

Cap_2_L1 = model.model.fc1.weight[:,-1].detach().numpy()

w1_2_L1 = model.model.fc1.weight[:,0].detach().numpy()
v1_2_L1 = model.model.fc1.weight[:,50].detach().numpy()

w2_2_L1 = model.model.fc1.weight[:,1].detach().numpy()

plt.figure()
# plt.hist(Cap2L1)
# plt.plot(range(len(Cap_2_L1)),Cap_2_L1)
plt.plot(range(len(w1_2_L1)),w1_2_L1,range(len(w2_2_L1)),w2_2_L1)
plt.show()
print(5)