from ML_Model_Defs import RF_model, NN_model, ANN
from Opt_Model_Defs import constraint_metrics, objective_metrics

# ml_params = {'Number Trees': 100,
#              'Max Depth': 50,
#              'Max Features': None}
#
# model = RF_model(ml_params,directories)
# model.load_data('Vanilla')
# model.fit()
# # model.load_model()
# model.predict('Test')
# model.eval_prediction(obj=objective_metrics,constraints=constraint_metrics())

directories = {'Instance Type': 'Weights_and_Values_and_Capacity',
               'problem': '0-1 Knapsack',
               'Opt Model': 'Vanilla'}

model_params = {'Batch Norm': True,
                'Dropout': False}

training_params = {'lr': 1e-4,
                   'Epochs': 1500,
                   'Training Batch Size': 256,
                   'Lagrange Step': 0.001,
                   'k Round': 25,
                   'Constraints': 'LDF',
                   'Clip Grad Norm': True,
                   'Max Grad Norm': 1,
                   'LM Step Scheduler': None,
                   'LM Update Interleave': 0}


model = NN_model(model_params, training_params, directories, ANN)
model.load_data_no_split()
model.load_model('model_params_best_AR.pt')
# model.load_model('model_params_best_loss.pt')
# model.load_model('model_params_best_0.5-normed_loss.pt')
# model.load_model('model_params_final.pt')
# model.predict('Test',idx_subset=('Quintile Splits',4))
# model.predict('Test',idx_subset=('Quintile Splits',4))
model.predict('Test')
# model.eval_prediction(obj=objective_metrics, constraints=constraint_metrics())