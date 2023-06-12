from ML_Model_Defs import RF_model, NN_model, ANN
from Opt_Model_Defs import constraint_metrics, objective_metrics

directories = {'Instance Type': 'Weights_and_Values_and_Capacity',
               'problem': '0-1 Knapsack',
               'Opt Model': 'Vanilla'}

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


# ml_params = {'lr': 1e-4,
#              'Epochs': 1000,
#              'Training Batch Size': 128,
#              'Constraints': 'LDF',
#              'Batch Norm': True,
#              'Dropout': False,
#              'Grid Search': False}
#
# model = NN_model(ml_params, directories, ANN)
# model.load_data('Vanilla')
# # model.fit(preload=False)
# model.load_model()
# model.predict('Test')
# model.eval_prediction(obj=objective_metrics,constraints=constraint_metrics())

def LM_Scheduler(epoch,AR,s):
    if AR < 1.15:
        return 0.1
    elif AR < 1.5:
        return 0.01
    else:
        return s

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


for lr in [1e-3,1e-4]:
    ml_params['lr'] = lr
    for batch_size in [256]:
        ml_params['Training Batch Size'] = batch_size
        for ls in [0.01,0.001]:
            ml_params['Lagrange Step'] = ls
            directories['Hyperparameters'] = f'LR: {lr}, BS: {batch_size}, LS: {ls}'

            model = NN_model(ml_params,directories,ANN)
            model.load_data('Vanilla')
            model.fit(preload=False)
            model.predict('Val')
            # model.eval_prediction(obj=objective_metrics,constraints=constraint_metrics())