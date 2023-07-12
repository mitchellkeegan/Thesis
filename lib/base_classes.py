from abc import ABC, abstractmethod
import os

from lib.custom_metrics import Approximation_Ratio

import lightgbm.sklearn
from gurobipy import Model
import pickle
from collections import defaultdict
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# class MyHeavyside(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,input,k):
#         ctx.save_for_backward(input)
#         ctx.k = k
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         (input,) = ctx.saved_tensors
#         k = ctx.k
#         grad_input = grad_output.clone()
#         return k * torch.exp(-k * (input - 0.5)) / (torch.exp(-k * (input - 0.5)) + 1) ** 2 * grad_input, None

# TODO Move this somewhere else
class MyRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        ctx.save_for_backward(input)
        ctx.k = k
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        k = ctx.k
        grad_input = grad_output.clone()
        return k*torch.exp(-k*(input-0.5))/(torch.exp(-k*(input-0.5))+1)**2 * grad_input, None

class base_opt_model(ABC):
    def __init__(self, opt_params):
        self.opt_params = opt_params
        self.base_dir = os.path.join('/home/mitch/Documents/Thesis Data/', self.opt_params['problem'])
        self.instance_dir = os.path.join(self.base_dir,'Instances',opt_params['instance folder'])

    # Overwrite as needed (E.g. for small vs large instances)
    def create_results_directory(self):
        self.results_directory = os.path.join(self.base_dir,
                                              'Opt Results',
                                              self.model_type,
                                              self.opt_params["instance folder"],
                                              str(self.opt_params["instance index"]))
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

    def optimize_model(self):
        self.model.optimize()

    def create_model(self):
        self.model = Model()
        if 'MIPGap' in self.opt_params:
            self.model.Params.MIPGap = self.opt_params['MIPGap']
        if 'TimeLimit' in self.opt_params:
            self.model.Params.TimeLimit = self.opt_params['TimeLimit']

    # Overwrite as needed (E.g. for small vs large instances)
    def solve_all_instances(self):
        instance_gen_info_file = os.path.join(self.instance_dir, 'Instance Generation Info.txt')

        if os.path.exists(instance_gen_info_file):
            with open(instance_gen_info_file, 'r') as f:
                self.available_instances = int(f.readline()[:-1])

            for instance in range(self.available_instances):
                self.setup_and_optimize(instance)
                self.save_model()
        else:
            print('No "Instance Generation Info.txt" file found.')


    def setup_and_optimize(self,index=None):
        self.create_model()
        self.load_instance(index)
        self.load_problem_specific_data()
        self.add_vars()
        self.add_objective()
        self.add_constraints()
        self.optimize_model()

    def load_problem_specific_data(self):
        # Not always required, can be used, for example, to construct sets which will be looped over in constraints
        pass

    def save_model_matrix(self):
        A = self.model.getA()

        sense = np.array(self.model.getAttr("Sense", self.model.getConstrs()))
        b = np.array(self.model.getAttr("RHS", self.model.getConstrs()))

        Aeq = A[sense == '=', :]
        Ale = A[sense == '<', :]
        Age = A[sense == '>', :]

        beq = b[sense == '=']
        ble = b[sense == '<']
        bge = b[sense == '>']

        if Aeq.shape[0] > 0:
            scipy.sparse.save_npz(os.path.join(self.results_directory,'Aeq.npz'),Aeq)

        if Ale.shape[0] > 0:
            scipy.sparse.save_npz(os.path.join(self.results_directory,'Ale.npz'),Ale)

        if Age.shape[0] > 0:
            scipy.sparse.save_npz(os.path.join(self.results_directory,'Age.npz'),Age)



    def save_model(self):
        self.create_results_directory()
        self.save_model_output()
        self.save_model_matrix()

        # Store the optimisation parameters used
        with open(os.path.join(self.results_directory, 'opt_params.pickle'), 'wb') as f:
            pickle.dump(self.opt_params, f, protocol=pickle.HIGHEST_PROTOCOL)

        status_dict = defaultdict(lambda x: '???')
        status_dict[2] = 'Optimal Solution Found'
        status_dict[3] = 'Infeasible'
        status_dict[9] = 'Time Limit Reached'

        # Store the date and time the solution was generated
        with open(os.path.join(self.results_directory, 'Info.txt'), 'w') as f:
            f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            f.write(f'\nObjective - {self.model.objVal:.2f}\n')
            f.write(f'Solve Time - {self.model.runTime:.2f}s\n')
            f.write(f'Status - {self.model.Status} ({status_dict[self.model.Status]})\n')
            f.write(f'NumVars - {self.model.NumVars}\n')
            f.write(f'NumConstrs - {self.model.NumConstrs}\n')
            f.write(f'MIPGap - {100 * self.model.MIPGap:.3f}%\n')

        # Store the solution such that it can be loaded back into a Gurobi model
        self.model.write(os.path.join(self.results_directory, 'solution.sol'))

    @abstractmethod
    def save_model_output(self):
        # Call vars_to_readable and then write model output to file(s)
        pass



    @abstractmethod
    def vars_to_readable(self):
        # Convert model output to a readable format for saving
        pass

    @abstractmethod
    def load_instance(self,index=None):
        pass

    @abstractmethod
    def add_vars(self):
        pass

    @abstractmethod
    def add_constraints(self):
        pass

    @abstractmethod
    def add_objective(self):
        pass

class ml_model(ABC):
    def __init__(self, model_params,training_params,directories):
        self.model_params = model_params
        self.training_params = training_params
        self.base_dir = os.path.join('/home/mitch/Documents/Thesis Data/', directories['problem'])
        self.instance_dir = os.path.join(self.base_dir,'Instances',directories['Instance Type'])
        self.opt_result_dir = os.path.join(self.base_dir,'Opt Results', directories['Opt Model'], directories['Instance Type'])
        self.ml_result_dir = os.path.join(self.base_dir,'ML Results', self.model_type, directories['Instance Type'])
        self.ml_results_base_dir = self.ml_result_dir
        if 'Hyperparameters' in directories:
            self.hyperparameters = directories['Hyperparameters']
            self.ml_result_dir = os.path.join(self.ml_result_dir, directories['Hyperparameters'])


        if not os.path.exists(self.ml_result_dir):
            os.makedirs(self.ml_result_dir)

        with open(os.path.join(self.instance_dir,'Instance Generation Info.txt'),'r') as f:
            self.available_instances = int(f.readline()[:-1])

        # Load in a dictionary which stores which parameters of the instances vary between instance and which are fixed
        with open(os.path.join(self.instance_dir,'param_types.pickle'),'rb') as f:
            self.param_types = pickle.load(f)

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_data(self):
        # Should maybe implement this as a dataloader class which is passed to the instance?
        # Should define self.X/Y_train/test/val and retrieve the objval of the solution
        pass

    @abstractmethod
    def fit(self):
        # Fit the model to the data
        pass

    @abstractmethod
    def predict(self,):
        pass

    @abstractmethod
    def print_problem_metrics(self, X, Y, coeff):
        pass

    def eval_prediction(self, obj=None, constraints=None):

        X, Y, pred = self.pred

        # X = self.input_dict[self.pred_on]
        # Y = self.output_dict[self.pred_on]
        # coeff = self.create_coefficient_dict(X)

        print('-'*10 + f'\n\nEVALUATING PREDICTION ON {self.pred_on} DATA\n\n')

        coeff = self.create_coefficient_dict(self.unnormalise_inputs(X))

        print('-' * 10)
        # problem_metrics = self.print_problem_metrics(X,Y,coeff)
        problem_metrics = self.print_problem_metrics(pred,Y,coeff)
        print('-' * 10)

        if constraints is not None:
            eq_constraints = constraints.eqc(coeff,pred,[])
            ineq_constraints = constraints.ineqc(coeff, pred,[])
        else:
            eq_constraints = []
            ineq_constraints = []

        if obj is not None:
            obj_metrics = obj(coeff,pred,Y)
        else:
            obj_metrics = []

        with open(os.path.join(self.ml_result_dir,'Prediction Metrics.txt'),'w') as f:
            f.write('EVAULATED on ' + self.pred_on +  ' DATA\n\n')

            f.write('-' * 10 + '\n')
            for s in problem_metrics:
                f.write(s + '\n')
            f.write('-' * 10 + '\n')
            for s in eq_constraints:
                f.write(s + '\n')
            f.write('-' * 10 + '\n')
            for s in ineq_constraints:
                f.write(s + '\n')
            f.write('-' * 10 + '\n')
            for s in obj_metrics:
                f.write(s + '\n')
            f.write('-' * 10 + '\n')


    @abstractmethod
    def create_coefficient_dict(self,X):
        # Create a coefficient dictionary for the inputs X
        # X is a B x N matrix where each row represents the data for one instance

        pass

    # Needs to be overwritten for augmented lagrangian models
    # Inputs are the B x n solution matrix X and B x M parameter matrix d
    # If self.create_coefficient_dict is defined the user can use it to unpack data into a dictionary of problem data
    # Output is a B x m matrix where m is the number of constraints
    def inequality_constraints(self,X,data,ctype='violation'):
        return torch.empty((X.shape[0],0))

    # Needs to be overwritten for augmented lagrangian models
    # Inputs are the B x n solution matrix X and B x M parameter matrix d
    # If self.create_coefficient_dict is defined the user can use it to unpack d into a dictionary of problem data
    def equality_constraints(self,X,data,ctype='violation'):
        return torch.empty((X.shape[0],0))

    def normalise_inputs(self,X):
        X_out = X.copy()

        for param, (X_max,X_min) in self.norm_groups.items():
            idx_to_normalise = self.input_dict['PARAM IDX'][param]
            X_out[:,idx_to_normalise] = (X_out[:,idx_to_normalise] - X_min)/(X_max-X_min + 1e-12)

        return X_out

    def unnormalise_inputs(self,X):
        # TODO Maybe rewrite code such that this function only ever takes either numpy arrays or torch tensors
        if torch.is_tensor(X):
            X_out = X.detach().clone()
        elif isinstance(X,np.ndarray):
            X_out = X.copy()

        for param, (X_max,X_min) in self.norm_groups.items():
            idx_to_normalise = self.input_dict['PARAM IDX'][param]
            X_out[:,idx_to_normalise] = X_min + X_out[:,idx_to_normalise] * (X_max - X_min)

        return X_out

class random_forest(ml_model):
    def __init__(self, ml_params, directories, ml_model_type=None):
        if ml_model_type is None:
            self.model_type = 'Random Forest'
        else:
            self.model_type = ml_model_type

        super().__init__(ml_params,directories)

    def fit(self):
        max_features = self.ml_params.get('Max Features', 'sqrt')
        num_trees = self.ml_params.get('Number Trees', 100)
        max_depth = self.ml_params.get('Max Depth', None)

        self.model = MultiOutputClassifier(RandomForestClassifier(n_estimators=num_trees,
                                                                  max_depth=max_depth,
                                                                  max_features=max_features))

        X_train = self.input_dict['TRAIN']
        Y_train = self.output_dict['TRAIN']

        self.model.fit(X_train,
                       Y_train)

    def predict(self, input='Test',idx_subset=None):
        # TODO Update this so that it can itulise idx_subset
        input = input.upper()

        if input in ['TEST', 'TRAIN', 'VAL']:
            X = self.input_dict[input]
            Y = self.output_dict[input]

        self.pred = (X,Y,self.model.predict(X))
        self.pred_on = input

    def save_model(self):
        with open(os.path.join(self.ml_result_dir,'Random Forest.pickle'),'wb') as f:
            pickle.dump(self.model,f)

        with open(os.path.join(self.ml_result_dir,'Info.txt'),'w') as f:
            f.write('ML PARAMS:\n')
            for k,v in self.ml_params.items():
                f.write(k + ': ' + str(v) + '\n')

    def load_model(self):
        with open(os.path.join(self.ml_result_dir,'Random Forest.pickle'),'rb') as f:
            self.model = pickle.load(f)

class neural_network(ml_model):
    def __init__(self, model_params, training_params, directories, ml_model_type=None):
        if ml_model_type is None:
            self.model_type = 'Neural Network'
        else:
            self.model_type = ml_model_type

        super().__init__(model_params,training_params,directories)

    def save_model(self,filename='model_params.pt'):

        # TODO: Have this save the model parameters (features, layers, etc..) in a way that can be easily loaded back in

        torch.save(self.model.state_dict(), os.path.join(self.ml_result_dir,filename))

        with open(os.path.join(self.ml_result_dir, 'Info.txt'), 'w') as f:
            f.write('TRAINING PARAMS:\n')
            for k, v in self.training_params.items():
                if not callable(v):
                    f.write(k + ': ' + str(v) + '\n')
            f.write('\nMODEL PARAMS:\n')
            for k, v in self.model_params.items():
                if not callable(v):
                    f.write(k + ': ' + str(v) + '\n')
            f.write('\nMODEL DETAILS\n')
            for k,v in self.model.model_details.items():
                f.write(k + ': ' + str(v) + '\n')

    def load_model(self,filename='model_params.pt'):
        self.model = self.forward_model(self.n_features,self.n_out,self.model_params)
        self.model.load_state_dict(torch.load(os.path.join(self.ml_result_dir,filename)))

    def fit(self,preload=False):

        # Load in training parameters
        n_epochs = self.training_params.get('Epochs',100)
        lr = self.training_params.get('lr',1e-3)
        aug_lagrangian = self.training_params.get('Constraints','None')
        train_batch_size = self.training_params.get('Training Batch Size',256)
        s = self.training_params.get('Lagrange Step',1)
        lm = torch.tensor(self.training_params.get('Initial Lagrange Multiplier',[[1.]]))
        lm_scheduler = self.training_params.get('LM Step Scheduler',None)
        k = self.training_params.get('k Round',25)
        clip_grad_norm = self.training_params.get('Clip Grad Norm',False)
        max_grad_norm = self.training_params.get('Max Grad Norm', 1)
        grid_search = self.training_params.get('Grid Search', False)



        assert (self.input_dict is not None and self.output_dict is not None), 'Please load in training data before training Neural Network Model\n'

        # Load in training data, normalise it and construct dataloader
        X_train = self.input_dict['TRAIN']
        Y_train = self.output_dict['TRAIN']
        X_train = self.normalise_inputs(X_train)
        Train_Dataset = TensorDataset(torch.from_numpy(X_train).float(),
                                      torch.from_numpy(Y_train.copy()).float())
        Train_Dataloader = DataLoader(Train_Dataset, batch_size=train_batch_size, shuffle=True)

        if 'VAL' in self.input_dict:
            X_val = self.input_dict['VAL']
            Y_val = self.output_dict['VAL']
            X_val = self.normalise_inputs(X_val)
            Val_Dataset = TensorDataset(torch.from_numpy(X_val).float(),
                                        torch.from_numpy(Y_val).float())
            Val_Dataloader = DataLoader(Val_Dataset, batch_size=train_batch_size, shuffle=False)

        n_features = X_train.shape[1]
        n_out = Y_train.shape[1]

        # Set up the model and optimizer
        if preload:
            self.load_model()
        else:
            self.model = self.forward_model(n_features,n_out,self.model_params)
        optimizer = torch.optim.Adam(self.model.parameters(),lr,weight_decay=0)

        # Set up loss Binary Cross-Entropy Loss
        BCEloss = nn.BCEWithLogitsLoss(reduce=False)

        # Set up logging a file to log the training process
        training_log_file = os.path.join(self.ml_result_dir,'Training Log.txt')
        open(training_log_file,'w').close()

        best_validation_loss = float('inf')
        best_cnormed_validation_loss = float('inf')
        best_validation_AR = float('inf')
        train_loss_history = []
        val_loss_history = []
        epochs_since_improvement = 0
        epochs_since_lr_decreased = 0

        if grid_search:
            print(f'\nFITTING MODEL WITH {self.hyperparameters}\n')

        append_learning_rate_to_log = False

        for epoch in range(n_epochs):
            epoch_total_loss = 0
            epoch_constraint_loss = 0

            epoch_label_loss = 0
            self.model.train()
            for x_batch, y_batch in Train_Dataloader:
                optimizer.zero_grad()
                output = self.model(x_batch)
                label_loss_individual = BCEloss(output,y_batch)
                # label_hook = label_loss_individual.register_hook(lambda grad: print('label loss grad: ',grad.norm(dim=1).tolist(),'\n',))
                label_loss = label_loss_individual.mean()
                # TODO: Wrap this + BCE loss into one loss function?
                if aug_lagrangian in ['ones','LDF']:
                    estimated_solution = MyRound.apply(torch.sigmoid(output),k)
                    # hook_handle_1 = estimated_solution.register_hook(lambda grad: print('rounded_sol grad norm: ', grad.norm(dim=1).tolist(),
                    #                                                                     '\nrounded_sol grad', grad))
                    g = self.inequality_constraints(estimated_solution, x_batch)
                    # print('Constraint Valid: ', torch.where(g==0)[0])
                    # hook_handle_2 = g.register_hook(lambda grad: print('v(g) grad: ',grad.flatten().tolist()))
                    h = self.equality_constraints(estimated_solution, x_batch)
                    c = torch.cat((g,h),dim=1)

                    if c.shape[1] < 1:
                        print('To relax constraint into objective please provide constraint definitions to .fit()\n')
                    else:
                        constraint_loss = torch.mean(lm @ c.t())
                        epoch_constraint_loss += constraint_loss.item()
                    loss = label_loss + constraint_loss
                else:
                    loss = label_loss

                epoch_total_loss += loss.item()
                epoch_label_loss += label_loss.item()

                loss.backward()
                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_grad_norm)
                optimizer.step()

                # hook_handle_1.remove()
                # hook_handle_2.remove()

            log_output = f'Epoch {epoch}: TRAINING LOSS  Total = {epoch_total_loss:.3f}, Base = {epoch_label_loss:.3f}, Constraint = {epoch_constraint_loss:.3f}'
            train_loss_history.append(epoch_total_loss)

            if 'VAL' in self.input_dict:
                val_total_loss = 0
                val_total_loss_cnormed = 0
                val_constraint_loss = 0
                val_normed_constraint_loss = 0
                val_label_loss = 0
                self.model.eval()
                AR = 0

                with torch.no_grad():
                    for x_batch, y_batch in Val_Dataloader:
                        output = self.model(x_batch)
                        label_loss = BCEloss(output,y_batch).sum()
                        estimated_solution = torch.round(torch.sigmoid(output))

                        coeff = self.create_coefficient_dict(self.unnormalise_inputs(x_batch))
                        AR += Approximation_Ratio(coeff,estimated_solution,y_batch,reduce='sum')

                        if aug_lagrangian in ['ones', 'LDF']:
                            g = self.inequality_constraints(estimated_solution, x_batch)
                            h = self.equality_constraints(estimated_solution, x_batch)
                            c = torch.cat((g, h), dim=1)

                            if c.shape[1] < 1:
                                print('To relax constraint into objective please provide constraint definitions to .fit()\n')
                            else:
                                constraint_loss = torch.sum(lm @ c.t())
                                normed_constraint_loss = torch.sum(c.t())
                                loss_cnormed = label_loss + normed_constraint_loss
                                loss = label_loss + constraint_loss
                            val_constraint_loss += constraint_loss.item()
                        else:
                            loss = label_loss

                        val_total_loss += loss.item()
                        val_total_loss_cnormed += loss_cnormed.item()
                        val_label_loss += label_loss.item()

                # Have only taken the sum of each metric over the batches. Divide through by the size of the validation set to get means
                AR = AR / (Y_val.shape[0])
                val_total_loss = val_total_loss / (Y_val.shape[0])
                val_total_loss_cnormed = val_total_loss_cnormed / (Y_val.shape[0])
                val_constraint_loss = val_constraint_loss / (Y_val.shape[0])
                val_label_loss = val_label_loss / (Y_val.shape[0])

                log_output += f' | VAL LOSS  Total = {val_total_loss:.3f}, ' \
                              f'Base = {val_label_loss:.3f}, ' \
                              f'Constraint = {val_constraint_loss:.3f}, ' \
                              f'Total Normed = {val_total_loss_cnormed:.3f}'
                log_output += f' | VAL METRICS  AR = {AR:.4f}'
                val_loss_history.append(val_total_loss)

                if AR >= best_validation_AR and val_total_loss_cnormed >= best_validation_loss and val_total_loss_cnormed >= best_cnormed_validation_loss:
                    epochs_since_improvement += 1
                    if epochs_since_improvement > 200:
                        break
                if AR < best_validation_AR:
                    self.save_model(filename='model_params_best_AR.pt')
                    best_validation_AR = AR
                    log_output += ' *BEST AR*'
                    epochs_since_improvement = 0
                if val_total_loss < best_validation_loss:
                    self.save_model(filename='model_params_best_loss.pt')
                    best_validation_loss = val_total_loss
                    log_output += ' *BEST LOSS*'
                    epochs_since_improvement = 0
                if val_total_loss_cnormed < best_cnormed_validation_loss:
                    self.save_model(filename='model_params_best_cnormed_loss.pt')
                    best_cnormed_validation_loss = val_total_loss_cnormed
                    log_output += ' *BEST CNORMED LOSS*'
                    epochs_since_improvement = 0

            else:
                # Save model every Epoch if there is no validation set available
                self.save_model()

            log_output += ' | '

            if aug_lagrangian == 'LDF':
                if lm_scheduler is not None:
                    s = lm_scheduler(epoch,best_validation_AR,s)

                # At the end of each epoch update the lagrange multipliers
                self.model.eval()
                with torch.no_grad():
                    for x_batch, _ in Train_Dataloader:
                        estimated_solution = torch.round(torch.sigmoid(self.model(x_batch)))
                        g = self.inequality_constraints(estimated_solution, x_batch)
                        h = self.equality_constraints(estimated_solution, x_batch)
                        c = torch.cat((g, h), dim=1)

                        # Sum across the batch dimension, results after all batches will be that
                        # the constraint will have an effect for each training sample
                        lm = lm + s*torch.mean(c, dim=0)
                log_output += f'LM = {lm[0][0]:.2f} '


            if append_learning_rate_to_log:
                log_output += f'LR = {lr} '
                append_learning_rate_to_log = False
            if grid_search:

                with open(training_log_file, 'a') as f:
                    f.write(log_output + '\n')
                if epoch % 50 == 0:
                    print(log_output)
            else:
                print(log_output)
        if grid_search:
            combined_params_dict = self.training_params | self.model_params
            training_params = ''
            for param, value in combined_params_dict.items():
                if param != 'Grid Search':
                    training_params += f'{param}={value} '
            with open(os.path.join(self.ml_results_base_dir, 'Grid Search Results AR.txt'), 'a') as f:
                f.write(training_params + f'{AR:.8f}\n')
            with open(os.path.join(self.ml_results_base_dir, 'Grid Search Results LOSS.txt'), 'a') as f:
                f.write(training_params + f'{best_validation_loss:.8f}\n')
            with open(os.path.join(self.ml_results_base_dir, 'Grid Search Results CNORMED LOSS.txt'), 'a') as f:
                f.write(training_params + f'{best_cnormed_validation_loss:.8f}\n')

        # self.model.load_state_dict(torch.load(os.path.join(self.ml_result_dir,'model_params.pt')))

        # TODO: Add support for automatically plotting different metrics
        # Save training graph to file
        fig, axs = plt.subplots()
        lns1 = axs.plot(range(len(train_loss_history)),train_loss_history)
        axs.set_ylabel('Loss')
        axs.set_xlabel('Epoch')
        axs.set_ylabel('Training')

        if len(val_loss_history) > 0:
            axs_val = axs.twinx()
            lns2 = axs_val.plot(range(len(val_loss_history)), val_loss_history, 'orange')
            axs_val.set_ylabel('Validation')
            # axs_val.legend()

        plt.savefig(os.path.join(self.ml_result_dir,'Training History.png'))
        if not grid_search:
            plt.show()

    def predict(self,input='Val', idx_subset=None):

        input = input.upper()

        if input in ['TEST', 'TRAIN', 'VAL']:
            X = self.normalise_inputs(self.input_dict[input])
            Y = self.output_dict[input]

        # Check if we only want to make predictions on some subset of the data
        if (idx_subset is not None) and (not self.training_params.get('Grid Search',False)):
            split_type, split = idx_subset
            split_idx = self.input_dict[split_type][input][split]
            X = X[split_idx,:]
            Y = Y[split_idx,:]

        My_Dataset = TensorDataset(torch.from_numpy(X).float(),
                                   torch.from_numpy(Y).float())
        My_Dataloader = DataLoader(My_Dataset, batch_size=1024, shuffle=False)

        pred = torch.empty((0,Y.shape[1]), dtype=torch.bool)

        self.model.eval()
        with torch.no_grad():
            for x_batch, _ in My_Dataloader:
                output = torch.round(torch.sigmoid(self.model(x_batch))).type(torch.bool)
                pred = torch.cat((pred,output))

        self.pred = (X,Y,pred.numpy())
        self.pred_on = input


class gradient_boosted_machine(ml_model):
    def __init__(self, ml_params, directories, ml_model_type=None):
        if ml_model_type is None:
            self.model_type = 'Gradient Boosted Machine'
        else:
            self.model_type = ml_model_type

        super().__init__(ml_params,directories)

    def save_model(self):
        pass

    def load_model(self):
        pass

    def fit(self):
        X_train = self.input_dict['TRAIN']
        Y_train = self.output_dict['TRAIN']
        X_val = self.input_dict['VAL']
        Y_val = self.output_dict['VAL']


    def predict(self):
        pass