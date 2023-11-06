from abc import ABC, abstractmethod
import os
import time
import yaml
import math
import functools

from lib.custom_metrics import Approximation_Ratio
from lib.autograd_funcs import MyRound

from gurobipy import Model
import pickle
from collections import defaultdict
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
# import scipy.sparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class base_opt_model(ABC):
    def __init__(self, opt_params):
        self.opt_params = opt_params
        self.base_dir = opt_params['base directory']

        self.opt_dir = os.path.join(self.base_dir,'Opt Results',self.model_type,self.opt_params["instance folder"])
        self.instance_dir = os.path.join(self.base_dir,'Instances',opt_params['instance folder'])
        self.gurobi_log_dir = os.path.join(os.path.dirname(self.opt_dir),'GurobiLogs')

        if not os.path.exists(self.opt_dir):
            os.makedirs(self.opt_dir)
        if not os.path.exists(self.gurobi_log_dir):
            os.makedirs(self.gurobi_log_dir)

        self.common_data_loaded = False

    # Overwrite as needed (E.g. for small vs large Instances)
    def create_results_directory(self):
        self.results_directory = os.path.join(self.base_dir,
                                              'Opt Results',
                                              self.model_type,
                                              self.opt_params["instance folder"],
                                              str(self.instance))
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

    def optimize_model(self):
        self.model.optimize()

    def create_model(self):
        self.model = Model()
        self.create_results_directory()
        if 'LogFile' in self.opt_params:
            if self.opt_params['LogFile']:
                self.model.params.LogFile = os.path.join(self.gurobi_log_dir,f'Gurobi Log {self.instance}.txt')
                # Clear out the logfile if it already exists
                if os.path.exists(self.model.params.LogFile):
                    open(self.model.params.LogFile,'w').close()
        if 'LogToConsole' in self.opt_params:
            self.model.Params.LogToConsole = self.opt_params['LogToConsole']
        if 'MIPGap' in self.opt_params:
            self.model.Params.MIPGap = self.opt_params['MIPGap']
        if 'TimeLimit' in self.opt_params:
            self.model.Params.TimeLimit = self.opt_params['TimeLimit']
        if 'threads' in self.opt_params:
            self.model.Params.Threads = self.opt_params['threads']
        if 'method' in self.opt_params:
            self.model.Params.method = self.opt_params['method']


    # Overwrite as needed (E.g. for small vs large Instances)
    def solve_all_instances(self):
        instance_gen_info_file = os.path.join(self.instance_dir, 'Instance Generation Info.txt')

        assert os.path.exists(instance_gen_info_file)

        with open(instance_gen_info_file, 'r') as f:
            self.available_instances = int(f.readline()[:-1])

        self.total_solve_time = 0

        self.instance = self.opt_params['instance index']

        # Before beginning to solve, wipe out old Gurobi logfiles
        for file in os.listdir(self.gurobi_log_dir):
            os.remove(os.path.join(self.gurobi_log_dir,file))

        # Store the parameters used for the optimisation
        with open(os.path.join(os.path.dirname(self.opt_dir),'opt_config.yaml'),'w') as f:
            yaml.dump(self.opt_params, f)

        wall_time = time.time()
        while self.instance < self.available_instances:
            self.setup_and_optimize()
            if self.model.Status in [2,9] and 100*self.model.MIPGap < 2:
                self.save_model()
            self.total_solve_time += self.model.Runtime
            self.instance += 1

        # end_time = time.time()
        # solve_time_per_instance = (end_time - start_time)/self.available_instances
        with open(os.path.join(self.opt_dir, 'Solve Time.txt'),'w') as f:
            f.write(f'{self.total_solve_time/self.available_instances:.5f} s')
        print(f'Total Solve Time: {self.total_solve_time:.2f}')
        print(f'Total Wall Time: {time.time() - wall_time:.2f}')



    def setup_and_optimize(self):
        print(f'Beginning to Solve Instance {self.instance}\n')
        self.create_model()
        self.load_instance()
        self.load_problem_specific_data()
        self.add_vars()
        self.add_objective()
        self.add_constraints()
        self.optimize_model()

    def load_problem_specific_data(self):
        # Not always required, can be used, for example, to construct sets which will be looped over in constraints
        pass

    # def save_model_matrix(self):
    #     A = self.model.getA()
    #
    #     sense = np.array(self.model.getAttr("Sense", self.model.getConstrs()))
    #     b = np.array(self.model.getAttr("RHS", self.model.getConstrs()))
    #
    #     Aeq = A[sense == '=', :]
    #     Ale = A[sense == '<', :]
    #     Age = A[sense == '>', :]
    #
    #     beq = b[sense == '=']
    #     ble = b[sense == '<']
    #     bge = b[sense == '>']
    #
    #     if Aeq.shape[0] > 0:
    #         scipy.sparse.save_npz(os.path.join(self.results_directory,'Aeq.npz'),Aeq)
    #
    #     if Ale.shape[0] > 0:
    #         scipy.sparse.save_npz(os.path.join(self.results_directory,'Ale.npz'),Ale)
    #
    #     if Age.shape[0] > 0:
    #         scipy.sparse.save_npz(os.path.join(self.results_directory,'Age.npz'),Age)

    def save_model(self):
        # self.create_results_directory()
        self.save_model_output()
        # self.save_model_matrix()

        # Store the optimisation parameters used
        # with open(os.path.join(self.results_directory, 'opt_params.pickle'), 'wb') as f:
        #     pickle.dump(self.opt_params, f, protocol=pickle.HIGHEST_PROTOCOL)

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

        #Store the solution such that it can be loaded back into a Gurobi model
        if self.opt_params['StoreSol'] and not self.opt_params['streamline']:
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
    def load_instance(self):
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

        self.base_dir = directories['Base Directory']
        self.instance_dir = os.path.join(self.base_dir,'Instances',directories['Instance Folder'])
        self.opt_result_dir = os.path.join(self.base_dir, 'Opt Results', directories['Opt Model'], directories['Instance Folder'])
        self.ml_results_base_dir = os.path.join(self.base_dir, 'ML Results', self.model_type, directories['Instance Folder'])

        if 'Hyperparameters' in directories:
            self.hyperparameter_description = directories['Hyperparameters']
            self.ml_result_dir = os.path.join(self.ml_results_base_dir, self.hyperparameter_description)
        else:
            self.ml_result_dir = self.ml_results_base_dir

        if not os.path.exists(self.ml_result_dir):
            os.makedirs(self.ml_result_dir)

        with open(os.path.join(self.instance_dir,'Instance Generation Info.txt'),'r') as f:
            self.available_instances = int(f.readline()[:-1])

        # Load in a dictionary which stores which parameters of the Instances vary between instance and which are fixed
        if os.path.exists(os.path.join(self.instance_dir,'param_types.pickle')):
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

    def eval_prediction(self,pred,pred_on):
        # TODO: Test constraint for examples with only one constraint per instance
        # TODO: Add stats about how many instances have no constraint violations

        predictions_stat_dir = os.path.join(self.ml_result_dir,'Prediction Metrics - ' + pred_on)
        if not os.path.exists(predictions_stat_dir):
            os.makedirs(predictions_stat_dir)

        print('-'*10 + f'\n\nEVALUATING PREDICTION ON {pred_on} DATA\n\n')

        # Write over the latex version
        open(os.path.join(predictions_stat_dir, 'Latex Version.txt'), 'w').close()

        if 'inequalities' in pred:
            for ineq, constraint_eval in pred['inequalities'].items():
                logout = ineq.upper() + ' - Constraint Violation Statistics\n'
                constraint_dim = constraint_eval.dim()
                constraints_per_instance = functools.reduce(lambda x1,x2: x1*x2, constraint_eval.shape[1:])

                num_violations_per_instance = (constraint_eval > 1e-12).float()
                total_violations = torch.sum(num_violations_per_instance)
                max_per_instance = constraint_eval

                # If there are multiple of this constraint type per instance, collapse those dimensions
                if constraint_dim > 1:
                    num_violations_per_instance = torch.sum(num_violations_per_instance,tuple(range(1,constraint_dim)))
                    max_per_instance = torch.amax(max_per_instance, tuple(range(1,constraint_dim)))

                mean_num_violations = torch.mean(num_violations_per_instance)
                max_num_violations = num_violations_per_instance.max()

                logout += (f'Violated on average {100*mean_num_violations/constraints_per_instance:.2f}% '
                           f'({math.floor(mean_num_violations)}/{constraints_per_instance}) '
                           f'of constraints per instance, maximum of {100*max_num_violations/constraints_per_instance:.2f}% ({math.floor(max_num_violations)}/{constraints_per_instance})\n')

                average_violation = torch.sum(constraint_eval * (constraint_eval>1e-12))/total_violations

                logout += f'Constraint Violation Statistics: MEAN - {average_violation:.2f} | MAX (ALL) - {torch.max(max_per_instance)} | AVERAGE MAX {torch.mean(max_per_instance):.2f}\n'

                print(logout)
                with open(os.path.join(predictions_stat_dir, 'Constraint - ' + ineq.upper() + '.txt'),'w') as f:
                    f.write(logout)
                with open(os.path.join(predictions_stat_dir, 'Latex Version.txt'),'a') as f:
                    f.write(ineq + f'& {100*mean_num_violations/constraints_per_instance:.1f} '
                                   f'& {100*max_num_violations/constraints_per_instance:.1f} '
                                   f'& {average_violation:.1f} '
                                   f'& {torch.mean(max_per_instance):.1f} \\\\ \n \hline \n')

        if 'AR' in pred:
            AR = pred['AR']
            mean_AR = AR.mean()
            min_AR = AR.min()
            max_AR = AR.max()
            with open(os.path.join(predictions_stat_dir, 'AR.txt'), 'w') as f:
                logout = f'AR Statistics - MEAN - {mean_AR:.5f} | MIN- {min_AR:.5f} | MAX {max_AR:.5f}\n'
                print(logout)
                f.write(logout)


    @abstractmethod
    def create_coefficient_dict(self,X):
        # Create a coefficient dictionary for the inputs X
        # X is a B x N matrix where each row represents the data for one instance
        pass

    # Needs to be overridden for augmented lagrangian models
    # Inputs are the B x n solution matrix X and B x M parameter matrix d
    # If self.create_coefficient_dict is defined the user can use it to unpack data into a dictionary of problem data
    # Output is a B x m matrix where m is the number of constraints
    def inequality_constraints(self,X,data,ctype='violation'):
        return torch.empty((X.shape[0],0))

    # Needs to be overridden for augmented lagrangian models
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
    def __init__(self, model_params, training_params, directories, ml_model_type=None):
        if ml_model_type is None:
            self.model_type = 'Random Forest'
        else:
            self.model_type = ml_model_type

        super().__init__(model_params,training_params,directories)

    def fit(self):
        max_features = self.model_params.get('Max Features', 'sqrt')
        num_trees = self.model_params.get('Number Trees', 100)
        max_depth = self.model_params.get('Max Depth', None)
        grid_search = self.training_params.get('Grid Search', False)

        self.model = MultiOutputClassifier(RandomForestClassifier(n_estimators=num_trees,
                                                                  max_depth=max_depth,
                                                                  max_features=max_features))

        X_train = self.input_dict['TRAIN']
        Y_train = self.output_dict['TRAIN']

        if 'VAL' in self.input_dict:
            X_val = self.input_dict['VAL']
            Y_val = self.output_dict['VAL']

        if grid_search:
            print(f'\nFITTING MODEL WITH {self.hyperparameters}\n')

        # Start a timer for the model fitting
        start = time.time()

        self.model.fit(X_train,
                       Y_train)

        # Record how long it took to train the random forest
        end_time = time.time()
        training_time = int((end_time - start) / 60)
        with open(os.path.join(self.ml_result_dir, 'Training Time.txt'), 'a') as f:
            f.write(f'{training_time} mins')

        self.save_model()

        # Evaluate the fitted random forest on the validation set (if one exists)
        if 'VAL' in self.input_dict:
            # log_output = 'VAL METRICS  '
            pred = self.model.predict(X_val)

            coeff = self.create_coefficient_dict(X_val)
            AR = Approximation_Ratio(coeff, pred, Y_val, reduce='mean')

            # log_output += f'AR = {AR:.4f} |'
            print(f'VAL METRICS  AR = {AR:.4f} |')

            if grid_search:
                combined_params_dict = self.training_params | self.model_params
                training_params = ''
                for param, value in combined_params_dict.items():
                    if param != 'Grid Search':
                        training_params += f'{param}={value} '
                with open(os.path.join(self.ml_results_base_dir, 'Grid Search Results AR.txt'), 'a') as f:
                    f.write(training_params + f'{AR:.8f}\n')


    def predict(self,input='Val', idx_subset=None):
        # TODO Update this so that it can utilise idx_subset
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
            for k,v in self.model_params.items():
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
        # TODO:
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
        self.model = self.forward_model(self.model_params, self.data_for_network())
        self.model.load_state_dict(torch.load(os.path.join(self.ml_results_base_dir,filename)))

    def fit(self,preload=None):

        # Load in training parameters
        n_epochs = self.training_params.get('Epochs',100)
        lr = self.training_params.get('lr',1e-3)
        aug_lagrangian = self.training_params.get('Constraints','None')
        train_batch_size = self.training_params.get('Training Batch Size',256)
        s = self.training_params.get('Lagrange Step',1)
        lm_scheduler = self.training_params.get('LM Step Scheduler',None)
        k = self.training_params.get('k Round',25)
        clip_grad_norm = self.training_params.get('Clip Grad Norm',False)
        max_grad_norm = self.training_params.get('Max Grad Norm', 1)
        grid_search = self.training_params.get('Grid Search', False)
        lm_update_interleave = self.training_params.get('LM Update Interleave', 0)
        ctype = self.training_params.get('ctype', 'violation')
        lm_delay = self.training_params.get('LM Delay',0)
        mu_norms = self.training_params.get('Lambda Norms',[1,10])
        log_interleave = self.training_params.get('Log Interleave',1)

        assert (self.input_dict is not None and self.output_dict is not None), 'Please load in training data before training Neural Network Model\n'

        # Load in training and validation sets, and construct dataloaders
        Train_Dataset, Val_Dataset = self.create_dataset('TRAIN'), self.create_dataset('VAL')
        Train_Dataloader = DataLoader(Train_Dataset, batch_size=train_batch_size, shuffle=True)
        if Val_Dataset is not None:
            Val_Dataloader = DataLoader(Val_Dataset, batch_size=train_batch_size, shuffle=False)

        # Set up the model and optimizer
        if preload is not None:
            self.load_model(preload)
        else:
            self.model = self.forward_model(self.model_params,self.data_for_network())
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f'Model has {n_params} parameters\n')

        # Check that the number of parameters detected by pytorch is the same as the expected number of parameters
        assert n_params == self.model.total_paramaters

        optimizer = torch.optim.Adam(self.model.parameters(),lr,weight_decay=0)

        lam = {'battery': torch.tensor(1)}

        if 'Initial Lam' in self.training_params:
            for k in lam.keys():
                lam[k] = torch.tensor(self.training_params['Initial Lam'])

        # Set up logging a file to log the training process
        training_log_file = os.path.join(self.ml_result_dir,'Training Log.txt')
        open(training_log_file,'w').close()

        best_validation_loss = float('inf')
        best_validation_AR = float('inf')
        best_cnormed_validation_loss = {mu: float('inf') for mu in mu_norms}

        val_loss_min_epoch = float('inf')
        val_AR_min_epoch = float('inf')
        val_cnormed_loss_epoch = {mu: float('inf') for mu in mu_norms}

        train_loss_history = []
        val_loss_history = []
        val_closs_history = {mu: [] for mu in mu_norms}

        epochs_since_improvement = 0
        epochs_since_lr_decreased = 0
        lm_update_interleave_counter = 0

        if grid_search:
            print(f'\nFITTING MODEL WITH {self.hyperparameter_description}\n')

        append_learning_rate_to_log = False

        start = time.time()

        for epoch in range(n_epochs):
            epoch_constraint_loss = 0
            epoch_label_loss = 0

            self.model.train()
            for batch in Train_Dataloader:
                x_batch, y_batch, auxiliary_batch = self.unpack_batch(batch)
                optimizer.zero_grad()
                output = self.model(x_batch)

                label_loss_individual = self.loss_function(output,y_batch)
                label_loss = label_loss_individual.mean()

                epoch_label_loss += label_loss_individual.sum().item()

                if epoch >= lm_delay and aug_lagrangian in ['ones','LDF']:
                    estimated_solution = self.train_decode(output)
                    coeff = self.create_coefficient_dict(self.unnormalise_inputs(x_batch), self.data_for_network())
                    decision_vars = self.construct_decision_variables(estimated_solution, coeff, mode='train')
                    g = self.inequality_constraints(decision_vars, coeff, ctype=ctype, mode='train')
                    constraint_loss_raw = self.constraint_loss_function(g)

                    # constraint_loss_raw holds the mean constraint violation over non batch dimensions (i.e. it has size batch_size)
                    battery_lam = lam['battery']
                    constraint_loss = battery_lam * constraint_loss_raw['battery']

                    loss = label_loss + constraint_loss.mean()
                    epoch_constraint_loss += constraint_loss.sum().item()

                else:
                    loss = label_loss

                loss.backward()
                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_grad_norm)
                optimizer.step()

            epoch_total_loss = epoch_label_loss + epoch_constraint_loss

            epoch_total_loss = epoch_total_loss / self.input_dict['dataset size']['TRAIN']
            epoch_label_loss = epoch_label_loss / self.input_dict['dataset size']['TRAIN']
            epoch_constraint_loss = epoch_constraint_loss / self.input_dict['dataset size']['TRAIN']

            log_output = f'Epoch {epoch}: TRAINING LOSS  Total = {epoch_total_loss:.3f}, Base = {epoch_label_loss:.3f}, Constraint = {epoch_constraint_loss:.3f}'
            train_loss_history.append(epoch_total_loss)

            if 'VAL' not in self.input_dict:
                # Save model every Epoch if there is no validation set available
                self.save_model()
            else:
                val_total_loss = 0
                val_constraint_loss = 0
                val_label_loss = 0
                normed_constraint_loss = {mu: 0 for mu in mu_norms}
                AR = 0

                self.model.eval()
                with torch.no_grad():
                    for batch in Val_Dataloader:
                        x_batch, y_batch, auxiliary_batch = self.unpack_batch(batch)
                        output = self.model(x_batch)
                        label_loss = self.loss_function(output,y_batch).sum()
                        val_label_loss += label_loss.item()

                        estimated_solution = self.test_decode(output)
                        coeff = self.create_coefficient_dict(self.unnormalise_inputs(x_batch), self.data_for_network())
                        decision_vars = self.construct_decision_variables(estimated_solution,coeff,mode='test')
                        predicted_objective = self.evaluate_objective(decision_vars,coeff)
                        AR += Approximation_Ratio(predicted_objective,auxiliary_batch,reduce='sum')

                        if aug_lagrangian in ['ones', 'LDF']:
                            estimated_solution = self.train_decode(output)
                            decision_vars = self.construct_decision_variables(estimated_solution, coeff, mode='train')
                            g = self.inequality_constraints(decision_vars, coeff, ctype=ctype, mode='train')
                            constraint_loss_raw = self.constraint_loss_function(g)

                            # TODO: Generalise in a way that still allows calculation of the mu-norms
                            battery_lam = lam['battery']
                            val_constraint_loss += battery_lam * constraint_loss_raw['battery'].sum().item()

                            for mu in mu_norms:
                                normed_constraint_loss[mu] += label_loss.item() + (mu * constraint_loss_raw['battery'].sum()).item()

                val_total_loss = val_constraint_loss + val_label_loss

                # Have only taken the sum of each metric over the batches. Divide through by the size of the validation set to get means
                AR = AR / self.input_dict['dataset size']['VAL']
                val_total_loss = val_total_loss / self.input_dict['dataset size']['VAL']
                val_constraint_loss = val_constraint_loss / self.input_dict['dataset size']['VAL']
                val_label_loss = val_label_loss / self.input_dict['dataset size']['VAL']

                val_loss_history.append(val_total_loss)

                for mu in normed_constraint_loss:
                    normed_constraint_loss[mu] = normed_constraint_loss[mu] / self.input_dict['dataset size']['VAL']
                    val_closs_history[mu].append(normed_constraint_loss[mu])

                log_output += f' | VAL LOSS  Total = {val_total_loss:.3f}, ' \
                              f'Base = {val_label_loss:.3f}, ' \
                              f'Constraint = {val_constraint_loss:.3f}'
                log_output += f' | VAL METRICS  AR = {AR:.4f}'

                save_time_start = time.time()

                if AR < best_validation_AR:
                    self.save_model(filename='model_params_best_AR.pt')
                    best_validation_AR = AR
                    log_output += ' *BEST AR*'
                    epochs_since_improvement = 0
                    val_loss_min_epoch = epoch

                if val_total_loss < best_validation_loss:
                    self.save_model(filename='model_params_best_loss.pt')
                    best_validation_loss = val_total_loss
                    log_output += ' *BEST LOSS*'
                    epochs_since_improvement = 0
                    val_AR_min_epoch = epoch

                for mu in mu_norms:
                    if normed_constraint_loss[mu] < best_cnormed_validation_loss[mu]:
                        self.save_model(filename=f'model_params_best_{mu}-normed_loss.pt')
                        best_cnormed_validation_loss[mu] = normed_constraint_loss[mu]
                        epochs_since_improvement = 0
                        val_cnormed_loss_epoch[mu] = epoch

                model_save_time = time.time() - save_time_start

                if epochs_since_improvement > 400:
                    break
                else:
                    epochs_since_improvement += 1


            log_output += ' | '
            if aug_lagrangian == 'LDF':
                lm_update_interleave_counter = 0
                epoch_constraint_eval = 0
                # Update Lagrange multipliers if the update interleave has been reached
                self.model.eval()
                with torch.no_grad():
                    for batch in Train_Dataloader:
                        x_batch, _, _ = self.unpack_batch(batch)
                        output = self.model(x_batch)

                        # Decode the output and use it to construct the degrees of violation of the constraints
                        estimated_solution = self.train_decode(output)
                        coeff = self.create_coefficient_dict(self.unnormalise_inputs(x_batch), self.data_for_network())
                        decision_vars = self.construct_decision_variables(estimated_solution, coeff, mode='train')
                        g = self.inequality_constraints(decision_vars, coeff, ctype=ctype, mode='train')

                        #TODO: Generalise the handling of constraints at evaluation time
                        # This evaluates to the mean over the constraint per batch, then summing over the batches
                        epoch_constraint_eval += torch.sum(torch.mean(g['battery min'] + g['battery capacity'], dim=(1,2)))

                epoch_constraint_eval = epoch_constraint_eval/self.input_dict['dataset size']['TRAIN']
                lam['battery'] = lam['battery'] + s*epoch_constraint_eval
                log_output += f'LM = {lam["battery"]:.2f} '

            if append_learning_rate_to_log:
                log_output += f'LR = {lr} '
                append_learning_rate_to_log = False
            if grid_search:
                with open(training_log_file, 'a') as f:
                    f.write(log_output + '\n')
                if epoch % log_interleave == 0:
                    print(log_output)
            else:
                with open(training_log_file, 'a') as f:
                    f.write(log_output + '\n')
                print(log_output)

        end_time = time.time()
        training_time = int((end_time - start)/60)
        with open(os.path.join(self.ml_results_base_dir, 'Training Time.txt'), 'a') as f:
            f.write(f'{training_time} mins\n')

        with open(os.path.join(self.ml_result_dir, 'Model Epochs.txt'),'a') as f:
            f.write(f'AR: {val_AR_min_epoch}\n')
            f.write(f'Loss: {val_loss_min_epoch}\n')
            for mu in mu_norms:
                f.write(f'{mu}-Normed Loss: {val_cnormed_loss_epoch[mu]}\n')

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
            for mu in mu_norms:
                with open(os.path.join(self.ml_results_base_dir, f'Grid Search Results {mu}-NORMED LOSS.txt'), 'a') as f:
                    f.write(training_params + f'{best_cnormed_validation_loss[mu]:.8f}\n')


        self.save_model(filename='model_params_final.pt')

        # TODO: Add support for automatically plotting different metrics
        # Save training graph to file
        fig, axs = plt.subplots()
        lns1 = axs.plot(range(len(train_loss_history)),train_loss_history,
                        range(len(train_loss_history)),val_loss_history)
        axs.set_ylabel('Loss')
        axs.set_xlabel('Epoch')


        # if len(val_loss_history) > 0:
        #     axs_val = axs.twinx()
        #     lns2 = axs_val.plot(range(len(val_loss_history)), val_loss_history, 'orange')
        #     axs_val.set_ylabel('Validation')
        #     # axs_val.legend()

        plt.savefig(os.path.join(self.ml_result_dir, 'Training History.png'))
        plt.savefig(os.path.join(self.ml_result_dir, 'Training History.eps'))

        # Save the constraint-normed losses
        for mu in mu_norms:
            fig1, axs1 = plt.subplots()
            lns1 = axs1.plot(range(len(val_closs_history[mu])), val_closs_history[mu])
            axs1.set_ylabel(f'{mu}-Normed Loss')
            axs1.set_xlabel('Epoch')

            plt.savefig(os.path.join(self.ml_result_dir, f'Training History {mu}-normed.png'))
            plt.savefig(os.path.join(self.ml_result_dir, f'Training History {mu}-normed.eps'))

        if not grid_search:
            plt.show()

    def predict(self,input_type='Val', idx_subset=None):

        input_type = input_type.upper()

        assert input_type in ['TEST', 'TRAIN', 'VAL']

        My_Dataset = self.create_dataset(input_type)
        My_Dataloader = DataLoader(My_Dataset, batch_size=256, shuffle=False)

        num_samples = self.input_dict['dataset size'][input_type]

        total_time = 0

        self.model.eval()
        first_batch = True
        start_time = time.time()
        with torch.no_grad():
            for batch in My_Dataloader:
                x_batch, _, auxiliary_batch = self.unpack_batch(batch)
                start_time = time.time()
                output = list(self.model(x_batch))
                total_time += time.time() - start_time

                estimated_solution = self.test_decode(output)
                coeff = self.create_coefficient_dict(self.unnormalise_inputs(x_batch), self.data_for_network())
                decision_vars = self.construct_decision_variables(estimated_solution, coeff)
                g = self.inequality_constraints(decision_vars,coeff,mode='test')
                predicted_objective = self.evaluate_objective(decision_vars, coeff)
                AR_batch = Approximation_Ratio(predicted_objective, auxiliary_batch, reduce='none')

                if first_batch:
                    pred = output
                    inequality_dict = g
                    AR = AR_batch
                    obj = predicted_objective
                    first_batch = False
                else:
                    for i in range(len(pred)):
                        pred[i] = torch.cat((pred[i], output[i]),dim=0)
                    for k in inequality_dict.keys():
                        inequality_dict[k] = torch.cat((inequality_dict[k],g[k]), dim=0)
                    AR = torch.cat((AR,AR_batch), dim=0)
                    obj = torch.cat((obj,predicted_objective), dim=0)


        end_time = time.time()
        inference_time_per_sample = total_time/num_samples
        print(f'{inference_time_per_sample:.5f} seconds')

        with open(os.path.join(self.ml_results_base_dir, 'Inference Time.txt'), 'a') as f:
            f.write(f'{inference_time_per_sample:.5f} seconds')

        return {'pred': tuple(pred),  'inequalities': inequality_dict, 'AR': AR, 'objective value': obj}, input_type

    def create_datasets(self, X, Y):
        Train_Dataset = self._create_dataset(self.normalise_inputs(self.input_dict['TRAIN']),
                                             self.output_dict['TRAIN'])
        Train_Dataset = self._create_dataset(self.normalise_inputs(self.input_dict['VAL']),
                                             self.output_dict['VAL'])

    def _create_dataset(self, X, Y):
        return TensorDataset(torch.from_numpy(X).float(),
                             torch.from_numpy(Y).float())


    def unpack_batch(self,batch):
        if len(batch) == 2:
            return batch[0], batch[1]
        if len(batch) >= 2:
            return batch[0], batch[1:]


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