import os
from lib.base_classes import random_forest, neural_network

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class RF_model(random_forest):
    def __init__(self, ml_params, training_params, directories, ml_model_type=None):
        super().__init__(ml_params, training_params, directories)

    def load_data(self, rng_seed=8983):

        with open(os.path.join(self.instance_dir, f'Instance Generation Info.txt')) as f:
            for line in f:
                line = line.split()
                if line[0] == 'n':
                    n_weights = int(line[1])

        X_dict = {p: np.zeros((self.available_instances, self.param_types['num'][p])) for p in
                  self.param_types['Varying']}
        self.fixed_params = {p: np.zeros((1, self.param_types['num'][p])) for p in self.param_types['Fixed']}

        for i in range(self.available_instances):
            with open(os.path.join(self.instance_dir, f'instance {i}.txt')) as f:
                for line in f:
                    line = line.split()
                    param = line[0]
                    dtype = float if param == 'W' else int
                    if param in X_dict:
                        X_dict[param][i, :] = np.asarray([dtype(v) for v in line[1:]])
                    if param in self.fixed_params:
                        self.fixed_params[param][0, :] = np.asarray([dtype(v) for v in line[1:]])

        self.n_out = n_weights

        Y = np.zeros((self.available_instances, self.n_out))
        for i in range(self.available_instances):
            with open(os.path.join(self.opt_result_dir, f'{i}/Weights Used.txt')) as f:
                for line in f:
                    line = line.split()
                    w = ([int(w) for w in line])
                    Y[i, w] = 1

        # Construct X in the order params are listed in self.param_types['Varying']
        X = np.concatenate(tuple(X_dict[param] for param in self.param_types['Varying']), 1)

        self.n_features = X.shape[1]

        # Shuffle the instances before splitting in train/validation/test sets
        # Only required if data was generated with an ordering
        rng = np.random.default_rng(seed=rng_seed)
        p = rng.permutation(self.available_instances)

        # Randomly split out the validation and test sets, but keep the training set ordered by instance
        val_split_index = int(self.available_instances * 0.1)
        test_split_index = int(self.available_instances * 0.2)

        p_val = np.sort(p[:val_split_index])
        p_test = np.sort(p[val_split_index:test_split_index])
        p_train = p[test_split_index:]

        high_to_low = False
        if high_to_low:
            X_train, Y_train = np.flip(X[p_train, :], axis=0), np.flip(Y[p_train, :], axis=0)
        else:
            X_train, Y_train = X[p_train, :], Y[p_train, :]

        X_test, Y_test = X[p_test, :], Y[p_test, :]
        X_val, Y_val = X[p_val, :], Y[p_val, :]

        self.input_dict = {'TRAIN': X_train,
                           'TEST': X_test,
                           'VAL': X_val,
                           'PARAM IDX': {}}

        self.input_dict['Quintile Splits'] = {'TRAIN': {},
                                              'VAL': {},
                                              'TEST': {}}

        # Construct the quintile splits
        quintile_split_points = [0, 6000, 12000, 18000, 24000, 30000]

        for i in range(5):
            idx1, idx2 = quintile_split_points[i], quintile_split_points[i + 1]
            self.input_dict['Quintile Splits']['VAL'][i] = np.nonzero((idx1 <= p_val) &
                                                                      (p_val < idx2))[0]
            self.input_dict['Quintile Splits']['TEST'][i] = np.nonzero((idx1 <= p_test) &
                                                                       (p_test < idx2))[0]

        # self.norm_groups = {}
        start_idx = 0

        for param in self.param_types['Varying']:
            self.input_dict['PARAM IDX'][param] = range(start_idx, start_idx + self.param_types['num'][param])
            # self.norm_groups[param] = (X[:, self.input_dict['PARAM IDX'][param]].max(),
            #                            X[:, self.input_dict['PARAM IDX'][param]].min())
            start_idx += self.param_types['num'][param]

        self.output_dict = {'TRAIN': Y_train,
                            'TEST': Y_test,
                            'VAL': Y_val}

    def create_coefficient_dict(self, X):
        return {param: X[:, idx] for param, idx in self.input_dict['PARAM IDX'].items()} | self.fixed_params

    def print_problem_metrics(self, pred, Y, coeff):
        shared_percent = np.ones(pred.shape[0])

        # Calculate on average how many correct weights are shared with the true solution
        shared = np.sum(Y * pred, axis=1)

        idx_actual_no_weights = (Y.sum(axis=1) == 0)
        idx_pred_no_weights = (pred.sum(axis=1) == 0)
        idx_normal = ~idx_actual_no_weights

        # These are the indices where the actual solution has no weights, but the predicted solution has weights :(
        idx_fail = idx_actual_no_weights & ~idx_pred_no_weights
        shared_percent[idx_fail] = 0.0

        shared_percent[idx_normal] = 100 * shared[idx_normal] / np.sum(Y[idx_normal, :], axis=1)

        max_shared = np.max(shared_percent)
        min_shared = np.min(shared_percent)
        mean_shared = np.mean(shared_percent)

        print(
            f'Shared Weights (Percentage): MEAN - {mean_shared:.2f} | MIN - {min_shared:.2f} | MAX - {max_shared:.2f}\n')

        return [
            f'Shared Weights (Percentage): MEAN - {mean_shared:.2f} | MIN - {min_shared:.2f} | MAX - {max_shared:.2f}']

    def inequality_constraints(self,X,data,ctype='violation'):
        # This method provides definitions for the inequality constraints to allow them to be relaxed into the objective function during training
        # Should define methods for h(x) and g(x) for constraints h(x)=0 and g(x)<=0

        # Extract coefficients for constraints from the data d
        coeff = self.create_coefficient_dict(data)

        w = torch.tensor(coeff['w']).float()
        W = torch.tensor(coeff['W']).float()

        # g = (torch.sum(X*w,1,keepdim=True) - W)/W
        g = (torch.sum(X * w, 1, keepdim=True) - W)

        # if g.requires_grad:
        #     print('g: ',g)
        #     g.register_hook(lambda grad: print('g grad: ',grad.flatten().tolist(),'\n',torch.where(grad==0)))

        if ctype == 'violation':
            return torch.relu(g)
        elif ctype == 'satisfiability':
            return g
        else:
            return None

class NN_model(neural_network):
    def __init__(self, ml_params, training_params, directories, forward_model, ml_model_type=None):
        self.forward_model = forward_model
        super().__init__(ml_params,training_params,directories)

    def load_data_no_split(self,dataset='TEST'):
        with open(os.path.join(self.instance_dir, f'Instance Generation Info.txt')) as f:
            for line in f:
                line = line.split()
                if line[0] == 'n':
                    n_weights = int(line[1])

        X_dict = {p: np.zeros((self.available_instances,self.param_types['num'][p])) for p in self.param_types['Varying']}
        self.fixed_params = {p: np.zeros((1,self.param_types['num'][p])) for p in self.param_types['Fixed']}

        for i in range(self.available_instances):
            with open(os.path.join(self.instance_dir,f'instance {i}.txt')) as f:
                for line in f:
                    line = line.split()
                    param = line[0]
                    dtype = float if param == 'W' else int
                    if param in X_dict:
                        X_dict[param][i,:] = np.asarray([dtype(v) for v in line[1:]])
                    if param in self.fixed_params:
                        self.fixed_params[param][0,:] = np.asarray([dtype(v) for v in line[1:]])

        self.n_out = n_weights

        Y = np.zeros((self.available_instances,self.n_out))
        for i in range(self.available_instances):
            with open(os.path.join(self.opt_result_dir,f'{i}/Weights Used.txt')) as f:
                for line in f:
                    line = line.split()
                    w = ([int(w) for w in line])
                    Y[i,w] = 1


        # Construct X in the order params are listed in self.param_types['Varying']
        X = np.concatenate(tuple(X_dict[param] for param in self.param_types['Varying']),1)

        self.n_features = X.shape[1]

        self.input_dict = {dataset: X,
                           'PARAM IDX': {}}

        self.input_dict['Quintile Splits'] = {dataset: {}}

        quintile_split_points = [0, 2000, 4000, 6000, 8000, 10000]

        p = np.asarray(range(10000))

        for i in range(5):
            idx1, idx2 = quintile_split_points[i], quintile_split_points[i + 1]
            self.input_dict['Quintile Splits'][dataset][i] = np.nonzero((idx1 <= p) &
                                                                      (p < idx2))[0]

        self.norm_groups = {}
        start_idx = 0

        for param in self.param_types['Varying']:
            self.input_dict['PARAM IDX'][param] = range(start_idx, start_idx + self.param_types['num'][param])
            self.norm_groups[param] = (X[:, self.input_dict['PARAM IDX'][param]].max(),
                                       X[:, self.input_dict['PARAM IDX'][param]].min())
            start_idx += self.param_types['num'][param]

        self.output_dict = {dataset: Y}

    def load_data(self,rng_seed=8983):

        with open(os.path.join(self.instance_dir, f'Instance Generation Info.txt')) as f:
            for line in f:
                line = line.split()
                if line[0] == 'n':
                    n_weights = int(line[1])

        X_dict = {p: np.zeros((self.available_instances,self.param_types['num'][p])) for p in self.param_types['Varying']}
        self.fixed_params = {p: np.zeros((1,self.param_types['num'][p])) for p in self.param_types['Fixed']}

        for i in range(self.available_instances):
            with open(os.path.join(self.instance_dir,f'instance {i}.txt')) as f:
                for line in f:
                    line = line.split()
                    param = line[0]
                    dtype = float if param == 'W' else int
                    if param in X_dict:
                        X_dict[param][i,:] = np.asarray([dtype(v) for v in line[1:]])
                    if param in self.fixed_params:
                        self.fixed_params[param][0,:] = np.asarray([dtype(v) for v in line[1:]])

        self.n_out = n_weights

        Y = np.zeros((self.available_instances,self.n_out))
        for i in range(self.available_instances):
            with open(os.path.join(self.opt_result_dir,f'{i}/Weights Used.txt')) as f:
                for line in f:
                    line = line.split()
                    w = ([int(w) for w in line])
                    Y[i,w] = 1


        # Construct X in the order params are listed in self.param_types['Varying']
        X = np.concatenate(tuple(X_dict[param] for param in self.param_types['Varying']),1)

        self.n_features = X.shape[1]

        # Shuffle the instances before splitting in train/validation/test sets
        # Only required if data was generated with an ordering
        rng = np.random.default_rng(seed=rng_seed)
        p = rng.permutation(self.available_instances)

        # Randomly split out the validation and test sets, but keep the training set ordered by instance
        val_split_index = int(self.available_instances*0.1)
        test_split_index = int(self.available_instances*0.2)

        p_val = np.sort(p[:val_split_index])
        p_test = np.sort(p[val_split_index:test_split_index])
        p_train = np.sort(p[test_split_index:])

        high_to_low = False
        if high_to_low:
            X_train, Y_train = np.flip(X[p_train,:],axis=0), np.flip(Y[p_train, :],axis=0)
        else:
            X_train, Y_train = X[p_train, :], Y[p_train, :]

        X_test, Y_test = X[p_test,:], Y[p_test, :]
        X_val, Y_val = X[p_val,:], Y[p_val, :]


        self.input_dict = {'TRAIN': X_train,
                           'TEST': X_test,
                           'VAL': X_val,
                           'PARAM IDX': {}}

        self.input_dict['Quintile Splits'] = {'TRAIN': {},
                                              'VAL':{},
                                              'TEST':{}}


        # Construct the quintile splits
        quintile_split_points = [0,6000,12000,18000,24000,30000]

        for i in range(5):
            idx1, idx2 = quintile_split_points[i], quintile_split_points[i + 1]
            self.input_dict['Quintile Splits']['VAL'][i] = np.nonzero((idx1 <= p_val) &
                                                                      (p_val < idx2))[0]
            self.input_dict['Quintile Splits']['TEST'][i] = np.nonzero((idx1 <= p_test) &
                                                                       (p_test < idx2))[0]

        self.norm_groups = {}
        start_idx = 0

        for param in self.param_types['Varying']:
            self.input_dict['PARAM IDX'][param] = range(start_idx,start_idx + self.param_types['num'][param])
            self.norm_groups[param] = (X[:,self.input_dict['PARAM IDX'][param]].max(),
                                       X[:,self.input_dict['PARAM IDX'][param]].min())
            start_idx += self.param_types['num'][param]

        self.output_dict = {'TRAIN': Y_train,
                           'TEST': Y_test,
                           'VAL': Y_val}

    def create_coefficient_dict(self,X):
        return {param: X[:,idx] for param, idx in self.input_dict['PARAM IDX'].items()} | self.fixed_params

    def print_problem_metrics(self, pred, Y, coeff):
        shared_percent = np.ones(pred.shape[0])

        # Calculate on average how many correct weights are shared with the true solution
        shared = np.sum(Y * pred, axis=1)

        idx_actual_no_weights = (Y.sum(axis=1) == 0)
        idx_pred_no_weights = (pred.sum(axis=1) == 0)
        idx_normal = ~idx_actual_no_weights

        # These are the indices where the actual solution has no weights, but the predicted solution has weights :(
        idx_fail = idx_actual_no_weights & ~idx_pred_no_weights
        shared_percent[idx_fail] = 0.0

        shared_percent[idx_normal] = 100 * shared[idx_normal]/np.sum(Y[idx_normal,:],axis=1)

        max_shared = np.max(shared_percent)
        min_shared = np.min(shared_percent)
        mean_shared = np.mean(shared_percent)

        print(f'Shared Weights (Percentage): MEAN - {mean_shared:.2f} | MIN - {min_shared:.2f} | MAX - {max_shared:.2f}\n')

        return [f'Shared Weights (Percentage): MEAN - {mean_shared:.2f} | MIN - {min_shared:.2f} | MAX - {max_shared:.2f}']

    def inequality_constraints(self,X,data,ctype='violation'):
        # This method provides definitions for the inequality constraints to allow them to be relaxed into the objective function during training
        # Should define methods for h(x) and g(x) for constraints h(x)=0 and g(x)<=0

        # Extract coefficients for constraints from the data d
        coeff = self.create_coefficient_dict(self.unnormalise_inputs(data))

        w = torch.tensor(coeff['w']).float()
        W = torch.tensor(coeff['W']).float()

        # g = (torch.sum(X*w,1,keepdim=True) - W)/W
        g = (torch.sum(X * w, 1, keepdim=True) - W)

        # if g.requires_grad:
        #     print('g: ',g)
        #     g.register_hook(lambda grad: print('g grad: ',grad.flatten().tolist(),'\n',torch.where(grad==0)))

        if ctype == 'violation':
            return torch.relu(g)
        elif ctype == 'satisfiability':
            return g
        else:
            return None

class ANN(nn.Module):
    def __init__(self,n_features,n_out,model_params):
        super().__init__()
        self.ANN_model = {}
        self.model_details = {}

        use_batch_norm = model_params.get('Batch Norm',False)
        use_dropout = model_params.get('Dropout',False)

        if n_features <= 500:
            self.fc1 = nn.Linear(n_features, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, n_out)
            self.model_details['Layers'] = f'{n_features} 512 256 {n_out} --- DROPOUT={use_dropout}|BATCHNORM={use_batch_norm}'

            self.bn1 = torch.nn.BatchNorm1d(512)
            self.bn2 = torch.nn.BatchNorm1d(256)
        else:
            self.fc1 = nn.Linear(n_features, 2048)
            self.fc2 = nn.Linear(2048, 1024)
            self.fc3 = nn.Linear(1024,n_out)
            self.model_details['Layers'] = f'{n_features} 2048 1024 {n_out}'

            self.bn1 = torch.nn.BatchNorm1d(2048)
            self.bn2 = torch.nn.BatchNorm1d(1024)

        self.dropout = nn.Dropout(0.3)
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

    def forward(self,x):
        x = F.relu(self.fc1(x))

        if self.use_batch_norm:
            x = self.bn1(x)
        if self.use_dropout:
            x = self.dropout(x)

        x = F.relu(self.fc2(x))

        if self.use_batch_norm:
            x = self.bn2(x)
        if self.use_dropout:
            x = self.dropout(x)

        logits = self.fc3(x)

        return logits
