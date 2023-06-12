from gurobipy import Model, quicksum, GRB, read
import math
import pickle
import os
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from collections import defaultdict

class base_model(ABC):
    def __init__(self,opt_params):

        self.opt_params = opt_params

        self.model = Model()
        if 'MIPGap' in opt_params:
            self.model.Params.MIPGap = opt_params['MIPGap']
        if 'TimeLimit' in opt_params:
            self.model.Params.TimeLimit = opt_params['TimeLimit']

    def create_results_directory(self,index):
        self.results_directory = f'Results/{self.model_type} - {index}'
        if not os.path.exists(self.results_directory):
            os.mkdir(self.results_directory)

    def optimize_model(self):
        self.model.optimize()

    def setup_and_optimize(self,index=None):
        self.load_instance(index)
        self.add_vars()
        self.add_objective()
        self.add_constraints()
        self.optimize_model()

    def save_model(self):

        weights_used = self.vars_to_readable()

        with open(os.path.join(self.results_directory,'Weights Used.txt'),'w') as f:
            for w in weights_used:
                f.write(f'{w} ')

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


    def load_instance(self,index=None):
        if index is None:
            index = self.opt_params['instance index']
        else:
            self.opt_params['instance index'] = index

        with open(os.path.join('Instances', f'test instance {index}')) as f:
            for line in f:
                line = line.split()
                if line[0] == 'n':
                    n = int(line[1])
                elif line[0] == 'W':
                    W = int(line[1])
                elif line[0] == 'v':
                    v = [int(x) for x in line[1:]]
                elif line[0] == 'w':
                    w = [int(x) for x in line[1:]]

        self.create_results_directory(index)

        self.instance_data = {'n': n,
                              'W': W,
                              'v': v,
                              'w': w}

    @abstractmethod
    def add_vars(self):
        pass

    @abstractmethod
    def add_constraints(self):
        pass

    @abstractmethod
    def add_objective(self):
        pass

class vanilla_IP(base_model):
    def __init__(self,opt_params):
        self.model_type = 'Vanilla'
        super().__init__(opt_params)

    def add_vars(self):
        n = self.instance_data['n']

        self.X = {i: self.model.addVar(vtype=GRB.BINARY)
                  for i in range(n)}

    def add_objective(self):
        n = self.instance_data['n']
        v = self.instance_data['v']


        self.model.setObjective(quicksum(v[i]*self.X[i] for i in range(n)), GRB.MAXIMIZE)

    def add_constraints(self):
        w = self.instance_data['w']
        n = self.instance_data['n']
        W = self.instance_data['W']

        self.model.addConstr(quicksum(w[i]*self.X[i] for i in range(n)) <= W)

    def vars_to_readable(self):
        n = self.instance_data['n']

        weights_used = [i for i in range(n) if self.X[i].X > 0.9]

        return weights_used