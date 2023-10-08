import numpy as np
from gurobipy import quicksum, GRB
import os
from math import floor

from lib.custom_metrics import Approximation_Ratio
from lib.base_classes import base_opt_model

class vanilla_IP(base_opt_model):
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

    def save_model_output(self):
        weights_used = self.vars_to_readable()

        with open(os.path.join(self.results_directory,'Weights Used.txt'),'w') as f:
            for w in weights_used:
                f.write(f'{w} ')

    def load_instance(self,index=None):
        if index is None:
            index = self.opt_params['instance index']
        else:
            self.opt_params['instance index'] = index

        with open(os.path.join(self.instance_dir, f'instance {index}.txt')) as f:
            for line in f:
                line = line.split()
                if line[0] == 'n':
                    n = int(line[1])
                elif line[0] == 'W':
                    W = floor(float(line[1]))
                elif line[0] == 'v':
                    v = [int(x) for x in line[1:]]
                elif line[0] == 'w':
                    w = [int(x) for x in line[1:]]

        with open(os.path.join(self.instance_dir, f'Instance Generation Info.txt')) as f:
            for line in f:
                line = line.split()
                if line[0] == 'n':
                    n = int(line[1])

        self.instance_data = {'n': n,
                              'W': W,
                              'v': v,
                              'w': w}

class constraint_metrics():
    def __init__(self):
        pass

    def eqc(self,coeff,pred,instances=None):
        return []

    def ineqc(self,coeff,pred,instances=None):
        W = coeff['W'].squeeze()
        w = coeff['w']

        n_instances,_ = pred.shape

        # diff = W - (w * pred).sum(axis=1,keepdims=True)
        diff = W - (w * pred).sum(axis=1)

        violated = diff < 0
        unviolated = diff >= 0

        violated_percent = -100*diff[violated]/W[violated]

        if violated_percent.sum() == 0:
            p = f'0/{len(W)} Inequality Constraints Violated'
            return [p]


        min_violation = violated_percent.min()
        max_violation = violated_percent.max()
        # mean_violation = violated_percent.mean()
        mean_violation = violated_percent[abs(violated_percent - np.mean(violated_percent)) < 6 * np.std(violated_percent)].mean()

        p1 = f'{violated.sum()}/{n_instances} Inequality Constraints Violated ({100*(violated.sum()/n_instances):.2f}%)'
        p2 = f'Constraint Violation Statistics:  MEAN - {mean_violation:.2f} | MIN - {min_violation:.2f} | MAX - {max_violation:.2f}'

        print(p1)
        print(p2)

        return [p1,p2]

def objective_metrics(coeff,pred,actual_solution,instances=None):

    n_samples = len(pred)

    v = coeff['v']

    obj_val_pred = (v * pred).sum(axis=1)
    obj_val_actual = (v * actual_solution).sum(axis=1)

    AR = Approximation_Ratio(coeff,pred,actual_solution,reduce='mean')


    diff = obj_val_actual - obj_val_pred

    optimal_idx = abs(diff) <= 1e-6

    under_idx = diff > 1e-6
    over_idx = diff < -1e-6

    idx_actual_has_weights = (obj_val_actual != 0)

    # Filter out cases where the actual solution has no weights, since this will lead to inf in the following calculations
    over_idx = over_idx & idx_actual_has_weights

    over_percent = 100 * (-diff[over_idx]/obj_val_actual[over_idx])
    under_percent = 100 * (diff[under_idx]/obj_val_actual[under_idx])

    p1 = f'Objective Statistics (%): UNDER OPTIMALITY - {100*under_idx.sum()/n_samples:.2f} | OPTIMAL - {100*optimal_idx.sum()/n_samples:.2f} | OVER OPTIMALITY - {100*over_idx.sum()/n_samples:.2f}'
    if len(over_percent) > 0:
        p2 = f'Over Objective Statistics (%): MIN - {over_percent.min():.2f} | MEAN - {over_percent.mean():.2f} | MAX - {over_percent.max():.2f}'
    else:
        p2 = 'No samples overshoot optimal objective'
    if len(under_percent) > 0:
        p3 = f'Under Objective Statistics (%): MIN - {under_percent.min():.2f} | MEAN - {under_percent.mean():.2f} | MAX - {under_percent.max():.2f}'
    else:
        p3 = 'No samples undershoot optimal objective'

    p4 = f'Approximation Ratio = {AR:.5f}'

    print(p1)
    print(p2)
    print(p3)
    print(p4)
    return [p1,p2,p3,p4]

