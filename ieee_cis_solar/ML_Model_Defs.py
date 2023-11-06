import os
import time
from collections import defaultdict
import itertools
from lib.base_classes import random_forest, neural_network
from lib.autograd_funcs import MyOnehot

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

import numpy as np
import pandas as pd
import pickle

class NN_model_base(neural_network):
    def load_problem_data(self):
        self.load_instance()
        self.load_forecasts()
        self.load_time_sets()

    def load_time_sets(self):
        dur = self.instance_data['dur']
        A_r = self.instance_data['A_r']
        A_o = self.instance_data['A_o']

        self.D_r = range(5)
        self.D_o = range(30)

        # Develop functionality for mapping between different time periods
        T_r = range(8 * 4)
        T_o = range(24 * 4)
        T_r_start = [range(8 * 4 - dur[a] + 1) for a in A_r]
        T_o_start = [range(24 * 4 - dur[a]) for a in A_o]

        T_start = T_r_start + T_o_start

        # There are 2880 times periods in general
        # Split it up into T_bus (business hours) and T_off (out of hours)
        T = range(2880)
        T_bus, T_off = [], []

        Day = 0  # Take Saturday to be the first day of the week (should maybe be sunday?)

        Weekday_business_hours = [set() for _ in range(5)]

        while Day < 30:
            # Check if it's a weekend
            if Day % 7 in [0, 6]:
                T_off.extend(list(range(4 * 24 * Day, 4 * 24 * (Day + 1))))
            else:
                T_off.extend(list(range(4 * 24 * Day,
                                        4 * 24 * Day + 9 * 4)))
                T_bus.extend(list(range(4 * 24 * Day + 9 * 4,
                                        4 * 24 * Day + 17 * 4)))
                T_off.extend(list(range(4 * 24 * Day + 17 * 4,
                                        4 * 24 * (Day + 1))))

                # Keep track of what day of the week each time period that falls within business hours is on
                Weekday_business_hours[Day % 7 - 1].update(range(4 * 24 * Day + 9 * 4, 4 * 24 * Day + 17 * 4))

            Day += 1

        def __T_2_To(T):
            if isinstance(T, int):
                d = T // 96
                t = T - d * 96
                return (d, t)
            try:
                return [(t // 96, t - 96 * (t // 96)) for t in T]
            except:
                raise Exception('__T_2_To only accepts ints or iterables')

        _T_2_To = __T_2_To(list(T))

        def T_2_To(T):
            if isinstance(T, int):
                return _T_2_To[T]
            try:
                return [_T_2_To[t] for t in T]
            except:
                raise Exception('T_2_To only accepts ints or iterables')

        def __T_2_Tr(T):
            if isinstance(T, int):
                for d in self.D_r:
                    if T in Weekday_business_hours[d]:
                        return (d, T % (4 * 24) - (9 * 4))
                return None
            try:
                dt = []
                for t in T:
                    valid_time = False
                    for d in self.D_r:
                        if t in Weekday_business_hours[d]:
                            dt.append((d, t % (4 * 24) - (9 * 4)))
                            valid_time = True
                            break
                    if not valid_time:
                        dt.append(None)
                return dt
            except:
                raise Exception('__T_2_Tr only accepts ints or iterables')

        _T_2_Tr = __T_2_Tr(list(T))

        def T_2_Tr(T):
            if isinstance(T, int):
                return _T_2_Tr[T]
            try:
                return [_T_2_Tr[t] for t in T]
            except:
                raise Exception('T_2_Tr only accepts ints or iterables')


        # Construct the inverses of T_2_Tr and T_2_To?

        self.time_sets = {'T': T,
                          'T_r': T_r,
                          'T_o': T_o,
                          'T_bus': T_bus,
                          'T_off': T_off,
                          'T_start': T_start,
                          'Weekday_business_hours': Weekday_business_hours,
                          'T_2_To': T_2_To,
                          'T_2_Tr': T_2_Tr}

    def load_forecasts(self):
        # Load in forecasted base demand and solar supply (should be sitting in the root instance folder)
        forecast_csv = os.path.join(os.path.dirname(self.instance_dir),'Forecasts-sample.csv')

        forecasts = pd.read_csv(forecast_csv, index_col=0, header=None).T

        base_load = forecasts[[col for col in forecasts.columns if col[:len('Building')] == 'Building']].sum(axis=1).to_list()
        solar_supply = forecasts[[col for col in forecasts.columns if col[:len('Solar')] == 'Solar']].sum(axis=1).to_list()

        # Forecasts are provided in UTC, shift into AEDT and fill in earlier hours
        base_load = base_load[:-11*4]
        solar_supply = solar_supply[:-11*4]

        solar_supply = solar_supply[13*4:24*4] + solar_supply
        base_load = base_load[13*4:24*4] + base_load

        # Load in pricing data
        if self.instance_dir.split('/')[-1] == 'smallArtificial':
            price_df = pd.read_csv(os.path.join(self.instance_dir,'price_forecasts.csv'))
            price = price_df.to_numpy()[:self.available_instances,1:]
        else:
            price_df = pd.read_csv(os.path.join(os.path.dirname(self.instance_dir),'PRICE_AND_DEMAND_202011_VIC1.csv'))

            price_30m = price_df['RRP'].to_list()

            price = []

            for rrp in price_30m:
                price.append(rrp)
                price.append(rrp)

        self.forecasts = {'base_load': base_load,
                          'solar_supply': solar_supply,
                          'price': price}

    def load_instance(self):
        n_small = 0
        n_large = 0
        prec = []
        dur = []
        p = []
        r_small = defaultdict(lambda: 0)
        r_large = defaultdict(lambda: 0)
        value = {}
        penalty = {}
        m = []
        cap = []
        eff = []

        if self.instance_dir.split('/')[-1] == 'smallArtificial':
            f = open(os.path.join(self.instance_dir,'phase2_instance_small.txt'),'r')
            self.loaded_common_instance = True
        else:
            index = self.instance
            f = open(os.path.join(self.instance_dir, f'phase2_instance_{index}.txt'))


        for line in f:
            line = line.split()
            if line[0] == 'ppoi':
                num_r = int(line[4])
                num_o = int(line[5])
                A_r = range(num_r)
                A_o = range(num_r, num_r + num_o)
                A = range(num_r + num_o)


            elif line[0] == 'b':
                n_small += int(line[2])
                n_large += int(line[3])
            elif line[0] == 's':
                continue
            elif line[0] == 'c':
                cap.append(int(line[3]))
                m.append(int(line[4]))
                eff.append(float(line[5]))
            elif line[0] == 'r':
                id = int(line[1])
                if line[3] == 'S':
                    r_small[id] = int(line[2])
                elif line[3] == 'L':
                    r_large[id] = int(line[2])
                else:
                    print('???')

                p.append(int(line[4]))
                dur.append(int(line[5]))

                if int(line[6]) > 0:
                    prec.append([int(x) for x in line[7:]])
                else:
                    prec.append([])

            elif line[0] == 'a':
                id = int(line[1]) + num_r
                if line[3] == 'S':
                    r_small[id] = int(line[2])
                elif line[3] == 'L':
                    r_large[id] = int(line[2])
                else:
                    print('???')

                p.append(int(line[4]))
                dur.append(int(line[5]))

                value[id] = int(line[6])
                penalty[id] = int(line[7])

                if int(line[8]) > 0:
                    prec.append([int(x) + num_r for x in line[9:]])
                else:
                    prec.append([])

        f.close()

        self.instance_data = {'n_small': n_small,
                              'n_large': n_large,
                              'prec': prec,
                              'dur': dur,
                              'p': p,
                              'r_small': r_small,
                              'r_large': r_large,
                              'value': value,
                              'penalty': penalty,
                              'A_r': A_r,
                              'A_o': A_o,
                              'A': A,
                              'cap': cap,
                              'm': m,
                              'eff': eff,
                              'B': range(len(eff))}

class NN_model(NN_model_base):
    def __init__(self, ml_params, training_params, directories, forward_model, ml_model_type=None):
        self.forward_model = forward_model
        super().__init__(ml_params,training_params,directories)

        self.CE_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.MSE_loss = torch.nn.MSELoss(reduction='none')


    def load_data(self,rng_seed=8983):
        self.load_problem_data()

        T = self.time_sets['T']
        T_r = self.time_sets['T_r']
        T_o = self.time_sets['T_o']
        D_r = self.D_r
        D_o = self.D_o

        A_r = self.instance_data['A_r']
        A_o = self.instance_data['A_o']
        B = self.instance_data['B']

        # Load in the price inputs
        X = self.forecasts['price']

        Y = {}
        # Battery states has dimension (batch, time, battery index)
        Y['Battery States'] = torch.zeros((self.available_instances,len(T),len(B)),
                                          dtype=torch.long)
        Y['recurring start times'] = torch.zeros((self.available_instances,len(A_r)),
                                                 dtype=torch.long)
        Y['oneoff start times'] = torch.zeros((self.available_instances,len(A_o)),
                                              dtype=torch.long)

        MIPGaps = []
        Objectives = []

        # Load in solutions for each instance and place into Y
        # for i in range(self.available_instances):
        for i in range(self.available_instances):
            with open(os.path.join(self.opt_result_dir,f'{i}/vars_for_ml.pickle'),'rb') as f:
                output_data = pickle.load(f)
                Y['Battery States'][i, :, 0] = torch.tensor(output_data['Battery States'][0]) - 1
                Y['Battery States'][i, :, 1] = torch.tensor(output_data['Battery States'][1]) - 1
                Y['recurring start times'][i, :] = torch.tensor(output_data['recurring start times'])
                Y['oneoff start times'][i, :] = torch.tensor(output_data['oneoff start times']) + 1   # add one since non scheduled tasks have class -1 -> 0
            with open(os.path.join(self.opt_result_dir,f'{i}/Info.txt'),'r') as f:
                for line in f:
                    line = line.split()
                    if line[0] == 'MIPGap':
                        MIPGaps.append(float(line[-1][:-1]))
                    if line[0] == 'Objective':
                        Objectives.append(float(line[-1]))

        MIPGaps = torch.tensor(MIPGaps)
        Objectives = torch.tensor(Objectives)

        rng = np.random.default_rng(seed=rng_seed)
        p = rng.permutation(self.available_instances)

        val_split_index = int(self.available_instances*0.1)
        test_split_index = int(self.available_instances*0.2)

        p_val = np.sort(p[:val_split_index])
        p_test = np.sort(p[val_split_index:test_split_index])
        p_train = np.sort(p[test_split_index:])

        X_train, X_test, X_val = X[p_train,:], X[p_test,:], X[p_val,:]

        Y_train = {k: outputs[p_train,:] for k,outputs in Y.items()}
        Y_test = {k: outputs[p_test, :] for k, outputs in Y.items()}
        Y_val = {k: outputs[p_val, :] for k, outputs in Y.items()}

        self.input_dict = {'TRAIN': X_train,
                           'TEST': X_test,
                           'VAL': X_val}

        self.output_dict = {'TRAIN': Y_train,
                            'TEST': Y_test,
                            'VAL': Y_val}

        # Add extra data which may be useful
        self.input_dict['dataset size'] = {'TRAIN': len(p_train),
                                           'TEST': len(p_test),
                                           'VAL': len(p_val)}
        self.output_dict['MIPGaps'] = {'TRAIN': MIPGaps[p_train],
                                       'TEST': MIPGaps[p_test],
                                       'VAL': MIPGaps[p_val]}
        self.output_dict['Objectives'] = {'TRAIN': Objectives[p_train],
                                          'TEST': Objectives[p_test],
                                          'VAL': Objectives[p_val]}

        self.norm_groups = (X.max(),X.min())

        self.model_params['n_features'] = len(T)
        self.model_params['T_r'] = len(T_r)*len(D_r)
        self.model_params['T_o'] = len(T) + 1       # Add one class to time series for nonscheduled activities
        self.model_params['n_recurring'] = len(A_r)
        self.model_params['n_oneoff'] = len(A_o)
        self.model_params['n_batteries'] = len(B)

    def create_dataset(self,dataset_type):
        if dataset_type in self.input_dict:
            Dataset = self._create_dataset(self.normalise_inputs(self.input_dict[dataset_type]),
                                                 self.output_dict[dataset_type],
                                                 self.output_dict['MIPGaps'][dataset_type],
                                                 self.output_dict['Objectives'][dataset_type])
        else:
            Dataset = None

        return Dataset

    def _create_dataset(self,X,Y,MIPGap,Objective):
        dataset = TensorDataset(torch.from_numpy(np.expand_dims(X,axis=1)).float(),
                                Y['recurring start times'],
                                Y['oneoff start times'],
                                Y['Battery States'],
                                MIPGap,
                                Objective)
        return dataset

    def unpack_batch(self,batch):
        return batch[0], batch[1:4], batch[4:]

    def loss_function(self,output,Y_true):
        recurring_output, recurring_labels = output[0], Y_true[0]
        oneoff_output, oneoff_labels = output[1], Y_true[1]
        battery_output, battery_labels = output[2], Y_true[2]

        # oneoff_probs = torch.zeros((64, 2880, 20))
        # for batch in range(64):
        #     for a in range(20):
        #         label_middle = oneoff_labels[]
        #         oneoff_probs[batch,oneoff_labels[],]

        recurring_loss = self.CE_loss(recurring_output, recurring_labels)
        oneoff_loss = self.CE_loss(recurring_output, recurring_labels)
        battery_loss = self.CE_loss(battery_output,battery_labels)

        return torch.mean(recurring_loss,dim=1) + torch.mean(oneoff_loss,dim=1) + torch.mean(battery_loss,dim=(1,2))

    def constraint_loss_function(self, g):
        # Return a vector with the constraint loss for each batch

        # Evaluate the constraint loss due to violation of the battery minimum
        battery_min_eval = g['battery min']
        battery_capacity_eval = g['battery capacity']


        raw_losses = {}
        raw_losses['battery'] = torch.mean(battery_capacity_eval + battery_min_eval, dim=(1,2))

        return raw_losses

    def train_decode(self, network_output):
        battery = torch.softmax(network_output[2],dim=1)

        # Moving the class dimension to the end to apply the onehot function
        # because I couldn't get vmap to work with batch dimensions in dim 0,2,3 (i.e. needed first 3 dimensions to be batched)
        battery_onehot = torch.movedim(MyOnehot(torch.movedim(battery,1,3))[0],3,1)

        return battery_onehot,

    def test_decode(self,network_output):
        # Return the following:
        # recurring - (batch x D_r x T_r x A_r) = 1 if class a in A_r is running in period t in T_r on day d in D_r
        # TODO Finish?

        recurring, oneoff, battery = tuple(torch.argmax(no,dim=1) for no in network_output)
        oneoff = oneoff - 1

        return recurring, oneoff, battery

    def data_for_network(self):
        if 'price' in self.forecasts:
            self.forecasts.pop('price')
        return self.instance_data | self.time_sets | self.forecasts

    def normalise_inputs(self,X):
        X_out = X.copy()

        X_max, X_min = self.norm_groups
        X_out = (X_out - X_min)/(X_max-X_min + 1e-12)

        return X_out

    def unnormalise_inputs(self,X):
        if torch.is_tensor(X):
            X_out = X.detach().clone()
        elif isinstance(X, np.ndarray):
            X_out = X.copy()

        X_max, X_min = self.norm_groups
        X_out =  X_min + X_out*(X_max - X_min)

        return X_out

    def create_coefficient_dict(self,X,fixed_params):
        # Create a dictionary of coefficients
        # Squeeze out the 'channel' dimension of X that was required for the network
        return {'price':torch.squeeze(X,dim=1)} | fixed_params

    def _construct_decision_variables_train(self, decoded_output, coeffs):
        battery_onehot, = decoded_output

        # Construct the battery charge/discharge states
        Z_c = battery_onehot[:, 0, :, :]
        Z_d = battery_onehot[:, 1, :, :]

        # Construct the battery charge at each timestep
        cap = torch.tensor(coeffs['cap']).reshape(1,2)
        m = torch.tensor(coeffs['m']).reshape(1,2)

        charge_or_discharge = m * (Z_c - Z_d)

        charge_or_discharge_cumulative = torch.cumsum(charge_or_discharge, dim=1)
        battery_storage = torch.zeros_like(Z_c)
        battery_storage[:, 0, :] = cap - Z_d[:, 0, :]
        battery_storage[:, 1:, :] = battery_storage[:, 0:1, :] + charge_or_discharge_cumulative[:, 1:, :]

        # battery_storage = torch.zeros_like(Z_c)
        # battery_storage[:,0,:] = cap - Z_d[:,0,:]
        #
        # n_timesteps = battery_storage.shape[1]
        #
        # for t in range(1,n_timesteps):
        #     battery_storage[:, t, :] = battery_storage[:, t - 1, :] + m * Z_c[:, t, :] - m * Z_d[:, t, :]



        return {'Z_c': Z_c,
                'Z_d': Z_d,
                'battery storage': battery_storage}

    def _construct_decision_variables_test(self, decoded_output, coeffs):
        # To evaluate the objective function and the constraints, we might want to following variables

        A_r = coeffs['A_r']
        A_o = coeffs['A_o']
        T = len(coeffs['T'])
        T_r = len(coeffs['T_r'])
        B = len(coeffs['B'])
        dur = coeffs['dur']
        p = coeffs['p']
        r_small = coeffs['r_small']
        r_large = coeffs['r_large']
        cap = coeffs['cap']
        m = coeffs['m']
        solar = coeffs['solar_supply']
        base_load = coeffs['base_load']
        eff = coeffs['eff']

        recurring, oneoff, battery = decoded_output

        batch_size, _ = recurring.shape

        # Create tensors of zeros, we will fill in timesteps with classes scheduled with ones
        # For the recurring schedule start with one week then use that to construct the full schedule
        X_r_week = torch.zeros((batch_size, len(A_r), 24*4*7),dtype=torch.bool)
        X_o = torch.zeros((batch_size, len(A_o), T),dtype=torch.bool)

        dur_o = torch.tensor([dur[a] for a in A_o])

        oneoff_active_batch_idx = []
        oneoff_active_class_idx = []
        oneoff_active_time_idx = []

        for batch in range(batch_size):
            start_times = oneoff[batch,:]
            end_times = torch.clamp(start_times + dur_o,max=T-1)
            diff = end_times-start_times
            diff[torch.where(start_times == -1)[0]] = 0     # Set diff to one if the classification was -1 (i.e. no task scheduled)
            oneoff_active_batch_idx.extend([batch]*diff.sum())
            oneoff_active_class_idx.extend(itertools.chain.from_iterable([[a]*diff[a] for a in range(len(A_o))]))
            oneoff_active_time_idx.extend(itertools.chain.from_iterable([list(range(start_times[a],end_times[a])) for a in range(len(A_o)) if diff[a]>0]))

        X_o[oneoff_active_batch_idx,oneoff_active_class_idx,oneoff_active_time_idx] = 1

        dur_r = torch.tensor([dur[a] for a in A_r])
        recurring_active_batch_idx = []
        recurring_active_class_idx = []
        recurring_active_time_idx = []

        T_9am = 9*4     # Number of timesteps to get to 9am
        T_day = 24*4
        d_r = torch.div(recurring, T_r, rounding_mode='floor')
        t_r = recurring % T_r

        for batch in range(batch_size):
            # Convert the start times into timesteps indexed on the week range(0,24*4)
            start_times = T_9am + T_day * d_r[batch, :] + t_r[batch,:]
            end_times = start_times + dur_r     # TODO: Figure out how to check for recurring classes which run past 5pm
            diff = end_times - start_times

            recurring_active_batch_idx.extend([batch] * diff.sum())
            recurring_active_class_idx.extend(itertools.chain.from_iterable([[a] * diff[a] for a in range(len(A_r))]))
            recurring_active_time_idx.extend(itertools.chain.from_iterable([list(range(start_times[a], end_times[a])) for a in range(len(A_r))]))

        X_r_week[recurring_active_batch_idx,recurring_active_class_idx,recurring_active_time_idx] = 1

        # Concatenate X_r_week together to get full timeseries
        X_r = torch.cat((torch.zeros((batch_size,len(A_r),T_day),dtype=torch.bool),) + tuple(X_r_week for _ in range(4)) + (X_r_week[:,:,:T_day],),dim=2)


        # Construct the class power demands
        demand_per_class_r = torch.tensor([p[a]*(r_small[a]+r_large[a]) for a in A_r]).reshape(1,len(A_r),1)
        demand_per_class_o = torch.tensor([p[a]*(r_small[a]+r_large[a]) for a in A_o]).reshape(1,len(A_o),1)
        class_demand = torch.sum(demand_per_class_r * X_r, dim=1) + torch.sum(demand_per_class_o * X_o, dim=1)

        # Construct the battery charge/discharge states
        Z_c = battery == 0
        Z_d = battery == 1

        # Get the battery storage at each timestep
        cap = torch.tensor(cap).reshape(1,2)
        m = torch.tensor(m).reshape(1,2)
        battery_storage = torch.zeros((batch_size,T,B))
        battery_storage[:,0,:] = cap
        for t in range(1,T):
            battery_storage[:,t,:] = battery_storage[:,t-1,:] + m*Z_c[:,t,:] - m*Z_d[:,t,:]

        # Calculate the grid power
        solar = torch.tensor(solar).reshape(1,T)
        base_load = torch.tensor(base_load).reshape(1, T)
        eff = torch.sqrt(torch.tensor(eff)).reshape(1,1,2)
        m = m.reshape(1,1,2)

        grid_power = class_demand + base_load - solar - torch.sum(m * (eff * Z_d - torch.div(Z_c,eff)), dim=2)

        return {'Y_o': oneoff,
                'Y_r': recurring,
                'X_r': X_r,
                'X_o': X_o,
                'class demand': class_demand,
                'Z_c': Z_c,
                'Z_d': Z_d,
                'battery storage': battery_storage,
                'grid power': grid_power}

    def construct_decision_variables(self, decoded_output, coeffs, mode='test'):
        if mode == 'test':
            return self._construct_decision_variables_test(decoded_output,coeffs)
        elif mode == 'train':
            return self._construct_decision_variables_train(decoded_output,coeffs)
        else:
            raise Exception('Inequality Constraints Method Requires a valid mode (train or test)')

    def evaluate_objective(self,decision_vars,coeffs):
        # TODO Add Penalty Term
        Y_o = decision_vars['Y_o']
        grid_power = decision_vars['grid power']
        A_o = coeffs['A_o']
        price = coeffs['price']
        value = coeffs['value']
        penalty = coeffs['penalty']
        T = len(coeffs['T'])
        T_off = coeffs['T_off']

        batch_size, _ = Y_o.shape

        power_cost = (0.25/1000) * torch.sum(grid_power * price, dim=1)

        # Calculate the value from scheduling oneoff classes
        value = torch.tensor([value[a] for a in A_o])
        class_scheduled = Y_o >= 0  # elements where Y_o == -1 means the class was not scheduled

        # Multiply scheduled classes by corresponding class value and the sum over the class to get total value from oneoff classes
        class_value = torch.sum((Y_o >= 0) * value, dim=1)

        # Calculate the total penalty from scheduling oneoff classes outside of business hours
        # Use torch.isin to create a boolean tensor with dim (batch_size x |A_o|) that's true when the corresponding class
        # is scheduled outside of business hours. Then multiply by penalty to get associated penalty and sum across A_o
        penalty = torch.tensor([penalty[a] for a in A_o])
        class_penalty = torch.sum(torch.isin(Y_o,torch.tensor(T_off)) * penalty, dim=1)

        return power_cost - class_value + class_penalty

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

    def _inequality_constraints_train(self, decision_vars, coeffs, ctype):

        if ctype == 'violation':
            v = lambda x: torch.relu(x)
        else:
            v = lambda x: x

        inequality_dict = {}

        battery_storage = decision_vars['battery storage']

        # Minimium storage constraint -C_{bt} <= 0 for all b,t
        inequality_dict['battery min'] = v(-battery_storage)

        # Minimium storage constraint C_{bt} - cap_b <= 0 for all b,t
        cap = torch.tensor(coeffs['cap']).reshape(1, 1, 2)
        inequality_dict['battery capacity'] = v(battery_storage - cap)

        return inequality_dict

    def _inequality_constraints_test(self, decision_vars, coeffs, ctype):
        # TODO: Update prececdence violations to consider days instead of timeslots

        # This method defines the inequality constraints of the model

        if ctype == 'violation':
            v = lambda x: torch.relu(x)
        else:
            v = lambda x: x

        battery_storage = decision_vars['battery storage']
        cap = coeffs['cap']

        inequality_dict = {}

        # Minimium storage constraint -C_{bt} <= 0 for all b,t
        inequality_dict['battery min'] = v(-battery_storage)

        # Minimium storage constraint C_{bt} - cap_b <= 0 for all b,t
        cap = torch.tensor(cap).reshape(1, 1, 2)
        inequality_dict['battery capacity'] = v(battery_storage - cap)

        inequality_dict['batter capacity or min'] = inequality_dict['battery min'] + inequality_dict['battery capacity']

        # Small classrooms available
        n_small = coeffs['n_small']
        A_r = coeffs['A_r']
        A_o = coeffs['A_o']
        r_small_r = torch.tensor([coeffs['r_small'][a] for a in A_r]).reshape(1, len(A_r), 1)
        r_small_o = torch.tensor([coeffs['r_small'][a] for a in A_o]).reshape(1, len(A_o), 1)
        X_r = decision_vars['X_r']
        X_o = decision_vars['X_o']

        inequality_dict['small rooms'] = v(
            torch.sum(r_small_r * X_r, dim=1) + torch.sum(r_small_o * X_o, dim=1) - n_small).float()

        # Large classrooms available
        n_large = coeffs['n_large']
        r_large_r = torch.tensor([coeffs['r_large'][a] for a in A_r]).reshape(1, len(A_r), 1)
        r_large_o = torch.tensor([coeffs['r_large'][a] for a in A_o]).reshape(1, len(A_o), 1)

        inequality_dict['large rooms'] = v(
            torch.sum(r_large_r * X_r, dim=1) + torch.sum(r_large_o * X_o, dim=1) - n_large).float()

        # Precedence Constraints
        # Structure precedence constraints as number of class in
        # i.e. for a given batch the constraint evaluation is a vector of length |A|
        # where the a^th element is the number of tasks in prec_a which do not satisfy the precedence constraint
        prec = coeffs['prec']
        A = coeffs['A']
        Y_o = decision_vars['Y_o']
        Y_r = decision_vars['Y_r']

        batch_size, _ = Y_o.shape
        prec_violated = torch.zeros((batch_size, len(A)), dtype=torch.float)

        # Add one to the array of max indices so that no class scheduled becomes class 0 in the onehot encoding, then cut off the unscheduled class index
        Y_o_onehot = F.one_hot(Y_o + 1, num_classes=2881)[:, :, 1:]

        # Get the day that each class is scheduled for
        Y_o_days = torch.div(Y_o, 96, rounding_mode='floor')
        Y_r_days = torch.div(Y_r, 96, rounding_mode='floor')

        for a in A_o:
            prec_a = prec[a]
            a = a - len(A_r)  # Offset so that we can properly index Y_o in range(len(A_o))
            # For each activity in prec_a, we want to check if it's scheduled before activity a
            for aa in prec_a:
                aa = aa - len(A_r)
                prec_violated[:, a + len(A_r)] += Y_o_days[:, a] < Y_o_days[:, aa]

        for a in A_r:
            prec_a = prec[a]
            for aa in prec_a:
                prec_violated[:, a] += Y_r_days[:, a] <= Y_r_days[:, aa]

        inequality_dict['precedence'] = prec_violated

        return inequality_dict

    def inequality_constraints(self,decision_vars,coeffs,mode='test',ctype='violation'):
        # TODO: This is kind of stupid, should find cleaner solution
        if mode == 'test':
            return self._inequality_constraints_test(decision_vars,coeffs,ctype)
        elif mode == 'train':
            return self._inequality_constraints_train(decision_vars,coeffs,ctype)
        else:
            raise Exception('Inequality Constraints Method Requires a valid mode (train or test)')

