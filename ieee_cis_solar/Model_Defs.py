from gurobipy import Model, quicksum, GRB, read
import math
import pickle
import os
import pandas as pd
from collections import defaultdict
from abc import ABC, abstractmethod
from datetime import datetime

class base_model(ABC):
    def __init__(self,opt_params):

        self.D_r = range(5)
        self.D_o = range(30)

        self.opt_params = opt_params

        self.model = Model()
        self.model.Params.MIPGap = opt_params['MIPGap']
        if opt_params['instance size'] == 'large':
            self.model.Params.Threads = opt_params['threads']
    
    def optimize_model(self):
        self.model.optimize()

    def load_model(self,instance_size, instance_index):
        results_directory = f'Results/{self.model_type} {self.opt_params["instance size"]} - {self.opt_params["instance index"]}'
        with open(os.path.join(results_directory, 'opt_params.pickle'), 'rb') as f:
            self.opt_params = pickle.load(f)

        self.load_instance(instance_size, instance_index)
        self.load_time_sets()

        # Create a fresh model and create variables for it
        self.add_vars()

        self.model.read(os.path.join(results_directory,'solution.sol'))
        self.model.optimize()

    def load_time_sets(self):
        dur = self.instance_data['dur']
        A_r = self.instance_data['A_r']
        A_o = self.instance_data['A_o']
        D_r = self.D_r


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

        Day = 0  # Take Saturday to be the first day of the week

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

        def T_2_To(T):
            if isinstance(T, int):
                d = T // 96
                t = T - d * 96
                return (d, t)
            if isinstance(T, list):
                return [(t // 96, t - 96 * (t // 96)) for t in T]

        def T_2_Tr(T):
            if isinstance(T, int):
                for d in D_r:
                    if T in Weekday_business_hours[d]:
                        return (d, T % (4 * 24) - (9 * 4))
                return None
            elif isinstance(T, list):
                dt = []
                for t in T:
                    for d in D_r:
                        if t in Weekday_business_hours[d]:
                            dt.append((d, T % (4 * 24) - (9 * 4)))
                        else:
                            dt.append(None)
            else:
                return None

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
        # Load in forecasted base demand and solar supply
        forecast_csv = 'Forecasts-sample.csv'

        forecasts = pd.read_csv(forecast_csv, index_col=0, header=None).T

        base_load = forecasts[[col for col in forecasts.columns if col[:len('Building')] == 'Building']].sum(axis=1).to_list()
        solar_supply = forecasts[[col for col in forecasts.columns if col[:len('Solar')] == 'Solar']].sum(axis=1).to_list()

        # Load in pricing data
        price_csv = 'PRICE_AND_DEMAND_202011_VIC1.csv'

        price_df = pd.read_csv(price_csv)

        price_30m = price_df['RRP'].to_list()

        price = []

        for rrp in price_30m:
            price.append(rrp)
            price.append(rrp)

        self.forecasts = {'base_load': base_load,
                          'solar_supply': solar_supply,
                          'price': price}

    def load_instance(self,size=None,index=None):
        n_small = 0
        n_large = 0
        prec = []
        dur = []
        p = []
        r_small = defaultdict(lambda: 0)
        r_large = defaultdict(lambda: 0)
        value = {}
        penalty = {}

        if size is None:
            size = self.opt_params['instance size']
            index = self.opt_params['instance index']

        with open(os.path.join('instances', f'phase2_instance_{size}_{index}.txt')) as f:
            for line in f:
                line = line.split()
                if line[0] == 'ppoi':
                    num_r = int(line[4])
                    num_o = int(line[5])
                    A_r = range(num_r)
                    A_o = range(num_r, num_r + num_o)
                    A = range(num_r + num_o)

                    B = range(int(line[2]))

                elif line[0] == 'b':
                    n_small += int(line[2])
                    n_large += int(line[3])
                elif line[0] == 's':
                    continue
                elif line[0] == 'c':
                    continue
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
                              'B': B}

    def setup_and_optimize(self):
        self.load_instance(self.opt_params['instance size'], self.opt_params['instance index'])
        self.load_time_sets()
        self.load_forecasts()
        self.add_vars()
        self.add_objective()
        self.add_constraints()
        self.optimize_model()

    def save_model(self):
        # Call after optimisation, saves solution to file in multiple formats, along with opt info

        dur = self.instance_data['dur']
        A_r = self.instance_data['A_r']
        A_o = self.instance_data['A_o']
        # T_2_To = self.time_sets['T_2_To']

        instance_size = self.opt_params['instance size']
        instance_index = self.opt_params['instance index']
        model_type = self.model_type

        results_directory = f'Results/{model_type} {instance_size} - {instance_index}'
        if not os.path.exists(results_directory):
            os.mkdir(results_directory)

        activity_start_times, grid_power = self.vars_to_readable()

        def Tro_2_Time(t):
            return f'{math.floor(t / 4)}:{t % 4 * 15}' + ('0' if t % 4 * 15 == 0 else '')

        with open(os.path.join(results_directory,'Schedule.txt'),'w') as f:
            for a, (d,t) in activity_start_times.items():
                f.write(f'({"r" if a in A_r else "o"}) a{a}: Day {d+1} {Tro_2_Time(t)}-{Tro_2_Time(t + dur[a])}\n')
        with open(os.path.join(results_directory,'Grid Power.txt'),'w') as f:
            for d, grid_power_day in enumerate(grid_power):
                line = f'Day {d+1}: '
                for t, power in enumerate(grid_power_day):
                    line += f'{power:.2f} '
                    if (t+1) % 4 == 0:
                        if (t+1)//4 in [9,15]:
                            line += '||||'
                        else:
                            line += '|'
                f.write(line + '\n')

        # Store the solution such that it can be loaded back into a Gurobi model
        self.model.write(os.path.join(results_directory, 'solution.sol'))

        # Store the optimisation parameters used
        with open(os.path.join(results_directory, 'opt_params.pickle'), 'wb') as f:
            pickle.dump(self.opt_params, f, protocol=pickle.HIGHEST_PROTOCOL)

        status_dict = defaultdict(lambda x: '???')
        status_dict[2] = 'Optimal Solution Found'
        status_dict[3] = 'Infeasible'
        status_dict[9] = 'Time Limit Reached'

        # Store the date and time the solution was generated
        with open(os.path.join(results_directory, 'Info.txt'), 'w') as f:
            f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            f.write(f'\nObjective - ${self.model.objVal:.2f}\n')
            f.write(f'Solve Time - {self.model.runTime:.2f}s\n')
            f.write(f'Status - {self.model.Status} ({status_dict[self.model.Status]})\n')
            f.write(f'NumVars - {self.model.NumVars}\n')
            f.write(f'NumConstrs - {self.model.NumConstrs}\n')
            f.write(f'MIPGap - {100 * self.model.MIPGap:.3f}%\n')



    @abstractmethod
    def add_vars(self):
        pass

    @abstractmethod
    def add_constraints(self):
        pass

    @abstractmethod
    def add_objective(self):
        pass

    @abstractmethod
    def vars_to_readable(self):
        # This function should convert the decision variables into:
        # Something saveable in a human readable format (for each important variable)
        pass


class vanilla_model(base_model):
    def __init__(self, opt_params):
        self.model_type = 'Base MIP'
        super().__init__(opt_params)

class column_gen(base_model):
    def __init__(self,opt_params):
        self.model_type = 'Column Generation'
        super().__init__(opt_params)

    def generate_columns(self):
        A = self.instance_data['A']
        A_r = self.instance_data['A_r']
        A_o = self.instance_data['A_o']
        dur = self.instance_data['dur']
        value = self.instance_data['value']
        penalty = self.instance_data['penalty']

        T_r = self.time_sets['T_r']
        T_o = self.time_sets['T_o']
        T_bus =self.time_sets['T_bus']

        D_r = self.D_r
        D_o = self.D_o

        theta = []
        class_value = []
        active_recurring_schedules = {}
        active_oneoff_schedules = {}
        G = {}

        for a in A:
            theta.append([])
            class_value.append([])
            G[a] = []

            if a in A_r:
                T_a = T_r
                D_a = D_r
            elif a in A_o:
                T_a = T_o
                D_a = D_o
            else:
                print('\n\n???????\n\n')

            k = 0  # Keep track of how many schedules have been created for activity a

            for d in D_a:
                G[a].append([])
                for t in T_a:
                    if t + dur[a] > len(T_a):
                        break
                    else:
                        G[a][d].append(k)
                        theta[a].append((d, t))
                        if a in A_o:
                            tt = d * 4 * 24 + t
                            # Check if class happens during business hours
                            if tt in T_bus and tt + dur[a] - 1 in T_bus:
                                class_value[a].append(value[a])
                            else:
                                class_value[a].append(value[a] - penalty[a])

                        for tt in range(t, t + dur[a]):
                            if a in A_r:
                                if (d, tt) not in active_recurring_schedules:
                                    active_recurring_schedules[d, tt] = []
                                active_recurring_schedules[d, tt].append((a, k))

                            if a in A_o:
                                if (d, tt) not in active_oneoff_schedules:
                                    active_oneoff_schedules[d, tt] = []
                                active_oneoff_schedules[d, tt].append((a, k))

                        k += 1

        K = [range(len(theta[a])) for a in A]

        self.column_info = {'theta': theta,
                            'class_value': class_value,
                            'ars': active_recurring_schedules,
                            'aos': active_oneoff_schedules,
                            'G': G,
                            'K': K}

    def add_vars(self):
        # Check if columns have been generated yet and generate if not
        if not hasattr(self,'column_info'):
            self.generate_columns()

        K = self.column_info['K']
        class_value = self.column_info['class_value']
        A = self.instance_data['A']
        A_o = self.instance_data['A_o']
        T = self.time_sets['T']
        price = self.forecasts


        self.X = {(a, k): self.model.addVar(vtype=GRB.BINARY, name=f'X[{a},{k}]')
                  for a in A
                  for k in K[a]}

        self.grid_power = {t: self.model.addVar(vtype=GRB.CONTINUOUS, name=f'grid[{t}]')
                           for t in T}

        self.class_demand = {t: self.model.addVar(vtype=GRB.CONTINUOUS, name=f'class[{t}]')
                             for t in T}

    def add_objective(self):

        K = self.column_info['K']
        class_value = self.column_info['class_value']
        A = self.instance_data['A']
        A_o = self.instance_data['A_o']
        T = self.time_sets['T']
        price = self.forecasts['price']

        oneoff_activity_payments = quicksum(self.X[a, k] * class_value[a][k] for a in A_o for k in K[a])
        grid_power_cost = (0.25 / 1000) * quicksum(self.grid_power[t] * price[t] for t in T)

        self.model.setObjective(grid_power_cost - oneoff_activity_payments)

    def add_constraints(self):

        K = self.column_info['K']
        G = self.column_info['G']
        active_oneoff_schedules = self.column_info['aos']
        active_recurring_schedules = self.column_info['ars']

        A_o = self.instance_data['A_o']
        A_r = self.instance_data['A_r']
        r_small = self.instance_data['r_small']
        r_large = self.instance_data['r_large']
        n_small = self.instance_data['n_small']
        n_large = self.instance_data['n_large']
        p = self.instance_data['p']
        prec = self.instance_data['prec']

        T = self.time_sets['T']
        T_2_To = self.time_sets['T_2_To']
        T_2_Tr = self.time_sets['T_2_Tr']

        base_load = self.forecasts['base_load']
        solar_supply = self.forecasts['solar_supply']

        # Choose one schedule for each recurring activity
        choose_one_schedule = {a: self.model.addConstr(quicksum(self.X[a, k] for k in K[a]) == 1)
                               for a in A_r}
        # Choose at most one schedule for each one-off activity
        choose_up_to_one_schedule = {a: self.model.addConstr(quicksum(self.X[a, k] for k in K[a]) <= 1)
                                     for a in A_o}

        # Calculate number of classrooms and power used in each time period
        for t in T:
            d, tt = T_2_To(t)

            small_rooms_used = quicksum(self.X[a, k] * r_small[a] for (a, k) in active_oneoff_schedules[d, tt])
            large_rooms_used = quicksum(self.X[a, k] * r_large[a] for (a, k) in active_oneoff_schedules[d, tt])
            power_used = quicksum(self.X[a, k] * p[a] * (r_small[a] + r_large[a]) for (a, k) in active_oneoff_schedules[d, tt])

            recurring_time = T_2_Tr(t)

            if recurring_time is not None:
                d, tt = recurring_time
                small_rooms_used += quicksum(self.X[a, k] * r_small[a] for (a, k) in active_recurring_schedules[d, tt])
                large_rooms_used += quicksum(self.X[a, k] * r_large[a] for (a, k) in active_recurring_schedules[d, tt])
                power_used += quicksum(self.X[a, k] * p[a] * (r_small[a] + r_large[a]) for (a, k) in active_recurring_schedules[d, tt])

            self.model.addConstr(small_rooms_used <= n_small)
            self.model.addConstr(large_rooms_used <= n_large)
            self.model.addConstr(self.class_demand[t] == power_used)

        # Precedence constraints
        recur_precedence = {(a, d): self.model.addConstr(len(prec[a]) * quicksum(self.X[a, k] for k in G[a][d]) <=
                                                         quicksum(self.X[aa, k] for aa in prec[a] for dd in range(d) for k in G[aa][dd]))
                            for a in A_r
                            for d in self.D_r}

        oneoff_precedence = {(a, d): self.model.addConstr(len(prec[a]) * quicksum(self.X[a, k] for k in G[a][d]) <=
                                                 quicksum(
                                                     self.X[aa, k] for aa in prec[a] for dd in range(d) for k in G[aa][dd]))
                             for a in A_o
                             for d in self.D_o}

        # Link grid power to other power supplies/demands
        equate_power = {t: self.model.addConstr(self.grid_power[t] + solar_supply[t] == base_load[t] + self.class_demand[t])
                        for t in T}

    def vars_to_readable(self):

        K = self.column_info['K']
        theta = self.column_info['theta']
        A = self.instance_data['A']
        A_r = self.instance_data['A_r']
        dur = self.instance_data['dur']
        T = self.time_sets['T']
        T_2_To = self.time_sets['T_2_To']

        activity_start_times = {}

        for a in A:
            printout = f'a{a}: '
            for k in K[a]:
                if self.X[a, k].X > 0.9:
                    d, t = theta[a][k]
                    if a in A_r:
                        t += 9 * 4
                    activity_start_times[a] = (d,t)
                    break


        grid_power = [[] for _ in self.D_o]
        for t in T:
            d, tt = T_2_To(t)
            grid_power[d].append(self.grid_power[t].X)

        return activity_start_times, grid_power