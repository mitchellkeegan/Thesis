import os
from collections import defaultdict
import pandas as pd
import pickle
from datetime import datetime

def load_instance(size,index):
    n_small = 0
    n_large = 0
    prec = []
    dur = []
    p = []
    r_small = defaultdict(lambda: 0)
    r_large = defaultdict(lambda: 0)
    value = {}
    penalty = {}

    with open(os.path.join('Instances', f'phase2_instance_{size}_{index}.txt')) as f:
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

    return n_small, n_large, prec, dur, p, r_small, r_large, value, penalty, A_r, A_o, A, B

def load_forecast():
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

    return base_load, solar_supply, price

def generate_columns(A, A_r, T_r, D_r, A_o, T_o, D_o, T_bus, dur, value, penalty):
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

    return theta, class_value, active_recurring_schedules, active_oneoff_schedules, G, K

def save_solution(model,opt_params):
    instance_size = opt_params['instance size']
    instance_index = opt_params['instance index']
    model_type = opt_params['name']

    results_directory = f'Results/{model_type} {instance_size} - {instance_index}'

    if not os.path.exists(results_directory):
        os.mkdir(results_directory)

    status_dict = defaultdict(lambda x: '???')
    status_dict[2] = 'Optimal Solution Found'
    status_dict[3] = 'Infeasible'
    status_dict[9] = 'Time Limit Reached'

    # Store the date and time the solution was generated
    with open(os.path.join(results_directory,'Info.txt'),'w') as f:
        f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        f.write(f'\nObjective - ${model.objVal:.2f}\n')
        f.write(f'Solve Time - {model.runTime:.2f}s\n')
        f.write(f'Status - {model.Status} ({status_dict[model.Status]})\n')
        f.write(f'NumVars - {model.NumVars}\n')
        f.write(f'NumConstrs - {model.NumConstrs}\n')
        f.write(f'MIPGap - {100*model.MIPGap:.3f}%\n')

    # Store the optimisation parameters used
    with open(os.path.join(results_directory,'opt_params.pickle'),'wb') as f:
        pickle.dump(opt_params, f, protocol=pickle.HIGHEST_PROTOCOL)

    model.write(os.path.join(results_directory,'solution.sol'))

    return