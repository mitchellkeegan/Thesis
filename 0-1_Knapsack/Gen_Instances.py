#!/home/mitch/anaconda3/envs/Thesis/bin/python

import os
import random
import pickle

n = 500
n_instances = 10000

v_min = 1
v_max = 2000

w_min = 1
w_max = 2000

w_mean = (w_min+w_max)/2

percent_used_min = 0.15
percent_used_max = 0.25

W_min = int(w_mean * percent_used_min * n)
W_max = int(w_mean * percent_used_max * n)

vary_weights = True
vary_values = True
vary_capacity = True
full_capacity_spread = True

strongly_correlated = True

base_dir = '/home/mitch/Documents/Thesis Data/0-1 Knapsack/Instances'

if vary_values:
    save_dir = os.path.join(base_dir,'Values_only')
    param_data = {'Varying': ['v'],
                   'Fixed': ['w','W']}

if vary_weights:
    save_dir = os.path.join(base_dir,'Weights_only')
    param_data = {'Varying': ['w'],
                   'Fixed': ['v', 'W']}

if vary_values and vary_weights:
    save_dir = os.path.join(base_dir,'Weights_and_Values')
    param_data = {'Varying': ['v','w'],
                   'Fixed': ['W']}

if vary_values and vary_weights and vary_capacity:
    save_dir = os.path.join(base_dir, 'Weights_and_Values_and_Capacity')
    param_data = {'Varying': ['v','w','W'],
                   'Fixed': []}

if strongly_correlated:
    save_dir = os.path.join(base_dir, 'Strongly_Correlated_BIG')
    param_data = {'Varying': ['v', 'w', 'W'],
                  'Fixed': []}

param_data['num'] = {'w':n, 'v': n, 'W': 1}

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

weights = [random.randint(w_min,w_max) for _ in range(n)]
values = [random.randint(v_min, v_max) for _ in range(n)]
W = random.randint(W_min, W_max)


for i in range(n_instances):
    rerolls = -1
    # Randomly change any varying parameters
    if vary_values:
        values = [random.randint(v_min, v_max) for _ in range(n)]
    if vary_weights:
        # Ensure optimal solution has at least one weight
        #TODO Look at relaxing this requirement
        weights = [random.randint(w_min, w_max) for _ in range(n)]
        # if min(weights) > W:
        #     print('No Viable Solution')
    if strongly_correlated:
        values = [weights[i] + w_max//10 for i in range(n)]
    if vary_capacity:
        if full_capacity_spread:
            W = (i+1)/(n_instances+1) * sum(weights)
        else:
            W = random.randint(W_min, W_max)
    with open(os.path.join(save_dir,f'instance {i}.txt'), 'w') as f:
        f.write(f'W {W}\n')
        f.write(f'v')
        for v in values:
            f.write(f' {v}')
        f.write('\n')
        f.write('w')
        for w in weights:
            f.write(f' {w}')

with open(os.path.join(save_dir, 'param_types.pickle'), 'wb') as f:
    pickle.dump(param_data, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(save_dir,'Instance Generation Info.txt'),'w') as f:
    f.write(f'{n_instances}\n')
    f.write(f'n {n}\n')
    f.write(f'w {w_min} {w_max}\n')
    f.write(f'v {v_min} {v_max}\n')
    f.write(f'W {W_min} {W_max}\n')
    f.write(f'Capacity Hardness {percent_used_min} {percent_used_max}\n')
    f.write('Varying - ')
    if vary_weights:
        f.write('Weights ')
    if vary_values:
        f.write('Values ')
    if vary_capacity:
        f.write('Capacity ')
    if full_capacity_spread:
        f.write('\nFull Capacity Spread')
    if strongly_correlated:
        f.write('\nStrongly Correlated')