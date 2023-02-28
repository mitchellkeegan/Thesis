import os
from collections import defaultdict
import pandas as pd


def load_instance(size,index):
    B_n_small = []
    B_n_large = []
    prec = []
    dur = []
    p = []
    r_small = defaultdict(lambda: 0)
    r_large = defaultdict(lambda: 0)
    value = {}
    penalty = {}

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
                B_n_small.append(int(line[2]))
                B_n_large.append(int(line[3]))
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

    return B_n_small, B_n_large, prec, dur, p, r_small, r_large, value, penalty, A_r, A_o, A, B

def load_forecast():
    # Load in forecasted base demand and solar supply
    forecast_csv = 'Forecasts-sample.csv'

    forecasts = pd.read_csv(forecast_csv, index_col=0, header=None).T

    base_load = [forecasts[f'Building{i}'].to_list() for i in [0, 1, 3, 4, 5, 6]]
    solar_supply = [forecasts[f'Solar{i}'].to_list() for i in [0, 1, 2, 3, 4, 5]]

    # Load in pricing data
    price_csv = 'PRICE_AND_DEMAND_202011_VIC1.csv'

    price_df = pd.read_csv(price_csv)

    price_30m = price_df['RRP'].to_list()

    price = []

    for rrp in price_30m:
        price.append(rrp)
        price.append(rrp)