from gurobipy import Model, quicksum, GRB
import math

from helper_functions import load_instance, load_forecast, generate_columns, save_solution


# TODO
# Add batteries
# Add quadratic term in objective

# Threads only applies for large instances
opt_params = {'name': 'Column Generation',
              'instance size': 'small',
              'instance index': 0,
              'threads': 1,
              'MIPGap': 0}

D_r = range(5)
D_o = range(30)

# Load in data instance
n_small, n_large, prec, dur, p, r_small, r_large, value, penalty, A_r, A_o, A, B = load_instance(opt_params['instance size'], opt_params['instance index'])

# Load in the forecast
base_load, solar_supply, price = load_forecast()

# Develop functionality for mapping between different time periods
T_r = range(8*4)
T_o = range(24*4)
T_r_start = [range(8*4-dur[a]+1) for a in A_r]
T_o_start = [range(24*4-dur[a]) for a in A_o]

T_start = T_r_start + T_o_start

# There are 2880 times periods in general
# Split it up into T_bus (business hours) and T_off (out of hours)
T = range(2880)
T_bus, T_off = [], []

Day = 0         # Take Saturday to be the first day of the week

Weekday_business_hours = [set() for _ in range(5)]

while Day < 30:
    # Check if it's a weekend
    if Day % 7 in [0,6]:
        T_off.extend(list(range(4*24*Day,4*24*(Day+1))))
    else:
        T_off.extend(list(range(4 * 24 * Day,
                                4 * 24 * Day + 9 * 4)))
        T_bus.extend(list(range(4 * 24 * Day + 9 * 4,
                                4 * 24 * Day + 17 * 4)))
        T_off.extend(list(range(4 * 24 * Day + 17 * 4,
                                4 * 24 * (Day+1))))

        # Keep track of what day of the week each time period that falls within business hours is on
        Weekday_business_hours[Day%7 - 1].update(range(4 * 24 * Day + 9 * 4, 4 * 24 * Day + 17 * 4))

    Day += 1

def T_2_To(T):
    if isinstance(T,int):
        d = T//96
        t = T - d*96
        return (d,t)
    if isinstance(T,list):
        return [(t//96,t - 96*(t//96)) for t in T]

def T_2_Tr(T):
    if isinstance(T,int):
        for d in D_r:
            if T in Weekday_business_hours[d]:
                return (d, T % (4*24) - (9*4))
        return None
    elif isinstance(T,list):
        dt = []
        for t in T:
            for d in D_r:
                if t in Weekday_business_hours[d]:
                    dt.append((d, T % (4*24) - (9*4)))
                else:
                    dt.append(None)
    else:
        return None

column_info = generate_columns(A, A_r, T_r, D_r, A_o, T_o, D_o, T_bus, dur, value, penalty)

theta, class_value, active_recurring_schedules, active_oneoff_schedules, G, K = column_info

m = Model()

m.Params.MIPGap = opt_params['MIPGap']

if opt_params['instance size'] == 'large':
    m.Params.Threads = opt_params['threads']

X = {(a,k): m.addVar(vtype=GRB.BINARY,name=f'X[{a},{k}]')
     for a in A
     for k in K[a]}

grid_power = {t: m.addVar(vtype=GRB.CONTINUOUS,name=f'grid[{t}]')
              for t in T}

class_demand = {t: m.addVar(vtype=GRB.CONTINUOUS,name=f'class[{t}]')
                for t in T}

oneoff_activity_payments = quicksum(X[a,k]*class_value[a][k] for a in A_o for k in K[a])

grid_power_cost = (0.25/1000)*quicksum(grid_power[t]*price[t] for t in T)

m.setObjective(grid_power_cost-oneoff_activity_payments)

# Choose one schedule for each recurring activity
choose_one_schedule = {a: m.addConstr(quicksum(X[a,k] for k in K[a]) == 1)
                       for a in A_r}
# Choose at most one schedule for each one-off activity
choose_up_to_one_schedule = {a: m.addConstr(quicksum(X[a,k] for k in K[a]) <= 1)
                             for a in A_o}

# Calculate number of classrooms used
for t in T:
    d, tt = T_2_To(t)

    small_rooms_used = quicksum(X[a,k] * r_small[a] for (a,k) in active_oneoff_schedules[d,tt])
    large_rooms_used = quicksum(X[a, k] * r_large[a] for (a, k) in active_oneoff_schedules[d, tt])
    power_used = quicksum(X[a,k] * p[a] * (r_small[a] + r_large[a]) for (a,k) in active_oneoff_schedules[d,tt])

    recurring_time = T_2_Tr(t)

    if recurring_time is not None:
        d, tt = recurring_time
        small_rooms_used += quicksum(X[a, k] * r_small[a] for (a, k) in active_recurring_schedules[d, tt])
        large_rooms_used += quicksum(X[a, k] * r_large[a] for (a, k) in active_recurring_schedules[d, tt])
        power_used += quicksum(X[a, k] * p[a] * (r_small[a] + r_large[a]) for (a, k) in active_recurring_schedules[d, tt])

    m.addConstr(small_rooms_used <= n_small)
    m.addConstr(large_rooms_used <= n_large)
    m.addConstr(class_demand[t] == power_used)

# Precedence constraints
recur_precedence = {(a,d): m.addConstr(len(prec[a]) * quicksum(X[a,k] for k in G[a][d]) <=
                                       quicksum(X[aa,k] for aa in prec[a] for dd in range(d) for k in G[aa][dd]))
                    for a in A_r
                    for d in D_r}

oneoff_precedence = {(a,d): m.addConstr(len(prec[a]) * quicksum(X[a,k] for k in G[a][d]) <=
                                        quicksum(X[aa,k] for aa in prec[a] for dd in range(d) for k in G[aa][dd]))
                     for a in A_o
                     for d in D_o}

# Link grid power to other power supplies/demands
equate_power = {t: m.addConstr(grid_power[t] + solar_supply[t] == base_load[t] + class_demand[t])
                for t in T}

m.optimize()

save_solution(m, opt_params)

def Tro_2_Time(t):
    return f'{math.floor(t / 4)}:{t % 4 * 15}' + ('0' if t % 4 * 15 == 0 else '')

# Find active schedules
for a in A:
    printout = f'a{a}: '
    for k in K[a]:
        if X[a,k].X > 0.9:
            d,t = theta[a][k]
            if a in A_r:
                t += 9*4
            printout += f'Day {d} {Tro_2_Time(t)}-{Tro_2_Time(t + dur[a])}'
            # activity_type = 'recurring' if a in A_r else 'oneoff'
            # printout += f'Day {d} {schedule_index_2_time(t,activity_type)}-{schedule_index_2_time(t + dur[a],activity_type)}'

    print(printout)
