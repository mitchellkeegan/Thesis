from gurobipy import Model, quicksum, GRB
import pandas as pd
import math

from helper_functions import load_instance, load_forecast


# TODO
# Add batteries
# Remove edge case of one-off activities ending outside of business hours not incurring a penalty
# Add quadratic term in objective
# Look at better solutions (optimize over full schedules)

instance_size = 'small'
instance_index = 0

D_r = range(5)
D_o = range(30)

# Load in data instance
n_small, n_large, prec, dur, p, r_small, r_large, value, penalty, A_r, A_o, A, B = load_instance(instance_size, instance_index)

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

t = 0

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


m = Model()

# Barrier method runs into memory issues quickly (the default option of running different algorithms concurrently
# is killed instantly by the OS)
# m.Params.Method = 1

X_r = {(a,d,t): m.addVar(vtype=GRB.BINARY)
       for a in A_r
       for d in D_r
       for t in T_r}

Y_r = {(a,d,t): m.addVar(vtype=GRB.BINARY)
       for a in A_r
       for d in D_r
       for t in T_start[a]}

X_o = {(a,d,t): m.addVar(vtype=GRB.BINARY)
       for a in A_o
       for d in D_o
       for t in T_o}

Y_o = {(a,d,t): m.addVar(vtype=GRB.BINARY)
       for a in A_o
       for d in D_o
       for t in T_start[a]}

grid_power = {t: m.addVar(vtype=GRB.CONTINUOUS)
              for t in T}

class_demand = {t: m.addVar(vtype=GRB.CONTINUOUS)
                for t in T}

oneoff_activity_payments = quicksum(quicksum(value[a]*Y_o[a,d,t] for (d,t) in T_2_To(T_bus))
                                    + quicksum((value[a]-penalty[a])*Y_o[a,d,t] for (d,t) in T_2_To(T_off) if t in T_start[a])
                                    for a in A_o)

grid_power_cost = (0.25/1000)*quicksum(grid_power[t]*price[t] for t in T)

m.setObjective(grid_power_cost-oneoff_activity_payments)


# Run each recurring activity once per week
recurring_once_per_week = {a: m.addConstr(quicksum(X_r[a,d,t]for d in D_r for t in T_r) == dur[a])
                           for a in A_r}
recurring_once_per_week_1 = {a: m.addConstr(quicksum(Y_r[a,d,t] for d in D_r for t in T_start[a]) == 1)
                           for a in A_r}

# Run each one-off activity at most once
oneoff_once_per_week = {a: m.addConstr(quicksum(X_o[a,d,t]for d in D_o for t in T_o) <= dur[a])
                        for a in A_o}
oneoff_once_per_week_1 = {a: m.addConstr(quicksum(Y_o[a,d,t] for d in D_o for t in T_start[a]) <= 1)
                           for a in A_o}

# Link x and y variables
recurring_activity_run_after_start = {(a,d,t): m.addConstr(quicksum(X_r[a,d,tt] for tt in range(t,t+dur[a])) >= dur[a]*Y_r[a,d,t])
                                      for (a,d,t) in Y_r}
oneoff_activity_run_after_start = {(a,d,t): m.addConstr(quicksum(X_o[a,d,tt] for tt in range(t,t+dur[a])) >= dur[a]*Y_o[a,d,t])
                                   for (a,d,t) in Y_o}

# Calculate power demand for classes
# Should find a better way to construct this, might be better to just do it in a loop
# calc_class_power_demand = {t: m.addConstr((0 if (tr := T_2_Tr(t)) is None else quicksum(p[a]*(r_small[a]+r_large[a])*X_r[(a,) + tr] for a in A_r)) +
#                                           quicksum(p[a]*(r_small[a]+r_large[a])*X_o[(a,) + T_2_To(t)] for a in A_o) == class_demand[t])
#                            for t in T}

# Calculate power demand and number of rooms used by all running classes at each timestep
for t in T:
    d,tt = T_2_To(t)

    demand_at_t = quicksum(p[a]*(r_small[a]+r_large[a])*X_o[a,d,tt] for a in A_o)
    small_rooms_used = quicksum(X_o[a,d,tt]*r_small[a] for a in A_o)
    large_rooms_used = quicksum(X_o[a,d,tt]*r_large[a] for a in A_o)

    recurring_time = T_2_Tr(t)

    if recurring_time is not None:
        d,tt = recurring_time
        demand_at_t += quicksum(p[a] * (r_small[a] + r_large[a]) * X_r[a, d, tt] for a in A_r)
        small_rooms_used += quicksum(X_r[a, d, tt] * r_small[a] for a in A_r)
        large_rooms_used += quicksum(X_r[a, d, tt] * r_large[a] for a in A_r)

    m.addConstr(demand_at_t == class_demand[t])
    m.addConstr(small_rooms_used <= n_small)
    m.addConstr(large_rooms_used <= n_large)

# Precedence constraints
recur_precedence = {(a,d): m.addConstr(len(prec[a]) * quicksum(Y_r[a,d,t] for t in T_start[a]) <=
                                       quicksum(Y_r[aa,dd,t] for aa in prec[a] for dd in range(d) for t in T_start[aa]))
                    for a in A_r
                    for d in D_r}

oneoff_precedence = {(a,d): m.addConstr(len(prec[a]) * quicksum(Y_o[a,d,t] for t in T_start[a]) <=
                                       quicksum(Y_o[aa,dd,t] for aa in prec[a] for dd in range(d) for t in T_start[aa]))
                    for a in A_o
                    for d in D_o}

# Link grid power to other power supplies/demands
equate_power = {t: m.addConstr(grid_power[t] + solar_supply[t] == base_load[t] + class_demand[t])
                for t in T}

m.optimize()

week_index_to_day = {0: 'Monday',
                     1: 'Tuesday',
                     2: 'Wednesday',
                     3: 'Thursday',
                     4: 'Friday'}

def Tr_2_Time(t):
    # This function takes in timesteps in range(0,32)
    # and returns the time between 9am-5pm as a string

    return f'{9+math.floor(t/4)}:{t%4*15}' + ('0' if t%4*15 == 0 else '')

def To_2_Time(t):
    # Takes in timesteps in range(0,96)
    # and returns the time between 00:00-24:00 as a string

    return f'{math.floor(t/4)}:{t%4*15}' + ('0' if t%4*15 == 0 else '')

for b in B:
    print(f'\n--------------- Building {b} Schedule ---------------')
    for d in D_r:
        printout = f'{week_index_to_day[d]}:'
        for a in A_r:
            for t in T_r_start[a]:
                if Y_r[a,b,d,t].X == 1:
                    printout += f' a{a} ({dur[a]},{sum(int(X_r[a,b,d,tt].X) for tt in range(t,t+dur[a]))},{f"S{r_small[a]}" if a in r_small else f"L{r_large[a]}"}) ' \
                                f'{Tr_2_Time(t)}-{Tr_2_Time(t+dur[a])} |'
        print(printout[:-1])

# for b in B:
#     print(f'\n--------------- Building {b} Small Capacity = {B_n_small[b]} ---------------')
#     for d in D_r:
#         printout = f'{week_index_to_day[d]}:'
#         for t in T_r:
#             printout += f' {sum(int(X_r[a,b,d,t].X) * r_small[a] for a in A_r if a in r_small)}'
#         print(printout)
#
# for b in B:
#     print(f'\n--------------- Building {b} Large Capacity = {B_n_large[b]} ---------------')
#     for d in D_r:
#         printout = f'{week_index_to_day[d]}:'
#         for t in T_r:
#             printout += f' {sum(int(X_r[a,b,d,t].X) * r_large[a] for a in A_r if a in r_large)}'
#         print(printout)

# Have a look at when the once-off activites are scheduled

print('\n--------------- One-Off Activity Schedules ---------------')
for a in A_o:
    printout = f'a{a}: '
    for d in D_o:
        for t in T_start[a]:
            for b in B:
                if Y_o[a,b,d,t].X > 0.9:
                    printout += f'Day {d} {To_2_Time(t)}-{To_2_Time(t+dur[a])}'

    print(printout)

print(5)