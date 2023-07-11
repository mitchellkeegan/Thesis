from Model_Defs import column_gen

# Threads parameter only applied for large instances
opt_params = {'instance size': 'large',
              'instance index': 4,
              'threads': 1,
              'MIPGap': 0,
              'TimeLimit': 600}

model = column_gen(opt_params)
model.setup_and_optimize()
model.save_model()
model.plot_results()
print(5)