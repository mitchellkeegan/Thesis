from Model_Defs import column_gen

opt_params = {'instance size': 'small',
              'instance index': 0,
              'threads': 1,
              'MIPGap': 0}

model = column_gen(opt_params)
model.setup_and_optimize()
model.save_model()
print(5)