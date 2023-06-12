from Opt_Model_Defs import vanilla_IP

# Threads parameter only applied for large instances
opt_params = {'instance index': 0,
              'threads': 1,
              'MIPGap': 0,
              'TimeLimit': 600,
              'problem': '0-1 Knapsack',
              'instance folder': 'Weights_and_Values_and_Capacity'}

model = vanilla_IP(opt_params)
model.solve_all_instances()
# for instance in range(model.available_instances):
#     model.setup_and_optimize(instance)
#     model.save_model()
# model.save_model()
# model.plot_results()