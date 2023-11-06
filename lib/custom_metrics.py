import numpy as np
import torch

def Approximation_Ratio(predicted_objective, auxiliary_info, reduce='sum', opt_type='min'):
    # TODO What do to if predicted objective is inside the bounds?
    if opt_type != 'min':
        raise Exception('Approximation Ratio Currently Only Implements Minimisation Problems')

    MIPGap, label_objective = auxiliary_info
    batch_size = predicted_objective.shape[0]

    upper_bound = label_objective
    lower_bound = upper_bound*(1-MIPGap/100)

    mid_bound = lower_bound + (upper_bound-lower_bound)/2

    AR = torch.ones(batch_size)

    idx_under = torch.nonzero(predicted_objective <= mid_bound)
    idx_over = torch.nonzero(predicted_objective > mid_bound)

    AR[idx_under] = upper_bound[idx_under]/predicted_objective[idx_under]
    AR[idx_over] = predicted_objective[idx_over] / lower_bound[idx_over]

    if reduce == 'sum':
        AR = torch.sum(AR)
    elif reduce == 'mean':
        AR = torch.mean(AR)
    elif reduce == 'none':
        pass
    else:
        raise Exception('Please provide a valid argument for reduce when calling Approximation_Ratio')
    return AR


# def Approximation_Ratio(coeff,pred,actual_solution,reduce):
#     v = coeff['v']
#
#     obj_val_pred = (v * pred).sum(axis=1)
#     obj_val_actual = (v * actual_solution).sum(axis=1)
#
#     approximation_ratio = np.ones(pred.shape[0])
#
#     # Check for Instances where the optimal solution is zero (i.e. cannot fit anything in knapsack)
#     # In this case set AR = 1 if the predicted solution is zero
#     # If the predicted solution is not zero, how to proceed?
#
#     idx_actual_no_weights = (obj_val_actual == 0)
#     idx_pred_no_weights = (obj_val_pred == 0)
#
#     # These are the Instances for which either the predicted solution incorrect but either
#     # the predicted or actual solution is zero (i.e. the approximation is undefined)
#     idx_fail = (idx_actual_no_weights & ~idx_pred_no_weights) | (idx_pred_no_weights & ~idx_actual_no_weights)
#
#     approximation_ratio[idx_fail] = 2
#
#     idx = ~(idx_actual_no_weights | idx_pred_no_weights)
#
#     R = np.stack((obj_val_pred / obj_val_actual,
#                   obj_val_actual / obj_val_pred), axis=1)[idx, :]
#     approximation_ratio[idx] = np.max(R, axis=1)
#
#     if reduce == 'mean':
#         approximation_ratio = np.mean(approximation_ratio)
#     elif reduce == 'sum':
#         approximation_ratio = np.sum(approximation_ratio)
#     elif reduce == 'None':
#         pass
#     else:
#         raise Exception('Please provide a valid argument for reduce when calling Approximation_Ratio')
#     return approximation_ratio