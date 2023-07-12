import numpy as np

def Approximation_Ratio(coeff,pred,actual_solution,reduce):
    v = coeff['v']

    obj_val_pred = (v * pred).sum(axis=1)
    obj_val_actual = (v * actual_solution).sum(axis=1)

    approximation_ratio = np.ones(pred.shape[0])

    # Check for instances where the optimal solution is zero (i.e. cannot fit anything in knapsack)
    # In this case set AR = 1 if the predicted solution is zero
    # If the predicted solution is not zero, how to proceed?

    idx_actual_no_weights = (obj_val_actual == 0)
    idx_pred_no_weights = (obj_val_pred == 0)

    # These are the instances for which either the predicted solution incorrect but either
    # the predicted or actual solution is zero (i.e. the approximation is undefined)
    idx_fail = (idx_actual_no_weights & ~idx_pred_no_weights) | (idx_pred_no_weights & ~idx_actual_no_weights)

    approximation_ratio[idx_fail] = 2

    idx = ~(idx_actual_no_weights | idx_pred_no_weights)

    R = np.stack((obj_val_pred / obj_val_actual,
                  obj_val_actual / obj_val_pred), axis=1)[idx, :]
    approximation_ratio[idx] = np.max(R, axis=1)

    if reduce == 'mean':
        approximation_ratio = np.mean(approximation_ratio)
    elif reduce == 'sum':
        approximation_ratio = np.sum(approximation_ratio)
    elif reduce == 'None':
        pass
    else:
        raise Exception('Please provide a valid argument for reduce when calling Approximation_Ratio')
    return approximation_ratio