import numpy as np
import pandas as pd
from pyswarms.single import GlobalBestPSO

import random

value = [403,	886,	814,	1151,	983,	629,	848,	1074,	839,	819,	1062,	762,	994,	950,	111,	914,	737,	1049,	1152,	1110,	973,	474,	824,	1013,	963,	1101,	1024,	816,	1063,	575,	1153,	447,	1117,	910,	1017,	931,	909,	1126,	1027,	871,	1052,	891,	375,	1131,	318,	705,	1048,	908,	1026,	1061]
weight = [94,	506,	416,	992,	649,	237,	457,	815,	446,	422,	791,	359,	667,	598,	7,	544,	334,	766,	994,	893,	633,	131,	428,	700,	617,	874,	720,	419,	794,	196,	997,	116,	908,	539,	707,	569,	537,	931,	726,	487,	772,	513,	81,	943,	58,	303,	764,	536,	724,	789]
capacity = 997  # max weight of knapsack
number_of_items = 50  # set of items to consider


item_range = range(number_of_items)


# PSO paramters
n_particles = 10
# n_processes = 4
iterations = 1000
options = {'c1': 2, 'c2': 2, 'w': 0.7}
dim = number_of_items
LB = [0] * dim
UB = [1] * dim
constraints = (np.array(LB), np.array(UB))
kwargs  = {'value':value,
           'weight': weight,
           'capacity': capacity
            }

def get_particle_obj(X, **kwargs):
    """ Calculates the objective function value which is
    total revenue minus penalty of capacity violations"""
    # X is the decision variable. X is vector in the lenght of number of items
    # $ value of items
    value = kwargs['value']
    # weight of items
    weight = kwargs['weight']
    # Total revenue
    revenue = sum([value[i]*np.round(X[i]) for i in item_range])
    # Total weight of selected items
    used_capacity = sum([kwargs['weight'][i]*np.round(X[i]) for i in item_range])
    # Total capacity violation with 100 as a penalty cofficient
    capacity_violation = 100 * min(0,capacity - used_capacity)
    # the objective function minimizes the negative revenue, which is the same
    # as maximizing the positive revenue
    return -1*(revenue + capacity_violation)

# Objective function
def objective_function(X, **kwargs):
    n_particles_ = X.shape[0]
    dist = [get_particle_obj(X[i], **kwargs) for i in range(n_particles_)]
    return np.array(dist)


KP_optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=dim, options=options, bounds=constraints, bh_strategy='periodic',  velocity_clamp = (-0.5,0.5), vh_strategy = 'invert')
best_cost, best_pos = KP_optimizer.optimize(objective_function, iters=iterations, n_processes= None, **kwargs)
print("\nThe total knapsack revenue is: "+str(-best_cost))
print("Indices of selected items:\t " + str(np.argwhere(np.round(best_pos)).flatten()))