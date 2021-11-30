import numpy as np
import pandas as pd
from pyswarms.single import GlobalBestPSO

import random


#knapPl_13_50_1000
# value = [234,	39,	1053,	351,	585,	78,	117,	234,	312,	156,	156,	273,	351,	351,	312,	468,	156,	585,	468,	1053,	702,	1053,	819,	234,	39,	1170,	234,	234,	351,	936,	468,	351,	351,	156,	351,	390,	468,	156,	312,	390,	468,	702,	234,	1170,	195,	117,	351,	936,	585,	78]
# weight = [114,	19,	873,	291,	485,	38,	97,	194,	152,	76,	76,	133,	291,	291,	152,	388,	76,	485,	388,	873,	582,	873,	679,	194,	19,	970,	194,	194,	291,	776,	388,	171,	171,	76,	171,	190,	388,	76,	152,	190,	388,	582,	114,	970,	95,	97,	291,	776,	485,	38]
# capacity = 970  
# number_of_items = 50 

#knapPl_11_100_1000_1
# value = [114,	38,	133,	95,	612,	171,	918,	408,	714,	510,	114,	76,	57,	204,	1020,	816,	190,	510,	114,	714,	152,	38,	918,	190,	612,	510,	306,	95,	816,	714,	114,	76,	918,	133,	102,	306,	612,	1020,	102,	1020,	918,	510,	918,	114,	76,	57,	918,	95,	612,	204,	918,	38,	19,	152,	76,	918,	816,	204,	204,	114,	95,	408,	76,	1020,	204,	510,	714,	114,	918,	510,	408,	306,	133,	133,	19,	114,	19,	306,	133,	1020,	408,	57,	152,	102,	57,	918,	306,	1020,	918,	102,	171,	38,	1020,	171,	171,	204,	171,	816,	152,	612]
# weight = [582,	194,	679,	485,	396,	873,	594,	264,	462,	330,	582,	388,	291,	132,	660,	528,	970,	330,	582,	462,	776,	194,	594,	970,	396,	330,	198,	485,	528,	462,	582,	388,	594,	679,	66,	198,	396,	660,	66,	660,	594,	330,	594,	582,	388,	291,	594,	485,	396,	132,	594,	194,	97,	776,	388,	594,	528,	132,	132,	582,	485,	264,	388,	660,	132,	330,	462,	582,	594,	330,	264,	198,	679,	679,	97,	582,	97,	198,	679,	660,	264,	291,	776,	66,	291,	594,	198,	660,	594,	66,	873,	194,	660,	873,	873,	132,	873,	528,	776,	396]
# capacity = 970  
# number_of_items = 100 

#knapPI_15_50_1000_1
value = [96,	507,	417,	993,	651,	237,	459,	816,	447,	423,	792,	360,	669,	600,	9,	546,	336,	768,	996,	894,	633,	132,	429,	702,	618,	876,	720,	420,	795,	198,	999,	117,	909,	540,	708,	570,	537,	933,	726,	489,	774,	513,	81,	945,	60,	303,	765,	537,	726,	789]
weight = [94,	506,	416,	992,	649,	237,	457,	815,	446,	422,	791,	359,	667,	598,	7,	544,	334,	766,	994,	893,	633,	131,	428,	700,	617,	874,	720,	419,	794,	196,	997,	116,	908,	539,	707,	569,	537,	931,	726,	487,	772,	513,	81,	943,	58,	303,	764,	536,	724,	789]
capacity = 997
number_of_items = 50

#knapPI_16_50_1000_1
# value = [403,	886,	814,	1151,	983,	629,	848,	1074,	839,	819,	1062,	762,	994,	950,	111,	914,	737,	1049,	1152,	1110,	973,	474,	824,	1013,	963,	1101,	1024,	816,	1063,	575,	1153,	447,	1117,	910,	1017,	931,	909,	1126,	1027,	871,	1052,	891,	375,	1131,	318,	705,	1048,	908,	1026,	1061]
# weight = [94,	506,	416,	992,	649,	237,	457,	815,	446,	422,	791,	359,	667,	598,	7,	544,	334,	766,	994,	893,	633,	131,	428,	700,	617,	874,	720,	419,	794,	196,	997,	116,	908,	539,	707,	569,	537,	931,	726,	487,	772,	513,	81,	943,	58,	303,	764,	536,	724,	789]
# capacity = 997  # max weight of knapsack
# number_of_items = 50  # set of items to consider


item_range = range(number_of_items)


# PSO paramters

# n_processes = 4
iterations = 1000

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

def main(pcls, w, run):
    n_particles = pcls
    options = {'c1': 2, 'c2': 2, 'w': w}
    KP_optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=dim, options=options, bounds=constraints, bh_strategy='periodic',  velocity_clamp = (-0.5,0.5), vh_strategy = 'invert')
    best_cost, best_pos = KP_optimizer.optimize(objective_function, iters=iterations, n_processes= None, **kwargs)
    print("\nThe total knapsack revenue is: "+str(-best_cost))
    print("Indices of selected items:\t " + str(np.argwhere(np.round(best_pos)).flatten()))
    d = {'beat_fitness': [-1*best_cost], 'best_solution': [str(np.argwhere(np.round(best_pos)).flatten())] }
    sol = pd.DataFrame(data= d)
    sol.to_csv("./solutions/each_run/sol_file_kp_15_50_1000_pcls_{0}_w_{1}_run_{2}.csv".format(pcls, w, run))
    fitness.append(-1*best_cost)


# [10, 30]
# [0.3, 0.7, 0.9]


for pcls in [10, 30]:
    for w in [0.3, 0.7, 0.9]:
        fitness = []
        for i in range(30):
            main(pcls, w, i)
        avg_fitness = sum(fitness)/30
        n_fun_calls = pcls * iterations
        ratio_avg_to_fun = avg_fitness/n_fun_calls
        print("avg_fitness:", avg_fitness)
        d = {'avg_fitness': [avg_fitness], 'n_fun_calls': [n_fun_calls], 'ratio_avg_to_fun' : [ratio_avg_to_fun] }
        df = pd.DataFrame(data= d)
        df.to_csv("./solutions/sol_file_kp_15_50_1000_pcls_{0}_w_{1}.csv".format(pcls, w))
