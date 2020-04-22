#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:10:36 2020

@author: fred
re-write test_GHZ with the new code
+ added pipeline for BEM
"""
import qiskit as qk
import numpy as np
import sys
sys.path.insert(0, '../core/')
import utilities as ut
import cost as cost
ut.add_path_GPyOpt()
import GPyOpt
import defined_circuits
import matplotlib.pyplot as plt


NB_SHOTS_DEFAULT = 8192
OPTIMIZATION_LEVEL_DEFAULT = 2
SINGAPORE_GATE_MAP_CYC_6 = [1,2,3,8,7,6] # Maybe put this in bem
SINGAPORE_GATE_MAP_CYC_6_EXTENDED = [2, 6, 10, 12, 14, 8] # Maybe put this in bem
ROCHESTER_GATE_MAP_GHZ_3_SWAPSx0 = [1,3,2]
ROCHESTER_GATE_MAP_GHZ_3_SWAPSx1 = [0,3,2]
ROCHESTER_GATE_MAP_GHZ_3_SWAPSx2 = [0,4,2]
ROCHESTER_GATE_MAP_GHZ_3_SWAPSx3 = [0,6,2]    
ROCHESTER_GATE_MAP_GHZ_3_SWAPSx4 = [5,6,2] # might actually be 5 dheck/rerun
ROCHESTER_GATE_MAP_GHZ_3_SWAPSx5 = [5,13,2] # might actually be 6 dheck/rerun
ROCHESTER_GATE_MAP_GHZ_3_SWAPSx6 = [9,13,2] # might actually be 7 dheck/rerun

TRANSPILER_SEED_DEFAULT = 10

# ===================
# Choose a backend using the custom backend manager and generate an instance
# ===================


bem = ut.BackendManager()
bem.get_current_status()
chosen_device = int(input('SELECT IBM DEVICE:'))
bem.get_backend(chosen_device, inplace=True)




gate_map_list = [ROCHESTER_GATE_MAP_GHZ_3_SWAPSx0,
                 ROCHESTER_GATE_MAP_GHZ_3_SWAPSx1,
                 ROCHESTER_GATE_MAP_GHZ_3_SWAPSx2,
                 ROCHESTER_GATE_MAP_GHZ_3_SWAPSx3,
                 ROCHESTER_GATE_MAP_GHZ_3_SWAPSx4,
                 ROCHESTER_GATE_MAP_GHZ_3_SWAPSx5,
                 ROCHESTER_GATE_MAP_GHZ_3_SWAPSx6]

ansatz0, nb_p, nb_q, x_sol = defined_circuits.GHZ_3qubits_6_params(0)
ansatz1, nb_p, nb_q, x_sol = defined_circuits.GHZ_3qubits_6_params(1)

multi_cost_gate = cost.Batch(gate_map = gate_map_list[:2], 
                             ansatz = ansatz0,
                             cost_function = cost.GHZPauliCost,
                             nb_params = nb_p, 
                             nb_qubits = nb_q,
                             be_manager = bem,
                             nb_shots = NB_SHOTS_DEFAULT,
                             optim_lvl = OPTIMIZATION_LEVEL_DEFAULT,
                             seed = TRANSPILER_SEED_DEFAULT)


multi_cost_antz = cost.Batch(gate_map = gate_map_list[0], 
                             ansatz = [ansatz0, ansatz1],
                             cost_function = cost.GHZPauliCost,
                             nb_params = nb_p, 
                             nb_qubits = nb_q,
                             be_manager = bem,
                             nb_shots = NB_SHOTS_DEFAULT,
                             optim_lvl = OPTIMIZATION_LEVEL_DEFAULT,
                             seed = TRANSPILER_SEED_DEFAULT)


multi_cost_cost = cost.Batch(gate_map = gate_map_list[0], 
                             ansatz = ansatz0,
                             cost_function = [cost.GHZPauliCost, cost.GHZWitness2Cost],
                             nb_params = nb_p, 
                             nb_qubits = nb_q,
                             be_manager = bem,
                             nb_shots = NB_SHOTS_DEFAULT,
                             optim_lvl = OPTIMIZATION_LEVEL_DEFAULT,
                             seed = TRANSPILER_SEED_DEFAULT)

multi_cost = multi_cost_cost


keep_going = str(input('Everything_look okay?'))
if keep_going != 'y':
    this_will_throw_and_error


NB_INIT = 5
NB_ITER = 5
DOMAIN_FULL = [(0, 2*np.pi) for i in range(nb_p)]
DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(DOMAIN_FULL)]
def dynamics_weight(n_iter):
        return max(0.000001, bo_args['acquisition_weight'] * (1 - n_iter / NB_ITER))


# ======================== /
#  Init each BO seperately
# ======================== /
bo_arg_list, bo_list = [], []
for cst in multi_cost.cost_list:
    cost_bo = lambda x: 1-cst(x) 
    bo_args = ut.gen_default_argsbo(f=cost_bo, domain=DOMAIN_FULL, nb_init=NB_INIT)
    bo_args.update({'acquisition_weight': 7}) # increase exploration
    bopt = GPyOpt.methods.BayesianOptimization(**bo_args)
    bopt.run_optimization(max_iter = 0, eps = 0) 

    bo_arg_list.append(bo_args)
    bo_list.append(GPyOpt.methods.BayesianOptimization(**bo_args))
    
 
# ======================== /
#  Run opt using the nice efficient class
# ======================== /
for ii in range(NB_ITER):
    x_new = multi_cost.get_new_param_points(bo_list)
    y_new = multi_cost(x_new)
    multi_cost.update_bo_inplace(bo_list, x_new, y_new)
    for bo in bo_list:
        bo.acquisition.exploration_weight = dynamics_weight(ii)

    print(ii)
    
 

for bopt in bo_list:
    bopt.run_optimization(max_iter = 0, eps = 0) 


for bo in bo_list:
    # Results found
    (x_seen, y_seen), (x_exp,y_exp) = bo.get_best()
    #fid_test(x_seen)
    #fid_test(x_exp)
    print(bo.model.model)
    bo.plot_convergence()
    plt.show()
    x_opt_pred = bo.X[np.argmin(bo.model.predict(bo.X, with_noise=False)[0])]





# ===================
# Get a baseline to compare to BO and save result
# ===================


# if type(x_sol) != type(None):
#     baseline_values = cost_cost.shot_noise(x_sol, 10)
# else:
#     baseline_values = None
# bopt_values = cost_cost.shot_noise(x_opt_pred, 10)


# ut.gen_pkl_file(cost_cost, Bopt, 
#                 baseline_values = baseline_values, 
#                 bopt_values = bopt_values, 
#                 dict_in = {'bo_args':bo_args})


