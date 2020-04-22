#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:10:36 2020
@author: Kiran
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


# ===================
# Defaults
# ===================
NB_SHOTS_DEFAULT = 8192
OPTIMIZATION_LEVEL_DEFAULT = 0
TRANSPILER_SEED_DEFAULT = 10
NB_INIT = 50
NB_ITER = 50


# ===================
# Choose a backend using the custom backend manager and generate an instance
# ===================
bem = ut.BackendManager()
bem.get_current_status()
chosen_device = int(input('SELECT IBM DEVICE:'))
bem.get_backend(chosen_device, inplace=True)


# ===================
# Use imports to generate Batch evaluation class, different init methods
# ===================
gate_map_list = [[1,3,2], [0,3,2]]
ansatz0, nb_p, nb_q, x_sol = defined_circuits.GHZ_3qubits_6_params(0)
ansatz1 = defined_circuits.GHZ_3qubits_6_params(1)[0]
ansatz2 = defined_circuits.GHZ_3qubits_6_params(2)[0]


# Left in for debugging - just making sure any deffinition works
multi_cost_gate = cost.Batch(gate_map = gate_map_list, 
                             ansatz = ansatz0,
                             cost_function = cost.GHZPauliCost,
                             nb_params = nb_p, 
                             nb_qubits = nb_q,
                             be_manager = bem,
                             nb_shots = NB_SHOTS_DEFAULT,
                             optim_lvl = OPTIMIZATION_LEVEL_DEFAULT,
                             seed = TRANSPILER_SEED_DEFAULT)

multi_cost_antz = cost.Batch(gate_map = gate_map_list[0], 
                             ansatz = [ansatz0, ansatz1, ansatz2],
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

# select single 
multi_cost = multi_cost_antz

# ======================== /
#  Default BO args - consider passing this into an extension of Batch class
# ======================== /
DOMAIN_FULL = [(0, 2*np.pi) for i in range(nb_p)]
DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(DOMAIN_FULL)]
def dynamics_weight(n_iter):
        return max(0.000001, bo_args['acquisition_weight'] * (1 - n_iter / NB_ITER))


# ======================== /
#  Init each BO seperately (might put this in Batch class, or extended class)
# ======================== /
bo_arg_list, bo_list = [], []
for cst in multi_cost.cost_list:
    cost_bo = lambda x: 1-cst(x) 
    bo_args = ut.gen_default_argsbo(f=cost_bo, domain=DOMAIN_FULL, nb_init=NB_INIT)
    bo_args.update({'acquisition_weight': 7}) # increase exploration
    bopt = GPyOpt.methods.BayesianOptimization(**bo_args)
    bopt.run_optimization(max_iter = 0, eps = 0) 
    
    # Opt runs on a list of bo args
    bo_arg_list.append(bo_args)
    bo_list.append(GPyOpt.methods.BayesianOptimization(**bo_args))
    
 
# ======================== /
#  Run opt using the nice efficient class (need to repackage)
# ======================== /
for ii in range(NB_ITER):
    x_new = multi_cost.get_new_param_points(bo_list)
    y_new = 1 - multi_cost(x_new)
    multi_cost.update_bo_inplace(bo_list, x_new, y_new)
    for bo in bo_list:
        bo.acquisition.exploration_weight = dynamics_weight(ii)
    print(ii)
    
 
# ======================== /
#  Print at results
# ======================== /
x_opt_pred = []
for bo in bo_list:
    bo.run_optimization(max_iter = 0, eps = 0) 
    (x_seen, y_seen), (x_exp,y_exp) = bo.get_best()
    #fid_test(x_seen)
    #fid_test(x_exp)
    print(bo.model.model)
    bo.plot_convergence()
    plt.show()
    x_opt_pred.append(bo.X[np.argmin(bo.model.predict(bo.X, with_noise=False)[0])])

# ======================== /
# Get a baseline to compare to BO and save result
# ======================== /
if type(x_sol) != ut.NoneType:
    baseline_values = multi_cost.shot_noise(x_sol, 10)
else:
    baseline_values = None
bopt_values = multi_cost.shot_noise(x_opt_pred, 10)


# ======================== /
# Save BO's in different files
# ======================== /
for cst, bo, bl_val, bo_val, bo_args in zip(multi_cost.cost_list,
                                            bo_list,
                                            baseline_values,
                                            bopt_values,
                                            bo_arg_list):
    ut.gen_pkl_file(cst, bo, 
                    baseline_values = bl_val, 
                    bopt_values = bo_val, 
                    info = 'cx' + str(cst.main_circuit.count_ops()['cx']) + '_',
                    dict_in = {'bo_args':bo_args,
                               'x_sol':x_sol})
    
    
