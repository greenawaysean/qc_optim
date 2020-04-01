#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:10:36 2020

@author: fred
re-write test_GHZ with the new code
"""
import qiskit as qk
import numpy as np
import sys
sys.path.insert(0, '../core/')
import utilities as ut
import cost as cost
ut.add_path_GPyOpt()
import GPyOpt
import copy

NB_SHOTS_DEFAULT = 8192
OPTIMIZATION_LEVEL_DEFAULT = 3
# ===================
# Choose a backend using the custom backend manager and generate an instance
# ===================
bem = ut.BackendManager()
bem.get_current_status()
chosen_device = int(input('SELECT IBM DEVICE:'))
bem.get_backend(chosen_device, inplace=True)
inst = bem.gen_instance_from_current(nb_shots=NB_SHOTS_DEFAULT, 
                                     optim_lvl=OPTIMIZATION_LEVEL_DEFAULT)
inst_test = bem.gen_instance_from_current(nb_shots=NB_SHOTS_DEFAULT, 
                                     optim_lvl=1)

# ===================
# Define ansatz and initialize costfunction
# Todo: generalize to abitrary nb of qubits
# ===================
def ansatz_easy(params):         
    """ Ansatz for which an ideal solution exist"""
    logical_qubits = qk.QuantumRegister(6, 'logicals')
    c = qk.QuantumCircuit(logical_qubits)
    c.ry(params[0],0)
    c.ry(params[1],1)
    c.ry(params[2],2)
    c.ry(params[3],3)
    c.ry(params[4],4)
    c.ry(params[5],5)
    c.barrier()
    c.cu1(np.pi,0,1)
    c.cu1(np.pi,2,3)
    c.cu1(np.pi,4,5)
    c.barrier()
    c.cu1(np.pi,1,2)
    c.cu1(np.pi,3,4)
    c.cu1(np.pi,5,0)
    c.barrier()
    return c
    

def ansatz_hard(params):
    """ Ansatz to be refined"""
    logical_qubits = qk.QuantumRegister(6, 'logicals')
    c = qk.QuantumCircuit(logical_qubits)
    c.ry(params[0],0)
    c.ry(params[1],1)
    c.ry(params[2],2)
    c.ry(params[3],3)
    c.ry(params[4],4)
    c.ry(params[5],5)
    c.barrier()
    c.cnot(0,1) 
    c.cnot(2,3) 
    c.cnot(4,5)
    c.cnot(1,2) 
    c.cnot(3,5)
    c.cnot(4,0)
    c.barrier()
    c.ry(params[6], 0)
    c.ry(params[7], 1)
    c.ry(params[8], 2)
    c.rx(params[9], 3)
    c.ry(params[10], 4)
    c.rx(params[11], 5)
    c.barrier()
    return c

### (ansatz, nb_params, nb_qubits, sol)
pb_infos = [(ansatz_easy, 6, 6, np.pi/2 * np.ones(shape=(6,))),
             (ansatz_hard, 12, 6, None)]
ansatz, nb_p, nb_q, x_sol = pb_infos[0]

# different cost functions 
# fid = cost.GraphCyclPauliCost(ansatz=ansatz, instance = inst, N=nb_q, nb_params=nb_p)
# fid_test = cost.GraphCyclPauliCost(ansatz=ansatz, instance = inst_test, N=nb_q, nb_params=nb_p)
# cost2 = cost.GraphCyclWitness2Cost(ansatz=ansatz, instance = inst, N=nb_q, nb_params=nb_p)
# cost1 = cost.GraphCyclWitness1Cost(ansatz=ansatz, instance = inst, N=nb_q, nb_params=nb_p)

cost_cost = cost.GraphCyclPauliCost(ansatz=ansatz, N=nb_q, instance=inst, nb_params=nb_p)


if bem.current_backend.name() != 'qasm_simulator':
    ansatz_transpiled = copy.deepcopy(cost_cost._main_circuit[0])
    #cost_transpiled = cost.GraphCyclPauliCost(ansatz=ansatz_transpiled, 
    #                                      N=nb_q, instance=inst_test, nb_params=nb_p)

    #print(cost_cost.compare_layout(cost_transpiled))
    print(cost_cost.check_depth(long_output=True))
    print(cost_cost.check_depth())
    print(cost_cost.check_layout())




keep_going = str(input('Everything_look okay?'))
if keep_going != 'y':
    this_will_throw_and_error
    



if x_sol is not None and bem.current_backend.name() == 'qasm_simulator':
    assert cost_cost(x_sol) == 1., "pb with ansatz/x_sol"
    # assert cost1(x_sol) == 1., "pb with ansatz/x_sol"
    # assert cost2(x_sol) == 1., "pb with ansatz/x_sol"
    


# ===================
# BO Optim
# Cost function defined by cost_cost
# ===================
# setu
    
print('''Warning: this assumes cost_cost is the default cost_function from here on''')
NB_INIT = 50
NB_ITER = 80
DOMAIN_FULL = [(0, 2*np.pi) for i in range(nb_p)]
DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(DOMAIN_FULL)]
cost_bo = lambda x: 1-cost_cost(x) 
bo_args = ut.gen_default_argsbo(f=cost_bo, domain=DOMAIN_FULL, nb_init=NB_INIT)
bo_args.update({'acquisition_weight': 7}) # increase exploration

#optim
Bopt = GPyOpt.methods.BayesianOptimization(**bo_args)    
print("start optim")
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)

# Results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
#fid_test(x_seen)
#fid_test(x_exp)
print(Bopt.model.model)
Bopt.plot_convergence()





## ===================
## Get a baseline to compare to BO and save result
## ===================

x_opt_pred = Bopt.X[np.argmin(Bopt.model.predict(Bopt.X, with_noise=False)[0])]

baseline_values = cost_cost.shot_noise(x_sol, 10)
bopt_values = cost_cost.shot_noise(x_opt_pred, 10)


ut.gen_pkl_file(cost_cost, Bopt, 
                baseline_values = baseline_values, 
                bopt_values = bopt_values, 
                dict_in = {'bo_args':bo_args})



# # ===================
# # BO Optim: no noise / Use of fidelity
# # 20/25 works
# #EPS = np.pi/2
# #DOMAIN_RED = [(x-EPS, x+EPS) for x in X_SOL]
# # ===================


# # ===================
# # BO Optim
# # No noise / Cost Function 2 
# # seems to work better with this split 30/70 not so much with 50/50 and more 
# # exploration
# # ===================

