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

NB_SHOTS_DEFAULT = 2048
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
inst_test = bem.gen_instance_from_current(nb_shots=8192, 
                                     optim_lvl=OPTIMIZATION_LEVEL_DEFAULT)

noise_model = bem.gen_noise_model_from_backend(name_backend='ibmq_essex')
inst_noisy = bem.gen_instance_from_current(nb_shots=NB_SHOTS_DEFAULT, 
                optim_lvl=OPTIMIZATION_LEVEL_DEFAULT, noise_model = noise_model)

# ===================
# Define ansatz and initialize costfunction
# Todo: generalize to abitrary nb of qubits
# ===================
def ansatz_easy(params):         
    """ Ansatz for which an ideal solution exist"""
    c = qk.QuantumCircuit(qk.QuantumRegister(1, 'a'), qk.QuantumRegister(1, 'b'),
                          qk.QuantumRegister(1,'c'),qk.QuantumRegister(1,'d'),
                          qk.QuantumRegister(1,'e'),qk.QuantumRegister(1,'f'))
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
    c = qk.QuantumCircuit(qk.QuantumRegister(1, 'a'), qk.QuantumRegister(1, 'b'),
                          qk.QuantumRegister(1,'c'),qk.QuantumRegister(1,'d'),
                          qk.QuantumRegister(1,'e'),qk.QuantumRegister(1,'f'))
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

# different cost functions (w/wo noise)
fid_test = cost.GraphCyclPauliCost(ansatz=ansatz, instance = inst_test, N=nb_q, nb_params=nb_p)
fid = cost.GraphCyclPauliCost(ansatz=ansatz, instance = inst, N=nb_q, nb_params=nb_p)
cost1 = cost.GraphCyclWitness1Cost(ansatz=ansatz, instance = inst, N=nb_q, nb_params=nb_p)
cost2 = cost.GraphCyclWitness2Cost(ansatz=ansatz, instance = inst, N=nb_q, nb_params=nb_p)
fid_noisy = cost.GraphCyclPauliCost(ansatz=ansatz, instance = inst_noisy, N=nb_q, nb_params=nb_p)
cost1_noisy = cost.GraphCyclWitness1Cost(ansatz=ansatz, instance = inst_noisy, N=nb_q, nb_params=nb_p)
cost2_noisy = cost.GraphCyclWitness2Cost(ansatz=ansatz, instance = inst_noisy, N=nb_q, nb_params=nb_p)

# different cost functions with a noisy instance

if x_sol is not None:
    assert fid_test(x_sol) == 1., "pb with ansatz/x_sol"
    assert cost1(x_sol) == 1., "pb with ansatz/x_sol"
    assert cost2(x_sol) == 1., "pb with ansatz/x_sol"

# ===================
# 1.a. BO Optim No noise (except sampling) + Fidelity
# (20/25 works)
#EPS = np.pi/2
#DOMAIN_RED = [(x-EPS, x+EPS) for x in X_SOL]
# ===================
# setup
NB_INIT = 50
NB_ITER = 50
DOMAIN = [(0, 2*np.pi) for i in range(nb_p)]
cost_bo = lambda x: 1-fid(x)
bo_args = ut.gen_default_argsbo(f=cost_bo, domain=DOMAIN, nb_init=NB_INIT)

#optim
Bopt = GPyOpt.methods.BayesianOptimization(**bo_args)    
print("start optim")
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)

# Results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
fid_test(x_seen)
fid_test(x_exp)
print(Bopt.model.model)
Bopt.plot_convergence()

# ===================
# 1.b. BO Optim No noise (except sampling) + Cost function 1 
# exploration has been crank up to 'acquisition_weight': 7
# ===================
# setup
NB_INIT = 50
NB_ITER = 50
DOMAIN = [(0, 2*np.pi) for i in range(nb_p)]
cost_bo = lambda x: 1-cost1(x)
bo_args = ut.gen_default_argsbo(f=cost_bo, domain=DOMAIN, nb_init=NB_INIT)
bo_args.update({'acquisition_weight': 7}) # increase exploration

#optim
Bopt = GPyOpt.methods.BayesianOptimization(**bo_args)    
print("start optim")
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)

# Results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
fid_test(x_seen)
fid_test(x_exp)
print(Bopt.model.model)
Bopt.plot_convergence()

# ===================
# 1.c BO Optim No noise (except sampling) + cost Function 2 
# ===================
# setup
NB_INIT = 50
NB_ITER = 50
DOMAIN = [(0, 2*np.pi) for i in range(nb_p)]
cost_bo = lambda x: 1-cost2(x)
bo_args = ut.gen_default_argsbo(f=cost_bo, domain=DOMAIN, nb_init=NB_INIT)

#optim
Bopt = GPyOpt.methods.BayesianOptimization(**bo_args)    
print("start optim")
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)

# Results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
fid_test(x_seen)
fid_test(x_exp)
print(Bopt.model.model)
Bopt.plot_convergence()

# ===================
# 2.a. BO Optim with NOISY SIMULATORS + Fidelity
# SLOW
# ===================
# setup
NB_INIT = 50
NB_ITER = 50
DOMAIN = [(0, 2*np.pi) for i in range(nb_p)]
cost_bo = lambda x: 1-fid_noisy(x)
bo_args = ut.gen_default_argsbo(f=cost_bo, domain=DOMAIN, nb_init=NB_INIT)

#optim
Bopt = GPyOpt.methods.BayesianOptimization(**bo_args)    
print("start optim")
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)

# Results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
fid_test(x_seen)
fid_test(x_exp)
print(Bopt.model.model)
Bopt.plot_convergence()

# ===================
# 2.b.  BO Optim with NOISY SIMULATORS + Cost1
# Exploration (acquisition weight)
# FAST 98-99% Fidelity test
# ===================
# setup
NB_INIT = 50
NB_ITER = 50
DOMAIN = [(0, 2*np.pi) for i in range(nb_p)]
cost_bo = lambda x: 1-cost1_noisy(x)
bo_args = ut.gen_default_argsbo(f=cost_bo, domain=DOMAIN, nb_init=NB_INIT)
bo_args.update({'acquisition_weight': 7}) # increase exploration

#optim
Bopt = GPyOpt.methods.BayesianOptimization(**bo_args)    
print("start optim")
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)

# Results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
fid_test(x_seen)
fid_test(x_exp)
print(Bopt.model.model)
Bopt.plot_convergence()


# ===================
# 2.c.  BO Optim with NOISY SIMULATORS + Cost2
# FAST + GOOD (Cost function during optim is far from 1, still final fid is 98-99%)
# ===================
# setup
NB_INIT = 50
NB_ITER = 50
DOMAIN = [(0, 2*np.pi) for i in range(nb_p)]
cost_bo = lambda x: 1-cost2_noisy(x)
bo_args = ut.gen_default_argsbo(f=cost_bo, domain=DOMAIN, nb_init=NB_INIT)
#bo_args.update({'acquisition_weight': 7})

#optim
Bopt = GPyOpt.methods.BayesianOptimization(**bo_args)    
print("start optim")
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)

# Results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
fid_test(x_seen)
fid_test(x_exp)
print(Bopt.model.model)
Bopt.plot_convergence()

## ===================
## Get a baseline to compare to BO
## ===================
#
#x_opt_guess =  np.array([3., 3., 2., 3., 3., 1.]) * np.pi/2
#x_opt_pred = Bopt.X[np.argmin(Bopt.model.predict(Bopt.X, with_noise=False)[0])]
#
#baseline_values = [qc.F(x_opt_guess) for ii in range(10)]
#bopt_values = [qc.F(x_opt_pred) for ii in range(10)]
#
#
#
#
#res_to_dill = gen_res(Bopt)
#dict_to_dill = {'Bopt_results':res_to_dill, 
#                'F_Baseline':baseline_values, 
#                'F_Bopt':bopt_values,
#                'Circ':qc.MAIN_CIRC[0],
#                'Device_config':qc.params_to_dict()}