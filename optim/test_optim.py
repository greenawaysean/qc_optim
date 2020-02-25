#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:39:09 2020

@author: fred
"""

import test_GHZ as qcirc
import sys, dill
sys.path.insert(0, '/home/fred/Desktop/GPyOpt/') ## Fork from https://github.com/FredericSauv/GPyOpt
import GPyOpt
import numpy as np
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise.errors import ReadoutError
from qiskit import IBMQ
# Global params
NB_SHOTS = 128

# ===================
# Noise models
# ===================
#noise in the readout (3q, p0g1 = p1g0 for the same qubit, but different over qubits)
ro_errors_proba = [[0.1, 0.1],[0.05,0.05],[0.15, 0.15]]
noise_ro = NoiseModel()
for n, ro in enumerate(ro_errors_proba):
    noise_ro.add_readout_error(ReadoutError([[1 - ro[0], ro[0]], [ro[1], 1 - ro[1]]]), [n])


ro_errors_proba_sym = [[0.1, 0.1],[0.1,0.1],[0.1, 0.1]]
noise_rosym = NoiseModel()
for n, ro in enumerate(ro_errors_proba_sym):
    noise_rosym.add_readout_error(ReadoutError([[1 - ro[0], ro[0]], [ro[1], 1 - ro[1]]]), [n])


provider = IBMQ.load_account()
device = provider.get_backend('ibmq_essex')
properties = device.properties()
noise_essex = noise.device.basic_device_noise_model(properties)

# ===================
# utility functions
# f_XXX wrapper function for the fidelity, ensure that they deal with several 
#               set of parameters (i.e. ndim(params)==2)
# 
# ===================
## Without noise
def f_average(params, shots = NB_SHOTS, noise_model = NoiseModel()):
    """  estimate of the fidelity"""
    if np.ndim(params) >1 : res = np.array([f_average(p, shots=shots, noise_model=noise_model) for p in params])
    else: 
        res = qcirc.F(params, shots = shots, noise_model=noise_model)
        print(res)
        res = 1 - np.atleast_1d(res)
    return res

def f_test(params, noise_model = NoiseModel()):
    if np.ndim(params) > 1 :
        res = np.array([f_test(p, noise_model=noise_model) for p in params])
    else:
        res = qcirc.F(params, shots=10000, noise_model=noise_model)
        print(res)
    return res

def get_best_from_bo(bo):
    """ Extract from a bo object the best set of parameters and fom
    based from both observed data and model"""
    x_obs = bo.x_opt
    y_obs = bo.fx_opt 
    pred = bo.model.predict(bo.X, with_noise=False)[0]
    x_pred = bo.X[np.argmin(pred)]
    y_pred = np.min(pred)
    return (x_obs, y_obs), (x_pred, y_pred)

def gen_res(bo):
    (x_obs, y_obs), (x_pred, y_pred) = get_best_from_bo(bo)
    res = {'x_obs':x_obs, 'x_pred':x_pred, 'y_obs':y_obs, 'y_pred':y_pred,
           'X':bo.X, 'Y':bo.Y, 'gp_params':bo.model.model.param_array,
           'gp_params_names':bo.model.model.parameter_names()}
    return res

# ===================
# BO setup
# ===================
NB_INIT = 25
X_SOL = np.pi/2 * np.array([1.,1.,2.,1.,1.,1.])
DOMAIN_DEFAULT = [(0, 2*np.pi) for i in range(6)]
EPS = np.pi/2
DOMAIN_RED = [(x-EPS, x+EPS) for x in X_SOL]
DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(DOMAIN_RED)]

BO_ARGS_DEFAULT = {'domain': DOMAIN_BO, 'initial_design_numdata':NB_INIT,
                   'model_update_interval':1, 'hp_update_interval':5, 
                   'acquisition_type':'LCB', 'acquisition_weight':5, 
                   'acquisition_weight_lindec':True, 'optim_num_anchor':5, 
                   'optimize_restarts':1, 'optim_num_samples':10000, 'ARD':False}

# ===================
# Ideal case: no noise
# 20/25 works
# ===================
NB_ITER = 25
Bopt = GPyOpt.methods.BayesianOptimization(f_average, **BO_ARGS_DEFAULT)    
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)
### Look at results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
print(f_test(x_seen))
print(f_test(x_exp))
print(Bopt.model.model)
Bopt.plot_convergence()


res_to_dill = gen_res(Bopt)
with open('_res.pkl', 'wb') as f:
    dill.dump(res_to_dill, f)

with open('_res.pkl', 'rb') as f:
    data_retrieved = dill.load(f)

# ===================
# Noise model essex
# ===================
NB_ITER = 50
f_essex = lambda x: f_average(x, shots=NB_SHOTS, noise_model=noise_essex)
Bopt_essex = GPyOpt.methods.BayesianOptimization(f_essex, **BO_ARGS_DEFAULT)    
Bopt_essex.run_optimization(max_iter = NB_ITER, eps = 0)
### Look at results found
(x_seen_essex, y_seen_essex), (x_exp_essex,y_exp_essex) = Bopt_essex.get_best()
f_test(x_seen_essex)
f_test(x_exp_essex)
print(Bopt_essex.model.model)
Bopt_essex.plot_convergence()


# ===================
# BO setup with noise in ro
# case 1 different readout errors for each qubit
# case 2 same readout errors for each qubit
# so far as difficult
# ===================
### Run optimization // case 1
f_noisy = lambda x: f_average(x, shots=1000, noise_model=noise_ro)
Bopt_noisy = GPyOpt.methods.BayesianOptimization(f_noisy, **BO_ARGS_DEFAULT)    
Bopt_noisy.run_optimization(max_iter = NB_ITER, eps = 0)
# Look at results found
(x_seen_noisy, y_seen_noisy), (x_exp_noisy,y_exp_noisy) = Bopt_noisy.get_best()
print(f_test(x_seen_noisy))
print(f_test(x_exp_noisy))
Bopt_noisy.plot_convergence()




### Run optimization // case 2 sy
f_noisy2 = lambda x: f_average(x, shots=1000, noise_model=noise_rosym)
Bopt_noisy2 = GPyOpt.methods.BayesianOptimization(f_noisy2, **BO_ARGS_DEFAULT)    
Bopt_noisy2.run_optimization(max_iter = NB_ITER, eps = 0)
# Look at results found
(x_seen_noisy2, y_seen_noisy2), (x_exp_noisy2,y_exp_noisy2) = Bopt_noisy2.get_best()
print(f_noisy2(x_exp_noisy2))
print(f_test(x_exp_noisy2))
Bopt_noisy2.plot_convergence()






