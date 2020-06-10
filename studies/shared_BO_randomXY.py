#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 8 09:15:32 2020
@author: Kiran
"""

import time
import numpy as np
import pandas as pd
import qiskit as qk
import seaborn as sns
import matplotlib.pyplot as plt
#sys.path.insert(0, '../qcoptim/')
from qcoptim import ansatz as az
from qcoptim import cost as cost
from qcoptim import utilities as ut
from qcoptim import optimisers as op




# ===================
# Defaults and global objects
# ===================
pi= np.pi
NB_SHOTS_DEFAULT = 2048
OPTIMIZATION_LEVEL_DEFAULT = 0
NB_SPINS = 7
NB_ANZ_DEPTH = 1
NB_TRIALS = 10
NB_CALLS = 200
NB_IN_IT_RATIO = 0.5
NB_OPT_VEC = [1,2]
SAVE_DATA = False

nb_init_vec = []
nb_iter_vec = []
for opt in NB_OPT_VEC:
    nb_init_vec.append(int((NB_CALLS * NB_IN_IT_RATIO) / opt))
    nb_iter_vec.append(int((NB_CALLS * (1 - NB_IN_IT_RATIO)) / opt))


simulator = qk.Aer.get_backend('qasm_simulator')
inst = qk.aqua.QuantumInstance(simulator,
                               shots=NB_SHOTS_DEFAULT,
                               optimization_level=OPTIMIZATION_LEVEL_DEFAULT)
Batch = ut.Batch(inst)

# ===================
# Generate ansatz for random XY
# ===================

hamiltonian = ut.gen_random_xy_hamiltonian(NB_SPINS,
                                           U = 1.0,
                                           J = 0.5,
                                           delta = 0.1,
                                           alpha = 2.0)
anz = az.RegularU2Ansatz(NB_SPINS,NB_ANZ_DEPTH)
xy_cost = cost.RandomXYCost(anz, inst, hamiltonian)

print("There are {} parameters to optimize".format(xy_cost.nb_params))


# ======================== /
#  Default BO optim args
# ======================== /
bo_args = ut.gen_default_argsbo(f=lambda x: .5, 
                                domain= [(0, 2*np.pi) for i in range(anz.nb_params)], 
                                nb_init=0,
                                eval_init=False)


# ======================== /
# Init runner (parallel runner is overkill here)
# ======================== /
df = pd.DataFrame()
np.random.seed(int(time.time()))
for trial in range(NB_TRIALS):
    this_data = []
    for opt, init, itt in zip(NB_OPT_VEC, nb_init_vec, nb_iter_vec):
        bo_args['nb_iter'] = itt
        bo_args['initial_design_numdata'] = init
    
        runner = op.ParallelRunner([xy_cost]*opt, 
                                   op.MethodBO, 
                                   optimizer_args = bo_args,
                                   share_init = False,
                                   method = 'shared')
    
    
        # # ========================= /
        # # And initilization:
        # ========================= /
        runner.next_evaluation_circuits()
        Batch.submit_exec_res(runner)
        runner.init_optimisers()
    
        # Run optimizer step by step
        for ii in range(itt):
            runner.next_evaluation_circuits()
            Batch.submit_exec_res(runner)
            runner.update()
            print(len(runner.optim_list[0].optimiser.X))
    
        # Get best_x
        x_opt_pred = [opt.best_x for opt in runner.optim_list]
    
        
        # Get bopt_results
        runner.shot_noise(x_opt_pred, nb_trials=5)
        Batch.submit_exec_res(runner)
        bopt_lines = runner._results_from_last_x()
        this_data.append(bopt_lines)
    
        print("opt: {}".format(np.mean(bopt_lines)))
    
    # Append results to dataframe (this is a mess - needs cleaning)
    point = []
    err = []
    for line in this_data:
        point.append(np.mean(np.ravel(line)))
        err.append(np.std(np.ravel(line)))
    dat = np.array([point, err, NB_OPT_VEC, [trial]*len(point)]).transpose()
    df_temp = pd.DataFrame(dat, columns = ['mean', 'std', 'nb_opt', 'trial'])
    df = df.append(df_temp)
    
   

plt.figure()
sns.set()
for ii in range(len(df)):
    m = df.iloc[ii]['mean']
    v = df.iloc[ii]['std']
    t = df.iloc[ii]['trial']
    o = df.iloc[ii]['nb_opt']
    plt.errorbar(o + 0.1*t/NB_TRIALS, m, yerr = v, fmt = 'r.', label='bopt')
plt.title('Shot noise: {} qubit random XY model'.format(NB_SPINS))
plt.xlabel('nb optims')
plt.ylabel('<H>')
plt.show()


plt.figure()
sns.pointplot(data = df, x = 'nb_opt', y = 'mean', join=False)
plt.title('Optimiser noise: {} qubit random XY model'.format(NB_SPINS))
plt.show()


plt.figure()
sns.boxplot(data = df, x = 'nb_opt', y = 'mean')
plt.title('Optimiser noise: {} qubit random XY model'.format(NB_SPINS))
plt.xlabel('nb of optimisers')
plt.ylabel('Solution')
plt.show()


if SAVE_DATA:
    import dill
    fname = '{}qubit_{}calls_{}ratio'.format(NB_SPINS,NB_CALLS,NB_IN_IT_RATIO).replace('.', 'p') + '.pkl'
    dict_to_dill = {'df':df,
                    'hamiltonian':hamiltonian,
                    'anz':anz}
    with open(fname, 'wb') as f:                                                                                                                                                                                                          
        dill.dump(dict_to_dill, f) 
    
# ========================= /
# Files:
    
# ========================= /    
    with open(fname, 'rb') as f:
        data = dill.load(f)