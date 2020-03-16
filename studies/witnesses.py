#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 19:41:03 2020

@author: fred
Goal: compare different fom for GHZ and graph states in terms of 
 + bias/variance (produce fig)
 + concentration (produce stat + scaling with N)
 
TODO:
    + generations of random state get more close to the target
    + implement sampling/measurements strategies
    + look at statistical properties

"""
import utilities_stabilizer as ut
import qutip as qt
import numpy as np
import matplotlib.pylab as plt

from scipy.stats import unitary_group


#####  Part I  #####
# Look at GHZ states
####################
N_ghz=4
N_states = 10000

ghz = ut.gen_ghz(N_ghz)
decomp = ut.gen_decomposition_paulibasis(ghz, N_ghz, threshold=1e-6, symbolic=True)
dims_ghz = ghz.dims
states_haar = [qt.rand_ket_haar(N=2**N_ghz, dims=dims_ghz) for _ in range(N_states)]

F = ut.gen_proj_ghz(N_ghz)
F1 = ut.gen_F1_ghz(N_ghz)
F2 = ut.gen_F2_ghz(N_ghz)

list_f = np.array([qt.expect(F, st) for st in states_haar])
list_f1 = np.array([qt.expect(F1, st) for st in states_haar])
list_f2 = np.array([qt.expect(F2, st) for st in states_haar])


fig, ax = plt.subplots()
ax.scatter(list_f, list_f1, label='f1')
ax.scatter(list_f, list_f2, label='f2')
ax.plot(list_f, list_f, label='f')
ax.legend()
ax.set_title('different foms')



#####  Part Ia  #####
# Look at the concentration
#####################
plt.hist(list_f)
plt.hist(list_f1)
plt.hist(list_f2)

avg, std = np.average(list_f), np.std(list_f)
avg1, std1 = np.average(list_f1), np.std(list_f1)
avg2, std2 = np.average(list_f2), np.std(list_f2)
std_norm = std / (1-avg)
std_norm1 = std1 / (1-avg1)
std_norm2 = std2 / (1-avg2)
print(std_norm)
print(std_norm1)
print(std_norm2)

#3,4,5,6
nb_q =[2,3,4,5,6,7,8,9]
res = np.array([[0.258276107091469, 0.3161054803737825, 0.3161054803737825],
                [0.1257447436332672,0.17622056376478779,0.19169664598402428],
                [0.06347733159484138, 0.10669941905274924, 0.12274957816578666],
                [0.03141659463343181, 0.06703660442251375, 0.07872189019318448],
                [0.015467538548765258, 0.04435145866191629,0.050931430704355414],
                [0.007794890212225759,0.03020906228369326,0.03331424973112738],
                [0.0038866049186859074,0.021085311335449268,0.0222141975578163],
                [0.0019566275973149382, 0.014968986939285224, 0.014815183106616713],
                [0.0009681666870207654,0.010561279603527155,0.009775006277963981]
])
log_res = np.log(res)
log_q = np.log(nb_q)

fig, ax = plt.subplots()
ax.plot(nb_q, log_res[:,0])
ax.plot(nb_q, log_res[:,1])
ax.plot(nb_q, log_res[:,2])


fig, ax = plt.subplots()
ax.plot(log_q, log_res[:,0],'o--')
ax.plot(log_q, log_res[:,1],'s--')
ax.plot(log_q, log_res[:,2],'v--')


fig, ax = plt.subplots()
ax.plot(nb_q[1:], res[1:,0]/res[:-1,0],'o--')
ax.plot(nb_q[1:], res[1:,1]/res[:-1,1],'s--')
ax.plot(nb_q[1:], res[1:,2]/res[:-1,2],'v--')
ax.set_yscale("log")

#####  Part Ib  #####
# Look at statistical properties 
#####################
stabgen_op = ut.gen_stab_gen_ghz(N_ghz)
N_repeat = 1000
N_meas = 100

# look at stat properties of operators themselves for 
# They have the same properties
estimates = [ut.estimate_op_bin(stabgen_op, states_haar, N_meas) for _ in range(N_repeat)]
#stats_avg = np.mean(estimates, axis=0)
#stats_std = np.std(estimates, axis=0)
#plt.hist(stats_std[0])
#plt.hist(stats_std[1])
#plt.hist(stats_std[2])
#np.average(stats_std[1])
#plt.hist(stats_avg[0])
#plt.hist(stats_avg[1])
#plt.hist(stats_avg[2])

expected = np.array([[qt.expect(op, st) for st in states_haar]for op in stabgen_op])
stats_avg = np.mean(expected, axis=1)
stats_std = np.std(expected, axis=1)

## strat estimate 1:







