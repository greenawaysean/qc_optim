#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 1 2020

@author: kiran

Script to quickly load and review results
    Loads results in f_name and looks at how things went
    + Want to change format to Lower case dict objects
"""
import qiskit as qk
import numpy as np
import dill
import matplotlib.pyplot as plt 
pi=np.pi

f_name = '_res_ibmq_singapore_GraphCyclPauliCost_MCb.pkl'
f = open(f_name, 'rb')
data = dill.load(f)
f.close()
del(f)

def load(f_in):
    """ This doesn't work yet. Looks like I might have to make a data class 
        (was hoping to avoid this)"""
    global f_name
    global data
    f_name = f_in
    f = open(f_name, 'rb')
    data = dill.load(f)
    f.close()
    del(f)
    return data


def print_all_keys(data=data):
    """ Prints a list of keys in the loaded file - just to see whats there"""
    keys = data.keys()
    running_tot = []
    for key in keys:
        if type(data[key]) == dict:
            sub_list = print_all_keys(data[key])
            running_tot.append(key)
            running_tot.append(sub_list)
        else:
            running_tot.append(key)
    return running_tot


def plot_convergence(data=data):
    """ Plots the convergence of the Bopt vales (hope I've done' this right)"""
    bopt = data['Bopt_results']
    X = bopt['X']
    Y = bopt['Y']
    plt.subplot(1, 2, 1)
    plt.plot(_diff_between_x(X))
    plt.xlabel('itt')
    plt.ylabel('|dX_i|')
    plt.subplot(1, 2, 2)
    plt.plot(Y)
    plt.xlabel('itt')
    plt.ylabel('Y')
    plt.show()


def _diff_between_x(X_in):
    """ Computes the euclidian distance between adjacent X values
    + Might need to vectorize this in future"""
    dX = X_in[1:] - X_in[0:-1]
    dX = [dx.dot(dx) for dx in dX]
    dX = np.sqrt(np.array(dX))
    return dX


def plot_baselines(data=data,
                   same_axis=False,
                   bins=30):
    """ Plots baseline histograms with mean and variance of each, comparing
        the Bopt values to the true baseilne values"""
    baseline = data['F_Baseline']
    bopt = data['F_Bopt']
    all_data = np.squeeze(np.concatenate([baseline, bopt]))
    x_min = 0.9*min(all_data)
    x_max = 1.1*max(all_data)
    plt.subplot(1, 2, 1)
    plt.hist(baseline, bins)
    plt.xlabel('Yobs')
    plt.ylabel('count')
    mean = np.round(np.mean(baseline), 4)
    std = np.round(np.std(baseline), 4)
    plt.title('mean: {} \n std: {}'.format(mean,std))
    if same_axis: plt.xlim(x_min, x_max)
    
    plt.subplot(1, 2, 2)    
    plt.hist(bopt, bins)
    plt.xlabel('Yobs')
    plt.ylabel('count')    
    mean = np.round(np.mean(bopt), 4)
    std = np.round(np.std(bopt), 4)
    plt.title('mean: {} \n std: {}'.format(mean,std))
    if same_axis: plt.xlim(x_min, x_max)
    
    plt.show()


def plot_circ(data=data):
    """ Displays quick info about the ansatz circuit:
        TODO: Add check for transpiled ansatz circuit
              Add log2phys mapping if avaliable"""
    circ = data['Ansatz']
    x_pred = data['Bopt_results']['x_pred']
    depth = data['Depth']
    circ = circ(x_pred)
    meta = data['Meta'][0]
    print('Backend: {}'.format(meta['backend_name']))
    print('Circuit depths = {} \pm {}'.format(np.mean(depth), np.std(depth)))
    print(circ)


def compare_pred(data=data,
                 x_sol=None):
    """ Compares Predicted, observed and analytic (input spesified) parameter 
        solutions. """
    bopt = data['Bopt_results']
    x_obs = bopt['x_obs']
    x_pred = bopt['x_pred']
    y_obs = np.round(bopt['y_obs'], 3)
    y_pred = np.round(bopt['y_pred'], 3)
    
    plt.plot(x_obs, 'rd', label='obs: ({})'.format(y_obs))
    plt.plot(x_pred, 'k*', label='pred: ({})'.format(y_pred))
    if x_sol != None:
        plt.plot(x_sol, 'bo', label='sol: ({})'.format(1))
        x_sol = np.array(x_sol)
        distance = _diff_between_x(np.array([x_sol, x_pred]))
    plt.legend()
    plt.xlabel('Parameter #')
    plt.ylabel('Parameter value')
    plt.title('Sol vs Seen (Dist = {})'.format(np.round(distance,4)))




if __name__ =='__main__':
    plot_baselines()
    plot_convergence()
    plot_circ()
    compare_pred(x_sol=[pi/2]*6)