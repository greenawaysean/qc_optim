#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Apr 20 13:15:21 2020

@author: kiran
basic script to holduseful ansatz function circuits, returns handels to circuits 
    and all info required to run them.  
    
"""
# ===================
# Define ansatz and initialize costfunction
# Todo: generalize to abitrary nb of qubits
# ===================
import qiskit as qk
import numpy as np
import dill



def from_saved_file(f_name):
    """ Does it's best to imort an ansatz cirucit from a saved file"""
    with open(f_name, 'rb') as f:
        data = dill.load(f)
    circ = data['ansatz']
    nb_params = len(data['bopt_results']['X'][0])
    try: nb_qubits = data['nb_qubits']
    except: nb_qubits = None
    try: x_sol = data['other']['x_sol']
    except : x_sol = None
    return circ, nb_params, nb_qubits, x_sol



def GHZ_3qubits_6_params(swaps = 0):
    """ Returns the 3 qubit GHZ state with different numbers of swaps"""
    x_sol =  np.array([3., 3., 2., 3., 3., 1.]) * np.pi/2
    if swaps == 0:
        return _GHZ_3qubits_6_params_x0, 6, 3, x_sol
    elif swaps == 1:
        return _GHZ_3qubits_6_params_x1, 6, 3, x_sol
    elif swaps == 2:
        return _GHZ_3qubits_6_params_x2, 6, 3, x_sol
    else:
        raise NotImplementedError()



def GraphCycl_6qubits_6params():
    """ Returns cyclic 6 cluster ansatz"""
    return (_GraphCycl_6qubits_6params, 6, 6, 
            np.pi/2 * np.ones(shape=(6,)))



def GraphCycl_6qubits_6params_inefficient():
    """ Returns cyclic 6 cluster ansatz with c-phase replaced by cry gates"""
    return (_GraphCycl_6qubits_6params_inefficient, 6, 6, 
            np.pi/2 * np.ones(shape=(6,)))



def GraphCycl_6qubits_24params():
    """ Returns cyclic 6 cluster ansatz with many free params 
        TODO: handle updating"""
    print('warning: not solution exists here')
    return (_GraphCycl_6qubits_24params, 24, 6, None)



def _GHZ_3qubits_6_params_x0(params, barriers = False):
    """ Returns function handle for 6 param ghz state"""
    logical_qubits = qk.QuantumRegister(3, 'logicals')
    c = qk.QuantumCircuit(logical_qubits)
    c.rx(params[0], 0)
    c.rx(params[1], 1)
    c.ry(params[2], 2)
    if barriers: c.barrier()
    c.cnot(0,2) 
    c.cnot(1,2) 
    if barriers: c.barrier()
    c.rx(params[3], 0)
    c.rx(params[4], 1)
    c.ry(params[5], 2)
    if barriers: c.barrier()
    return c

def _GHZ_3qubits_6_params_x1(params, barriers = False):
    """ Returns function handle for 6 param ghz state 1 swap"""
    logical_qubits = qk.QuantumRegister(3, 'logicals')
    c = qk.QuantumCircuit(logical_qubits)
    c.ry(params[2], 0)
    c.rx(params[1], 1)
    c.rx(params[0], 2)
    c.swap(0, 2)
    if barriers: c.barrier()
    c.cnot(0,2) 
    c.cnot(1,2) 
    if barriers: c.barrier()
    c.rx(params[3], 0)
    c.rx(params[4], 1)
    c.ry(params[5], 2)
    if barriers: c.barrier()
    return c

def _GHZ_3qubits_6_params_x2(params, barriers = False):
    """ Returns function handle for 6 param ghz state 2 swaps"""
    logical_qubits = qk.QuantumRegister(3, 'logicals')
    c = qk.QuantumCircuit(logical_qubits)
    c.ry(params[2], 0) 
    c.rx(params[0], 1) 
    c.rx(params[1], 2) 
    c.swap(0, 1)
    c.swap(1, 2)
    if barriers: c.barrier()
    c.cnot(0,2) 
    c.cnot(1,2) 
    if barriers: c.barrier()
    c.rx(params[3], 0)
    c.rx(params[4], 1)
    c.ry(params[5], 2)
    if barriers: c.barrier()
    return c
    
def _GraphCycl_6qubits_6params(params, barriers = False):        
    """ Returns handle to cyc6 cluster state with c-phase gates"""
    logical_qubits = qk.QuantumRegister(6, 'logicals')
    c = qk.QuantumCircuit(logical_qubits)
    c.ry(params[0],0)
    c.ry(params[1],1)
    c.ry(params[2],2)
    c.ry(params[3],3)
    c.ry(params[4],4)
    c.ry(params[5],5)
    if barriers: c.barrier()
    c.cz(0,1)
    c.cz(2,3)
    c.cz(4,5)
    if barriers: c.barrier()
    c.cz(1,2)
    c.cz(3,4)
    c.cz(5,0)
    if barriers: c.barrier()
    return c

def _GraphCycl_6qubits_6params_inefficient(params, barriers = False):        
    """ Returns handle to cyc6 cluster state with cry() gates"""
    logical_qubits = qk.QuantumRegister(6, 'logicals')
    c = qk.QuantumCircuit(logical_qubits)
    c.ry(params[0],0)
    c.ry(params[1],1)
    c.ry(params[2],2)
    c.ry(params[3],3)
    c.ry(params[4],4)
    c.ry(params[5],5)
    if barriers: c.barrier()
    c.crz(np.pi, 0,1)
    c.crz(np.pi, 2,3)
    c.crz(np.pi, 4,5)
    if barriers: c.barrier()
    c.crz(np.pi, 1,2)
    c.crz(np.pi, 3,4)
    c.crz(np.pi, 5,0)
    if barriers: c.barrier()
    return c

def _GraphCycl_6qubits_24params(params, barriers = False):
    """ Ansatz to be refined, too many params - BO doens't converge"""
    logical_qubits = qk.QuantumRegister(6, 'logicals')
    c = qk.QuantumCircuit(logical_qubits)
    c.h(0)
    c.h(1)
    c.h(2)
    c.h(3)
    c.h(4)
    c.h(5)
    if barriers: c.barrier()
    c.ry(params[0], 0)
    c.ry(params[1], 1)
    c.ry(params[2], 2)
    c.ry(params[3], 3)
    c.ry(params[4], 4)
    c.ry(params[5], 5)
    if barriers: c.barrier()
    c.cnot(0,1) 
    c.cnot(2,3) 
    c.cnot(4,5)
    if barriers: c.barrier()
    c.rz(params[6], 0)
    c.rz(params[7], 1)
    c.rz(params[8], 2)
    c.rz(params[9], 3)
    c.rz(params[10], 4)
    c.rz(params[11], 5)
    if barriers: c.barrier()
    c.cnot(1,2) 
    c.cnot(3,4)
    c.cnot(5,0)
    if barriers: c.barrier()
    c.u2(params[12], params[13], 0)
    c.u2(params[14], params[15], 1)
    c.u2(params[16], params[17], 2)
    c.u2(params[18], params[19], 3)
    c.u2(params[20], params[21], 4)
    c.u2(params[22], params[23], 5)
    if barriers: c.barrier()
    return c