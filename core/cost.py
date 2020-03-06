#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:11:28 2020

@author: fred
"""
import qiskit as qk
import numpy as np
#import itertools as it

pi =np.pi
NB_SHOTS_DEFAULT = 256

class Cost():
    """
    This base class defines all the ingredients necessary to evaluate a cost 
    function based on an ansatz circuit:
        + what should be measured and how many times
        + how the measurements outcomes(counts) should be aggregated to return 
          an estimate of the cost
        + how should be the full(ansatz+measurements) circuit generated and 
          transpiled
        + what results to keep 
    
    Logic of computing the cost are defined by the followings (which are not
    implement in the base class but should be implemented in the subclasses):
        + self._list_meas is a list of M strings indicating all the measurement 
            settings required
        + self._meas_func is a single function taking as an input the list of 
            the M outputs from the execution of the circuits (counts dictionaries) 
            and returning a single value
    
    Other bits of logic have been included:
        + transpile
          ++ if False the circuit is only transpiled once. Then 
             in this case 'args' should contain 
          ++if True the circuit is transpiled at each call of the function
        + keep_res 
          ++ if True the results from the execution of the circuits are kept 
             (appended) in self._res
                    
    Questions: 
        + should we pass backend or instance
        + Do we need to pass the number of shots in the instance when transpiling or can they 
        be specified at exec time
    append_measurements
    
    Terminology:
        + ansatz: callable (function so far but may be extended to an object) 
                 which takes parameter as input and return a circuit
        + circuit: quantum circuit 
        + measurable circuit: circuit with measurement operations
        
    
    """

    def __init__(self, ansatz, N,  instance, nb_params, fix_transpile = True,
                  keep_res = True, verbose = True, noise_model = None,
                  **args):
        """ initialize the cost function taking as input parameters:
            + ansatz 
            + N <int>: the number of qubits
            + simulator
        """
        self.ansatz = ansatz
        self.instance = instance
        self.nb_qubits = N #may be redundant
        self.dim = np.power(2,N)
        self.nb_params = nb_params # maybe redundant
        self.fix_transpile = fix_transpile
        self.verbose = verbose
        self._keep_res = keep_res
        self._res = []

        # These methods needs to be implemented in the subclasses
        self._list_meas = self._gen_list_meas()  
        self._meas_func = self._gen_meas_func() 

        # measurable circuits are transpiled at initialization
        self._gen_qk_vars()
        self._transpile_measurable_circuits(self._qk_vars)
    
    def __call__(self, params):
        """ Estimate the CostFunction for some parameters"""
        if np.ndim(params) > 1 :
            res = np.array([self.__call__(p) for p in params])
        else:
            #bind the values of the parameters to the circuit
            bound_circs = bind_params(self._main_circuit, params, self._qk_vars)
            
            #See if one can add a noise model here and the number of parameters
            results = self.instance.execute(bound_circs, had_transpiled=self.fix_transpile)
            
            if self._keep_res: self._res.append(results.to_dict())
            counts = [results.get_counts(ii) for ii in range(len(self._list_meas))]
            res = self._meas_func(counts)
            
        if self.verbose: print(res)
        return np.squeeze(res) 
    
    def _transpile_measurable_circuits(self, params):
        """ Transpile all the measurable circuits"""
        instance = self.instance
        ansatz = self.ansatz
        meas_settings = self._list_meas
        list_circ_meas = gen_meas_circuits(ansatz, meas_settings, params)
        self._main_circuit = instance.transpile(list_circ_meas)
        print('Measuremable circuits have been transpiled')

    def _gen_qk_vars(self):
        """ Generate qiskit variables to be bound to a circuit"""
        name_params = ['R'+str(i) for i in range(self.nb_params)]
        self._qk_vars = [qk.circuit.Parameter(n) for n in name_params]
        
    def _init_res(self):
        self._res = []

    def _gen_list_meas(self):
        raise NotImplementedError()
        
    def _gen_meas_func(self):
        raise NotImplementedError()

    

# Subclasses: GHZ related costs
class GHZPauliCost(Cost):
    """ Cost = fidelity w.r.t. a N-qubit GHZ state, estimated based on the 
    expected values of N-fold Pauli operators (e.g. 'XXY')
    """   
    # Hardcoded list of measurements settings for GHZ of different sizes
    # {'nb_qubits':(meas_strings, weights)}, wehere the measurement string is a 
    # list of Pauli operators and the weights correspond to the decomposition 
    # of the GHZ state in the 
    # Could be automated
    _GHZ_PAULI_DECOMP = {
    '2':(
            ['xx', 'yy', 'zz'], 
            np.array([1.,-1.,1.])/2**2
            ),
    '3':(
            ['1zz','xxx','xyy','yxy','yyx','z1z','zz1'], 
            np.array([1., 1., -1., -1., -1., 1.,1.])/2**3
            ),
    '4':( 
            ['11zz','1z1z','1zz1','xxxx','xxyy','xyxy','xyyx','yxxy','yxyx','yyxx','yyyy',
           'z11z','z1z1','zz11','zzzz'],
            np.array([1.,1.,1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,1.,1.,1.,1.,1.])/2**4
            )}
        
    def _gen_list_meas(self):
        return self._GHZ_PAULI_DECOMP[str(self.nb_qubits)][0]
    
    def _gen_meas_func(self):
        """ expected parity associated to each of the measurement settings"""
        weights = self._GHZ_PAULI_DECOMP[str(self.nb_qubits)][1]
        dim = self.dim
        def meas_func(counts):
            return (1+np.dot([expected_parity(c) for c in counts], weights))/dim

        return meas_func


class GHZWitness1Cost(Cost):
    """ Cost based on witnesses for genuine entanglement ([guhne2005])
    Stabilizer generators S_l of GHZ are (for n=4) S = <XXXX, ZZII, IZZI, IIZZ>
    To estimate S_1 to S_n only requires two measurement settings: XXXX, ZZZZ
    Cost =  (S_1 - 1)/2 + Prod_l>1 [(S_l + 1)/2] """   
    
    def _gen_list_meas(self):
        """ two measurement settings ['x...x', 'z...z']"""
        N = self.nb_qubits
        list_meas = ['x'*N, 'z'*N]
        return list_meas
    
    def _gen_meas_func(self):
        """ functions defining how outcome counts should be used """
        N = self.nb_qubits
        def meas_func(counts):
            S1 = freq_even(counts[0])
            S2 = np.array([freq_even(counts[1], indices=[i,i+1]) for i in range(N-1)])
            return 0.5*(S1-1) + np.prod((S2+1)/2)
        return meas_func

class GHZWitness2Cost(Cost):
    """ Exactly as GHZWitness1Cost except that Cost =  Sum_l[S_l] - (N-1)I """   
    
    def _gen_list_meas(self):
        """ two measurement settings ['x...x', 'z...z']"""
        N = self.nb_qubits
        list_meas = ['x'*N, 'z'*N]
        return list_meas
    
    def _gen_meas_func(self):
        """ functions defining how outcome counts should be used """
        N = self.nb_qubits
        def meas_func(counts):
            S1 = freq_even(counts[0])
            S2 = np.array([freq_even(counts[1], indices=[i,i+1]) for i in range(N-1)])
            return S1 + np.sum(S2) - (N -1)
        return meas_func
    
    
    
    
# ------------------------------------------------------
# Functions to compute expected values based on measurement outcomes counts as 
# returned by qiskit
# ------------------------------------------------------
def freq_even(count_result, indices=None):
    """ return a frequency of +1 eigenvalues:
    +1 e.v. corresponds to the case where the number of 0 in the outcome is even
    
    indices: list<integer>
             if not None it allows to consider only selected elements of the string
    """
    nb_odd, nb_even = 0, 0
    for k, v in count_result.items():
        sub_k = get_substring(k, indices)
        nb_even += v * (sub_k.count('0')%2 == 0)
        nb_odd += v * (sub_k.count('0')%2)
    return nb_even / (nb_odd + nb_even)

def expected_parity(results):
    """ return the estimated expected value of the parity operator:
    P = P+ - P-
    """
    return 2 * freq_even(results) - 1


def get_substring(string, list_indices=None):
    """ probably already exist or better way.. still"""
    if list_indices == None:
        return string
    else:
        return "".join([string[ind] for ind in list_indices])

# ------------------------------------------------------
# Some functions to deals with appending measurement and param bindings  
# ------------------------------------------------------
def append_measurements(circ, measurements):
    """ Assumes circ returns an instance of the relevant circuit"""
    num_classical = len(measurements.replace('1',''))
    if num_classical > 0:
        cr = qk.ClassicalRegister(num_classical, 'classical')
        circ.add_register(cr)
    ct_m = 0
    ct_q = 0
    for basis in measurements:
        if basis == 'z':
            circ.measure(ct_q, ct_m)
            ct_m+=1
        elif basis == 'x':
            circ.u3(pi/2, 0, 0, ct_q)
            circ.measure(ct_q, ct_m)
            ct_m+=1
        elif basis == 'y':
            circ.u3(pi/2, -pi/2, -pi/2, ct_q)
            circ.measure(ct_q, ct_m)
            ct_m+=1
        elif basis == '1':
            pass
        ct_q+=1
    return circ


def gen_meas_circuits(ansatz, meas_settings, params):
    """ Return a list of measurable circuit with same parameters but with 
    different measurement settings"""
    c_list = [append_measurements(ansatz(params), m) for m in meas_settings]
    return c_list


def bind_params(circ, param_values, param_variables):
    """ Take a list of circuits with bindable parameters and bind the values 
    passed according to the param_variables
    Returns the list of circuits with bound values DOES NOT MODIFY INPUT
    (i.e. hardware details??)
    """
    if type(circ) != list: 
        circ = [circ]
    val_dict = {key:val for key,val in zip(param_variables, param_values)}
    bound_circ = [cc.bind_parameters(val_dict) for cc in circ]
    return bound_circ  

