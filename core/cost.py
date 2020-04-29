#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:11:28 2020

@author: fred
TODO: (SOON) Better impliment the measurements
TODO: (VERYSOON) Move bind_parameters to utils as it's used in optimisers
TODO: (SOON) implement more general graph states 
TODO: (SOON) PROBLEM WITH WitnessesCost1 SHOULD NOT BE USED
DONE: See new class - Batch Concatenated Cost Functions
TODO: (LATER) ability to deal with different number of shots 
TODO: (LATER) implement sampling (of the measurement settings) strategy

CHANGES
* Cost now conforms to CostInterface
* Cost.meas_func now changed to cost.evaluate_cost
* cost.evaluate_cost is now a general function that calls self._meas_func that was 
    generated in the sub classes
    cost.evaluate_cost accepts a qk.results OBJECT IFF
* Ansatz inputs now it's own class that holds lots of useful info

Choice of noise_models, initial_layouts, nb_shots, etc.. is done through the 
quantum instance passed when initializing a Cost, i.e. it is outside of the
scope of the classes here
"""
import abc
import pdb
import copy
import numpy as np
import qiskit as qk
import utilities as ut

#import itertools as it
pi =np.pi

#======================#
# Basic cost interface
#======================#
class CostInterface(metaclass=abc.ABCMeta):
    """ Impliments interface that can be used in batch processing"""
    @property
    @abc.abstractmethod
    def meas_circuits(self):
        """ Returns a list of measurement circs required to evaluate cost function"""
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def qk_vars(self):
        """ Returns a list of qiskit.circuit.parameter.Parameter objects for the paramaterised circs"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def evaluate_cost(self, results : qk.result.result.Result, 
                      name = None):
        """ Returns the result of the cost function from a qk results object, 
            optional to spesify a name to give the results list
            TODO: extend to allow list of names"""
        raise NotImplementedError
    
   

#======================#
# Base class
#======================#
class Cost(CostInterface):
    """
    This base class defines all the ingredients necessary to evaluate a cost 
    function based on an ansatz circuit:
        + what should be measured and how many times
        + how the measurements outcomes(counts) should be aggregated to return 
          an estimate of the cost
        + how should be the full(ansatz+measurements) circuit generated and 
          transpiled
        
    Logic of computing the cost are defined by the followings (which are not
    implemented in the base class but should be implemented in the subclasses):
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
        + shot_noise
          ++ added to see shot noise to better estimate difference in results (maybe this is overkill)
          
    Bits removed:
        - No longer accepts pretranspiled circuits, ansatz must now be class and
            dealing with pre-transpiling should happen there. 

    Terminology:
        + ansatz: Object of ansatz class that has all needed properties, some 
        + circuit: quantum circuit 
        + qk_vars qiskit.parameter objects
        + measurable circuit: circuit with measurement operations
        
    """
    def __init__(self, ansatz, instance, 
                 fix_transpile = True, # maybe redundent now
                  keep_res = False, 
                  verbose = True, 
                  noise_model = None,
                  debug = False, 
                  error_correction = False,
                  name = None, **args):
        """ initialize the cost function taking as input parameters:
            + ansatz : either a function taking parameters as input and 
                       returning a circuit, 
                       or a transpiled circuit
            + N <int>: the number of qubits
            + instance (QuantumInstance)
        """
        if debug: pdb.set_trace()
        self.name = name
        self.ansatz = ansatz
        self.instance = instance
        self.nb_qubits = ansatz.nb_qubits  # may be redundant
        self.dim = np.power(2,ansatz.nb_qubits)
        self.nb_params = ansatz.nb_params # maybe redundant
        self.fix_transpile = fix_transpile # is it needed
        self.verbose = verbose
        self._keep_res = keep_res
        self._res = []
        # These methods needs to be implemented in the subclasses
        #--------------------------------------
        self._list_meas = self._gen_list_meas()  
        self._meas_func = self._gen_meas_func() 
        #--------------------------------------
        self.main_circuit = ansatz.circuit
        self._untranspiled_main_circuit = copy.deepcopy(ansatz.circuit)
        self._qk_vars = ansatz.params
        self._meas_circuits = gen_meas_circuits(self._untranspiled_main_circuit, 
                                                self._list_meas)
        self._meas_circuits = self.instance.transpile(self._meas_circuits)
        self._label_circuits()
        #--------------------------------------
        self.err_corr = error_correction
        if(self.err_corr):
            raise NotImplementedError
    
    def __call__(self, params, debug=False):
        """ Estimate the CostFunction for some parameters - Has a known buy:
            if number of measurement settings > max_job_size """
        if debug: pdb.set_trace()
        # reshape the inputs
        params_reshaped = np.atleast_2d(params)
        nb_meas = len(self._list_meas) #number of meas taken per set of parameters
        nb_params = len(params_reshaped) #number of different parameters
        
        # List of all the circuits to be ran
        bound_circs = []
        for p in params_reshaped:
            bound_circs += bind_params(self._meas_circuits, p, self._qk_vars)
        results = self.instance.execute(bound_circs, 
                                        had_transpiled=self.fix_transpile)         
        counts = [results.get_counts(i) for i in range(len(bound_circs))]
        counts = np.reshape(counts, newshape=[nb_params, nb_meas])
        
        # reshape the output
        res = np.array([self._meas_func(c) for c in counts]) 
        if np.ndim(res) == 1: 
            res = res[:,np.newaxis]
        if self.verbose: print(res)
        return res 

    def _gen_qk_vars(self):
        raise NotImplementedError("This function is now in the Ansatz class")        
                    
    def _init_res(self):
        """ Flush the res accumulated so far """
        self._res = []

    def _gen_list_meas(self):
        """ To be implemented in the subclasses """
        raise NotImplementedError()
        
    def _gen_meas_func(self):
        """ To be implemented in the subclasses """
        raise NotImplementedError()
    
    def _label_circuits(self):
        """ Gives (random name) to all circuits to they can be identified in the results obj"""
        if self.name == None:
            self.name = 'circuit_' + ut.gen_random_str(5)
        self.main_circuit.name = self.name
        for c in self._meas_circuits:
            c.name = self.name
            
    @property
    def meas_circuits(self):
        """ Returns list of measurement circuits needed to evaluate the cost function"""
        circs = self._meas_circuits
        return circs
    
    @property
    def qk_vars(self):
        """ Returns parameter objects in the circuit"""
        return self._qk_vars
    
    def evaluate_cost(self, results_obj, name = None):
        """ Returns cost value from results object/count list"""
        count_list = []
        if name == None:
            name = self.name
        for ii in range(len(results_obj.results)):
            if name in results_obj.results[ii].header.name:
                count_list.append(results_obj.get_counts(ii))
        return self._meas_func(count_list)

    def shot_noise(self, params, nb_experiments=8):
        """ Sends a single job many times to see shot noise"""        
        params = [params for ii in range(nb_experiments)]
        return self.__call__(params)
    
    def check_layout(self):
        """ Draft, check if all the meas_circuit have the same layout
        TODO: remove except if really needed
        """
        ref = self.main_circuit
        test = [compare_layout(ref, c) for c in self._meas_circuits]
        return np.all(test)

    def compare_layout(self, cost2, verbose=True):
        """ Draft, goal compare transpiled circuits (self._maincircuit)
        and ensure they have the same layout"""
        test1 = self.check_layout()
        test2 = cost2.check_layout()
        test3 = compare_layout(self.main_circuit, cost2.main_circuit)
        if verbose: 
            print("self: same layout - {}".format(test1))
            print("cost2: same layout - {}".format(test2))
            print("self and cost2: same layout - {}".format(test3))
        return test1 * test2 *test3
    
    def check_depth(self, long_output=False, delta = 1):
        """ Check the depths of the measurable circuits, are all within a delta
        """
        depth = [c.depth() for c in self._meas_circuits]
        test = (max(depth) - min(depth)) <=delta
        return test
    
    def get_depth(self, num=None):
        """ Get the depth of the circuit(s)
        if num=None main_circuit / num=-1 all the meas_circ / else meas_circ[num] 
        """
        circ = self._return_circuit(num)
        depth = [c.depth() for c in circ]
        return depth
    
    def compare_depth(self, cost2, verbose=True, delta=0):
        """ Draft, goal compare transpiled circuits (self._maincircuit)
        and ensure they have the same layout"""
        depth1 = self.check_depth(long_output=True)
        depth2 = cost2.check_depth(long_output=True)
        test1 = np.abs(max(depth1) - max(depth2)) <= delta
        test2 = np.abs(min(depth1) - min(depth2)) <= delta
        test = test1 and test2
        if verbose: 
            print("self and cost2: same depth - {}".format(test))
            print("self min-max: {} and {}".format(min(depth1), max(depth1)))
            print("cost2 min-max: {} and {}".format(min(depth2), max(depth2)))
        return test

    def draw(self, num=None, depth = False):
        """ Draw one of the circuit 
        if num=None main_circuit / num=-1 all the meas_circ / else meas_circ[num] 
        """
        circs = self._return_circuit(num)
        for c in circs:
            print(c)
            if depth:
                print(c.depth())
        
    def _return_circuit(self, num=None):
        """ Return a list of circuits according to num following the convention:
        if num=None main_circuit / num=-1 all the meas_circ / else meas_circ[num] 
        """
        if num is None:
            circ = [self.main_circuit]
        elif num >= 0:
            circ = [self._meas_circuits[num]]
        elif num == -1:
            circ = self._meas_circuits
        return circ


def compare_layout(circ1, circ2):
    """ Draft, define a list of checks to compare transpiled circuits
        not clear what the rules should be (or what would be a better name)
        So far: compare the full layout"""
    test = True
    test &= (circ1._layout.get_physical_bits() == circ2._layout.get_physical_bits())
    test &= (circ1.count_ops()['cx'] == circ2.count_ops()['cx'])
    return test


#======================#
# Subclasses: GHZ related costs
#======================#
class GHZPauliCost(Cost):
    """ Cost = fidelity w.r.t. a N-qubit GHZ state, estimated based on the 
    expected values of N-fold Pauli operators (e.g. 'XXY')
    """   
    # Hardcoded list of measurements settings for GHZ of different sizes
    # {'nb_qubits':(meas_strings, weights)}, wehere the measurement string is a 
    # list of Pauli operators and the weights correspond to the decomposition 
    # of the GHZ state in the Pauli tensor basis (up to a constant 1/dim)
    # It could be automated to deal with arbitrary size state
    _GHZ_PAULI_DECOMP = {
    '2':(
            ['xx', 'yy', 'zz'], 
            np.array([1.,-1.,1.])
            ),
    '3':(
            ['1zz','xxx','xyy','yxy','yyx','z1z','zz1'], 
            np.array([1., 1., -1., -1., -1., 1.,1.])
            ),
    '4':( 
            ['11zz','1z1z','1zz1','xxxx','xxyy','xyxy','xyyx','yxxy','yxyx',
             'yyxx','yyyy','z11z','z1z1','zz11','zzzz'],
            np.array([1.,1.,1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,1.,1.,1.,1.,1.])
            )
        }
        
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
    

#======================#
# Subclasses: Graph states
#======================#    
class GraphCyclPauliCost(Cost):
    """ A N-qubit Cyclical graph has edges = [[1,2],[2,3],...,[N-1,N],[N,1]]
    Cost = fidelity, estimated based on the expected values of the N-fold Pauli 
    operators (e.g. 'XXY')
    """   
    # Hardcoded list of measurements settings for Cyclical graph states of 
    #different sizes {'nb_qubits':(meas_strings, weights)}, wehere the measurement 
    # string is a list of Pauli operators and the weights correspond to the 
    # decomposition of the target state in the Pauli tensor basis (up to a constant 1/dim)
    # It could be automated to deal with arbitrary size state
    _CYCLICAL_PAULI_DECOMP = {
    '2':(
            ['1x','x1','xx'], 
            np.array([1,1,1])
            ),
    '3':(
            ['1yy','xxx','xzz','y1y','yy1','zxz','zzx'], 
            np.array([1,-1,1,1,1,1,1])
            ),
    '4':( 
            ['1x1x','1yxy','1zxz','x1x1','xxxx','xy1y','xz1z','y1yx','yxy1','yyzz',
             'yzzy','z1zx','zxz1','zyyz','zzyy'],
            np.array([1,-1,1,1,1,-1,1,-1,-1,1,1,1,1,1,1])
            ),
    '5':( 
            ['11zxz','1x1yy','1xzzx','1yxxy','1yy1x','1zxz1','1zyyz','x1xzz','x1yy1',
             'xxxxx','xxy1y','xy1yx','xyzzy','xz11z','xzzx1','y1x1y','y1yxx','yxxy1',
             'yxyzz','yy1x1','yyz1z','yz1zy','yzzyx','z11zx','z1zyy','zx1xz','zxz11',
             'zyxyz','zyyz1','zzx1x','zzyxy'],
            np.array([1,1,1,1,1,1,1,1,1,-1,1,1,-1,1,1,1,1,1,-1,1,1,1,-1,1,1,1,1,-1,1,1,-1])
            ),
    '6':( 
            ['111zxz','11zxz1','11zyyz','1x1x1x','1x1yxy','1xz1zx','1xzzyy','1yxxxy',
             '1yxy1x','1yy1yy','1yyzzx','1zx1xz','1zxz11','1zyxyz','1zyyz1','x1x1x1',
             'x1xz1z','x1yxy1','x1yyzz','xxxxxx','xxxy1y','xxy1yx','xxyzzy','xy1x1y',
             'xy1yxx','xyz1zy','xyzzyx','xz111z','xz1zx1','xzzxzz','xzzyy1','y1x1yx',
             'y1xzzy','y1yxxx','y1yy1y','yxxxy1','yxxyzz','yxy1x1','yxyz1z','yy1xzz',
             'yy1yy1','yyz11z','yyzzx1','yz11zy','yz1zyx','yzzx1y','yzzyxx','z111zx',
             'z11zyy','z1zx1x','z1zyxy','zx1xz1','zx1yyz','zxz111','zxzzxz','zyxxyz',
             'zyxyz1','zyy1xz','zyyz11','zzx1yy','zzxzzx','zzyxxy','zzyy1x'],
            np.array([1,1,1,1,-1,1,1,-1,-1,1,1,1,1,-1,1,1,1,-1,1,1,-1,-1,1,-1,-1,
                      -1,1,1,1,1,1,-1,1,-1,1,-1,1,-1,-1,1,1,1,1,1,-1,1,1,1,1,1,
                      -1,1,1,1,1,1,-1,1,1,1,1,1,1])
            )
        }
        
    def _gen_list_meas(self):
        return self._CYCLICAL_PAULI_DECOMP[str(self.nb_qubits)][0]
    
    def _gen_meas_func(self):
        """ expected parity associated to each of the measurement settings"""
        weights = self._CYCLICAL_PAULI_DECOMP[str(self.nb_qubits)][1]
        dim = self.dim
        def meas_func(counts):
            return (1+np.dot([expected_parity(c) for c in counts], weights))/dim
        return meas_func


class GraphCyclWitness1Cost(Cost):
    """ Cost function based on the construction of witnesses for genuine 
    entanglement ([guhne2005])
    Stabilizer generators S_l of cyclical graph states are (for N=4 qubits) 
        S = <XZIZ, ZXZI, IZXZ, ZIZX>
    To estimate S_1 to S_N only requires two measurement settings: XZXZ, ZXZX
    Cost =  (S_1 - 1)/2 + Prod_l>1 [(S_l + 1)/2] 
    !!! ONLY WORK FOR EVEN N FOR NOW !!!
    !!! PROBABLY WRONG (or at least not understood clearly) !!!
    """
    def _gen_list_meas(self):
        """ two measurement settings ['zxzx...zxz', 'xzxzx...xzx']"""
        N = self.nb_qubits
        if (N%2): 
            raise NotImplementedError("ATM cannot deal with odd N")
        else:
            meas_odd = "".join(['zx'] * (N//2))
            meas_even = "".join(['xz'] * (N//2))
        return [meas_odd, meas_even]
    
    def _gen_meas_func(self):
        raise Warning("This is likely broken be careful")
        """ functions defining how outcome counts should be used """
        N = self.nb_qubits
        if (N%2): 
            raise NotImplementedError("ATM cannot deal with odd N")
        else:
            ind_odd = [[i, i+1, i+2] for i in range(0,N-2, 2)] + [[0, N-2, N-1]]  
            ind_even = [[i, i+1, i+2] for i in range(1,N-2, 2)] + [[0, 1, N-1]]
            def meas_func(counts):
                counts_odd, counts_even = counts[0], counts[1]
                S_odd = np.array([expected_parity(counts_odd, indices=i) for i in ind_odd])
                S_even = np.array([expected_parity(counts_even, indices=i) for i in ind_even])
                return 0.5*(S_even[-1]-1) + np.prod((S_odd+1)/2) * np.prod((S_even[:-1]+1)/2) 
        return meas_func


class GraphCyclWitness2Cost(Cost):
    """ Exactly as GraphCyclWitness1Cost except that:
        Cost =  Sum_l[S_l] - (N-1)I """   
    def _gen_list_meas(self):
        """ two measurement settings ['zxzx...zxz', 'xzxzx...xzx']"""
        N = self.nb_qubits
        if (N%2): 
            raise NotImplementedError("ATM cannot deal with odd N")
        else:
            meas1 = "".join(['zx'] * (N//2))
            meas2 = "".join(['xz'] * (N//2))
        return [meas1, meas2]
    
    def _gen_meas_func(self):
        """ functions defining how outcome counts should be used """
        N = self.nb_qubits
        if (N%2): 
            raise NotImplementedError("ATM cannot deal with odd N")
        else:
            ind_odd = [[i, i+1, i+2] for i in range(0,N-2, 2)] + [[0, N-2, N-1]]  
            ind_even = [[i, i+1, i+2] for i in range(1,N-2, 2)] + [[0, 1, N-1]]
            def meas_func(counts):
                counts_odd, counts_even = counts[0], counts[1]
                S_odd = np.array([expected_parity(counts_odd, indices=i) for i in ind_odd])
                S_even = np.array([expected_parity(counts_even, indices=i) for i in ind_even])
                return np.sum(S_odd) + np.sum(S_even) - (N-1)
        return meas_func


class GraphCyclWitness2FullCost(Cost):
    """ Same cost function as GraphCyclWitness2Cost, except that the measurement
    settings to obtain the expected values of the generators S_l have been
    splitted into N measurent settings (rather than 2), and now each measurement
    settings involved only 3 measurements instead of N
    -> measurement outcomes should be less noisy as less measurements are
       involved per measurement settings
    """   
    def _gen_list_meas(self):
        """ N measurement settings ['xz1..1z', 'zxz1..1', .., 'z1..1zx' ]"""
        N = self.nb_qubits
        list_meas = []
        for ind in range(N):    
            meas = ['1'] * N
            meas[(ind-1) % N] = 'z'
            meas[ind % N] = 'x'
            meas[(ind+1) % N] = 'z'
            list_meas.append(''.join(meas))
        return list_meas
    
    def _gen_meas_func(self):
        """ functions defining how outcome counts should be used """
        N = self.nb_qubits
        def meas_func(counts):
            exp = [expected_parity(c) for c in counts]
            return np.sum(exp)  - (N-1)
        return meas_func


class GraphCyclWitness3Cost(Cost):
    """ Exactly as GraphCyclWitness1Cost except that Cost =  XXX
    To implement"""   
    
    def _gen_list_meas(self):
        """ N measurement settings ['xz1..1z', 'zxz1..1', .., 'z1..1zx' ]"""
        N = self.nb_qubits
        list_meas = []
        for ind in range(N):    
            meas = ['1'] * N
            meas[(ind-1) % N] = 'z'
            meas[ind % N] = 'x'
            meas[(ind+1) % N] = 'z'
            list_meas.append(''.join(meas))
        return list_meas
    
    def _gen_meas_func(self):
        """ functions defining how outcome counts should be used """
        N = self.nb_qubits
        def meas_func(counts):
            exp = [expected_parity(c) for c in counts]
            return np.sum(exp)  - (N-1)
        return meas_func


# ------------------------------------------------------
# Functions to compute expected values based on measurement outcomes counts as 
# returned by qiskit
# ------------------------------------------------------
def freq_even(count_result, indices=None):
    """ return the frequency of +1 eigenvalues:
    The +1 e.v. case corresponds to the case where the number of 0 in the 
    outcome string is even
    
    indices: list<integer>
             if not None it allows to consider only selected elements of the 
             outcome string
    """
    nb_odd, nb_even = 0, 0
    for k, v in count_result.items():
        k_invert = k[::-1]
        sub_k = get_substring(k_invert, indices)
        nb_even += v * (sub_k.count('1')%2 == 0)
        nb_odd += v * (sub_k.count('1')%2)
    return nb_even / (nb_odd + nb_even)


def expected_parity(results,indices=None):
    """ return the estimated value of the expectation of the parity operator:
    P = P+ - P- where P+(-) is the projector 
    Comment: Parity operator ircuit.quantumcircuit.QuantumCircuitircuit.quantumcircuit.QuantumCircuitmay nor be the right name
    """
    return 2 * freq_even(results, indices=indices) - 1


def get_substring(string, list_indices=None):
    """ return a substring comprised of only the elements associated to the 
    list of indices
    Comment: probably already exist or there may be a better way"""
    if list_indices == None:
        return string
    else:
        return "".join([string[ind] for ind in list_indices])


# ------------------------------------------------------
# Some functions to deals with appending measurement and param bindings  
# ------------------------------------------------------
def append_measurements(circuit, measurements, logical_qubits=None):
    """ Append measurements to one circuit:
        TODO: Replace with Weighted pauli ops?"""
    circ = copy.deepcopy(circuit)
    num_creg = len(measurements.replace('1',''))
    if num_creg > 0:
        cr = qk.ClassicalRegister(num_creg, 'classical')
        circ.add_register(cr)
    if logical_qubits is None: 
        logical_qubits = np.arange(circ.num_qubits)
    creg_idx = 0
    for qb_idx, basis in enumerate(measurements):
        qubit_number = logical_qubits[qb_idx]
        if basis == 'z':
            circ.measure(qubit_number, creg_idx)
            creg_idx += 1
        elif basis == 'x':
            circ.u2(0.0, pi, qubit_number)  # h
            circ.measure(qubit_number, creg_idx)
            creg_idx += 1
        elif basis == 'y':
            circ.u1(-np.pi / 2, qubit_number)  # sdg
            circ.u2(0.0, pi, qubit_number)  # h
            circ.measure(qubit_number, creg_idx)
            creg_idx += 1
        elif basis != '1':
            raise NotImplementedError('measurement basis {} not understood').format(basis)
    return circ


def gen_meas_circuits(main_circuit, meas_settings, logical_qubits=None):
    """ Return a list of measurable circuit based on a main circuit and
    different settings"""
    c_list = [append_measurements(main_circuit.copy(), m, logical_qubits) 
                  for m in meas_settings] 
    return c_list


def bind_params(circ, param_values, param_variables):
    """ Take a list of circuits with bindable parameters and bind the values 
    passed according to the param_variables
    Returns the list of circuits with bound values DOES NOT MODIFY INPUT
    (i.e. hardware details??)
    """
    if type(circ) != list: circ = [circ]
    val_dict = {key:val for key,val in zip(param_variables, param_values)}
    bound_circ = [cc.bind_parameters(val_dict) for cc in circ]
    return bound_circ  





            
        
#%%
# -------------------------------------------------------------- #

if __name__ == '__main__':
    from qiskit.test.mock import FakeRochester
    import ansatz as anz
    fake = FakeRochester() # not working
    simulator = qk.Aer.get_backend('qasm_simulator')
    backends = [simulator]
    sim = simulator
    for sim in backends:
        #-----#
        # Verif conventions
        #-----#
        ansatz = anz.AnsatzFromFunction(anz._GHZ_3qubits_6_params_cx0)
        
        bound_circ = bind_params(ansatz.circuit, [1,2,3,4,5,6], ansatz.circuit.parameters)
        
        inst = qk.aqua.QuantumInstance(sim, shots=8192, optimization_level=3)
        transpiled_cir = inst.transpile(bound_circ)[0]
        m_c = gen_meas_circuits(transpiled_cir, ['zzz'])
        res = inst.execute(m_c)
        counts = res.get_counts()
        
        #-----#
        # GHZ
        #-----#
        # Create an ansatz capable of generating a GHZ state (not the most obvious 
        # one here) with the set of params X_SOL

        X_SOL = np.pi/2 * np.array([1.,1.,2.,1.,1.,1.])
        X_LOC = np.pi/2 * np.array([1., 0., 4., 0., 3., 0.])
        X_RDM = np.random.uniform(0.0, 2*pi, size=(6,1))
        
        # Create an instance
        sim = qk.Aer.get_backend('qasm_simulator')
        inst = qk.aqua.QuantumInstance(sim, shots=8192, optimization_level=3)
        
        # Verif the values of the different GHZ cost
        # Fidelity
        ghz_cost = GHZPauliCost(ansatz=ansatz, instance = inst)
        assert ghz_cost(X_SOL) == 1.0, "For this ansatz, parameters, cost function should be one"
        assert np.abs(ghz_cost(X_LOC) - 0.5) < 0.1, "For this ansatz and parameters, the cost function should be close to 0.5 (up to sampling error)"
        
        test_batch = ghz_cost([X_SOL] * 10)
        
        # Witnesses inspired cost functions: they are different compared to the fidelity
        # but get maximized only when the state is the right one
        ghz_witness1 = GHZWitness1Cost(ansatz=ansatz, instance = inst, N=3, nb_params=6)
        assert ghz_witness1(X_SOL) == 1.0, "For this ansatz, parameters, cost function should be one"
        assert np.abs(ghz_witness1(X_LOC) - 0.31) < 0.1, "For this ansatz and parameters, the cost function should be close to 0.31 (up to sampling error)"

        ghz_witness2 = GHZWitness2Cost(ansatz=ansatz, instance = inst, N=3, nb_params=6)
        assert ghz_witness2(X_SOL) == 1.0, "For this ansatz, parameters, cost function should be one"
        assert np.abs(ghz_witness2(X_LOC) + 0.5) < 0.1, "For this ansatz and parameters, the cost function should be close to 0.31 (up to sampling error)"    
        
        
        
        #-----#
        # Cyclical graph states
        #-----#

        ansatz = anz.AnsatzFromFunction(anz._GraphCycl_6qubits_6params)
        X_SOL = np.pi/2 * np.ones(ansatz.nb_params) # sol of the cycl graph state for this ansatz
        X_RDM = np.array([1.70386471,1.38266762,3.4257722,5.78064,3.84102323,2.37653078])
        #X_RDM = np.random.uniform(low=0., high=2*np.pi, size=(N_params,))
        
        # Create an instance
        sim = qk.Aer.get_backend('qasm_simulator')
        inst = qk.aqua.QuantumInstance(sim, shots=8192, optimization_level=3)
        graph_cost = GraphCyclPauliCost(ansatz=ansatz, instance = inst)
        
        fid_opt = graph_cost(X_SOL)
        fid_rdm = graph_cost(X_RDM)
        assert fid_opt == 1.0, "For this ansatz, parameters, cost function should be one"
        assert (fid_opt-fid_rdm) > 1e-4, "For this ansatz, parameters, cost function should be one"
        
        if False: # don't test the broken witness (raises warning now)
            graph_cost1 = GraphCyclWitness1Cost(ansatz=ansatz, instance = inst)
            cost1_opt = graph_cost1(X_SOL)
            cost1_rdm = graph_cost1(X_RDM)
            assert cost1_opt == 1.0, "For this ansatz, parameters, cost function should be one"
            assert  (fid_rdm - cost1_rdm) > 1e-4, "cost function1 should be lower than true fid"
        
        graph_cost2 = GraphCyclWitness2Cost(ansatz=ansatz, instance = inst)
        cost2_opt = graph_cost2(X_SOL)
        cost2_rdm = graph_cost2(X_RDM)
        assert cost2_opt == 1.0, "For this ansatz, parameters, cost function should be one"
        assert  (fid_rdm - cost2_rdm) > 1e-4, "cost function should be lower than true fid"
        
        graph_cost2full = GraphCyclWitness2FullCost(ansatz=ansatz, instance = inst)
        cost2full_opt = graph_cost2full(X_SOL)
        cost2full_rdm = graph_cost2full(X_RDM)
        assert cost2full_opt == 1.0, "For this ansatz, parameters, cost function should be one"
        assert  (fid_rdm - cost2_rdm) > 1e-4, "cost function should be lower than true fid"
        assert  np.abs(cost2full_rdm - cost2_rdm) < 0.1, "both cost function should be closed"
        
        X_SOL = np.pi/2 * np.array([1.,1.,2.,1.,1.,1.])
        circs0 = ghz_cost.meas_circuits
        circs1 = ghz_witness2.meas_circuits
        circs0 = bind_params(circs0, X_SOL, ghz_cost.qk_vars)
        circs1 = bind_params(circs1, X_SOL, ghz_witness2.qk_vars)
        circs = circs0 + circs1
        res = inst.execute(circs, had_transpiled=True)
        
        assert ghz_cost.evaluate_cost(res) == 1.0, "For passing in results object, check the solutions are correct"
        assert ghz_witness2.evaluate_cost(res) == 1.0, "For passing in results object, check the solutions are correct"
