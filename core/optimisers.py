
# list of * contents
__all__ = [
    'Optimiser',
    'ParallelOptimizer',
]

import sys
import numpy as np
import utilities as ut
import copy
import cost
from abc import ABC, abstractmethod
pi = np.pi

class Optimiser(ABC):
    """
    Interface for an Optimiser class. The optimiser must have two methods:
    one that returns a set of quantum circuits and another that takes new 
    data in and updates the internal state of the optimiser. It also has a
    property `prefix` that is used as an ID by the batch class.
    """

    @abstractmethod
    def next_evaluation_circuits(self):
        """ 
        Return the next set of Cost function evaluations, in the form of 
        executable qiskit quantum circuits
        """
        raise NotImplementedError

    @abstractmethod
    def update(self,results_obj):
        """ Process a new set of data in the form of a results object """
        raise NotImplementedError
    
    @abstractmethod
    def init_params(self):
        """ Returns parameters required to initialize optimiser"""
        raise NotImplementedError

    @property
    def prefix(self):
        return self._prefix

    @property
    def iter(self):
        return self._iter

    @property
    def best_x(self):
        return self._best_x

    def __init__(self):
        self._iter = 0 # keep track of iterations
        self._prefix = ut.gen_random_str(5) # make (almost certainly) unique id
        self._type = None # type
        self._iter_max = None
        self._best_x = None


class OptimiserBO(Optimiser):
    """
    Creates a warapper around GPyOpt.methods.BayesianOptimization """
    
    def __init__(self, optimiser):
        """
        Creates a warapper around GPyOpt.methods.BayesianOptimization to be used
        in ParallelOptimiser. 
        TODO: Better input checks? 
        
        Parameters
        ----------
        optimiser: Must be an instance of 
        """
        if str(optimiser.__class__) != "<class 'type'>" or not hasattr(optimiser, 'mro'):
            raise TypeError
        raise Warning('This is experimental for now, only an example')
        self.optimiser = optimiser
        self._prefix = ut.safe_string.gen(3)
    
    def next_evaluation_circuits(self):
        """
        Really only returns the next evaluation points requested by this optimiser
        TODO: RENAME
        """
        x_new = np.atleast_2d(np.squeeze(self.optimiser._compute_next_evaluations()))
        raise Warning('Currenty does not give format requried by _gen_circs_from_param')
        return x_new
    
    def update(self, x_new, y_new):
        """
        Updates the interal state of the optimiser with the data provided
        
        Parametres:
        -------------
        x_new: Parameter points that were requested/provided
        
        y_new: Cost functino evalutations for those parameter points
        """
        self.optimiser.X = np.vstack((self.optimiser.X, x_new))
        self.optimiser.Y = np.vstack((self.optimiser.Y, y_new))
        self.optimiser._update_model(self.optimiser.normalization_type)
        


class SPSA(Optimiser):
    """ Implementation of the Simultaneous Perturbation Stochastic Algo,
    Implemented to perform minimization (can be extended for maximization)
    """
    def __init__(
        self,
        cost_obj,
        domain,
        x_init,
        **spsa_args,
        ):
        """ 
        Parameters
        ----------
            cost_obj : Cost object
            domain: list of tuples or None 
                specify the domain of the optimization
                if None the parameters are unbounded but needs to provide x_init
            x_init: 1D/2D np.array
                if None x_init is uniformly drawn from the domain
            verbose : bool, optional
                Set level of output of the object
            spsa_args : dict
                typical_config = {'tol': 0, 'a':1, 'b':0.628, 's':0.602, 't':0.101,'A':0}
        Comments
        ----------
        Implementation follows [Spall98] (with alpha->s and gamma->t)
        + additional restricted domain
        """
        self.cost_obj = cost_obj
        self.nb_params = cost_obj.nb_params

        if domain is None:
            self._x_min, self._x_max = -np.inf, np.inf
            assert x_init is not None, "If domain is None, x_init should be specified"
        else:
            assert len(domain) == self.nb_params, "Length of domain and nb_params do not match"
            self._x_min, self._x_max = np.array(domain)[:,0], np.array(domain)[:,1]
            if x_init is None:
                x_init = np.array([np.random.uniform(*d) for d in self.domain])

        self.domain = domain
        self.x_init = x_init
        self._best_x = x_init
        self.verbose = verbose
        self.spsa_args = spsa_args
        self._x = [x_init] # track x
        self._x_mp = [] # track x -/+ perturbations
        self._x_mp_names = [] 
        
        #Scheduleof the perturbations and step sizes
        a, A, s, b, t = [self.spsa_args[k] for k in ['a', 'A','s','b','t']]
        self._alpha_schedule = lambda k: a / np.power(k+1+A, s)
        self._beta_schedule = lambda k: b/np.power(k+1, t) 
        

    def next_evaluation_circuits(self):
        """ Needs evaluation of 2 points: x_m and x_p"""
        b_k = self._beta_schedule(self._iter) # size of the perturbation
        eps = np.sign(np.random.uniform(0, 1, self.nb_params) - 0.5) # direction of the perturbation
        x_last = self._x[-1]
        x_p = np.clip(x_last + b_k * eps, self._x_min, self._x_max)
        x_m = np.clip(x_last - b_k * eps, self._x_min, self._x_max)
        x_mp = [x_m, x_p]
        self._x_pm.append(x_mp)
        name_params = ['xm', 'xp']
        self._x_mp_names.append(name_params)
        return self.cost_obj.bind_params_to_meas(x_mp, name_params)

    def update(self, X, Y):
        """ Process a new set of data in the form of a results object """
        assert len(X) == 2
        assert len(Y) == 2
        x_m, x_p = X
        y_m, y_p = Y
        g_k = np.squeeze((y_m - y_p))/(x_p - x_m) #finite diff gradient approx 
        a_k = self.alpha_schedule(self._iter) # step size
        self._best_x = np.clip(self.x[-1] + a_k * g_k, self._x_min, self._x_max)
        self._x.append(self._best_x)
        self._iter += 1

            
    def _update_from_obj(self, results_obj, infos=None):
        """ 
        Update the state of the optimizer
        Parameters
        ----------
        nb_iter : int
            Number of steps
        infos: None or dic
            informations needed to process the dictionnary 
            If dic it should be{'x':[x_m, x_p], 'names':[name_x_m, name_x_p])
            If None
        """

        if infos is None:
            names = self._x_mp_names[-1]
            x =  self._x_mp_names[-1]
        else:
            names = infos['names']
            x =  infos['x']

        assert len(names) == 2
        assert len(x) == 2
        y_m, y_p = self.cost.evaluate_cost(results_obj, name = [names])
        x_m, x_p = x
        a_k = self.alpha_schedule(self._iter) # step size
        self._best_x = np.clip(self.x[-1] + a_k * g_k, self._x_min, self._x_max)
        self._x.append(self._best_x)
        self._iter += 1

    def run(self, nb_iter):
        """ 
        Run the full optimization (execution is provided by the cost.instance)
        Parametersgedit
        ----------
        nb_iter : int
            Number of steps
        """
        assert self.iter == 0
        for n in range(nb_iter):
            bound_circuits = self.next_evaluation_circuits()
            res_obj = self.cost.instance.execute(bound_circuits)
            self._update_from_obj(res_obj)

    def verify(self, nb = 8):
        """ evaluate the cost nb times for the best_x
        """
        res = self.cost.shot_noise(self._best_x, nb_experiments=nb)
        return res
        

class ParallelOptimizer():
    """ 
    Class that wraps a set of quantum optimisation tasks. It separates 
    out the cost function evaluation requests from the updating of the 
    internal state of the optimisers to allow aggregation of quantum 
    jobs. It also supports different information sharing approaches 
    between the set of optimisers (see 'method' arg under __init__)

    TODO
    ----
    _gen_optim_list : add check for list of optim args? 1/optim?
    _cross_evaluation : allow vectorized verion for fast evaluation?
    gen_init_circuits : Update init points to take into accout domain 
        (see ut.get_default_args)
    gen_init_circuits : Making something like this automatic for quick 
        compling measurement circuits
    init_optimisers : allow for more than one initial 
    next_evaluation_circuits  : Put interface for _compute_next_ev....
    update & init_optimisers : generalise beyond BO optimisers
    update : implement by-hand updating of dynamic weights?
    """

    def __init__(self, 
                 cost_objs,
                 optimizer, # to replace default BO, extend to list? 
                 optimizer_args, # also allow list of input args
                 method = 'shared',
                 share_init = True,
                 nb_init = 10,
                 ): 
        """ 
        Parameters
        ----------
        cost_objs : list of Cost objects
            Cost functions being max/minimised by the internal optimsers
        optimizer : **class/list of classes under some interface?**
            Class(es) of individual internal optimiser objects
        optimizer_args : { dict, list of dicts }
            The initialisation args to pass to the internal optimisation 
            objects, either a single set to be passed to all or a list to
            be distributed over the optimisers
        method : {'independent','shared','random','left','right'}
            This controls the evaluation sharing of the internal optimiser 
            objects, cases:
                'independent' : The optimiser do not share data, each only 
                    recieves its own evaluations.
                'shared' :  Each optimiser obj gains access to evaluations 
                    of all the others. 
                'random1' : The optimsers do not get the evaluations others 
                    have requested, but in addition to their own they get an 
                    equivalent number of randomly chosen parameter points 
                'random2' : The optimisers do not get the evaluations others 
                    have requested, but in addition to their own they get an 
                    equivalent number of randomly chosen parameter points. 
                    These points are not chosen fully at random, but instead 
                    if x1 and x2 are opt[1] and opt[2]'s chosen evaluations 
                    respectively then opt[1] get an additional point y2 that 
                    is |x2-x1| away from x1 but in a random direction, 
                    similar for opt[2], etc. (Only really relevant to BO.)
                'left', 'right' : Implement information sharing but in a 
                    directional way, so that (using 'left' as an example) 
                    opt[1] gets its evaluation as well as opt[0]; opt[2] gets 
                    its point as well as opt[1] and opt[0], etc. To ensure all 
                    BO's get an equal number of evaluations this is padded 
                    with random points. These points are not chosen fully at 
                    random, they are chosen in the same way as 'random2' 
                    described above. (Only really relevant to BO.)
        share_init : boolean, optional
            Do the optimiser objects share initialisation data, or does each
            generate their own set?
        nb_init : int or keyword 'max', default 'max'
            (BO) Sets the number of initial data points to feed into the BO 
            before starting iteration rounds. If set to 'max' it will 
            generate the maximum number of initial points such that it 
            submits `init_jobs` worth of circuits to a qiskit backend.
        init_jobs : int, default 1
            (BO) The number of qiskit jobs to use to generate initial data. 
            (Most real device backends accept up to 900 circuits in one job.)
        """
        # make (almost certainly) unique id
        self._prefix = ut.gen_random_str(5) 

        # check the method arg is recognised
        if not method in ['independent','shared','left','right']:
            print('method '+f'{method}'+' not recognised, please choose: '
                +'"independent", "shared", "left" or "right".',file=sys.stderr)
            raise ValueError
        elif method in ['random1','random2']:
            raise NotImplementedError

        # store inputs
        self.cost_objs = cost_objs
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.method = method
        self._share_init = share_init
        self.nb_init = nb_init
        
        # make internal assets
        self.optim_list = self._gen_optim_list()
        self._sharing_matrix = self._gen_sharing_matrix()
        self.circs_to_exec = None
        self._parallel_x = {}
        self._parallel_id = {}
        self._last_results_obj = None
        self._last_x_new = None
        # unused currently
        self._initialised = False

    @property    
    def prefix(self):
        return self._prefix

    
    def _gen_optim_list(self):
        """ 
        Generate the list of internal optimser objects

        Comments:
        ---------
        Not really needed as a whole seperate function for now, but might be 
        useful dealing with different types of optmizers
        """
        optim_list =  [self.optimizer(**self.optimizer_args) for ii in range(len(self.cost_objs))]
        return optim_list
    
    
    def _gen_sharing_matrix(self):
        """ 
        Generate the sharing tuples based on sharing mode
        """
        nb_optim = len(self.optim_list)
        if self.method == 'shared':
            return [(ii, jj, jj) for ii in range(nb_optim) for jj in range(nb_optim)]
        elif self.method == 'independent':
            return [(ii, ii, 0) for ii in range(nb_optim)]
        elif self.method == 'left':
            tuples = []
            for consumer_idx in range(nb_optim):
                for generator_idx in range(nb_optim):
                    if consumer_idx >= generator_idx:
                        # higher indexed optims consume the evaluations generated by
                        # lower indexed optims
                        tuples.append((consumer_idx,generator_idx,generator_idx))
                    else:
                        # lower indexed optims generate extra 'padding' evaluations so
                        # that they recieve the same number of new data points
                        tuples.append((consumer_idx,consumer_idx,generator_idx))
            # sanity check
            assert len(tuples)==nb_optim*nb_optim
            return tuples
        elif self.method == 'right':
            tuples = []
            for consumer_idx in range(nb_optim):
                for generator_idx in range(nb_optim):
                    if consumer_idx <= generator_idx:
                        # higher indexed optims consume the evaluations generated by
                        # lower indexed optims
                        tuples.append((consumer_idx,generator_idx,generator_idx))
                    else:
                        # lower indexed optims generate extra 'padding' evaluations so
                        # that they recieve the same number of new data points
                        tuples.append((consumer_idx,consumer_idx,generator_idx))
            # sanity check
            assert len(tuples)==nb_optim*nb_optim
            return tuples


    def _get_padding_circuits(self):
        """
        Different sharing modes e.g. 'left' and 'right' require padding
        of the evaluations requested by the optimisers with other random
        points, generate those circuits here
        """
        def _find_min_dist(a,b):
            """
            distance is euclidean distance, but since the values are angles we want to
            minimize the (element-wise) differences over optionally shifting one of the
            points by ±2\pi
            """
            disp_vector = np.minimum((a-b)**2,((a+2*np.pi)-b)**2)
            disp_vector = np.minimum(disp_vector,((a-2*np.pi)-b)**2)
            return np.sqrt(np.sum(disp_vector))

        circs_to_exec = []
        for consumer_idx,requester_idx,pt_idx in self._sharing_matrix:
            # case where we need to generate a new evaluation
            if (consumer_idx==requester_idx) and not (requester_idx==pt_idx):

                # get the points that the two optimsers indexed by
                # (`consumer_idx`==`requester_idx`) and `pt_idx` chose for their evals
                generator_pt = self._parallel_x[requester_idx,requester_idx]
                pt = self._parallel_x[pt_idx,pt_idx]
                # separation between the points
                dist = _find_min_dist(generator_pt,pt)
                
                # generate random vector in N-d space then scale it to have length we want, 
                # using 'Hypersphere Point Picking' Gaussian approach
                random_displacement = np.random.normal(size=self.cost_objs[requester_idx].ansatz.nb_params)
                random_displacement = random_displacement * dist/np.sqrt(np.sum(random_displacement**2))
                # element-wise modulo 2\pi
                new_pt = np.mod(generator_pt+random_displacement,2*np.pi)

                # make new circuit
                this_id = ut.gen_random_str(8)
                named_circs = ut.prefix_to_names(self.cost_objs[requester_idx].meas_circuits, 
                    this_id)
                circs_to_exec += cost.bind_params(named_circs, new_pt, 
                    self.cost_objs[requester_idx].ansatz.params)
                self._parallel_id[requester_idx,pt_idx] = this_id
                self._parallel_x[requester_idx,pt_idx] = new_pt

        return circs_to_exec


    def _cross_evaluation(self, 
                          cst_eval_idx, 
                          optim_requester_idx, 
                          point_idx=None, 
                          results_obj=None):
        """ 
        Evaluate the results of an experiment allowing sharing of data 
        between the different internal optimisers

        Parameters
        ----------
        cst_eval_idx : int
            Index of the optim/cost function that we will evaluate the 
            point against
        optim_requester_idx : int
            Index of the optim that requested the point being considered
        point_idx : int, optional, defaults to optim_requester_idx
            Subindex of the point inside the set of points that optim 
            optim_requester_idx requested
        results_obj : Qiskit results obj, optional, defaults to last got
            The experiment results to use
        """
        if results_obj is None:
            results_obj = self._last_results_obj
        if point_idx is None:
            point_idx = optim_requester_idx
        circ_name = self._parallel_id[optim_requester_idx,point_idx]
        cost_obj = self.cost_objs[cst_eval_idx]
        x = self._parallel_x[optim_requester_idx,point_idx]
        y = cost_obj.evaluate_cost(results_obj, name = circ_name)
        #print(f'{cst_eval_idx}'+' '+f'{optim_requester_idx}'+' '+f'{point_idx}'+':'+f'{circ_name}')
        return x, y
    

    def _gen_circuits_from_params(self, x_new):
        """
        Creates measurement circuits from supplied parameter points, assumes input 
        is of the form x_new = [[p11,p12...], [p21, p22.,,], ...] where pij is the 
        j'th parameter point requested bit circuit i. Input structure need not be square
                
        Parameters:
        ---------------
        x_new: Nested list of parameter points assumed at least 3d, need not be square
        """
        circs_to_exec = []
        cost_list = self.cost_objs
        self._last_x_new = x_new

        for cst_idx, (cst, points) in enumerate(zip(cost_list, x_new)):
            print(cst.qk_vars)
            idx_points = [ut.safe_string.gen(4) for _ in points]
            circs_to_exec += cst.bind_params_to_meas(points, idx_points)
            self._parallel_x.update({(cst_idx,pt_idx):pt for pt_idx, pt in enumerate(points) })
            self._parallel_id.update({(cst_idx,pt_idx):idx for pt_idx, idx in enumerate(idx_points) })
        self.circs_to_exec = circs_to_exec
        return circs_to_exec        



    def _results_from_last_x(self):
        """
        If specific points were requested, then this returns an array of the same 
        dimentions. WARNING THIS IGNORES ALL CROSS SHARING    
        """
        results = []
        for cst_idx,cst in enumerate(self.cost_objs):
            sub_results = []
            for pt in range(len(self._last_x_new[cst_idx])):
                sub_results.append(self._cross_evaluation(cst_idx,cst_idx,pt)[1])
            results.append(sub_results)
        return results
            


    def gen_init_circuits(self):
        """ 
        Generates circuits to gather initialisation data for the optimizers
        """
        circs_to_exec = []
        if self._share_init:
            cost_list = [self.cost_objs[0]] # maybe run compatability check here? 
        else:
            cost_list = self.cost_objs
        x_new = []
        for cst_idx,cst in enumerate(cost_list):
            # meas_circuits = cst.meas_circuits
            # qk_params = meas_circuits[0].parameters
            points = self._get_random_points_in_domain(size=self.nb_init)
            x_new.append(points)
            # #self._parallel_x.update({ (cst_idx,p_idx):p for p_idx,p in enumerate(points) })
            # for pt_idx,pt in enumerate(points):
            #     this_id = ut.gen_random_str(8)
            #     named_circs = ut.prefix_to_names(meas_circuits, this_id)
            #     circs_to_exec += cost.bind_params(named_circs, pt, qk_params)
            #     self._parallel_x[cst_idx,pt_idx] = pt
            #     self._parallel_id[cst_idx,pt_idx] = this_id
        circs_to_exec = self._gen_circuits_from_params(x_new)
        self.circs_to_exec = circs_to_exec
        return circs_to_exec


    def _get_random_points_in_domain(self,size=1):
        """ 
        Generate a requested number of random points distributed uniformly
        over the domain of the BO parameters.
        TODO: Deal with different optimizers having different domains?
        """
        if type(self.optimizer_args) is list:
            raise NotImplementedError

        for idx,dirn in enumerate(self.optimizer_args['domain']):
            assert int(dirn['name'])==idx, 'BO domain dims not being returned in correct order.'
            assert dirn['type']=='continuous', 'BO domain is not continuous, this is not supported.'

            dirn_min = dirn['domain'][0]
            dirn_diff = dirn['domain'][1]-dirn_min
            if idx==0:
                rand_points = dirn_min + dirn_diff*np.array([np.random.random(size=size)]).T
            else:
                _next = dirn_min + dirn_diff*np.array([np.random.random(size=size)]).T
                rand_points = np.hstack((rand_points,_next))

        return rand_points
    
    
    def init_optimisers(self, results_obj = None): 
        """ 
        Take results object to initalise the internal optimisers 

        Parameters
        ----------
        results_obj : Qiskit results obj
            The experiment results to use
        """
        if results_obj == None:
            results_obj = self._last_results_obj
        self._last_results_obj= results_obj
        nb_optim = len(self.optim_list)
        nb_init = self.nb_init
        if self._share_init:
            sharing_matrix = [(cc,0,run) for cc in range(nb_optim) for run in range(nb_init)]
        else:
            sharing_matrix = [(cc,cc,run) for cc in range(nb_optim) for run in range(nb_init)]
        for evl, req, run in sharing_matrix:
            x, y = self._cross_evaluation(evl, req, run)
            opt = self.optim_list[evl]
            opt.X = np.vstack((opt.X, x))
            opt.Y = np.vstack((opt.Y, y))
            # Replace with self.optim_list[eval].update(x, y)
        [opt.run_optimization(max_iter = 0, eps = 0) for opt in self.optim_list]
            

    def next_evaluation_circuits(self):
        """ 
        Return the set of executable (i.e. transpiled and bound) quantum 
        circuits that will carry out cost function evaluations at the 
        points requested by each of the internal optimisers
        
        Parameters
        ----------
        x_new : list of x vals, optional
            An iterable with exactly 1 param point per cost function, if None
            is passed the function will query the internal optimisers
        """
        self._parallel_id = {}
        self._parallel_x = {}
        x_new = [opt._compute_next_evaluations() for opt in self.optim_list]
        # Kiran Replace with 
        # x_new = [opt.next_params_points() for opt in self.optim_list]
        circs_to_exec = self._gen_circuits_from_params(x_new)
        # circs_to_exec += self._get_padding_circuits()

        # sanity check on number of circuits generated
        if self.method in ['independent','shared']:
            assert len(self._parallel_id.keys())==len(self.cost_objs),('Should have '
                +f'{len(self.cost_objs)}'+' circuits, but instead have '
                +f'{len(self._parallel_id.keys())}')
        elif self.method in ['random1','random2']:
            assert len(self._parallel_id.keys())==len(self.cost_objs)**2,('Should have '
                +f'{len(self.cost_objs)**2}'+' circuits, but instead have '
                +f'{len(self._parallel_id.keys())}')
        elif self.method in ['left','right']:
            assert len(self._parallel_id.keys())==len(self.cost_objs)*(len(self.cost_objs)+1)//2,('Should have '
                +f'{len(self.cost_objs)*(len(self.cost_objs)+1)//2}'
                +' circuits, but instead have '+f'{len(self._parallel_id.keys())}')

        self.circs_to_exec = circs_to_exec
        return circs_to_exec
            
    

    def update(self, results_obj):
        """ 
        Update the internal state of the optimisers, currently specific
        to Bayesian optimisers
            
        Parameters
        ----------
        results_obj : Qiskit results obj
            The experiment results to use
        """
        # Kiran: replace with 
        self._last_results_obj = results_obj
        for evl, req, par in self._sharing_matrix:
            x, y = self._cross_evaluation(evl, req, par)
            opt = self.optim_list[evl]
            opt.X = np.vstack((opt.X, x))
            opt.Y = np.vstack((opt.Y, y))

        for opt in self.optim_list:
            opt._update_model(opt.normalization_type)

def check_cost_objs_consistency(cost_objs):
    """
    Carry out some error checking on the Cost objs passed to the class
    constructor. Fix small fixable errors and crash for bigger errors.
    
    Parameters
    ----------
    cost_objs : list of cost objs 
        The cost objs passed to __init__

    Returns
    -------
    new_cost_objs : list of cost objs
        Possibly slightly altered list of cost objs
    """

    # TODO: Only makes sense to do this if the cost objs are based on 
    # the WeightedPauliOps class. Should check that and skip otherwise

    new_cost_objs = []
    for idx,op in enumerate(cost_objs):

        if idx>0:
            assert op.num_qubits==num_qubits, ("Cost operators passed to"
                +" do not all have the same number of qubits.")

            if not len(op.paulis)==len(test_pauli_set):
                # the new qubit op has a different number of Paulis than the previous
                new_pauli_set = set([ p[1] for p in op.paulis ])
                if len(op.paulis)>len(test_pauli_set):
                    # the new operator set has more paulis the previous
                    missing_paulis = list(new_pauli_set - test_pauli_set)
                    paulis_to_add = [ [op.atol*10,p] for p in missing_paulis ]
                    wpo_to_add = wpo(paulis_to_add)
                    # iterate over previous qubit ops and add new paulis
                    for prev_op in qubit_ops:
                        prev_op.add(wpo_to_add)
                    # save new reference pauli set
                    test_pauli_set = new_pauli_set
                else:
                    # the new operator set has less paulis than the previous
                    missing_paulis = list(test_pauli_set - new_pauli_set)
                    paulis_to_add = [ [op.atol*10,p] for p in missing_paulis ]
                    wpo_to_add = wpo(paulis_to_add)
                    # add new paulis to current qubit op
                    op.add(wpo_to_add)
        else:
            test_pauli_set = set([ p[1] for p in op.paulis ])
            num_qubits = op.num_qubits

        new_cost_objs.append(op)

    return new_cost_objs



class FullOptimiserBO(ParallelOptimizer):
    """ Perhaps a different way of handeling runable optimiser with no overhead"""
    def _subclass_init(self):
        """ Doesn't work yet, but something like this?"""
        self.optimizer = [OptimiserBO(GPyOpt.methods.BayesianOptimization)]
        ...