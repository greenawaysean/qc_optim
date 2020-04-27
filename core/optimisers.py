
# list of * contents
__all__ = [
    'Optimiser',
    'BayesianOptim',
]

from abc import ABC, abstractmethod

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

    @property
    def prefix(self):
        return self._prefix

class BayesianOptim(Optimiser):
    """
    """

    def __init__(
        self,
        cost_obj,
        nb_init='max',
        init_jobs=1,
        verbose=False,
        **bo_args,
        ):
        """ 
        Parameters
        ----------
        cost_obj : Cost object
            These define the parallelised BO tasks being performed.
        nb_init : int or keyword 'max', default 'max'
            (BO) Sets the number of initial data points to feed into the BO 
            before starting iteration rounds. If set to 'max' it will 
            generate the maximum number of initial points such that it 
            submits `init_jobs` worth of circuits to a qiskit backend.
        init_jobs : int, default 1
            (BO) The number of qiskit jobs to use to generate initial data. 
            (Most real device backends accept up to 900 circuits in one job.)
        verbose : bool, optional
            Set level of output of the object
        bo_args : dict
            Additional args to pass to the GPyOpt BayesianOptimization init
        """

        # make (hopefully) unique id
        import time
        self._prefix = str(hash(time.time()))[:16]

        # unpack other args
        self.cost_obj = cost_obj
        self.nb_init = nb_init
        self.init_jobs = init_jobs
        self.verbose = verbose
        self.bo_args = bo_args

        # stop the BO trying to make function calls when created
        self.bo_args['initial_design_numdata'] = 0

        # still needs initial data
        self._initialised = False

    def next_evaluation_circuits(self):
        """ 
        Get the next set of evaluation circuits.

        Returns
        -------
        bound_circuits : list of executable qiskit circuits
            This is the set of circuits that need to be evaluated in order
            to update the internal state of the optimiser
        """
        # save where the evaluations were requested
        self._last_x = self._next_evaluation_points()

        return self._bind_circuits(self._last_x)

    def update(self,results_obj,experiment_name=''):
        """ 
        Update the internal state of the underlying GPyOpt BO object with
        new data. If this object has not previously recieved initialisation
        data this call spawns the GPyOpt BO object.

        Parameters
        ----------
        results_obj : qiskit results object
            This contains the outcomes of the qiskit experiment or 
            simulation and is used to evaluate the cost object at the 
            requested points
        experiment_name : optional string
            Passed to the cost object's evaluation function in order to 
            specify which qiskit experiment contains the relevant data
        """
        # evaluate cost values
        y_new = np.zeros((self._last_x.shape[0],1))
        for idx in range(y_new.size):
            y_new[idx] = self.cost_obj.meas_func(results_obj,experiment_name=experiment_name)
           
        # add data to internal GPyOpt BO object
        if not self._initialised:
            self._BO = GPyOpt.methods.BayesianOptimization(lambda x: None, # blank cost function
                                                           X=self._last_x, 
                                                           Y=y_new, 
                                                           **self.bo_args)
            self._initialised = True
        else:
            self._BO.X = np.vstack((self._BO.X,self._last_x))
            self._BO.Y = np.vstack((self._BO.Y,y_new))

    def _next_evaluation_points(self):
        """ 
        Returns the next set of points to be evaluated for in the form
        specified by the underlying GPyOpt BO class:
            x : 2d array where with one row for each point to be evaluated

        If the BO class has not yet recieved initialisation data this is
        a number of random points (set by __init__) distributed uniformly
        over the domain of the BO, else it is yielded by the GPyOpt BO obj.
        """

        if not self._initialised:
            # if 'max' initial points work out number of evaluations 
            _nb_init = self.nb_init
            if _nb_init=='max':
                _nb_init = self.init_jobs*900//len(self.cost_obj.meas_circuits)
            # get random points over domain
            return self._get_random_points_in_domain(size=_nb_init)
        else:
            # get next evaluation from BO
            return self.BO._compute_next_evaluations()

    def _get_random_points_in_domain(self,size=1):
        """ 
        Generate a requested number of random points distributed uniformly
        over the domain of the BO parameters.
        """
        for idx,dirn in enumerate(self.bo_args['domain']):
            assert int(dirn['name'])==idx, 'BO domain not being returned in correct order.'
            assert dirn['type']=='continuous', 'BO domain is not continuous, this is not supported.'

            dirn_min = dirn['domain'][0]
            dirn_diff = dirn['domain'][1]-dirn_min
            if idx==0:
                rand_points = dirn_min + dirn_diff*np.array([np.random.random(size=size)]).T
            else:
                _next = dirn_min + dirn_diff*np.array([np.random.random(size=size)]).T
                rand_points = np.hstack((rand_points,_next))

        return rand_points

    def _bind_circuits(self,params_values):
        """
        Binds parameter values, getting the transpiled measurment circuits
        and the qiskit parameter objects from `self.cost_obj`
        """
        if np.ndim(params_values)==1:
            params_values = [params_values]

        # package and bind circuits
        bound_circs = []
        for pidx,p in enumerate(params_values):
            for cc in self.cost_obj.meas_circuits:
                tmp = cc.bind_parameters(dict(zip(self.cost_obj.qk_vars, p)))
                tmp.name = str(pidx) + tmp.name
                bound_circs.append(tmp)
            
        return bound_circs

class BayesianOptimParallel(Optimiser):
    """
    """

    def __init__(
        self,
        cost_objs,
        info_sharing_mode='shared',
        nb_init='max',
        init_jobs=1,
        verbose=False,
        **bo_args,
        ):
        """ 
        Parameters
        ----------
        cost_objs : list of Cost objects
            These define the parallelised BO tasks being performed.
        info_sharing_mode : {'independent','shared','random','left','right'}
            (BO) This controls the evaluation sharing of the BO 
            instances, cases:
                'independent' : The BO do not share data, each only recieves 
                    its own evaluations.
                'shared' :  Each BO obj gains access to evaluations of all 
                    of the others. 
                'random1' : The BO do not get the evaluations others have 
                    requested, but in addition to their own they get an 
                    equivalent number of randomly chosen parameter points 
                'random2' : The BO do not get the evaluations others have 
                    requested, but in addition to their own they get an 
                    equivalent number of randomly chosen parameter points. 
                    These points are not chosen fully at random, but instead 
                    if x1 and x2 are BO[1] and BO[2]'s chosen evaluations 
                    respectively then BO[1] get an additional point y2 that 
                    is |x2-x1| away from x1 but in a random direction, 
                    similar for BO[2], etc.
                'left', 'right' : Implement information sharing but in a 
                    directional way, so that (using 'left' as an example) 
                    BO[1] gets its evaluation as well as BO[0]; BO[2] gets 
                    its point as well as BO[1] and BO[0], etc. To ensure all 
                    BO's get an equal number of evaluations this is padded 
                    with random points. These points are not chosen fully at 
                    random, they are chosen in the same way as 'random2' 
                    described above.
        nb_init : int or keyword 'max', default 'max'
            (BO) Sets the number of initial data points to feed into the BO 
            before starting iteration rounds. If set to 'max' it will 
            generate the maximum number of initial points such that it 
            submits `init_jobs` worth of circuits to a qiskit backend.
        init_jobs : int, default 1
            (BO) The number of qiskit jobs to use to generate initial data. 
            (Most real device backends accept up to 900 circuits in one job.)
        verbose : bool, optional
            Set level of output of the object
        bo_args : dict
            Additional args to pass to the GPyOpt BayesianOptimization init
        """
        
        # make unique id
        import time
        self._prefix = str(hash(time.time()))[:16]

        # run checks and cleanup on cost objs
        _cost_objs = self._enforce_cost_objs_consistency(cost_objs)

        # unpack other args
        self.cost_obj = cost_obj
        self.nb_init = nb_init
        self.init_jobs = init_jobs
        self.verbose = verbose
        self.bo_args = bo_args

        # spawn individual BOpt objs
        self.BOpts = [ BayesianOptim(c,**bo_args) for c in _cost_objs ]

        # still needs initial data
        self._initialised = False

    def _enforce_cost_objs_consistency(self,cost_objs):
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
                    +" BOptParallel do not all have the same number of qubits.")

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
