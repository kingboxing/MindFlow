#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:13:08 2024

@author: bojin

Generalized Riccati Equation Solver

This module provides the GRiccatiDAE2Solver class for solving generalized Riccati equations
for index-2 systems using the M.E.S.S. library.
"""

from ..Deps import *
from joblib import Parallel, delayed

try:
    import pymess as mess
except ImportError:
    MESS = False
    
from ..OptimControl.SystemModel import StateSpaceDAE2
from ..LinAlg.Utils import distribute_numbers


class GRiccatiDAE2Solver:
    """
    This class solves a generalized Riccati equation for an Index-2 system and computes
    various performance metrics such as H2 norm.
    """
    def __init__(self, ssmodel):
        """
        Initialize the Riccati solver with a state-space model.

        Parameters
        ----------
        ssmodel : dict or StateSpaceDAE2
            State-space model or dictionary containing the system matrices.

        Description
        ----------
        MESS_DIRECT_DEFAULT_LU 0, the same as UMFPACK
        
        MESS_DIRECT_SPARSE_LU 1, too long time to check
        
        MESS_DIRECT_LAPACKLU 2, 
        
        MESS_DIRECT_UMFPACK_LU: 3, normal
            
        MESS_DIRECT_SUPERLU_LU : 4, check error occured, kernel died
        
        MESS_DIRECT_CSPARSE_LU 5, too long time to check
        
        MESS_DIRECT_BANDED_LU 6,
        
        MESS_DIRECT_MKLPARDISO_LU 7,


        """
        self.Model = ssmodel
        self.param = {
                    'solver_type': 'riccati_solver',
                    'mess_options': mess.Options(),
                    'riccati_solver': {}
        }
        self._default_param()
        
    def _default_param(self):
        """
        Set default parameters for the Riccati solver.
        """
        param_default = {'type': mess.MESS_OP_NONE,
                         'adi.memory_usage': mess.MESS_MEMORY_HIGH,
                         'adi.paratype': mess.MESS_LRCFADI_PARA_ADAPTIVE_V,
                         'adi.output': 0,
                         'adi.res2_tol': 1e-8,
                         'adi.maxit': 2000,
                         'nm.output': 1,
                         'nm.singleshifts': 0,
                         'nm.linesearch': 1,
                         'nm.res2_tol': 1e-5,
                         'nm.maxit': 30,
                         #'nm.k0': None #Initial Feedback
                         'lusolver':mess.MESS_DIRECT_UMFPACK_LU
                         }
        self.update_parameters(param_default)
        
    def update_parameters(self, param):
        """
        Update the Riccati solver parameters.

        Parameters
        ----------
        param : dict
            A dictionary containing solver parameters to update.
        """
        self.param['riccati_solver'].update(param)
        for key, value in param.items():
            if key != 'lusolver':
                setattr(self.param['mess_options'], key, value)
            else:
                mess.direct_select(self.param['riccati_solver']['lusolver'])
        
    def _assign_model(self, ssmodel):
        """
        Assign the state-space model.

        Parameters
        ----------
        ssmodel : dict or StateSpaceDAE2
            State-space model.

        Raises
        ------
        TypeError
            If the input is not a valid state-space model.
        """
        
        if isinstance(ssmodel, dict):
            self._M = ssmodel['M']
            self._A = ssmodel['A']
            self._G = ssmodel['G']
            self._B = ssmodel['B']
            self._C = ssmodel['C']
        elif isinstance(self.ssmodel, StateSpaceDAE2):
            self._M = ssmodel.M
            self._A = ssmodel.A
            self._G = ssmodel.G
            self._B = ssmodel.B
            self._C = ssmodel.C
        else:
            raise TypeError('Invalid type for state-space model.')
        
            
    def solve_riccati(self, delta = -0.02):
        """
        Solve the generalized Riccati equation.
        Provides the factor Z of the solution X = Z * Z^T.

        Parameters
        ----------
        delta : float, optional
            A real, negative scalar for shift-invert. Default is -0.02.

        Returns
        -------
        status : M.E.S.S. status object
            Status of the Riccati equation solver.
        """

        self._assign_model(self.Model)
        eqn = mess.EquationGRiccatiDAE2(self.param['mess_options'], self._M, self._A, self._G, self._B, self._C, delta)
        self.facZ, self.status = mess.lrnm(eqn, self.param['mess_options'])
        return self.status
    
    def squared_h2norm(self, MatQ, pid = None, chunk_size = 5000):
        """
        Compute the squared H2 norm of the solution.
        H2-Norm = Q * Z * Z^T * Q^T, where Q = M^(1/2).

        Parameters
        ----------
        MatQ : scipy.sparse matrix
            Factor of the weight matrix used to evaluate the H2 norm.
        pid : int, optional
            Number of parallel processes. Default is None (uses available cores).
        chunk_size : int, optional
            Maximum number of columns to process per iteration. Default is 5000.

        Returns
        -------
        float
            Squared H2 norm.
        """
        # the number of cpus
        num_cores = multiprocessing.cpu_count()
        # if pid is None then pid = max_num_cpu - 1
        pid = pid or num_cores - 1
        # the number of columns
        n = self.facZ.shape[1]
        
        def h2norm_partial(start, end):
            return np.sum(np.square(MatQ @ self.facZ[:, start:end]))
        
        if n <= chunk_size or pid == 1:
            return h2norm_partial(0, n)
        else:
            chunk_alloc = distribute_numbers(n, pid)
            h2norms = np.zeros(pid) # square of h2 norm from each processing
            # Parallel computation of H2 norm
            def h2norm_parallel(proc):
                # start index for this process
                size = chunk_alloc[proc]
                ind_s = int(np.sum(chunk_alloc[:proc]))
                # computes 5000 columns at the same time over all processes
                iters=int(np.ceil(1.0 * size * pid/chunk_size)) # second division for mempry saving
                # pass if size is zero
                if iters:
                    chunk_alloc_2nd = distribute_numbers(size, iters)
                    sub_h2norm=0
                    for j in range(iters):
                        # pass if chunk_alloc_2nd[j] is zero
                        if chunk_alloc_2nd[j]:
                            # start and end indices
                            ind_s += int(np.sum(chunk_alloc_2nd[:j]))
                            ind_e = ind_s+chunk_alloc_2nd[j]
                            # h2norm for this part of facZ[:,ind_s:ind_z]
                            sub_h2norm += h2norm_partial(ind_s, ind_e)
                    h2norms[proc]=sub_h2norm
            # Parallel computation of H2 norm
            Parallel(n_jobs=pid,require='sharedmem')(delayed(h2norm_parallel)(proc) for proc in range(pid))
            return np.sum(h2norms)
    
    def normvec_T(self, K):
        """
        Compute the contribution of each estimator/regulator on the H2 Norm?

        Parameters
        ----------
        K : numpy array
            Feedback matrix.

        Returns
        -------
        numpy array
            Diagonal matrix representing the contribution of each estimator.
        """
        
        if K.shape[1] !=  self.facZ.shape[0]:
            K = K.T
        norm_T = K @ self.facZ
        
        return np.diag(norm_T@norm_T.T)
