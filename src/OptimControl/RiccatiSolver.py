#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:13:08 2024

@author: bojin
"""

from src.Deps import *
try:
    import pymess as mess
except ImportError:
    MESS = False
from src.OptimControl.SystemModel import StateSpaceDAE2


class GRiccatiDAE2Solver:
    def __init__(self, ssmodel):
        """
        Solver for a generalized Riccati equation of a Index 2 system
        
        MESS_DIRECT_DEFAULT_LU 0, the same as UMFPACK
        
        MESS_DIRECT_SPARSE_LU 1, too long time to check
        
        MESS_DIRECT_LAPACKLU 2, 
        
        MESS_DIRECT_UMFPACK_LU: 3, normal
            
        MESS_DIRECT_SUPERLU_LU : 4, check error occured, kernel died
        
        MESS_DIRECT_CSPARSE_LU 5, too long time to check
        
        MESS_DIRECT_BANDED_LU 6,
        
        MESS_DIRECT_MKLPARDISO_LU 7,

        Parameters
        ----------
        ssmodel : dict or StateSpaceDAE2
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.Model = ssmodel
        self.param['solver_type']='riccati_solver'
        self.param['mess_options'] = mess.Options()
        self.param['riccati_solver']={'method': 'lu',
                                      'lusolver':mess.MESS_DIRECT_UMFPACK_LU}
        self._default_param()
        
    def _assign_model(self, ssmodel):
        """
        

        Parameters
        ----------
        ssmodel : dict or StateSpaceDAE2
            DESCRIPTION.

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        None.

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
            raise TypeError('Worong type of state-space model')
            
    def _default_param(self):
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
                         }
        self.param['riccati_solver'].update(param_default)
        
        for key, value in self.param['riccati_solver'].items():
            eval("self.param['mess_options']."+ key +" = value")
            
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
            eval("self.param['mess_options']."+ key +" = value")
    
    
    def solve(self, delta = -0.02):
        """
        solve Riccati equation
        solution X = Z * Z^T

        Parameters
        ----------
        delta : float, optional
            real and negativ, common choice -0.02. The default is -0.02.

        Returns
        -------
        status : Status Class in M.E.S.S
            the status structure mess_status.

        """
        self._assign_model(self.Model)
        mess.direct_select(self.param['riccati_solver']['lusolver'])
        eqn = mess.EquationGRiccatiDAE2(self.param['mess_options'], self._M, self._A, self._G, self._B, self._C, delta)
        self.facZ, self.status = mess.lrnm(eqn, self.param['mess_options'])
        return status
    
    def h2norm(self, MatQ, pid = None):
        """
        compute the h2 norm of the solution
        
        H2-Norm = Q * Z * Z^T * Q^T
        Q = M^(1/2)

        Parameters
        ----------
        MatQ : scipy.sparse.matrix
            Factor of weight matrix to evaluate H2 norm.
        pid : int, optional
            Number of multiprocessing. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        num_cores = multiprocessing.cpu_count()
        if pid is None:
            pid=num_cores-1
            
        n = np.shape(self.MatZ)[1]
        dn = int(np.ceil(1.0*n/pid))
        h2n_proc = np.zeros(pid)
        ind_all=range(n)
        
        def h2norm_partial(j):
            h2n = np.sum(np.square(MatQ*self.facZ[:,j]))
            return h2n
        
        def h2norm_parallel(i):
            h2n=0
            ind=ind_all[dn*i:dn*i+dn]
            ddn=int(np.ceil(5000.0/pid)) # computes 5000 columns each iteration
            divide=int(np.ceil(1.0*np.size(ind)/ddn)) # second division for mempry reduction
            for j in range(divide):
                h2n += h2norm_partial(ind[ddn*j:ddn*(j+1)])
            h2n_proc[i]=h2n
            
        Parallel(n_jobs=pid,require='sharedmem')(delayed(h2norm_parallel)(i) for i in range(pid))
        
        return np.sum(h2n_proc)
    
    def normvec_T(self, K):
        """
        pending

        Parameters
        ----------
        K : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        norm_T = K * self.facZ
        
        return np.diag(norm_T*norm_T.transpose())
