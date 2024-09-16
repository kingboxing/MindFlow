#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:19:23 2024

@author: bojin

Linear Quadratic Estimation (LQE) Solver

This module provides the LQESolver class for solving the LQE problem for Index-2 systems,
utilizing the M.E.S.S. library for solving generalized Riccati equations.
"""

from src.Deps import *
from src.OptimControl.RiccatiSolver import GRiccatiDAE2Solver
from src.OptimControl.BernoulliSolver import BernoulliFeedback
from src.LinAlg.Utils import distribute_numbers
try:
    import pymess as mess
except ImportError:
    MESS = False
 
class LQESolver(GRiccatiDAE2Solver):
    """
    Solver for the Linear Quadratic Estimation (LQE) problem of an Index-2 system.
    """
    def __init__(self, ssmodel):
        """
        Initialize the LQE solver with the state-space model.

        Parameters
        ----------
        ssmodel : dict or StateSpaceDAE2
            State-space model or dictionary containing the system matrices.
        """
        
        super().__init__(self._initialise_model(ssmodel))
        self._k0=BernoulliFeedback(ssmodel)
        self.param['solver_type']='lqe_solver'
        self._default_param()
        
    def _initialise_model(self, ssmodel):
        """
        Initialize the model with necessary matrices.

        Parameters
        ----------
        ssmodel : dict or StateSpaceDAE2
            State-space model.

        Returns
        -------
        dict
            Initialized model with required matrices.
        """
        
        model = {'A': ssmodel['A'],
                 'M': ssmodel['M'],
                 'G': ssmodel['G'],
                 'B': ssmodel['B'],
                 'C': ssmodel['C']}
        return model
        
        
    def _default_param(self):
        """
        Initialize default parameters for the LQE solver.
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
            
    def init_feedback(self):
        """
        Initialize the feedback using Bernoulli feedback.
        """
        param = {'nm.k0': self._k0.solve(transpose=True)}
        self.update_parameters(param)
        
    def sensor_noise(self, alpha):
        """
        Set the magnitude of sensor noise modeled as uncorrelated zero-mean Gaussian white noise.

        noise weight matrix: V^{-1/2} in the formulation
        
        Parameters
        ----------
        alpha : int, float, or 1D array
            Magnitude of sensor noise.
        """
        n=self.Model['C'].shape[0]
        if isinstance(alpha, (int, np.integer, float, np.floating)):
            mat = 1.0/alpha * np.identity(n)
        elif isinstance(alpha, (list, tuple, np.ndarray)):
            mat = np.diag(np.reciprocal(alpha[:n]))
        else:
            raise TypeError('Invalid type for sensor noise.')
        
        self.Model['C_pen_mat'] = mat
        self.Model['C'] = mat @ self.Model['C'] 
        
    def disturbance(self, Bd=None):
        """
        Set the square root of the random disturbance covariance matrix.

        Parameters
        ----------
        Bd : 1D array or ndarray
            Disturbance matrix to set.
        """
        
        if Bd is not None and Bd.shape[0] == self.Model['B'].shape[0]:
            self.Model['B'] = Bd
        else:
            raise ValueError('Invalid shape for disturbances')
                
    def estimator(self):
        """
        Compute the estimator from the solution of the LQE problem.

        Returns
        -------
        Kf : numpy array
            Kalman filter gain matrix (H2 estimator).
        """
        
        return self.facZ @ (self.facZ.T @  self.Model['C'].T @ self.Model['C_pen_mat'])
    
    def solve(self, MatQ=None, delta = -0.02, pid = None, Kr = None):
        """
        Solve the LQE problem once and return results.

        Parameters
        ----------
        MatQ : sparse matrix, optional
            Factor of the weight matrix used to evaluate the H2 norm. Default is None.
        delta : float, optional
            Shift-invert parameter. Default is -0.02.
        pid : int, optional
            Number of processes to use for H2 norm computation. Default is None.
        Kr : numpy array, optional
            Linear Quadratic Regulator matrix. Default is None.

        Returns
        -------
        Kf : numpy array
            H2 estimator (Kalman Filter).
        Status : list
            Status of each iteration.
        H2NormS : float
            Square of H2 norm.
        norm_lqr : numpy array
            LQR norm, if computed.
        """
        
        return self.iter_solve(1, MatQ, delta, pid, Kr)
    
    def iter_solve(self, num_iter = 1, MatQ=None, delta = -0.02, pid = None, Kr = None):
        """
        Solve the LQE problem iteratively using accumulation method.

        Parameters
        ----------
        num_iter : int, optional
            Number of iterations to perform. Default is 1.
        MatQ : sparse matrix, optional
            Factor of the weight matrix used to evaluate the H2 norm. Default is None.
        delta : float, optional
            Shift-invert parameter. Default is -0.02.
        pid : int, optional
            Number of processes to use for H2 norm computation. Default is None.
        Kr : numpy array, optional
            Linear Quadratic Regulator matrix. Default is None.

        Returns
        -------
        Kf : numpy array
            H2 estimator (Kalman Filter).
        Status : list
            Status of each iteration.
        H2NormS : float
            Square of H2 norm.
        norm_lqr : numpy array
            LQR norm, if computed.
        """
        # get original matrices
        B = self.Model['B'].copy()
        # allocate the number of disturbance modes
        n = B.shape[1]
        alloc = distribute_numbers(n, num_iter)
        # initilise results
        Kf=np.zeros(self.Model['C'].T.shape)
        Status=[]
        H2NormS=0
        norm_lqr=None
        
        for i in range(num_iter):
            if i > 0:
                self.Model['A'] += - sp.csr_matrix(self.Model['M'] @ Kf_sub) @ sp.csr_matrix(self.Model['C'])
            self.Model['A'].sort_indices() # sort data indices, prevent unmfpack error -8
            
            # allocate disturbance modes
            ind_s = int(np.sum(alloc[:i]))
            ind_e = ind_s + alloc[i]
            self.disturbance(B[:, ind_s:ind_e])
            
            # now self._B = B[:, ind_s:ind_e]
            status_sub = self.solve_riccati(delta)
            Kf_sub = self.estimator()
            
            # collect results
            Status.append(status_sub)
            Kf += Kf_sub
            
            if MatQ is not None:
                H2NormS += self.squared_h2norm(MatQ, pid)
            
            if Kr is not None:
                norm_lqr = self.normvec_T(Kr) if i == 0 else norm_lqr + self.normvec_T(Kr)

            # delete results to save memory
            del self.facZ
            gc.collect()
        
        Status.append({'alloc_mode': alloc})
        return Kf, Status, H2NormS, norm_lqr
