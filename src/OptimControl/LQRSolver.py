#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:19:50 2024

@author: bojin

Linear Quadratic Regulator (LQR) Solver

This module provides the LQRSolver class for solving the LQR problem for Index-2 systems,
utilizing the M.E.S.S. library to solve generalized Riccati equations.
"""

from ..Deps import *
from ..OptimControl.RiccatiSolver import GRiccatiDAE2Solver
from ..OptimControl.BernoulliSolver import BernoulliFeedback
from ..LinAlg.Utils import distribute_numbers
from ..Params.Params import DefaultParameters

class LQRSolver(GRiccatiDAE2Solver):
    """
    Solver for the Linear Quadratic Regulator (LQR) problem for Index-2 systems.
    """

    def __init__(self, ssmodel):
        """
        Initialize the LQR solver with the state-space model.

        Parameters
        ----------
        ssmodel : dict or StateSpaceDAE2
            State-space model containing system matrices.
        """
        self._k0 = BernoulliFeedback(ssmodel)
        super().__init__(self._initialise_model(ssmodel))
        self.control_penalty(beta=1)  # initialise Key 'B_pen_mat' in ssmodel

    def _apply_default_param(self):
        """
        Initialize default parameters for the LQR solver.

        Returns
        -------
        None
        """
        self.param = DefaultParameters().parameters['lqr_pymess']
        self.param['mess_options'] = mess.Options()
        self.param['initial_feedback'] = self._k0.param
        param_default = self.param['riccati_solver']
        self.update_riccati_params(param_default)

    def _initialise_model(self, ssmodel):
        """
        Initialize the model with necessary matrices.

        Parameters
        ----------
        ssmodel : dict or StateSpaceDAE2
            State-space model containing system matrices.

        Returns
        -------
        dict
            Initialized model with system matrices.
        """

        model = {'A': ssmodel['A'],
                 'M': ssmodel['M'],
                 'G': ssmodel['G'],
                 'B': ssmodel['B'],
                 'C': ssmodel['C']}
        return model

    def init_feedback(self):
        """
        Initialize feedback using Bernoulli feedback solver.
        """
        param = {'nm': {'k0': self._k0.solve(transpose=False)}}
        self.update_riccati_params(param)

    def control_penalty(self, beta):
        """
        Set the magnitude of control penalty in the LQR formulation.
        
        penalisation of the control signal: R^{-1/2} in the formulation

        Parameters
        ----------
        beta : int, float, or 1D array
            Magnitude of control penalty.

        Raises
        ------
        TypeError
            If the provided beta is not a valid type.
        """

        n = self.Model['B'].shape[1]
        if isinstance(beta, (int, np.integer, float, np.floating)):
            mat = 1.0 / beta * np.identity(n)
        elif isinstance(beta, (list, tuple, np.ndarray)):
            mat = np.diag(np.reciprocal(beta[:n]))
        else:
            raise TypeError('Invalid type for control penalty.')

        self.Model['B_pen_mat'] = mat
        self.Model['B'] = self.Model['B'] @ mat

    def measure(self, Cz=None):
        """
        Set the square root of the response weight matrix for the LQR formulation.

        Parameters
        ----------
        Cz : 1D array or ndarray
            Matrix used to represent the response weight.
        
        Raises
        ------
        ValueError
            If the shape of Cz does not match the shape of the existing C matrix.
        """

        if Cz is not None and Cz.shape[1] == self.Model['C'].shape[1]:
            self.Model['C'] = Cz
        else:
            raise ValueError('Incorrect shape for measurement matrix.')

    def regulator(self):
        """
        Compute the linear quadratic regulator (LQR) from the solution of the LQR problem.

        Returns
        -------
        Kr : numpy array
            Linear Quadratic Regulator matrix.
        """

        return self.Model['B_pen_mat'] @ self.Model['B'].T @ self.facZ @ self.facZ.T

    def solve(self, MatQ=None, delta=-0.02, pid=None, Kf=None):
        """
        Solve the LQR problem and return results.

        Parameters
        ----------
        MatQ : sparse matrix, optional
            Factor of the weight matrix used to evaluate the H2 norm. Default is None.
        delta : float, optional
            Shift-invert parameter for solving Riccati equation. Default is -0.02.
        pid : int, optional
            Number of processes for H2 norm computation. Default is None.
        Kf: numpy array, optional
            Kalman Filter estimator matrix. Default is None.

        Returns
        -------
        Kr : numpy array
            Linear Quadratic Regulator matrix.
        Status : list
            Status of each iteration.
        H2NormS : float
            Square of the H2 norm.
        norm_lqe : numpy array
            LQE norm.
        """
        return self.iter_solve(1, MatQ, delta, pid, Kf)

    def iter_solve(self, num_iter=1, MatQ=None, delta=-0.02, pid=None, Kf=None):
        """
        Solve the LQR problem iteratively using an accumulation method.

        Parameters
        ----------
        num_iter : int, optional
            Number of iterations to perform. Default is 1.
        MatQ : sparse matrix, optional
            Matrix used to compute the H2 norm. Default is None.
        delta : float, optional
            Shift-invert parameter for solving Riccati equation. Default is -0.02.
        pid : int, optional
            Number of processes for H2 norm computation. Default is None.
        Kf : numpy array, optional
            Kalman Filter estimator matrix. Default is None.

        Returns
        -------
        Kr : numpy array
            Linear Quadratic Regulator matrix.
        Status : list
            Status of each iteration.
        H2NormS : float
            Square of the H2 norm.
        norm_lqe : numpy array
            LQE norm.
        """
        # get original matrices
        C = self.Model['C'].copy()
        # allocate the number of measure modes
        n = C.shape[0]
        alloc = distribute_numbers(n, num_iter)
        # initilise results
        Kr = np.zeros(self.Model['B'].T.shape)
        Status = []
        H2NormS = 0
        norm_lqe = None

        for i in range(num_iter):
            if i > 0:
                self.Model['A'] += - sp.csr_matrix(self.Model['B']) @ sp.csr_matrix(Kr_sub @ self.Model['M'])
            self.Model['A'].sort_indices()  # sort data indices, prevent unmfpack error -8
            # allocate measure modes
            ind_s = int(np.sum(alloc[:i]))
            ind_e = ind_s + alloc[i]
            self.measure(C[ind_s:ind_e, :])
            # now self._C = C[:, ind_s:ind_e]
            status_sub = self.solve_riccati(delta)
            Kr_sub = self.regulator()
            # collect results
            Status.append(status_sub)
            Kr += Kr_sub

            if MatQ is not None:
                H2NormS = self.squared_h2norm(MatQ, pid) if i == 0 else H2NormS + self.squared_h2norm(MatQ, pid)

            if Kf is not None:
                norm_lqe = self.normvec_T(Kf) if i == 0 else norm_lqe + self.normvec_T(Kf)

            # delete results to save memory
            if num_iter > 1:
                del self.facZ
                gc.collect()

        Status.append({'alloc_mode': alloc})
        return Kr, Status, H2NormS, norm_lqe
