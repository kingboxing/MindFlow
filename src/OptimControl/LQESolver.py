#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:19:23 2024

@author: bojin

Linear Quadratic Estimation (LQE) Solver

This module provides the LQESolver class for solving the LQE problem for Index-2 systems,
utilizing the M.E.S.S. library for solving generalized Riccati equations.
"""

from ..Deps import *
from ..OptimControl.RiccatiSolver import GRiccatiDAE2Solver
from ..OptimControl.BernoulliSolver import BernoulliFeedback
from ..LinAlg.Utils import distribute_numbers
from ..Params.Params import DefaultParameters
from ..Interface.Py2Mat import start_matlab


class LQESolver(GRiccatiDAE2Solver):

    def __init__(self, model, method='nm', backend='python'):
        """
        Initialize the LQE solver with a state-space model.

        Parameters
        ----------
        model : dict or StateSpaceDAE2
            State-space model or dictionary containing system matrices.
        method : str, optional
            Method used to solve generalized Riccati equations. Default is 'nm'.
        backend : str, optional
            Backend used to solve generalized Riccati equations. Default is 'python'.
        """
        self._k0 = BernoulliFeedback(model)
        super().__init__(model, method, backend)
        self._lqe_default_parameters()
        self.sensor_noise(alpha=1)  # initialise Key 'C_pen_mat' in model

    def _lqe_default_parameters(self):
        """
        Set default parameters based on the method and backend.

        pending for further optimization
        """
        backend = self.param['backend']
        self.param['solver_type'] = 'lqe_solver'
        if backend == 'python':
            self.param['riccati_solver']['type'] = 0
        elif backend == 'matlab':
            self.param['riccati_solver']['eqn']['type'] = 'N'

    def _feedback_pencil(self, U, V, mode):
        """
        Assemble the feedback pencil.
        Matrix A can have the form A = Ã + U*V' if U (eqn.U) and V (eqn.V) are provided U and V are dense (n x m3)
        matrices and should satisfy m3 << n

        Parameters
        ----------
        U : dense (n x m3) matrix
            for feedback pencil.
        V : dense (n x m3) matrix
            for feedback pencil.
        mode : int, optional
            Mode of assembling feedback pencil. Default is 0.

        """
        U = -U # negative feedback
        if mode == 0:
            self.eqn['A'] += sp.csr_matrix(U) @ sp.csr_matrix(V.T)
            self.eqn['A'].eliminate_zeros()
            self.eqn['A'].sort_indices()  # sort data indices, prevent unmfpack error -8
        elif mode == 1 and self.param['backend'] == 'matlab':
            self.param['riccati_solver']['eqn'].update({'haveUV': True, 'sizeUV1': 'default'})
            if 'U' in self.eqn and 'V' in self.eqn:
                # In LQE, V should always be measure matrix C
                if V.shape[1] < self.eqn['V'].shape[1] and np.array_equal(V, self.eqn['V'][:, -V.shape[1]:]):
                    # when measure matrix C has already stacked and keep the same, then U only requires addition
                    zeros = np.zeros((U.shape[0], self.eqn['U'].shape[1] - U.shape[1]))
                    U = np.hstack((zeros, U)) + self.eqn['U']
                    self.eqn.update({'U': U})
                else:  # otherwise U, V stack, and dimensions increase
                    U = np.hstack((self.eqn['U'], U))
                    V = np.hstack((self.eqn['V'], V))
                    self.eqn.update({'U': U, 'V': V})
            else: # 1st initialisation
                self.eqn.update({'U': U, 'V': V})
        else:
            raise ValueError('Invalid mode for feedback accumulation.')

    def init_feedback(self):
        """
        Initialize the feedback using Bernoulli feedback.
        """
        method = self.param['method']
        backend = self.param['backend']
        k0 = self._k0.solve(transpose=True)  # (k * n)
        if backend == 'python':
            self.param['riccati_solver'][method]['k0'] = k0
        elif backend == 'matlab':
            self.param['riccati_solver'][method]['K0'] = k0

    def sensor_noise(self, alpha):
        """
        Set the magnitude of sensor noises that are modeled as uncorrelated zero-mean Gaussian white noise.

        noise weight matrix: Q^{-1/2} in M.E.S.S or }V^{-1/2} in the paper

        Parameters
        ----------
        alpha : int, float, or 1D array
            Magnitude of sensor noise.
        """
        n = self.eqn['C'].shape[0]
        if isinstance(alpha, (int, np.integer, float, np.floating)):
            mat = 1.0 / alpha * np.identity(n)
        elif isinstance(alpha, (list, tuple, np.ndarray)):
            mat = np.diag(np.reciprocal(alpha[:n]))
        else:
            raise TypeError('Invalid type for sensor noise.')

        if self.param['backend'] == 'python' or not self.param['riccati_solver']['LDL_T']:
            self.eqn['C_pen_mat'] = mat
            self.eqn['C'] = mat @ self.eqn['C']
        else:
            self.eqn['Q'] = np.linalg.inv(mat.T @ mat)

    def disturbance(self, B, beta=None):
        """
        Set the square root of the random disturbance covariance matrix.

        Parameters
        ----------
        B : 1D array or ndarray
            Disturbance/Input matrix to set.
        beta : int, float, or 1D array
            Magnitude of disturbances. Default is None.
        """
        if B.shape[0] != self.eqn['B'].shape[0]:
            raise ValueError('Invalid shape of input matrix B')

        self.eqn['B'] = B
        n = self.eqn['B'].shape[1]
        if beta is None:
            mat = np.identity(n)
        elif isinstance(beta, (int, np.integer, float, np.floating)):
            mat = beta * np.identity(n)
        elif isinstance(beta, (list, tuple, np.ndarray)):
            mat = np.diag(beta[0:n])
        else:
            raise TypeError('Invalid type for input scale.')

        self.eqn['R'] = mat @ mat.T  # renew due to iterative solve and for consistency
        if self.param['backend'] == 'python' or not self.param['riccati_solver']['LDL_T']:
            self.eqn['B_pen_mat'] = mat
            self.eqn['B'] = self.eqn['B'] @ mat

    def estimator(self):
        """
        Compute the estimator from the solution of the LQE problem.

        Kf = X * C' * Q^{-1} = Z * Z' * C' * Q^{-1}

        K = (E * Kf)' where K is returned by self.iter_solver and M.E.S.S solver

        Returns
        -------
        Kf : numpy array
            Kalman filter gain matrix (H2 estimator).
        """
        facZ = self._solution_factor()
        if facZ is None:
            print('No solution factor returned since RADI method is used with only initial K0 provided.')
            return np.nan
        # may have issue for Q, R and LDL^T formulation/options
        if self.param['backend'] == 'matlab' and self.param['riccati_solver']['LDL_T']:
            Kf = facZ @ (facZ.T @ self.eqn['C'].T @ np.linalg.inv(self.eqn['Q']))
        else:
            Kf = facZ @ (facZ.T @ self.eqn['C'].T @ self.eqn['C_pen_mat'])
        return Kf

    def solve(self):
        """
        Solve the LQE problem once and return results.
        """

        return self.iter_solve(num_iter=1)

    def iter_solve(self, num_iter=1, MatQ=None, pid=None, Kr=None, mode=0):
        """
        Solve the LQE problem iteratively using accumulation method.

        Parameters
        ----------
        num_iter : int, optional
            Number of iterations to perform. Default is 1.
        MatQ : sparse matrix, optional
            Factor of the weight matrix used to evaluate the H2 norm. Default is None.
        pid : int, optional
            Number of processes to use for H2 norm computation. Default is None.
        Kr : numpy array, optional
            Linear Quadratic Regulator matrix. Default is None.
        mode : int, optional
            Method to perform accumulation method. Default is 0 (i.e. direct assemble).
            if mode = 1 and backend = matlab, utilise M.M.E.S.S. formulation A = Ã + U*V'

        Return a dict with following keys:
        ----------
        K : numpy array
            H2 estimator (Kalman Filter).
        status : list
            Status of each iteration.
        size: tuple or list
            Size of solution factor at each iteration.
        alloc_mode : list
            The number of disturbance modes allocated to each iteration.
        sqnorm_sys : float
            Square of H2 norm.
        sqnorm_lqr : numpy array
            LQR norm, if computed.
        """
        # get original matrices
        B = self.eqn['B'].copy()
        # allocate the number of disturbance modes
        alloc = distribute_numbers(B.shape[1], num_iter)
        if num_iter > 1:
            eng = start_matlab()
        else:
            eng = None
        # initialise results
        status = []
        Zshape = []
        K = np.zeros(self.eqn['C'].T.shape)
        sqnorm_sys = None
        sqnorm_lqr = None

        for i in range(num_iter):
            # allocate disturbance modes
            ind_s = int(np.sum(alloc[:i]))
            ind_e = ind_s + alloc[i]
            self.disturbance(B[:, ind_s:ind_e])
            # now self.eqn['B'] = B[:, ind_s:ind_e]
            self.solve_riccati(engine=eng)

            if num_iter > 1:
                # feedback accumulation
                Kf_sub = self.estimator()
                if Kf_sub is not np.nan:
                    U = self.eqn['M'] @ Kf_sub
                elif self.param['backend'] == 'matlab':
                    U = self.facZ['K'].T
                K += U
                self._feedback_pencil(U, self.eqn['C'].T, mode)
                # get status and size of solution
                status.append(self.status)
                if self.param['backend'] == 'python':
                    Zshape.append(self.facZ.shape)
                elif self.param['backend'] == 'matlab':
                    Zshape.append(np.shape(self.facZ['Z']))
                # accumulate squared norms
                if MatQ is not None:
                    sqnorm_sys = self.sys_energy(MatQ, pid) if i == 0 else sqnorm_sys + self.sys_energy(MatQ, pid)
                if Kr is not None:
                    sqnorm_lqr = self.gain_energy(Kr) if i == 0 else sqnorm_lqr + self.gain_energy(Kr)
                # delete results to save memory
                del self.facZ
                gc.collect()
            else:
                return None

        eng.quit()
        output = {'K': K, 'size': Zshape, 'status': status, 'alloc_mode': alloc, 'sqnorm_sys': sqnorm_sys,
                  'sqnorm_lqr': sqnorm_lqr}
        return output


class LQENMSolver_pymess(GRiccatiDAE2Solver):
    """
    Solver for the Linear Quadratic Estimation (LQE) problem of an Index-2 system.
    """

    def __init__(self, model):
        """
        Initialize the LQE solver with the state-space model.

        Parameters
        ----------
        model : dict or StateSpaceDAE2
            State-space model or dictionary containing the system matrices.
        """
        self._k0 = BernoulliFeedback(model)
        super().__init__(self._initialise_model(model))
        self.sensor_noise(alpha=1)  # initialise Key 'C_pen_mat' in model

    def _apply_default_param(self):
        """
        Initialize default parameters for the LQE solver.
        """
        self.param = DefaultParameters().parameters['lqe_pymess']
        self.param['mess_options'] = mess.Options()
        self.param['initial_feedback'] = self._k0.param
        param_default = self.param['riccati_solver']
        self.update_riccati_params(param_default)

    def _initialise_model(self, model):
        """
        Initialize the model with necessary matrices.

        Parameters
        ----------
        model : dict or StateSpaceDAE2
            State-space model.

        Returns
        -------
        dict
            Initialized model with required matrices.
        """

        model = {'A': model['A'],
                 'M': model['M'],
                 'G': model['G'],
                 'B': model['B'],
                 'C': model['C']}
        return model

    def init_feedback(self):
        """
        Initialize the feedback using Bernoulli feedback.
        """
        param = {'nm': {'k0': self._k0.solve(transpose=True)}}
        self.update_riccati_params(param)

    def sensor_noise(self, alpha):
        """
        Set the magnitude of sensor noise modeled as uncorrelated zero-mean Gaussian white noise.

        noise weight matrix: V^{-1/2} in the formulation
        
        Parameters
        ----------
        alpha : int, float, or 1D array
            Magnitude of sensor noise.
        """
        n = self.Model['C'].shape[0]
        if isinstance(alpha, (int, np.integer, float, np.floating)):
            mat = 1.0 / alpha * np.identity(n)
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

        return self.facZ @ (self.facZ.T @ self.Model['C'].T @ self.Model['C_pen_mat'])

    def solve(self, MatQ=None, delta=-0.02, pid=None, Kr=None):
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
        sqnorm_sys : float
            Square of H2 norm.
        sqnorm_lqr : numpy array
            LQR norm, if computed.
        """

        return self.iter_solve(1, MatQ, delta, pid, Kr)

    def iter_solve(self, num_iter=1, MatQ=None, delta=-0.02, pid=None, Kr=None):
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
        sqnorm_sys : float
            Square of H2 norm.
        sqnorm_lqr : numpy array
            LQR norm, if computed.
        """
        # get original matrices
        B = self.Model['B'].copy()
        # allocate the number of disturbance modes
        n = B.shape[1]
        alloc = distribute_numbers(n, num_iter)
        # initilise results
        Kf = np.zeros(self.Model['C'].T.shape)
        Status = []
        sqnorm_sys = None
        sqnorm_lqr = None

        for i in range(num_iter):
            if i > 0:
                self.Model['A'] += - sp.csr_matrix(self.Model['M'] @ Kf_sub) @ sp.csr_matrix(self.Model['C'])
            self.Model['A'].sort_indices()  # sort data indices, prevent unmfpack error -8

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
                sqnorm_sys = self.squared_h2norm(MatQ, pid) if i == 0 else sqnorm_sys + self.squared_h2norm(MatQ, pid)

            if Kr is not None:
                sqnorm_lqr = self.normvec_T(Kr) if i == 0 else sqnorm_lqr + self.normvec_T(Kr)

            # delete results to save memory
            if num_iter > 1:
                del self.facZ
                gc.collect()

        Status.append({'alloc_mode': alloc})
        return Kf, Status, sqnorm_sys, sqnorm_lqr
