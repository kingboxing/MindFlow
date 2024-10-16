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
from ..Interface.Py2Mat import start_matlab


class LQRSolver(GRiccatiDAE2Solver):
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
        self._lqr_default_parameters()
        self.control_penalty(alpha=1)  # initialise Key 'B_pen_mat' in model

    def _lqr_default_parameters(self):
        """
        Set default parameters based on the method and backend.

        pending for further optimization
        """
        backend = self.param['backend']
        self.param['solver_type'] = 'lqr_solver'
        if backend == 'python':
            self.param['riccati_solver']['type'] = 1
        elif backend == 'matlab':
            self.param['riccati_solver']['eqn']['type'] = 'T'

    def _feedback_pencil_deprecated(self, U, V, mode):
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
        V = -V  # negative feedback
        if mode == 0:
            self.eqn['A'] += sp.csr_matrix(U) @ sp.csr_matrix(V.T)
            self.eqn['A'].eliminate_zeros()
            self.eqn['A'].sort_indices()  # sort data indices, prevent unmfpack error -8
        elif mode == 1 and self.param['backend'] == 'matlab':
            self.param['riccati_solver']['eqn'].update({'haveUV': True, 'sizeUV1': 'default'})
            if 'U' in self.eqn and 'V' in self.eqn:
                # In LQR, U should always be actuation matrix B
                if U.shape[1] < self.eqn['U'].shape[1] and np.array_equal(U, self.eqn['U'][:, -U.shape[1]:]):
                    # when actuation matrix B has already stacked and keep the same, then V only requires addition
                    zeros = np.zeros((V.shape[0], self.eqn['V'].shape[1] - V.shape[1]))
                    V = np.hstack((zeros, V)) + self.eqn['V']
                    self.eqn.update({'V': V})
                else:  # otherwise U, V stack, and dimensions increase
                    U = np.hstack((self.eqn['U'], U))
                    V = np.hstack((self.eqn['V'], V))
                    self.eqn.update({'U': U, 'V': V})
            else:  # 1st initialisation
                self.eqn.update({'U': U, 'V': V})
        else:
            raise ValueError('Invalid mode for feedback accumulation.')

    def init_feedback(self):
        """
        Initialize the feedback using Bernoulli feedback.
        """
        method = self.param['method']
        backend = self.param['backend']
        k0 = self._k0.solve(transpose=False)  # (k * n)
        if backend == 'python':
            self.param['riccati_solver'][method]['k0'] = k0
        elif backend == 'matlab':
            self.param['riccati_solver'][method]['K0'] = k0

    def control_penalty(self, alpha):
        """
        Set the magnitude of control penalty in the LQR formulation.

        penalisation of the control signal: R^{-1/2} in the formulation

        Parameters
        ----------
        alpha : int, float, or 1D array
            Magnitude of control penalty.

        Raises
        ------
        TypeError
            If the provided alpha is not a valid type.
        """

        n = self.eqn['B'].shape[1]
        if isinstance(alpha, (int, np.integer, float, np.floating)):
            mat = 1.0 / alpha * np.identity(n)
        elif isinstance(alpha, (list, tuple, np.ndarray)):
            mat = np.diag(np.reciprocal(alpha[:n]))
        else:
            raise TypeError('Invalid type for control penalty.')

        if self.param['backend'] == 'python' or not self.param['riccati_solver']['LDL_T']:
            self.eqn['B_pen_mat'] = mat
            self.eqn['B'] = self.eqn['B'] @ mat
        else:
            self.eqn['R'] = np.linalg.inv(mat @ mat.T)

    def measurement(self, C, beta=None):
        """
        Set the square root of the response weight matrix for the LQR formulation.

        Parameters
        ----------
        C : 1D array or ndarray
            Measurement/Output matrix that represents the response weight.

        beta : int, float, or 1D array
            Magnitude of response weight. Default is None.

        Raises
        ------
        ValueError
            If the shape of Cz does not match the shape of the existing C matrix.
        """
        if C.shape[1] != self.eqn['C'].shape[1]:
            raise ValueError('Invalid shape of output matrix C.')

        self.eqn['C'] = C
        n = self.eqn['C'].shape[0]
        if beta is None:
            mat = np.identity(n)
        elif isinstance(beta, (int, np.integer, float, np.floating)):
            mat = beta * np.identity(n)
        elif isinstance(beta, (list, tuple, np.ndarray)):
            mat = np.diag(beta[0:n])
        else:
            raise TypeError('Invalid type for output scale.')

        self.eqn['Q'] = mat.T @ mat  # renew due to iterative solve and for consistency
        if self.param['backend'] == 'python' or not self.param['riccati_solver']['LDL_T']:
            self.eqn['C_pen_mat'] = mat
            self.eqn['C'] = mat @ self.eqn['C']

    def regulator(self):
        """
        Compute the linear quadratic regulator (LQR) from the solution of the LQR problem.

        Kr = R^{-1} * B' * X = R^{-1} * B' * Z * Z'

        K = (Kr * E ) where K is returned by self.iter_solver and M.E.S.S solver

        Returns
        -------
        Kr : numpy array
            Linear Quadratic Regulator matrix.
        """
        facZ = self._solution_factor()
        if facZ is None:
            print('No solution factor returned since RADI method is used with only initial K0 provided.')
            return np.nan

        # may have issue for Q, R and LDL^T formulation/options
        if self.param['backend'] == 'matlab' and self.param['riccati_solver']['LDL_T']:
            Kr = np.linalg.inv(self.eqn['R']) @ self.eqn['B'].T @ facZ @ facZ.T
        else:
            Kr = self.eqn['B_pen_mat'] @ self.eqn['B'].T @ facZ @ facZ.T
        return Kr

    def solve(self):
        """
        Solve the LQE problem once and return results.
        """

        return self.iter_solve(num_iter=1)

    def iter_solve(self, num_iter=1, MatQ=None, pid=None, Kf=None, mode=0):
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
        Kf : numpy array, optional
            Linear Kalman Filter matrix. Default is None.
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
        sqnorm_lqe : numpy array
            LQR norm, if computed.
        """

        # get original matrices
        C = self.eqn['C'].copy()
        # allocate the number of disturbance modes
        alloc = distribute_numbers(C.shape[0], num_iter)
        if num_iter > 1:
            eng = start_matlab()
        else:
            eng = None
        # initialise results
        status = []
        Zshape = []
        K = np.zeros(self.eqn['B'].T.shape) # k * n
        sqnorm_sys = None
        sqnorm_lqe = None

        for i in range(num_iter):
            # allocate disturbance modes
            ind_s = int(np.sum(alloc[:i]))
            ind_e = ind_s + alloc[i]
            self.measurement(C[ind_s:ind_e, :])
            # now self.eqn['B'] = B[:, ind_s:ind_e]
            self.solve_riccati(engine=eng)

            if num_iter > 1:
                # feedback accumulation
                Kr_sub = self.regulator()
                if Kr_sub is not np.nan:
                    V = (Kr_sub @ self.eqn['M'])
                elif self.param['backend'] == 'matlab':
                    V = self.facZ['K']
                K += V
                self._feedback_pencil(self.eqn['B'], V.T, mode)
                # get status and size of solution
                status.append(self.status)
                if self.param['backend'] == 'python':
                    Zshape.append(self.facZ.shape)
                elif self.param['backend'] == 'matlab':
                    Zshape.append(np.shape(self.facZ['Z']))
                # accumulate squared norms
                if MatQ is not None:
                    sqnorm_sys = self.sys_energy(MatQ, pid) if i == 0 else sqnorm_sys + self.sys_energy(MatQ, pid)
                if Kf is not None:
                    sqnorm_lqe = self.gain_energy(Kf) if i == 0 else sqnorm_lqe + self.gain_energy(Kf)
                # delete results to save memory
                del self.facZ
                gc.collect()
            else:
                return None

        eng.quit()
        output = {'K': K, 'size': Zshape, 'status': status, 'alloc_mode': alloc, 'sqnorm_sys': sqnorm_sys,
                  'sqnorm_lqr': sqnorm_lqe}
        return output

