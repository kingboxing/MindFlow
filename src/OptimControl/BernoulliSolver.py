#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:11:39 2024

@author: bojin

Bernoulli Feedback Control Solver：
This module provides the BernoulliFeedback class for computing feedback control
using a generalized algebraic Bernoulli equation and eigen-decomposition techniques.

"""
from ..Deps import *
from ..LinAlg.Utils import eigen_decompose, sort_complex, assemble_dae2, assemble_sparse
from ..OptimControl.SystemModel import StateSpaceDAE2
from ..Params.Params import DefaultParameters


class BernoulliFeedback:
    """
    This class computes feedback control to flip unstable eigenvalues to Left-Half Plane in a dynamical system
    by solving a generalized Bernoulli equation.
    """

    def __init__(self, model):
        """
        Initialize the BernoulliFeedback solver with the state-space model.

        Parameters
        ----------
        model : dict
            Dictionary containing the mass matrix 'Mass', the state matrix 'State', and
            optionally other components of the state-space model.
        """
        self._assign_model(model)
        self.param = DefaultParameters().parameters['bernoulli_feedback']

    def _assign_model(self, model):
        """
        Assign the state-space model.

        Mass = E_full = | M   0 |      State = A_full = | A   G  |
                        | 0   0 |                       | GT  Z=0|
        Parameters
        ----------
        model : dict or StateSpaceDAE2
            State-space model.

        Raises
        ------
        TypeError
            If the input is not a valid state-space model.
        """
        if isinstance(model, StateSpaceDAE2):
            model = model.model

        if isinstance(model, dict):
            self.model = model
            if 'E_full' not in model or 'A_full' not in model:
                self.A_full, self.E_full = assemble_dae2(self.model)
            else:
                self.E_full = self.model['E_full']
                self.A_full = self.model['A_full']
        else:
            raise TypeError('Invalid type for state-space model.')

    def _real_projection_basis(self, vals, vecs):
        """
        Form a real-valued projection basis from complex eigenvalues and eigenvectors.

        Parameters
        ----------
        vals : numpy array
            Array of complex eigenvalues.
        vecs : numpy array
            Array of complex eigenvectors.

        Returns
        -------
        Basis : numpy array
            Real-valued projection basis.
        """

        arr = np.concatenate((np.real(vals), np.abs(np.imag(vals))))
        mat = np.concatenate((np.real(vecs), np.imag(vecs)), axis=1)
        _, indices_unique = np.unique(arr, return_index=True)

        Basis = mat[:, indices_unique]
        return Basis

    def _bi_eigen_decompose(self):
        """
        Compute right and left eigenvectors for the unstable eigenvalues and return projection bases.

        Returns
        -------
        HL : numpy array
            Real projection basis for the left eigenvectors.
        HR : numpy array
            Real projection basis for the right eigenvectors.
        """
        param = self.param['bernoulli_feedback']
        vals_r, vecs_r = eigen_decompose(self.A_full, M=self.E_full, k=param['k'], sigma=param['sigma'],
                                         solver_params=param)
        vals_l, vecs_l = eigen_decompose(self.A_full.transpose(), M=self.E_full.transpose(), k=param['k'],
                                         sigma=param['sigma'], solver_params=param)

        # Sort eigenvalues and eigenvectors
        vals_rs, ind_r = sort_complex(vals_r)
        vals_ls, ind_l = sort_complex(vals_l)
        vecs_rs = vecs_r[:, ind_r]
        vecs_ls = vecs_l[:, ind_l]

        if np.sum(np.abs(vals_rs - vals_ls)) > 1e-10:
            raise ValueError('Left and right eigenvalues are not equal.')
        # Count the number of elements where the real part is greater than 0
        count = np.sum(np.real(vals_rs) > 0)
        info(f'Note: {count} unstable eigenvalues and {vals_r.size - count} stable eigenvalues are computed.')

        # Form projection basis for unstable eigenvalues
        ind_us = np.real(vals_rs) > 0  # indices of unstable eigenvalues
        HL = self._real_projection_basis(vals_ls[ind_us], vecs_ls[:, ind_us])  # left
        HR = self._real_projection_basis(vals_rs[ind_us], vecs_rs[:, ind_us])  # right

        return HL, HR

    def _bernoulli_solver(self, transpose):
        """
        Solve the generalized algebraic Bernoulli equation and form feedback vector.

        Parameters
        ----------
        transpose : bool, optional
            If True, solve the transposed system (for LQE). Default is False.

        Returns
        -------
        Feedback : numpy array
            Feedback vector of shape (n, k).
        """
        if not transpose:
            B = self.model['B']
        else:
            B = self.model['C'].T

        HL, HR = self._bi_eigen_decompose()

        # Compute reduced system matrices
        M_tilda = HL.T.conj() @ self.E_full @ HR
        A_tilda = HL.T.conj() @ self.A_full @ HR
        B_tilda = HL.T.conj() @ np.pad(B, ((0, self.E_full.shape[0] - B.shape[0]), (0, 0)), 'constant',
                                       constant_values=0)

        if transpose:  # LQE problem
            M_tilda = M_tilda.T
            A_tilda = A_tilda.T
            B_tilda = HR.T.conj() @ np.pad(B, ((0, self.E_full.shape[0] - B.shape[0]), (0, 0)), 'constant',
                                           constant_values=0)

        Xare = sla.solve_continuous_are(A_tilda, B_tilda, np.zeros_like(A_tilda), np.identity(B_tilda.shape[1]),
                                        e=M_tilda, s=None, balanced=True)

        if transpose:  # LQE problem
            return self.E_full.T @ HR @ Xare @ B_tilda

        return self.E_full @ HL @ Xare @ B_tilda

    def solve(self, transpose=False):
        """
        Solve the Bernoulli equation and compute the feedback vector to stabilize the system.

        Parameters
        ----------
        transpose : bool, optional
            If True, solve the transposed problem. Default is False.

        Returns
        -------
        Feedback : numpy array
            Feedback vector of shape (k, m).
        """
        feedback = self._bernoulli_solver(transpose)
        info('Bernoulli Feedback Solver Finished.')
        if transpose:
            return feedback[:self.model['C'].shape[1], :].T

        return feedback[:self.model['B'].shape[0], :].T

    def validate_eigs(self, k0, k=3, sigma=0.0, param=None, transpose=False):
        """
        Perform eigen-decomposition on the state-space model with feedback applied.

        Parameters
        ----------
        k0 : numpy array
            Feedback vector computed using self.solve().
        k : int, optional
            Number of eigenvalues to compute. Default is 3.
        sigma : float, optional
            Shift-invert parameter. Default is 0.0.
        param : dict, optional
            Additional parameters for the eigenvalue solver. Default is None.
        transpose : bool, optional
            Whether a transposed problem is solved. Default is False.
        """
        if param is None:
            param = {}
        # Set up matrices
        if not transpose:
            M = self.E_full
            A = self.A_full
            BC = self.model['B']
        else:
            M = self.E_full.T
            A = self.A_full.T
            BC = self.model['C'].T

        if k0 is not None:
            B_sp = sp.csr_matrix(np.pad(BC, ((0, M.shape[0] - BC.shape[0]), (0, 0)), 'constant', constant_values=0))
            K_sp = sp.csr_matrix(np.pad(k0, ((0, 0), (0, M.shape[0] - k0.shape[1])), 'constant', constant_values=0))
            B_sp.eliminate_zeros()
            K_sp.eliminate_zeros()
            Mat = B_sp @ K_sp
            return eigen_decompose(A - Mat, M, k=k, sigma=sigma, solver_params=param)
        else:
            return eigen_decompose(A, M, k=k, sigma=sigma, solver_params=param)
