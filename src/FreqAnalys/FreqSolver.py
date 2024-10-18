#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:32:49 2024

@author: bojin

FreqSolver Module

This module provides classes for solving frequency responses of linearized Navier-Stokes systems.

"""

from ..Deps import *

from ..FreqAnalys.FreqSolverBase import FreqencySolverBase
from ..LinAlg.MatrixOps import AssembleMatrix, AssembleVector, ConvertVector, ConvertMatrix, InverseMatrixOperator
from ..LinAlg.Utils import allclose_spmat, get_subspace_info, find_subspace_index


#%%
class FrequencyResponse(FreqencySolverBase):
    """
    Solve Frequency Response of a linearized Navier-Stokes system with given input and output vectors.
    """

    def __init__(self, mesh, Re=None, order=(2, 1), dim=2, constrained_domain=None):
        """
        Initialize the FrequencyResponse solver.

        Parameters
        ----------
        mesh : Mesh
            The mesh of the flow field.
        Re : float, optional
            Reynolds number. Default is None.
        order : tuple, optional
            Order of finite elements. Default is (2, 1).
        dim : int, optional
            Dimension of the flow field. Default is 2.
        constrained_domain : SubDomain, optional
            Constrained domain defined in FEniCS (e.g., for periodic boundary conditions). Default is None.
        """
        super().__init__(mesh, Re, order, dim, constrained_domain)
        self.param = self.param['frequency_response']

    def _initialize_solver(self):
        """
        Initialize the Inverse Matrix Operator and LU solver.

        """
        param = self.param[self.param['solver_type']]
        self.LHS = self.pencil[0] + self.pencil[1].multiply(1j)
        Mat = self.param['feedback_pencil']
        if self.element.dim > self.element.mesh.topology().dim():  #quasi-analysis
            self.LHS += self.pencil[2].multiply(1j)

        if param['method'] == 'lu':
            info(f"LU decomposition using {param['lusolver'].upper()} solver...")
            self.Linv = InverseMatrixOperator(self.LHS, Mat=Mat, lusolver=param['lusolver'], echo=param['echo'])
            info('Done.')

        elif param['method'] == 'krylov':
            #self.Minv=precondition_jacobi(L, useUmfpack=useUmfpack)
            pass  # Krylov solver implementation pending

    def _check_vec(self, b):
        """
        Validate and flatten the 1-D vector for solving.

        Parameters
        ----------
        b : numpy array
            The vector to be validated. Should be a 1-D numpy array.

        Raises
        ------
        ValueError
            The shape of the array.

        Returns
        -------
        b : numpy array
            Flattened 1D array.
            
        """

        if np.size(b.shape) == 1 or np.min(b.shape) == 1:
            b = np.asarray(b).flatten()
        else:
            raise ValueError('Please give 1D input/output array')

        return b

    def _solve_linear_system(self, input_vec, output_vec=None):
        """
        Solve the linearized system.

        Parameters
        ----------
        input_vec : numpy array
            Input vector for the system.
        output_vec : numpy array, optional
            Output vector for the system. Default is None (return flow state).

        Returns
        -------
        numpy array
            Frequency response vector (actuated by input_vec and measured by output_vec).

        """
        iv = self._check_vec(input_vec)
        self.state = self.Linv.matvec(iv)

        if output_vec is None:
            return self.state.reshape(-1, 1)  # Reshape to column vector
        else:
            ov = self._check_vec(output_vec).reshape(1, -1)  # Reshape to row vector
            return ov @ self.state  # Matrix multiplication (equivalent to * and np.matrix)

    def solve(self, s=None, input_vec=None, output_vec=None, Re=None, Mat=None, sz=None, reuse=False):
        """
        Solve the frequency response of the linearized Navier-Stokes system.

        Parameters
        ----------
        s : int, float, complex, optional
            The Laplace variable s, also known as the operator variable in the Laplace domain.
        input_vec : numpy array, optional
            1-D input vector. Default is None.
        output_vec : numpy array, optional
            1-D output vector. Default is None.
        Re : float, optional
            Reynolds number. Default is None.
        Mat : scipy.sparse matrix or dict with real matrices U, V, optional
            Feedback matrix Mat = U * V'. Default is None.
        sz : complex or tuple/list of complex, optional
           Spatial frequency parameters for quasi-analysis of the flow field. Default is None.
        reuse : bool, optional
            Whether to reuse previous computations. Default is False.

        Returns
        -------
        None.

        """
        param = self.param[self.param['solver_type']]

        if Re is not None:
            self.eqn.Re = Re

        if not reuse:
            self._form_LNS_equations(s, sz)
            if Mat is None or sp.issparse(Mat):
                self._assemble_pencil(Mat=Mat, symmetry=param['symmetry'], BCpart=param['BCpart'])
            elif isinstance(Mat, dict) and 'U' in Mat and 'V' in Mat:
                self.param['feedback_pencil'] = Mat
                self._assemble_pencil(Mat=None, symmetry=param['symmetry'], BCpart=param['BCpart'])
            else:
                raise ValueError(
                    'Invalid Type of feedback matrix Mat (Can be a sparse matrix or a dict containing U and V).')
            self._initialize_solver()

        self.gain = self._solve_linear_system(input_vec, output_vec)
        gc.collect()
