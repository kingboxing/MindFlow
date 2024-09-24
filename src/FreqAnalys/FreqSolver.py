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
from ..LinAlg.MatrixOps import AssembleMatrix, AssembleVector, ConvertVector,ConvertMatrix, InverseMatrixOperator
from ..LinAlg.Utils import allclose_spmat, get_subspace_info, find_subspace_index

#%%
class FrequencyResponse(FreqencySolverBase):
    """
    Solve Frequency Response of a linearized Navier-Stokes system with given input and output vectors.
    """
    def __init__(self, mesh, Re=None, order=(2,1), dim=2, constrained_domain=None):
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
        self.param['solver_type']='frequency_response'
        self.param['frequency_response']={'method': 'lu', 
                                        'lusolver': 'mumps',
                                        'echo': False}
    
    def _initialize_solver(self):
        """
        Initialize the Inverse Matrix Operator and LU solver.

        """
        param = self.param[self.param['solver_type']]
        self.LHS = self.pencil[0] + self.pencil[1].multiply(1j)
        if self.element.dim > self.element.mesh.topology().dim(): #quasi-analysis
            self.LHS += self.pencil[2].multiply(1j)
        
        if param['method'] == 'lu':
            info(f"LU decomposition using {param['lusolver'].upper()} solver...")
            self.Linv=InverseMatrixOperator(self.LHS,lusolver=param['lusolver'], echo=param['echo'])
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
        
        if np.size(b.shape)==1 or np.min(b.shape)==1:
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
        iv=self._check_vec(input_vec)
        self.state = self.Linv.matvec(iv)
        
        if output_vec is None:
            return self.state.reshape(-1, 1)  # Reshape to column vector
        else:
            ov=self._check_vec(output_vec).reshape(1, -1) # Reshape to row vector
            return ov @ self.state  # Matrix multiplication (equivalent to * and np.matrix)
    
    def solve(self, s=None, input_vec=None, output_vec=None, Re=None, Mat=None, sz=None):
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
        Mat : scipy.sparse matrix, optional
            Feedback matrix (if control is applied). Default is None.
        sz : complex or tuple/list of complex, optional
           Spatial frequency parameters for quasi-analysis of the flow field. Default is None.

        Returns
        -------
        None.

        """
        
        rebuild=False
        
        if Re is not None and (self.eqn.Re is None or not np.allclose(self.eqn.Re, Re, atol=1e-12)):
            # if not yet set Re or the current Re has a value different from the previous on
            self.eqn.Re = Re
            rebuild=True
        else:
            # if Re is None, raise error
            if self.eqn.Re is None:
                raise ValueError('Please indicate a Reynolds number')
            # else use previous Re (Re = None means reuse Reynolds number)
            
        if s is not None and (not hasattr(self.eqn, 's') or not np.allclose(self.eqn.s, s, atol=1e-12)):
            # if not yet set s or the current s has a value different from the previous one
            rebuild=True
        else:
            # if s not yet set and the current one is None, raise error
            if not hasattr(self.eqn, 's'):
                raise ValueError('Please indicate value of the Laplace variable s')
            # else use previous s (s = None means resue the value)
               
            
        if Mat is not None and (not hasattr(self, 'Mat') or self.Mat is None or not allclose_spmat(self.Mat, Mat, atol=1e-12)):
            # if self.Mat doesn't exist (1st solve) or self.Mat (previous Mat) is not equal to current Mat
            self.Mat=Mat
            rebuild=True
        else:
            if not (hasattr(self, 'Mat') and self.Mat is None):
                self.Mat=Mat
                rebuild=True
            #else: current Mat is None and the previous Mat is None (not 1st solve), do nothing; otherwise do above command


        if rebuild:
            self._form_LNS_equations(s, sz)
            self._assemble_pencil(Mat)
            self._initialize_solver()
        
        self.gain=self._solve_linear_system(input_vec, output_vec)
        gc.collect()
        