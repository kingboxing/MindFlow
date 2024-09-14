#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:49:56 2024

@author: bojin

"""

"""
EigenSolver Module

This module provides a class for performing eigenvalue and eigenvector analysis on the Navier-Stokes system.
"""

from src.Deps import *

from src.FreqAnalys.FreqSolverBase import FreqencySolverBase
from src.LinAlg.MatrixOps import AssembleMatrix, AssembleVector, InverseMatrixOperator


class EigenAnalysis(FreqencySolverBase):
    """
    Perform eigenvalue and eigenvector analysis of the linearised Navier-Stokes system.
    """
    def __init__(self, mesh, Re=None, order=(2,1), dim=2, constrained_domain=None):
        """
        Initialize the EigenAnalysis solver.

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
        self.param['solver_type']='eigen_solver'
        # see scipy.sparse.linalg.eigs for details of parameters
        self.param['eigen_solver']={'method': 'lu', 
                                    'lusolver': 'mumps',
                                    'echo':False,
                                    'which': 'LM',
                                    'v0': None,
                                    'ncv': None,
                                    'maxiter': None,
                                    'tol': 0,
                                    'return_eigenvectors': True,
                                    'OPpart': None}

    def _initialize_solver(self, sigma, inverse):
        """
        Initialize the solver for the eigenvalue problem.
        
        Ax = λ * M * x, A=-NS
        Boundary Condition: u = 0

        Parameters
        ----------
        sigma : float
            Shift value for the shift-invert method. 
        inverse : bool
            Whether to compute the inverse of the RHS eigenvectors.
        """
        # Set up the eigenvalue problem: A*x = lambda * M * x, with A = -NS (Navier-Stokes)
        self.A = -self.pencil[0]
        self.M = self.pencil[1] # check if symmtery
        if self.element.dim > self.element.mesh.topology().dim(): #quasi-analysis
            self.A += -self.pencil[2].multiply(1j)
            
        param=self.param[self.param['solver_type']]
        
        # shift-invert operator
        OP = self.A - sigma * self.M if sigma else self.A
        
        if inverse: # inversed eigenvalue?
            OP=OP.T
            
        if sigma is not None:
            if param['method'] == 'lu':
                info(f"LU decomposition using {param['lusolver'].upper()} solver...")
                self.OPinv=InverseMatrixOperator(OP,lusolver=param['lusolver'], echo=param['echo'])
                info('Done.')
            else:
                pass # pending for iterative solvers
                
    def _solve_implicit(self, k, sigma):
        """
        Solve the Shift-Invert Mode eigenvalue problem using scipy.sparse.linalg.eigs

        Parameters
        ----------
        k : int
            Number of eigenvalues to compute.
        sigma : float
            Shift value for the shift-invert method.

        Returns
        -------
        tuple
            Eigenvalues and eigenvectors.
        """
        param = self.param[self.param['solver_type']]
        OPinv = self.OPinv if sigma is not None else None
        
        if sigma is not None:
            info('Internal Shift-Invert Mode Solver is on')
        return spla.eigs(self.A, k=k, M=self.M, Minv=None, OPinv=OPinv, sigma=sigma, which=param['which'],
                                                                v0=param['v0'],ncv=param['ncv'], maxiter=param['maxiter'], tol=param['tol'],
                                                                return_eigenvectors=param['return_eigenvectors'], OPpart=param['OPpart'])
            
    def _solve_explicit(self, k, sigma, inverse):
        """

        Solve the eigenvalue problem using the explicit expression of Shift-Invert Mode

        Parameters
        ----------
        k : int
            Number of eigenvalues to compute. 
        sigma : real or complex
            Shift value for the shift-invert method. 
        inverse : bool
            Whether to compute the inverse of the RHS eigenvectors. 
        Returns
        -------
        tuple
            Eigenvalues and eigenvectors.
        """
        info('Explicit Shift-Invert Mode Solver is on')
        if sigma is None:
            sigma = 0.0
            
        if inverse:
            expr = spla.aslinearoperator(self.M.transpose())*self.OPinv
        else:
            expr = self.OPinv*spla.aslinearoperator(self.M)
            
        if isinstance(sigma, (complex, np.complexfloating)): # if sigma is complex, then convert expr to complex
            expr = expr.astype(complex) # now eigs computes w'[i] = 1/(w[i]-sigma).
        
        vals, vecs = spla.eigs(expr, k=k, M=None, Minv=None, OPinv=None, sigma=None, which=param['which'],
                                               v0=param['v0'],ncv=param['ncv'], maxiter=param['maxiter'], tol=param['tol'],
                                               return_eigenvectors=param['return_eigenvectors'], OPpart=param['OPpart'])
        # Adjust eigenvalues with sigma
        vals=1.0/vals+sigma 
        
        return vals, vecs
        
        
    def solve(self, k=3, sigma=0.0, Re=None, Mat=None, solve_type='implicit', inverse=False, reuse=False, BCpart=None, sz = None):
        """
        Solve for k eigenvalues and eigenvectors of system.

        Parameters
        ----------
        k : int, optional
            Number of eigenvalues to find. Default is 3.
        sigma : real or complex, optional
            Shift value for the shift-invert method. Default is 0.0.
        Mat : scipy.sparse matrix, optional
            Feedback matrix if control is applied. Default is None.
        solve_type : str, optional
            Solver type ('implicit' or 'explicit'). Default is 'implicit'.
        inverse : bool, optional
            Whether to compute the inverse of the RHS eigenvectors. Default is False.
        reuse : bool, optional
            If True, reuse the previous solver and matrices. Default is False.
        BCpart : ‘r’ or ‘i’/None, optional
            The homogenous boundary conditions are applied in the real (matrix NS) or imag (matrix M) part of system. The default is None.
        sz : complex or tuple/list of complex, optional
            Spatial frequency parameters for quasi-analysis of the flow field. Default is None.
            
            if BCpart == 'r':
                A is not full-rank, M is not full rank
            elif BCpart == 'i'/None:
                A is full-rank, M is not full rank, sigma = 0.0, enable shift-invert mode
                vals_orig = 1.0 / vals_si, vals_si computed and vals_orig returned

        """
        
        param=self.param[self.param['solver_type']]
        
        # if sigma is None:
        #     BCpart = 'r'
            
        # if sigma == 0.0:
        #     BCpart = 'i'
            
        if Re is not None:
            self.eqn.Re = Re
            
        if not reuse:
            self._form_LNS_equations(s=1.0j, sz = sz)
            self._assemble_pencil(Mat, symmetry=True, BCpart=BCpart)
            self._initialize_solver(sigma, inverse)
        
        if solve_type.lower() == 'implicit' and not inverse:
            # Solve the eigenvalue problem using an implicit method
            self.vals, self.vecs = self._solve_implicit(k, sigma)
        elif solve_type.lower() == 'explicit':
            # Solve the eigenvalue problem explicitly
            self.vals, self.vecs = self._solve_explicit(k, sigma, inverse)
        else:
            raise ValueError("Invalid solve type. Use 'implicit' or 'explicit'. while using Shift-Invert Mode (Explicit solver for inverse=True)")
            
        gc.collect()





    
    