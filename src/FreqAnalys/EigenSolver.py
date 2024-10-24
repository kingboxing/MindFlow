#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the `EigenAnalysis` class for performing eigenvalue and eigenvector analysis on linearized Navier-Stokes systems.

The class extends `FrequencySolverBase` and is designed to compute the stability characteristics of fluid flow by
finding eigenvalues and eigenvectors of the linearized system. It supports methods such as the shift-invert technique
for efficient computation of eigenvalues near a target value.

Classes
-------
- EigenAnalysis:
    Performs eigenvalue and eigenvector analysis of the linearized Navier-Stokes system.

Dependencies
------------
- FEniCS
- NumPy
- SciPy

Ensure that all dependencies are installed and properly configured.

Examples
--------
Typical usage involves creating an instance of `EigenAnalysis`, setting up the problem domain, boundary conditions, base flow, and solving for the eigenvalues and eigenvectors.

```python
from FERePack.FreqAnalys.EigenSolver import EigenAnalysis
import numpy as np

# Define mesh and parameters
mesh = ...  # Define your mesh
Re = 100.0

# Initialize the eigenvalue analysis solver
solver = EigenAnalysis(mesh, Re=Re, order=(2, 1), dim=2)

# Set boundary conditions
solver.set_boundary(bc_list)
solver.set_boundarycondition(bc_list)

# Set base flow
solver.set_baseflow(ic=base_flow_function)

# Solve for the leading eigenvalues and eigenvectors
solver.solve(k=5, sigma=0.0)

# Access the computed eigenvalues and eigenvectors
eigenvalues = solver.vals
eigenvectors = solver.vecs
```

Notes
--------
- The EigenAnalysis class can utilize LU decomposition or iterative solvers for the eigenvalue problem.
- Supports the shift-invert method for computing eigenvalues near a target (shift) value sigma.
- The solver can handle feedback matrices and spatial frequency parameters for quasi-analysis.
"""

from ..Deps import *

from ..FreqAnalys.FreqSolverBase import FreqencySolverBase
from ..LinAlg.MatrixOps import AssembleMatrix, AssembleVector, InverseMatrixOperator


class EigenAnalysis(FreqencySolverBase):
    """
    Performs eigenvalue and eigenvector analysis of the linearized Navier-Stokes system.

    This class extends `FrequencySolverBase` and is used to compute the stability characteristics of the flow by finding eigenvalues and eigenvectors of the linearized system matrix. It supports methods like the shift-invert technique to efficiently compute eigenvalues near a specified target value.

    Parameters
    ----------
    mesh : dolfin.Mesh
        The computational mesh of the flow domain.
    Re : float, optional
        The Reynolds number of the flow. Default is None.
    order : tuple of int, optional
        The polynomial orders of the finite element spaces for velocity and pressure, respectively. Default is (2, 1).
    dim : int, optional
        Dimension of the flow field. Default is 2.
    constrained_domain : dolfin.SubDomain, optional
        A constrained domain for applying periodic boundary conditions or other constraints. Default is None.

    Attributes
    ----------
    vals : numpy.ndarray
        The computed eigenvalues of the system.
    vecs : numpy.ndarray
        The computed eigenvectors corresponding to the eigenvalues.

    Methods
    -------
    solve(k=3, sigma=0.0, Re=None, Mat=None, sz=None, reuse=False)
        Solve for the eigenvalues and eigenvectors of the system.

    Notes
    -----
    - The class supports both implicit and explicit methods for solving the eigenvalue problem.
    - Shift-invert technique is used to find eigenvalues near a specified shift `sigma`.
    - Input parameters allow for flexibility in specifying the number of eigenvalues, shift, and feedback matrices.
    """

    def __init__(self, mesh, Re=None, order=(2, 1), dim=2, constrained_domain=None):
        """
        Initialize the EigenAnalysis solver.

        Parameters
        ----------
        mesh : dolfin.Mesh
            The computational mesh of the flow domain.
        Re : float, optional
            The Reynolds number of the flow. Default is None.
        order : tuple of int, optional
            The polynomial orders of the finite element spaces for velocity and pressure, respectively. Default is (2, 1).
        dim : int, optional
            Dimension of the flow field. Default is 2.
        constrained_domain : dolfin.SubDomain, optional
            A constrained domain for applying periodic boundary conditions or other constraints. Default is None.
        """
        super().__init__(mesh, Re, order, dim, constrained_domain)
        self.param = self.param['eigen_solver']

    def _initialize_solver(self, sigma, inverse):
        """
        Initialize the solver for the eigenvalue problem.

        This method sets up the eigenvalue problem in the form `A * x = λ * M * x`,
        where `A` is the system matrix (i.e. -NS), `M` is the mass matrix, and `λ` represents the eigenvalues.
        It supports the shift-invert method by modifying the operator `A` with a shift `sigma`.

        Parameters
        ----------
        sigma : float or complex
            Shift value for the shift-invert method. If `None`, no shift is applied.
        inverse : bool
            If True, computes the eigenvalues of the inverse problem (i.e. the inverse of the RHS eigenvectors).

        Notes
        -----
        - If `sigma` is provided, the operator is shifted to target eigenvalues near `sigma`.
        - If `inverse` is True, the transpose of the operator is used.
        """
        # Set up the eigenvalue problem: A*x = lambda * M * x, with A = -NS (Navier-Stokes)
        self.A = -self.pencil[0]
        self.M = self.pencil[1]  # check if symmtery
        if self.element.dim > self.element.mesh.topology().dim():  #quasi-analysis
            self.A += -self.pencil[2].multiply(1j)

        param = self.param[self.param['solver_type']]
        Mat = self.param['feedback_pencil']

        # shift-invert operator
        OP = self.A - sigma * self.M if sigma else self.A

        if inverse:  # inversed eigenvalue?
            OP = OP.T

        if sigma is not None:
            if param['method'] == 'lu':
                info(f"LU decomposition using {param['lusolver'].upper()} solver...")
                self.OPinv = InverseMatrixOperator(OP, Mat=Mat, lusolver=param['lusolver'], echo=param['echo'])
                info('Done.')
            else:
                pass  # pending for iterative solvers

    def _solve_implicit(self, k, sigma):
        """
        Solve the eigenvalue problem using the implicit shift-invert method.

        Parameters
        ----------
        k : int
            Number of eigenvalues to compute.
        sigma : float or complex
            Shift value for the shift-invert method.

        Returns
        -------
        vals : numpy.ndarray
            Computed eigenvalues.
        vecs : numpy.ndarray
            Computed eigenvectors.

        Notes
        -----
        - Utilizes `scipy.sparse.linalg.eigs` to compute eigenvalues and eigenvectors.
        - The operator is shifted to target eigenvalues near `sigma`.
        """
        param = self.param[self.param['solver_type']]
        OPinv = self.OPinv if sigma is not None else None
        Mat = self.param['feedback_pencil']
        if Mat is None:
            A = self.A
        else:
            A = spla.aslinearoperator(self.A) + spla.aslinearoperator(Mat['U']) @ spla.aslinearoperator(Mat['V'].T)

        if sigma is not None:
            info('Internal Shift-Invert Mode Solver is on')
        return spla.eigs(A, k=k, M=self.M, Minv=None, OPinv=OPinv, sigma=sigma, which=param['which'],
                         v0=param['v0'], ncv=param['ncv'], maxiter=param['maxiter'], tol=param['tol'],
                         return_eigenvectors=param['return_eigenvectors'], OPpart=param['OPpart'])

    def _solve_explicit(self, k, sigma, inverse):
        """
        Solve the eigenvalue problem using the explicit shift-invert formulation.

        Parameters
        ----------
        k : int
            Number of eigenvalues to compute.
        sigma : float or complex
            Shift value for the shift-invert method.
        inverse : bool
            If True, computes the eigenvalues of the inverse problem (i.e. the inverse of the RHS eigenvectors).

        Returns
        -------
        vals : numpy.ndarray
            Computed eigenvalues.
        vecs : numpy.ndarray
            Computed eigenvectors.

        Notes
        -----
        - Computes the inverse of the operator explicitly.
        - Adjusts the eigenvalues with the shift `sigma`.
        """
        info('Explicit Shift-Invert Mode Solver is on')
        param = self.param[self.param['solver_type']]

        if sigma is None:
            sigma = 0.0

        if inverse:
            expr = spla.aslinearoperator(self.M.transpose()) * self.OPinv
        else:
            expr = self.OPinv * spla.aslinearoperator(self.M)

        if isinstance(sigma, (complex, np.complexfloating)):  # if sigma is complex, then convert expr to complex
            expr = expr.astype(complex)  # now eigs computes w'[i] = 1/(w[i]-sigma).

        vals, vecs = spla.eigs(expr, k=k, M=None, Minv=None, OPinv=None, sigma=None, which=param['which'],
                               v0=param['v0'], ncv=param['ncv'], maxiter=param['maxiter'], tol=param['tol'],
                               return_eigenvectors=param['return_eigenvectors'], OPpart=param['OPpart'])
        # Adjust eigenvalues with sigma
        vals = 1.0 / vals + sigma

        return vals, vecs

    def solve(self, k=3, sigma=0.0, Re=None, Mat=None, sz=None, reuse=False):
        """
        Solve for the eigenvalues and eigenvectors of the system.

        Parameters
        ----------
        k : int, optional
            Number of eigenvalues to compute. Default is 3.
        sigma : float or complex, optional
            Shift value for the shift-invert method. Default is 0.0.
        Re : float, optional
            Reynolds number. If provided, updates the Reynolds number before solving. Default is None.
        Mat : scipy.sparse matrix or dict with keys 'U' and 'V', optional
            Feedback matrix `Mat = U * V.T`. Can be provided as a sparse matrix or a dictionary containing 'U' and 'V'. Default is None.
        sz : complex or tuple/list of complex, optional
            Spatial frequency parameters for quasi-analysis of the flow field. Default is None.
        reuse : bool, optional
            If True, reuses previous computations (e.g., LU factorization) if available. Default is False.

        Returns
        -------
        None

        Notes
        -----
        - The method assembles the system matrices and initializes the solver if `reuse` is False.
        - After solving, the computed eigenvalues and eigenvectors are stored in `self.vals` and `self.vecs`, respectively.
        - Supports both implicit and explicit methods, as well as handling feedback matrices.
        - if BCpart == 'r', A is not full-rank, M is not full rank;
        - if BCpart == 'i'/None, A is full-rank, M is not full rank. Set sigma = 0.0 to enable shift-invert mode,
          and vals_orig = 1.0 / vals_si where vals_si will be computed and vals_orig returned
        """

        param = self.param[self.param['solver_type']]

        # if sigma is None:
        #     BCpart = 'r'

        # if sigma == 0.0:
        #     BCpart = 'i'

        if Re is not None:
            self.eqn.Re = Re

        if not reuse:
            self._form_LNS_equations(s=1.0j, sz=sz)
            if Mat is None or sp.issparse(Mat):
                self._assemble_pencil(Mat=Mat, symmetry=param['symmetry'], BCpart=param['BCpart'])
            elif isinstance(Mat, dict) and 'U' in Mat and 'V' in Mat:
                self.param['feedback_pencil'] = {'U': -Mat['U'], 'V': Mat['V']}
                self._assemble_pencil(Mat=None, symmetry=param['symmetry'], BCpart=param['BCpart'])
            else:
                raise ValueError(
                    'Invalid Type of feedback matrix Mat (Can be a sparse matrix or a dict containing U and V).')
            self._initialize_solver(sigma, param['inverse'])

        if param['solve_type'].lower() == 'implicit' and not param['inverse']:
            # Solve the eigenvalue problem using an implicit method
            self.vals, self.vecs = self._solve_implicit(k, sigma)
        elif param['solve_type'].lower() == 'explicit':
            # Solve the eigenvalue problem explicitly
            self.vals, self.vecs = self._solve_explicit(k, sigma, param['inverse'])
        else:
            raise ValueError(
                "Invalid solve type. Use 'implicit' or 'explicit'. while using Shift-Invert Mode (Explicit solver for inverse=True)")

        gc.collect()
