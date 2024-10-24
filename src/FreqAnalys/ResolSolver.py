#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the `ResolventAnalysis` class for performing resolvent analysis on linearized Navier-Stokes systems.

The class extends `FrequencySolverBase` and is designed to compute the most amplified modes (singular modes) of the
system under harmonic forcing at specified frequencies. This analysis is essential for understanding flow
receptivity, transition to turbulence, and designing flow control strategies.

Classes
-------
- ResolventAnalysis:
    Performs resolvent analysis of the linearized Navier-Stokes system.

Dependencies
------------
- FEniCS
- NumPy
- SciPy

Ensure that all dependencies are installed and properly configured.

Examples
--------
Typical usage involves creating an instance of `ResolventAnalysis`, setting up the problem domain, boundary conditions, base flow, and solving for the leading singular modes of the resolvent operator.

```python
from FERePack.FreqAnalys.ResolSolver import ResolventAnalysis
import numpy as np

# Define mesh and parameters
mesh = ...  # Define your mesh
Re = 100.0
frequency = 1.0  # Frequency in Hz
omega = 2 * np.pi * frequency  # Angular frequency

# Initialize the resolvent analysis solver
solver = ResolventAnalysis(mesh, Re=Re, order=(2, 1), dim=2)

# Set boundary conditions
solver.set_boundary(bc_list)
solver.set_boundarycondition(bc_list)

# Set base flow
solver.set_baseflow(ic=base_flow_function)

# Solve for the leading singular modes
s = 1j * omega
solver.solve(k=5, s=s)

# Access the computed singular values and modes
singular_values = solver.energy_amp
force_modes = solver.force_mode
response_modes = solver.response_mode
```

Notes
--------
- The ResolventAnalysis class computes the leading singular values and corresponding force and response modes of the resolvent operator.
- The resolvent operator is defined as the transfer function from forcing to response in the linearized Navier-Stokes system.
- Supports restriction of the analysis to specific spatial regions (subdomains) via the bound parameter.

"""

from ..Deps import *

from ..FreqAnalys.FreqSolverBase import FreqencySolverBase
from ..LinAlg.MatrixOps import AssembleMatrix, AssembleVector, InverseMatrixOperator
from ..LinAlg.MatrixAsm import MatP, MatM, MatQ, MatD
from ..LinAlg.Utils import assign2


class ResolventAnalysis(FreqencySolverBase):
    """
    Performs resolvent analysis of the linearized Navier-Stokes system.

    This class extends `FrequencySolverBase` and is used to compute the most amplified modes of the system under
    harmonic forcing at specified frequencies. It calculates the singular values and corresponding singular modes (
    force and response modes) of the resolvent operator, which characterizes how input disturbances are amplified by
    the flow dynamics.

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
    energy_amp : numpy.ndarray
        The singular values (energy amplification) of the resolvent operator.
    force_mode : numpy.ndarray
        The force modes (left singular vectors) of the resolvent operator.
    response_mode : numpy.ndarray
        The response modes (right singular vectors) of the resolvent operator.

    Methods
    -------
    solve(k, s, Re=None, Mat=None, bound=[None, None], sz=None, reuse=False)
        Solve for the leading singular values and modes of the resolvent operator.
    save(k, s, path)
        Save the k-th singular mode as a time series to a specified path.

    Notes
    -----
    - The resolvent operator maps forcing to response in the frequency domain.
    - Singular value decomposition (SVD) is used to find the most amplified modes.
    - The analysis can be restricted to specific subdomains for both forcing and response.
    """

    def __init__(self, mesh, Re=None, order=(2, 1), dim=2, constrained_domain=None):
        """
        Initialize the ResolventAnalysis solver.

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
        self.param = self.param['resolvent_solver']

    def _initialize_solver(self, bound=None):
        """
        Initialize the inverse matrix operators and LU solvers for the resolvent analysis.

        This method sets up the necessary operators for computing the resolvent operator and its singular value decomposition.

        Parameters
        ----------
        bound : str or None, optional
            Specifies the subdomain restriction for the input (forcing). Default is None.

        Notes
        -----
        - Constructs the inverse of the linearized Navier-Stokes operator.
        - Constructs the input energy matrix `Qf` and its inverse.
        """
        param = self.param[self.param['solver_type']]
        Mat = self.param['feedback_pencil']
        # matrix of the resolvent operator
        self.LHS = self.pencil[0] + self.pencil[1].multiply(1j)
        if self.element.dim > self.element.mesh.topology().dim():  #quasi-analysis
            self.LHS += self.pencil[2].multiply(1j)

        # Prolongation matrix for subdomain restriction (input)
        Df = MatD(self.element, bound)
        Qf = Df.transpose() * MatQ(self.element) * Df  # Input energy matrix of size m x m

        self.mats = {'Df': Df}

        if param['method'] == 'lu':
            info(f"LU decomposition using {param['lusolver'].upper()} solver...")
            self.Linv = InverseMatrixOperator(self.LHS, Mat=Mat, lusolver=param['lusolver'], trans='N',
                                              echo=param['echo'])
            self.LinvH = InverseMatrixOperator(self.LHS, A_lu=self.Linv.A_lu, Mat=Mat, lusolver=param['lusolver'],
                                               trans='H', echo=param['echo'])

            self.Qfinv = InverseMatrixOperator(Qf, lusolver=param['lusolver'], trans='N', echo=param['echo'])
            info('Done.')

        elif param['method'] == 'krylov':
            #self.Minv=precondition_jacobi(L, useUmfpack=useUmfpack)
            pass  # Krylov solver implementation pending

    def _initialize_expr(self, bound=None):
        """
        Initialize the operator expression for the resolvent analysis:
        u = P^T * L^-1 * P * M * f
        D = u^H * Qu * u
          = (M^T * P^T) * L^-H * (P * Qu * P^T) * L^-1 * (P * M)

         if bound on forcing, f = Df * f_s, Qf = Df^T * Qf * Df
         if bound on velocity, u_s = Du^T * u, Qu = Du^T * Qf * Du

        This method constructs the composite operator whose singular values and vectors are to be computed.

        Parameters
        ----------
        bound : str or None, optional
            Specifies the subdomain restriction for the response. Default is None.

        Notes
        -----
        - Constructs operators for both forcing and response restrictions.
        - The operator expression represents the resolvent operator in terms of available matrices.
        """
        P = MatP(self.element)  # Prolongation matrix for the entire space of size nxk
        M = MatM(self.element, bcs=self.boundary_condition.bc_list)  # Weight matrix with bcs for forcing of size k x k
        Qu = MatQ(self.element)  # # Kinetic energy matrix of size k x k
        Du = MatD(self.element, bound)  # prolongation mat for subdomain restriction on velocity subspace

        DQu = Du * Du.transpose()  # square mat for subdomain restriction on velocity subspace

        # Matrix for forcing and response terms
        PM = P * M * self.mats['Df']
        PQPT = P * DQu * Qu * DQu * P.transpose()

        # Resolvent operator expression
        self.expr = spla.aslinearoperator(PM.transpose()) * self.LinvH * spla.aslinearoperator(
            PQPT) * self.Linv * spla.aslinearoperator(PM)

        self.mats.update({'P': P, 'PM': PM})

    def _format_solution(self, s, vals, vecs):
        """
        Process and normalize the solution obtained from the SVD.

        Parameters
        ----------
        s : complex
            The Laplace variable (frequency).
        vals : numpy.ndarray
            Singular values obtained from the SVD.
        vecs : numpy.ndarray
            Singular vectors (force modes) obtained from the SVD.

        Notes
        -----
        - Sorts the singular values and vectors in descending order.
        - Normalizes the response modes based on the energy amplification.
        """
        imag_max = np.max(np.abs(np.imag(vals / np.real(vals))))
        if imag_max > 1e-9:
            info(f'Large imaginary part at s = {s} with max. imag. part (relative) = {imag_max}')

        # Sort eigenvalues and corresponding eigenvectors in descending order
        vals = np.real(vals)
        index = vals.argsort()[::-1]
        self.energy_amp = vals[index]
        vecs = vecs[:, index]

        # Normalize eigenvectors with energy # may not necessary
        Qf = self.Qfinv.A_lu.operator
        self.response_mode = np.zeros((self.mats['P'].shape[0], vecs.shape[1]), dtype=vecs.dtype)

        for ind in range(self.energy_amp.size):
            # # Normalize eigenvector energy
            # vecs_energy = np.dot(vecs[:,ind].T.conj(), Qf.dot(vecs[:,ind]))
            # vecs[:, ind] = vecs[:,ind] / np.sqrt(np.real(vecs_energy))
            # print(vecs_energy)

            # # Compute response mode from normalized prolonged force mode
            self.response_mode[:, ind] = self.Linv.A_lu.solve(self.mats['PM'].dot(vecs[:, ind])) / np.sqrt(
                self.energy_amp[ind])

        self.force_mode = self.mats['P'].dot(self.mats['Df'].dot(vecs))

    def solve(self, k, s, Re=None, Mat=None, bound=[None, None], sz=None, reuse=False):
        """
        Solve for the leading singular values and modes of the resolvent operator.

        Parameters
        ----------
        k : int
            Number of singular values and modes to compute.
        s : complex
            The Laplace variable `s`, typically `s = sigma + i*omega`, where `omega` is the angular frequency.
        Re : float, optional
            Reynolds number. If provided, updates the Reynolds number before solving. Default is None.
        Mat : scipy.sparse matrix or dict with keys 'U' and 'V', optional
            Feedback matrix `Mat = U * V.T`. Can be provided as a sparse matrix or a dictionary containing 'U' and 'V'. Default is None.
        bound : list or tuple, optional
            Subdomain restrictions for the response and input, respectively. Default is [None, None].
            - `bound[0]`: Subdomain for response restriction.
            - `bound[1]`: Subdomain for input (forcing) restriction.
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
        - After solving, the computed singular values and modes are stored in `self.energy_amp`, `self.force_mode`, and `self.response_mode`.
        """

        param = self.param[self.param['solver_type']]

        if Re is not None:
            self.eqn.Re = Re

        if not reuse:
            self._form_LNS_equations(s=s, sz=sz)
            if Mat is None or sp.issparse(Mat):
                self._assemble_pencil(Mat=Mat, symmetry=param['symmetry'], BCpart=param['BCpart'])
            elif isinstance(Mat, dict) and 'U' in Mat and 'V' in Mat:
                self.param['feedback_pencil'] = Mat
                self._assemble_pencil(Mat=None, symmetry=param['symmetry'], BCpart=param['BCpart'])
            else:
                raise ValueError(
                    'Invalid Type of feedback matrix Mat (Can be a sparse matrix or a dict containing U and V).')
            self._initialize_solver(bound=bound[1])  # bound for forcing
            self._initialize_expr(bound=bound[0])  # bound for response

        Qf = self.Qfinv.A_lu.operator

        vals, vecs = spla.eigs(
            self.expr, k=k, M=Qf, Minv=self.Qfinv, OPinv=None, sigma=None, which=param['which'],
            v0=param['v0'], ncv=param['ncv'], maxiter=param['maxiter'], tol=param['tol'],
            return_eigenvectors=param['return_eigenvectors'], OPpart=param['OPpart']
        )

        self._format_solution(s, vals, vecs)

    def save(self, k, s, path):
        """
        Save the k-th singular mode (force and response) as a time series to a specified path.

        Parameters
        ----------
        k : int
            Index of the mode to save (0-based indexing).
        s : complex
            The Laplace variable `s`, typically `s = sigma + i*omega`, where `omega` is the angular frequency.
        path : str
            Directory path where the mode will be saved.

        Notes
        -----
        - The saved time series contains the real and imaginary parts of the force and response modes.
        - The modes are stored at time steps:
            - Time 0.0: Real part of the force mode.
            - Time 1.0: Imaginary part of the force mode.
            - Time 2.0: Real part of the response mode.
            - Time 3.0: Imaginary part of the response mode.
        """

        force = self.force_mode[:, k]
        response = self.response_mode[:, k]
        # savepath = 'path/cylinder_mode(k)_Re(nu)_Omerga(omega)'
        savepath = path + '/resolvent_mode_' + str(k) + 'th_Re' + str(self.eqn.Re).zfill(3) + '_s' + str(s)
        timeseries_r = TimeSeries(savepath)

        # store the mode
        mode = self.element.w
        assign2(mode, np.real(force))
        timeseries_r.store(mode.vector(), 0.0)
        assign2(mode, np.imag(force))
        timeseries_r.store(mode.vector(), 1.0)

        assign2(mode, np.real(response))
        timeseries_r.store(mode.vector(), 2.0)
        assign2(mode, np.imag(response))
        timeseries_r.store(mode.vector(), 3.0)
