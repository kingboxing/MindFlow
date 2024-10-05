#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 20:18:46 2024

@author: bojin

Resolvent Analysis Module

This module provides a class for performing resolvent analysis on the linearized Navier-Stokes system.
"""

from ..Deps import *

from ..FreqAnalys.FreqSolverBase import FreqencySolverBase
from ..LinAlg.MatrixOps import AssembleMatrix, AssembleVector, InverseMatrixOperator
from ..LinAlg.MatrixAsm import MatP, MatM, MatQ, MatD
from ..LinAlg.Utils import assign2


class ResolventAnalysis(FreqencySolverBase):
    """
    Perform resolvent analysis of the linearized Navier-Stokes system.
    """

    def __init__(self, mesh, Re=None, order=(2, 1), dim=2, constrained_domain=None):
        """
        Initialize the ResolventAnalysis solver.

        Parameters
        ----------
        mesh : Mesh
            Mesh of the flow field.
        Re : float, optional
            Reynolds number. Default is None.
        order : tuple, optional
            Order of finite elements. Default is (2, 1).
        dim : int, optional
            Dimension of the flow field. Default is 2.
        constrained_domain : SubDomain, optional
            Constrained domain defined in FEniCS (e.g., periodic boundary conditions). Default is None.
        """

        super().__init__(mesh, Re, order, dim, constrained_domain)
        self.param = self.param['resolvent_solver']

    def _initialize_solver(self, bound=None):
        """
        Initialize the Inverse Matrix Operator and LU solver.

        Parameters
        ----------
        bound : str or None, optional
            Restricted subdomain definition for input/force. Default is None.
        """
        param = self.param[self.param['solver_type']]
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
            self.Linv = InverseMatrixOperator(self.LHS, lusolver=param['lusolver'], trans='N', echo=param['echo'])
            self.LinvH = InverseMatrixOperator(self.LHS, A_lu=self.Linv.A_lu, lusolver=param['lusolver'], trans='H',
                                               echo=param['echo'])
            self.Qfinv = InverseMatrixOperator(Qf, lusolver=param['lusolver'], trans='N', echo=param['echo'])
            info('Done.')

        elif param['method'] == 'krylov':
            #self.Minv=precondition_jacobi(L, useUmfpack=useUmfpack)
            pass  # Krylov solver implementation pending

    def _initialize_expr(self, bound=None):
        """
        Initialize the expression for the resolvent analysis.
        
        u = P^T * L^-1 * P * M * f
        D = u^H * Qu * u
          = (M^T * P^T) * L^-H * (P * Qu * P^T) * L^-1 * (P * M)
          
         if bound on forcing, f = Df * f_s, Qf = Df^T * Qf * Df
         if bound on velocity, u_s = Du^T * u, Qu = Du^T * Qf * Du

         Parameters
        ----------
        bound : str
            Restrict subdomain definition for the response.

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
        Process the solution and normalize the eigenvectors.

        Parameters
        ----------
        s : complex
            The Laplace variable.
        vals : numpy array
            Eigenvalues of the resolvent operator.
        vecs : numpy array
            Eigenvectors of the resolvent operator.
        """
        imag_max = np.max(np.abs(np.imag(vals / np.real(vals))))
        if imag_max > 1e-9:
            info('Large imaginary part at s = {s} with max. imag. part (relative) = {imag_max}')

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

    def solve(self, k, s, Re=None, Mat=None, bound=[None, None], reuse=False, sz=None):
        """
        Solve the resolvent problem.

        Parameters
        ----------
        k : int
            Number of eigenvalues to compute.
        s : complex
            The Laplace variable.
        Re : float, optional
            Reynolds number. Default is None.
        Mat : scipy.sparse matrix, optional
            Feedback matrix. Default is None.
        bound : list or tuple, optional
            Subdomain restrictions for the response and input, respectively. Default is [None, None].
        reuse : bool, optional
            Whether to reuse previous computations. Default is False.
        sz : complex or tuple/list of complex, optional
            Spatial frequency parameters for quasi-analysis of the flow field. Default is None.
        """

        self.s = s
        param = self.param[self.param['solver_type']]

        if Re is not None:
            self.eqn.Re = Re

        if not reuse:
            self._form_LNS_equations(s=s, sz=sz)
            self._assemble_pencil(Mat=Mat, symmetry=param['symmetry'], BCpart=param['BCpart'])
            self._initialize_solver(bound=bound[1])  # bound for forcing
            self._initialize_expr(bound=bound[0])  # bound for response

        Qf = self.Qfinv.A_lu.operator

        vals, vecs = spla.eigs(
            self.expr, k=k, M=Qf, Minv=self.Qfinv, OPinv=None, sigma=None, which=param['which'],
            v0=param['v0'], ncv=param['ncv'], maxiter=param['maxiter'], tol=param['tol'],
            return_eigenvectors=param['return_eigenvectors'], OPpart=param['OPpart']
        )

        self._format_solution(s, vals, vecs)

    def save(self, k, path):
        """
        Save the k-th singular mode as a time series.
        
        time 0 is the real part of the force, time 1 is the imag part of the force,
        time 2 is the real part of the response, time 3 is the imag part of the response

        Parameters
        ----------
        k : int
            Index of the mode to save.
        path : str
            Directory path of the folder to save the mode.
        """

        force = self.force_mode[:, k]
        response = self.response_mode[:, k]
        # savepath = 'path/cylinder_mode(k)_Re(nu)_Omerga(omega)'
        savepath = path + '/resolvent_mode_' + str(k) + 'th_Re' + str(self.eqn.Re).zfill(3) + '_s' + str(self.s)
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
