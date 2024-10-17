#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 20:38:07 2024

@author: bojin

StateSpaceDAE2 Module

This module provides the StateSpaceDAE2 class to build and assemble the state-space model of
linearized Navier-Stokes equations (DAE2 type).
"""

from ..Deps import *

from ..FreqAnalys.FreqSolverBase import FreqencySolverBase
from ..LinAlg.MatrixAsm import IdentMatBC, MatP, IdentMatProl
from ..LinAlg.Utils import del_zero_cols, eigen_decompose, convert_to_2d, assemble_dae2, assemble_sparse
from ..Params.Params import DefaultParameters


class StateSpaceDAE2(FreqencySolverBase):
    """
    Assemble state-space model of linearized Navier-Stokes equations (DAE2 type)
    in the following block form.
    
    The model structure is:
    | M   0 | d(|vel|     = | A    G  | |vel| + | B | u
    | 0   0 |   |pre|)/dt   | GT  Z=0 | |pre|   | 0 |
    
    y = | C   0 | |vel|
                  |pre|
                  
    """

    def __init__(self, mesh, Re=None, order=(2, 1), dim=2, constrained_domain=None):
        """
        Initialize the StateSpaceDAE2 model.

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
            Constrained domain defined in FEniCS. Default is None.
        """
        super().__init__(mesh, Re, order, dim, constrained_domain)
        self.model = {}
        self.param = DefaultParameters().parameters['state_space_model']

    def _initialize_prolmat(self):
        """
        Initialize prolongation matrices and store them in self.model['Prol'].

        This includes:
        - Prolongation matrix to exclude boundary conditions (P_nbc)
        - Prolongation matrix to exclude velocity subspace and boundary conditions (P_nvel_bc)
        - Prolongation matrix to exclude pressure subspace and boundary conditions (P_npre_bc)
        """

        # prolongation matrix without columns represnet boundary conditions
        P_nbc = del_zero_cols(IdentMatBC(self.element, self.boundary_condition.bc_list))
        # prolongation matrices without columns represent pressure subspace and velocity subspace
        P_npre = MatP(self.element)
        P_nvel = IdentMatProl(self.element, index=list(range(self.element.dim)))
        # prolongation matrix without columns represent pressure subspace and boundary conditions
        P_npre_bc = del_zero_cols(P_nbc.T * P_npre * P_npre.T * P_nbc)
        # prolongation matrix without columns represent velocity subspace and boundary conditions
        P_nvel_bc = del_zero_cols(P_nbc.T * P_nvel * P_nvel.T * P_nbc)

        self.model.update({'Prol': (P_nbc, P_nvel_bc, P_npre_bc)})

    def _initialize_statespace(self):
        """
        Initialize the matrices of linearized Navier-Stokes equations for the state-space model.
        The matrices are constructed by excluding boundary condition rows and columns.
        """
        # RHS and LHS matrices (linearized Navier-Stokes)
        State = -self.pencil[0]
        Mass = self.pencil[1]
        # If this is a quasi-model (e.g., for 3D flow using 2D assumptions)
        if self.element.dim > self.element.mesh.topology().dim():  #quasi-analysis
            State += -self.pencil[2].multiply(1j)  # note that state matrix is complex here

        # Prolongation matrix without boundary conditions
        P_nbc = self.model['Prol'][0]
        State = P_nbc.T * State * P_nbc
        Mass = P_nbc.T * Mass * P_nbc
        # initilise block matrices
        self._initialize_block(Mass, State)

    def _initialize_block(self, Mass, State):
        """
        Extract block matrices for the state-space model.

        This creates the following structure:
        Mass = | M   0 |      State = | A   G  |
               | 0   0 |              | GT  Z=0|

        Parameters
        ----------
        Mass : scipy.sparse matrix
            The mass matrix after excluding boundary conditions.
        State : scipy.sparse matrix
            The state matrix after excluding boundary conditions.
        """
        P_nvel_bc = self.model['Prol'][1]
        P_npre_bc = self.model['Prol'][2]

        M = P_npre_bc.T * Mass * P_npre_bc
        A = P_npre_bc.T * State * P_npre_bc

        G = P_npre_bc.T * State * P_nvel_bc
        GT = P_nvel_bc.T * State * P_npre_bc

        Z = P_nvel_bc.T * State * P_nvel_bc

        # update state-space model
        M.sort_indices()
        A.sort_indices()
        G.sort_indices()
        GT.sort_indices()
        Z.sort_indices()
        self.model.update({
            'M': M,
            'A': A,
            'G': G,
            'GT': GT,
            'Z': Z
        })

    def _assemble_statespace(self):
        """
        Assemble state-space matrices from block matrices.

        Mass = E_full = | M   0 |     State = A_full =  | A   G  |
                        | 0   0 |                       | GT  Z=0|
        """
        # assemble block matrix of mass and state matrices
        A_full, E_full = assemble_dae2(self.model)
        # Update state-space model with assembled matrices
        self.model.update({'E_full': E_full, 'A_full': A_full})

    def _assemble_IO(self, input_vec=None, output_vec=None):
        """
        Assemble input and output vectors for the state-space model in the correct order.

        Parameters
        ----------
        input_vec : numpy array, optional
            Input/actuation vector in the velocity subspace for the state-space model. Default is None.
        output_vec : numpy array, optional
            Output/measurement vector of velocity subspace for the state-space model. Default is None.
        """

        P_nbc = self.model['Prol'][0]
        P_npre_bc = self.model['Prol'][2]

        # assemble input vector in a correct order
        B = P_npre_bc.T @ (P_nbc.T @ input_vec) if input_vec is not None else np.zeros(
            (P_npre_bc.shape[1], 1))
        # assemble output vector in a correct order
        C = (output_vec @ P_nbc) @ P_npre_bc if output_vec is not None else np.zeros((1, P_npre_bc.shape[1]))

        # update state-space model
        self.model.update({'B': convert_to_2d(B, axis=1), 'C': convert_to_2d(C, axis=0)})
        self.model.update({'R': np.identity(self.model['B'].shape[1]), 'Q': np.identity(self.model['C'].shape[0])})

    def _assign_attr(self):
        """
        Assign attributions for Riccato Solver.

        Mass = E_full = | M   0 |     State = A_full =  | A   G  |
                        | 0   0 |                       | GT  Z=0|
        """
        self.A = self.model['A']
        self.M = self.model['M']
        self.G = self.model['G']
        self.Z = self.model['Z']
        self.B = self.model['B']
        self.C = self.model['C']
        if 'A_full' in self.model and 'E_full' in self.model:
            self.A_full = self.model['A_full']
            self.E_full = self.model['E_full']
        if 'U' in self.model and 'V' in self.model:
            self.U = self.model['U']
            self.U = self.model['V']
        if 'Q' in self.model and 'R' in self.model:
            self.Q = self.model['Q']
            self.R = self.model['R']

    def assemble_model(self, input_vec=None, output_vec=None, Re=None, Mat=None, sz=None, reuse=False):
        """
        Assemble the complete state-space model.

        Parameters
        ----------
        input_vec : numpy array, optional
            Input/actuation vector for the state-space model. Default is None.
        output_vec : numpy array, optional
            Output/measurement vector for the state-space model. Default is None.
        Re : float, optional
            Reynolds number.
        Mat : scipy.sparse matrix or dict with Mat = U * V, optional
            Feedback matrix (negative feedback).
        sz : complex or tuple/list of complex, optional
            Spatial frequency parameters for quasi-analysis of the flow field. Default is None.
        reuse : bool, optional
            Whether to reuse the previous state-space model. Default is False.
        """

        if Re is not None:
            self.eqn.Re = Re

        if not reuse:
            self._form_LNS_equations(1.0j, sz)
            if not Mat or sp.issparse(Mat):
                self._assemble_pencil(Mat)
            elif isinstance(Mat, dict):
                self._assemble_pencil()
                self.model.update({'U': -Mat['U'], 'V': Mat['V']})  # update Mat={'U': U, 'V':V}
            else:
                raise TypeError(
                    'Invalid Type of feedback matrix Mat (Can be a sparse matrix or a dict containing U and V).')
            self._initialize_prolmat()
            self._initialize_statespace()
            #self._assemble_statespace()
            self._assemble_IO(input_vec, output_vec)
            self._assign_attr()

    def validate_eigs(self, k=3, sigma=0.0, param={}):
        """
        Perform eigen-decomposition on the state-space model: λ*M*x = A*x.

        Parameters
        ----------
        k : int, optional
            Number of eigenvalues to compute. Default is 3.
        sigma : float, optional
            Shift-invert parameter. Default is 0.0.
        param : dict, optional
            Additional parameters for the eigenvalue solver.
        """

        # Set up matrices
        A_ = assemble_sparse([[self.model['A'], self.model['G']], [self.model['G'].T, None]])
        E_ = assemble_sparse([[self.model['M'], None], [None, self.model['Z']]])
        return eigen_decompose(A_, E_, k=k, sigma=sigma, solver_params=param)
