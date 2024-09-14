#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 20:38:07 2024

@author: bojin

StateSpaceDAE2 Module

This module provides the StateSpaceDAE2 class to build and assemble the state-space model of
linearized Navier-Stokes equations (DAE2 type).
"""

from src.Deps import *

from src.FreqAnalys.FreqSolverBase import FreqencySolverBase
from src.LinAlg.MatrixAsm import IdentMatBC, MatP, IdentMatProl
from src.LinAlg.MatrixOps import InverseMatrixOperator
from src.LinAlg.Utils import del_zero_cols, eigen_decompose


class StateSpaceDAE2(FreqencySolverBase):
    """
    Assemble state-space model of linearized Navier-Stokes equations (DAE2 type).
    
    The model structure is:
    | M   0 | d(|vel|     = | A    G  | |vel| + | B | u
    | 0   0 |   |pre|)/dt   | GT  Z=0 | |pre|   | 0 |
    
    y = | C   0 | |vel|
                  |pre|
                  
    """
    def __init__(self, mesh, Re=None, order=(2,1), dim=2, constrained_domain=None):
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
        self.SSModel = {}
        self.param['solver_type']='state_space_model'
        self.param['state_space_model']={}
        
    def _initialize_prolmat(self):
        """
        Initialize prolongation matrices and store them in self.SSModel['Prol'].

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
        P_npre_bc = del_zero_cols(P_nbc.transpose() * P_npre * P_npre.transpose() * P_nbc)
        # prolongation matrix without columns represent velocity subspace and boundary conditions
        P_nvel_bc = del_zero_cols(P_nbc.transpose() * P_nvel * P_nvel.transpose() * P_nbc)
        
        self.SSModel.update({'Prol': (P_nbc, P_nvel_bc, P_npre_bc)})
        
    def _initialize_statespace(self):
        """
        Initialize the matrices of linearized Navier-Stokes equations for the state-space model.
        The matrices are constructed by excluding boundary condition rows and columns.
        """
        # RHS and LHS matrices (linearized Navier-Stokes)
        State = -self.pencil[0]
        Mass = self.pencil[1]
        # If this is a quasi-model (e.g., for 3D flow using 2D assumptions)
        if self.element.dim > self.element.mesh.topology().dim(): #quasi-analysis
            State += -self.pencil[2].multiply(1j) # note that state matrix is complex here

        # Prolongation matrix without boundary conditions
        P_nbc = self.SSModel['Prol'][0]
        State=P_nbc.transpose() * State * P_nbc
        Mass=P_nbc.transpose() * Mass * P_nbc
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
        P_nvel_bc = self.SSModel['Prol'][1]
        P_npre_bc = self.SSModel['Prol'][2]
        

        M=P_npre_bc.transpose()*Mass*P_npre_bc
        A=P_npre_bc.transpose()*State*P_npre_bc
        
        G =P_npre_bc.transpose()*State*P_nvel_bc
        GT=P_nvel_bc.transpose()*State*P_npre_bc
        
        Z=P_nvel_bc.transpose()*State*P_nvel_bc
        
        # update state-space model
        self.SSModel.update({
                            'M': M,
                            'A': A,
                            'G': G,
                            'GT': GT,
                            'Z': Z
                            })
    
    def _assemble_statespace(self):
        """
        Assemble state-space matrices from block matrices.

        Mass = | M   0 |      State = | A   G  |
               | 0   0 |              | GT  Z=0|
        """
        # assemble block matrix of mass and state matrices
        Mass_b = sp.bmat([[self.SSModel['M'], None], [None, self.SSModel['Z']]],format='csr')
        State_b = sp.bmat([[self.SSModel['A'], self.SSModel['G']], [self.SSModel['GT'] , self.SSModel['Z']]],format='csr')
        # Eliminate zeros for efficiency
        Mass_b.eliminate_zeros()
        State_b.eliminate_zeros()
        
        # Update state-space model with assembled matrices
        self.SSModel.update({'Mass': Mass_b, 'State': State_b})
        
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

        P_nbc = self.SSModel['Prol'][0]
        P_npre_bc = self.SSModel['Prol'][2]

        # assemble input vector in a correct order
        B=P_npre_bc.transpose() @ (P_nbc.transpose() @ input_vec) if input_vec is not None else np.zeros((P_npre_bc.shape[1],1))
         # assemble output vector in a correct order
        C=(output_vec @ P_nbc) @ P_npre_bc if output_vec is not None else np.zeros((1, P_npre_bc.shape[1]))

        # update state-space model
        self.SSModel.update({'B': B, 'C': C})
        
    def _assign_attr(self):
        """
        Assign attributions for Riccato Solver.

        Mass = | M   0 |      State = | A   G  |
               | 0   0 |              | GT  Z=0|
        """
        self.A = self.SSModel['A']
        self.M = self.SSModel['M']
        self.G = self.SSModel['G']
        self.B = self.SSModel['B']
        self.C = self.SSModel['C']
    
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
        Mat : scipy.sparse matrix, optional
            Feedback matrix.
        sz : complex or tuple/list of complex, optional
            Spatial frequency parameters for quasi-analysis of the flow field. Default is None.
        reuse : bool, optional
            Whether to reuse the previous state-space model. Default is False.
        """

        if Re is not None:
            self.eqn.Re = Re
        
        if not reuse:
            self._form_LNS_equations(1.0j, sz)
            self._assemble_pencil(Mat)
            self._initialize_prolmat()
            self._initialize_statespace()
            self._assemble_statespace()
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
        M=self.SSModel['Mass']
        A=self.SSModel['State']
            
        return eigen_decompose(A, M, k=k, sigma=sigma, solver_params=param)

 