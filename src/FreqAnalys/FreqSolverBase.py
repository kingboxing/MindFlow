#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:07:07 2024

@author: bojin
"""


from src.Deps import *

from src.NSolver.SolverBase import NSolverBase
from src.BasicFunc.ElementFunc import TaylorHood
from src.BasicFunc.Boundary import SetBoundary, SetBoundaryCondition
from src.BasicFunc.InitialCondition import SetInitialCondition
from src.Eqns.NavierStokes import Incompressible
from src.LinAlg.MatrixOps import AssembleMatrix, AssembleVector, ConvertVector,ConvertMatrix, AssembleSystem, TransposePETScMat, InverseMatrixOperator, SparseLUSolver

#%%
class FreqencySolverBase(NSolverBase):
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
        
        element = TaylorHood(mesh = mesh, order = order, dim = dim, constrained_domain = constrained_domain) # initialise finite element space
        super().__init__(mesh, element, Re, None, None)
       
        # boundary condition
        self.boundary_condition = SetBoundaryCondition(self.element.functionspace, self.boundary)
        # init param
        self.param={}
        
    def set_baseflow(self, ic, timestamp=0.0):
        """
        Set the base flow around which Navier-Stokes equations are linearized.

        Parameters
        ----------
        ic : str or Function
            Path or FEniCS function stores the base flow.
        timestamp : float, optional
            Timestamp of the base flow saved in the time-series file if ic is a path. Default is 0.0.

        Returns
        -------
        None.

        """
        
        SetInitialCondition(0, ic=ic, fw=self.eqn.fw[0], timestamp=timestamp)
        
    def _form_LNS_equations(self, s, sz = None):
        """
        Form the UFL expression of a linearized Navier-Stokes system.

        Parameters
        ----------
        s : int, float, complex
            The Laplace variable s, also known as the operator variable in the Laplace domain.
        sz : complex or tuple/list of complex, optional
            Spatial frequency parameters for quasi-analysis of the flow field. Default is None.

        Returns
        -------
        None.

        """
        if self.element.dim == self.element.mesh.topology().dim():
            # form Steady Linearised Incompressible Navier-Stokes Equations
            leqn=self.eqn.SteadyLinear()
            feqn=self.eqn.Frequency(s)
            
            for key in self.has_traction_bc.keys():
                leqn += self.BoundaryTraction(self.eqn.tp, self.eqn.tu, self.eqn.nu, mark=self.has_traction_bc[key][0], mode=self.has_traction_bc[key][1])
            
            self.LNS=(leqn+feqn[0], feqn[1]) # (real part, imag part)
            
        elif self.element.dim > self.element.mesh.topology().dim(): # quasi-analysis
            # form quasi-Steady Linearised Incompressible Navier-Stokes Equations
            leqn_r, leqn_i=self.eqn.QuasiSteadyLinear(sz)
            feqn=self.eqn.Frequency(s)
            
            for key in self.has_traction_bc.keys():
                leqn_r += self.BoundaryTraction(self.eqn.tp, self.eqn.tu, self.eqn.nu, mark=self.has_traction_bc[key][0], mode=self.has_traction_bc[key][1])
            
            self.LNS=(leqn_r+feqn[0], feqn[1], leqn_i)
            
        
    def _assemble_pencil(self, Mat=None, symmetry=False, BCpart=None):


        """
        Assemble the matrix pencil (sM+NS) of the linear system.
        
        u=(sM+NS)^-1*f where RHS=f, Resp=u
        
        Boundary Condition: u = value/0.0
        
        Parameters
        ----------
        Mat : scipy.sparse matrix, optional
            Feedback matrix (if control is applied: sM+NS+BC). The default is None.
        symmetry : bool, optional
            if assemble matrices in a symmetric fashion. The default is False.
        BCpart : ‘r’ or ‘i’, optional
            The homogenous boundary conditions are applied in the real (matrix NS) or imag (matrix M) part of system. The default is 'i'.
        Returns
        -------
        None.

        """
        if symmetry: # for homogeneous bcs, e.g. eigen analysis
            dummy_rhs = inner(Constant((0.0,)*self.eqn.dim),self.eqn.v)*dx
            Ar = AssembleSystem(self.LNS[0], dummy_rhs, self.boundary_condition.bc_list)[0]
            Ai = AssembleSystem(self.LNS[1], dummy_rhs, self.boundary_condition.bc_list)[0]
            I_bc = AssembleSystem(Constant(0.0)*self.LNS[1], dummy_rhs,self.boundary_condition.bc_list)[0]

        else:
            Ar = AssembleMatrix(self.LNS[0],self.boundary_condition.bc_list) # assemble the real part
            Ai = AssembleMatrix(self.LNS[1],self.boundary_condition.bc_list) # assemble the imag part
            I_bc = AssembleMatrix(Constant(0.0)*self.LNS[1],self.boundary_condition.bc_list) # matrix has only ones in diagonal for rows specified by the boundary condition

        if Mat is not None:# feedback loop
            Ar += Mat # Mat need to be BC applied
            temp_mat = ConvertMatrix(Ar,flag='Mat2PETSc')
            [bc.apply(temp_mat) for bc in self.boundary_condition.bc_list]
            
            if symmetry:
                temp_mat = TransposePETScMat(temp_mat)
                [bc.apply(temp_mat) for bc in self.boundary_condition.bc_list]
                temp_mat = TransposePETScMat(temp_mat)
            
            Ar = ConvertMatrix(temp_mat,flag='PETSc2Mat')
            
        if BCpart is None or BCpart.lower() =='i':
            Ai = Ai-I_bc
        elif BCpart.lower() =='r':
            Ar = Ar-I_bc
        else:
            raise ValueError("BCpart must be one of ('r','i')")
            
        # Complex matrix with boundary conditions
        pencil = (Ar.tocsc(), Ai.tocsc())
        
        if self.element.dim > self.element.mesh.topology().dim():
            if symmetry:
                Ai_quasi = AssembleSystem(self.LNS[2], dummy_rhs, self.boundary_condition.bc_list)[0] - I_bc
            else:
                Ai_quasi= AssembleMatrix(self.LNS[2],self.boundary_condition.bc_list) - I_bc
                
            pencil += (Ai_quasi.tocsc(),)
        # Convert to CSC format for efficient LU decomposition
        self.pencil = pencil
