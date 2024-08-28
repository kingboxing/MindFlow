#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:32:49 2024

@author: bojin
"""

from src.Deps import *

from src.NSolver.SolverBase import NSolverBase
from src.BasicFunc.ElementFunc import TaylorHood
from src.BasicFunc.Boundary import SetBoundary, SetBoundaryCondition
from src.BasicFunc.InitialCondition import SetInitialCondition
from src.Eqns.NavierStokes import Incompressible
from src.LinAlg.MatrixOps import AssembleMatrix, AssembleVector

class FrequencyResponse(NSolverBase):
    def __init__(self, mesh, Re=None, const_expr=None, order=(2,1), dim=2, constrained_domain=None):
        """
        

        Parameters
        ----------
        mesh : TYPE
            DESCRIPTION.
        Re : TYPE, optional
            DESCRIPTION. The default is None.
        const_expr : TYPE, optional
            DESCRIPTION. The default is None.
        order : TYPE, optional
            DESCRIPTION. The default is (2,1).
        dim : TYPE, optional
            DESCRIPTION. The default is 2.
        constrained_domain : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        element = TaylorHood(mesh = mesh, order = order, dim = dim, constrained_domain = constrained_domain) # initialise finite element space
        NSolverBase.__init__(self, mesh, element, Re, const_expr, None)
       
        # boundary condition
        self.boundary_condition = SetBoundaryCondition(self.element.functionspace, self.boundary)
        # init param
        self.param['frequency_solver']={}
        
    def set_baseflow(self, ic, timestamp=0.0):
        """
        Set initial condition

        Parameters
        ----------
        ic : TYPE, optional
            DESCRIPTION. The default is None.
        timestamp : TYPE, optional
            DESCRIPTION. The default is 0.0.

        Returns
        -------
        None.

        """
        
        SetInitialCondition(0, ic=ic, fw=self.eqn.fw[0], timestamp=timestamp)
        
        
    def __LNSEqn(self, s):
        """
        

        Parameters
        ----------
        s : int, float, complex
            the Laplace variable s is also known as operator variable in the Laplace domain.

        Returns
        -------
        None.

        """
        
        # form Steady Linearised Incompressible Navier-Stokes Equations
        leqn=self.eqn.SteadyLinear()
        feqn=self.eqn.Frequency(s)
        for key in self.has_traction_bc.keys():
            leqn+=self.BoundaryTraction(self.eqn.tp, self.eqn.tu, self.eqn.nu, mark=self.has_traction_bc[key][0], mode=self.has_traction_bc[key][1])
        
        self.LNS=(leqn+feqn[0], feqn[1]) # (real part, imag part)

    def __Assemble_LHS(self, Mat=None):
        """
        LHS=(sI-A)
        RHS=f
        Resp=u
        in
        (sI-A)u=f
        
        Parameters
        ----------
        Mat : scipy.sparse.csr_matrix, optional
            Feedback matrix added. The default is None.

        Returns
        -------
        None.

        """
        # assemble the real part
        Ar = AssembleMatrix(self.LNS[0],self.boundary_condition.bc_list)
        if Mat is not None:# feedback loop
            Ar += Mat
        # assemble the imag part
        Ai = AssembleMatrix(self.LNS[1],self.boundary_condition.bc_list) 
        # matrix has only ones in diagonal for rows specified by the boundary condition
        I_bc = AssembleMatrix(Constant(0.0)*self.LNS[1],self.boundary_condition.bc_list)
        # Complex matrix L with boundary conditions
        LHS = Ar + Ai.multiply(1j)-I_bc.multiply(1j)
        # Change the format to CSC, it will be more efficient while LU decomposition
        self.LHS = LHS.tocsc()
    
    def __Solver_Init(self, method = 'lu', lusolver='mumps'):
        if method == 'lu':
            info('LU decomposition using '+lusolver.upper() +' solver...')
            self.Linv=MatInv(self.LHS,lusolver=lusolver, echo=False)
            info('Done.')
            
        elif method == 'krylov':
            #self.Minv=precondition_jacobi(L, useUmfpack=useUmfpack)
            pass
    
    def solve(self, s):
        pass

    




