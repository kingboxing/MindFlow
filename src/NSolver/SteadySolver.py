#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:10:37 2023

@author: bojin
"""

from src.Deps import *

from src.NSolver.SolverBase import NSolverBase

from src.BasicFunc.ElementFunc import TaylorHood
from src.BasicFunc.Boundary import SetBoundary, SetBoundaryCondition
from src.BasicFunc.InitialCondition import SetInitialCondition
from src.Eqns.NavierStokes import Incompressible

"""
This module provides the classes that solve Navier-Stokes equations
"""

class NewtonSolver(NSolverBase):
    """
    Solver of steady Navier-Stokes equations using Newton method


    Examples
    ----------------------------
    here is a snippet code shows how to use this class

    >>> see test 'CylinderBaseFlow.py'

    """
    
    def __init__(self, mesh, Re=None, const_expr=None, order=(2,1), dim=2, constrained_domain=None):
        """
        Initial Steady Incompressible Navier-Stokes Newton Solver

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
        # init solver
        element = TaylorHood(mesh = mesh, order = order, dim = dim, constrained_domain = constrained_domain) # initialise finite element space
        NSolverBase.__init__(self, mesh, element, Re, const_expr, None)
       
        # boundary condition
        self.boundary_condition = SetBoundaryCondition(self.element.functionspace, self.boundary)
        # init param
        self.param['newton_solver']={}

    def __SNSEqn(self):
        """
        UFL expression of steady Naviar-Stokes equations in the weak form

        Returns
        -------
        None.

        """
        # form Steady Incompressible Navier-Stokes Equations
        seqn=self.eqn.SteadyNonlinear()
        self.SNS=seqn
        
        " pending for dealing with free boundary/ zero boundary traction condition in bc_list"
        # if self.boundary_condition.has_free_bc is False: 
        #     self.SNS+=seqn[1] 
        # else:
        #     self.has_free_bc=self.boundary_condition.has_free_bc
        
        for key in self.has_traction_bc.keys():
            self.SNS+=self.BoundaryTraction(self.eqn.p, self.eqn.u, self.eqn.nu, mark=self.has_traction_bc[key][0], mode=self.has_traction_bc[key][1])

    def __NewtonMethod(self):
        """
        Initialise Newton solver
        
        Returns
        -------
        None.
        """
        J = derivative(self.SNS, self.element.w, self.element.tw) # Jacobian matrix
        problem = NonlinearVariationalProblem(self.SNS, self.element.w, self.boundary_condition.bc_list, J) # Nonlinear problem
        self.solver = NonlinearVariationalSolver(problem) # Nonlinear solver
        self.solver.parameters.update({'newton_solver':self.param['newton_solver']})
        
    def initial(self, ic=None, timestamp=0.0):
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

        
    def parameters(self, param):
        """
        Set solve parameters

        Parameters
        ----------
        param : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # update solver parameters
        self.param.update(param)


    def solve(self, Re=None, const_expr=None):
        """
        Solve the problem

        Parameters
        ----------
        Re : TYPE, optional
            DESCRIPTION. The default is None.
        const_expr : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if Re is not None:
            self.eqn.Re = Re
        if const_expr is not None:
            self.eqn.const_expr = const_expr
            
        self.__SNSEqn()
        self.__NewtonMethod()
        self.solver.solve()
        
        