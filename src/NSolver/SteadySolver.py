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
    
    def __init__(self, mesh, Re=None, sourceterm=None, bodyforce=None, order=(2,1), dim=2, constrained_domain=None):
        """
        Initial Steady Incompressible Navier-Stokes Newton Solver

        Parameters
        ----------
        mesh : TYPE
            DESCRIPTION.
        Re : TYPE, optional
            DESCRIPTION. The default is None.
        sourceterm : TYPE, optional
            DESCRIPTION. The default is None.
        bodyforce : TYPE, optional
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
        NSolverBase.__init__(self, mesh, element, Re, sourceterm, bodyforce)
       
        # boundary condition
        self.BCs = SetBoundaryCondition(self.element.functionspace, self.boundary)
        self.set_boundarycondition=self.BCs.set_boundarycondition


    def __SNSEqn(self):
        """
        UFL expression of steady Naviar-Stokes equations in the weak form

        Returns
        -------
        None.

        """
        # form Steady Incompressible Navier-Stokes Equations
        self.SNS=self.eqn.SteadyNonlinear()
        # force act on the body
        self.force_exp=self.eqn.force_init() 

    def __NewtonMethod(self):
        """
        Initialise Newton solver
        
        Returns
        -------
        None.
        """
        J = derivative(self.SNS, self.element.w, self.element.tw) # Jacobian matrix
        problem = NonlinearVariationalProblem(self.SNS, self.element.w, self.BCs.bcs, J) # Nonlinear problem
        self.solver = NonlinearVariationalSolver(problem) # Nonlinear solver
        self.solver.parameters.update(self.param)
        
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
        
        SetInitialCondition(0, ic=ic, fw=self.w, timestamp=timestamp)

        
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

    def force(self, mark=None, direction=None):
        """
        Get the force on the body (lift or drag)

        Parameters
        ----------------------------
        bodymark : int
            the boundary mark of the body

        direction: int
            0 means X direction and 1 means Y direction

        Returns
        ----------------------------
        force : Fx or Fy

        """
        return assemble((self.force_exp[direction]) * self.eqn.ds(mark))

    def solve(self, Re=None, sourceterm=None):
        """
        Solve the problem

        Parameters
        ----------
        Re : TYPE, optional
            DESCRIPTION. The default is None.
        sourceterm : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if Re is not None:
            self.eqn.Re = Re
        if sourceterm is not None:
            self.eqn.sourceterm = sourceterm
            
        self.__SNSEqn()
        self.__NewtonMethod()
        self.solver.solve()
        
        