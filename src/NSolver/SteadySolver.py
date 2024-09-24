#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:10:37 2023

@author: bojin
"""

from ..Deps import *

from ..NSolver.SolverBase import NSolverBase

from ..BasicFunc.ElementFunc import TaylorHood
from ..BasicFunc.Boundary import SetBoundary, SetBoundaryCondition
from ..BasicFunc.InitialCondition import SetInitialCondition

"""
This module provides the classes that solve Navier-Stokes equations
"""

class NewtonSolver(NSolverBase):
    """
    Solver for steady Navier-Stokes equations using the Newton method.

    """
    
    def __init__(self, mesh, Re=None, const_expr=None, order=(2,1), dim=2, constrained_domain=None):
        """
        Initialize the Steady Incompressible Navier-Stokes Newton Solver.

        Parameters
        ----------
        mesh : Mesh
            The computational mesh.
        Re : float, optional
            The Reynolds number. The default is None.
        const_expr : Expression, Function or Constant, optional
            The time-invariant source term for the flow field. The default is None.
        order : tuple, optional
            The order of the finite element. The default is (2, 1).
        dim : int, optional
            The dimension of the problem (2D or 3D). The default is 2.
        constrained_domain : SubDomain, optional
            Domain for applying constraints (e.g., for periodic boundary conditions). The default is None.
        """
        # init solver
        element = TaylorHood(mesh = mesh, order = order, dim = dim, constrained_domain = constrained_domain) # initialise finite element space
        super().__init__(mesh, element, Re, const_expr, time_expr=None)
       
        # boundary condition
        self.boundary_condition = SetBoundaryCondition(self.element.functionspace, self.boundary)
        # Initialize solver parameters
        self.param['solver_type']='newton_solver'
        self.param['newton_solver']={}
        
    def _form_SINS_equations(self):
        """
        Formulate the steady Navier-Stokes equations in their weak form.

        Returns
        -------
        None
        """
        # Steady Incompressible Navier-Stokes Equations
        self.SNS = self.eqn.SteadyNonlinear()
        
        ## dealing with free boundary/ zero boundary traction condition in bc_list
        for key, value in self.has_traction_bc.items():
            self.SNS += self.BoundaryTraction(self.eqn.p, self.eqn.u, self.eqn.nu, mark=value[0], mode=value[1])
            
    def _initialize_newton_solver(self):
        """
        Initialize the Newton solver for the nonlinear Navier-Stokes problem.

        Returns
        -------
        None
        """
        J = derivative(self.SNS, self.element.w, self.element.tw) # Jacobian matrix
        problem = NonlinearVariationalProblem(self.SNS, self.element.w, self.boundary_condition.bc_list, J) # Nonlinear problem
        self.solver = NonlinearVariationalSolver(problem) # Nonlinear solver
        self.solver.parameters.update({'newton_solver':self.param['newton_solver']})
        #self.solver.parameters.update(self.param['newton_solver'])

    def initial(self, ic=None, timestamp=0.0):
        """
        Set the initial condition for the simulation.

        Parameters
        ----------
        ic : str or Function, optional
            The initial condition as a file path or a FEniCS function. The default is None.
        timestamp : float, optional
            The timestamp for retrieving the initial condition from a time series. The default is 0.0.
        """
        
        SetInitialCondition(flag=0, ic=ic, fw=self.eqn.fw[0], timestamp=timestamp)

    def update_parameters(self, param):
        """
        Update the solver parameters.

        Parameters
        ----------
        param : dict
            A dictionary containing solver parameters to update.
        """
        self.param.update(param)
        
    def solve(self, Re=None, const_expr=None):
        """
        Solve the steady Navier-Stokes equations using the Newton method.

        Parameters
        ----------
        Re : float, optional
            The Reynolds number. The default is None.
        const_expr : Expression or Function, optional
            The time-invariant source term for the flow field. The default is None.

        Returns
        -------
        None
        """
        
        if Re is not None:
            self.eqn.Re = Re
        if const_expr is not None:
            self.eqn.const_expr = const_expr
            
        self._form_SINS_equations()
        self._initialize_newton_solver()
        self.solver.solve()
        gc.collect()

        
        