#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the `NewtonSolver` class for solving steady-state incompressible Navier-Stokes equations using the Newton-Raphson method.

The solver is designed to handle complex fluid flow problems in two or three dimensions, leveraging the FEniCS finite element library for discretization and solution.

Classes
-------
- NewtonSolver:
    Solver for steady Navier-Stokes equations using the Newton method.

Dependencies
------------
- FEniCS
- NumPy
- SciPy

Ensure that all dependencies are installed and properly configured.

Examples
--------
Typical usage involves creating an instance of `NewtonSolver`, setting up the problem domain, boundary conditions, and solving the Navier-Stokes equations.

```python
from FERePack.NSolver.SteadySolver import NewtonSolver

# Define mesh and parameters
mesh = ...
Re = 100.0
const_expr = ...

# Initialize the solver
solver = NewtonSolver(mesh, Re=Re, const_expr=const_expr, order=(2, 1), dim=2)

# Set boundary conditions
solver.set_boundary(bc_list)
solver.set_boundarycondition(bc_list)

# Solve the steady Navier-Stokes equations
solver.solve()

# Retrieve the solution
u, p = solver.flow.split()
```

Notes
--------

- This solver uses the Newton-Raphson method for solving the nonlinear steady-state Navier-Stokes equations.
- The finite element spaces are constructed using the Taylor-Hood elements by default.
- Boundary conditions and initial conditions can be specified as per FEniCS conventions.
"""

from ..Deps import *

from ..NSolver.SolverBase import NSolverBase

from ..BasicFunc.ElementFunc import TaylorHood
from ..BasicFunc.Boundary import SetBoundary, SetBoundaryCondition
from ..BasicFunc.InitialCondition import SetInitialCondition


class NewtonSolver(NSolverBase):
    """
    Solver for steady-state incompressible Navier-Stokes equations using the FEniCS build-in Newton method.

    This class extends the `NSolverBase` class and implements a solver for the steady Navier-Stokes equations using
    the FEniCS build-in Newton method. It is suitable for solving complex fluid flow problems in two or three dimensions.

    Parameters
    ----------
    mesh : dolfin.Mesh
        The computational mesh.
    Re : float, optional
        The Reynolds number. Default is None.
    const_expr : dolfin.Expression, dolfin.Function, or dolfin.Constant, optional
        The time-invariant source term for the flow field (e.g., body forces). Default is None.
    order : tuple of int, optional
        The order of the finite element spaces for velocity and pressure, respectively. Default is (2, 1).
    dim : int, optional
        Dimension of the flow field. Default is 2.
    constrained_domain : dolfin.SubDomain, optional
        Constrained subdomain for periodic boundary conditions. Default is None.

    Attributes
    ----------
    element : object
        The finite element object defining the function spaces.
    boundary_condition : SetBoundaryCondition
        Object for handling boundary conditions.
    solver : dolfin.NonlinearVariationalSolver
        The Newton solver for the nonlinear problem.

    Methods
    -------
    initial(ic=None, timestamp=0.0)
        Set the initial condition for the simulation.
    update_parameters(param)
        Update the solver parameters.
    solve(Re=None, const_expr=None)
        Solve the steady Navier-Stokes equations using the Newton method.

    Notes
    -----
    - The solver uses Taylor-Hood elements for discretization by default.
    - The Newton method is employed to solve the nonlinear steady-state equations.
    - Boundary conditions and source terms should be specified using FEniCS conventions.
    """

    def __init__(self, mesh, Re=None, const_expr=None, order=(2, 1), dim=2, constrained_domain=None):
        """
        Initialize the Steady Incompressible Navier-Stokes Newton Solver.

        Parameters
        ----------
        mesh : dolfin.Mesh
            The computational mesh.
        Re : float, optional
            The Reynolds number. Default is None.
        const_expr : dolfin.Expression, dolfin.Function, or dolfin.Constant, optional
            The time-invariant source term for the flow field (e.g., body forces). Default is None.
        order : tuple of int, optional
            The order of the finite element spaces for velocity and pressure, respectively. Default is (2, 1).
        dim : int, optional
            Dimension of the flow field. Default is 2.
        constrained_domain : dolfin.SubDomain, optional
            Constrained subdomain for periodic boundary conditions. Default is None.
        """
        # init solver
        element = TaylorHood(mesh=mesh, order=order, dim=dim,
                             constrained_domain=constrained_domain)  # initialise finite element space
        super().__init__(mesh, element, Re, const_expr, time_expr=None)

        # boundary condition
        self.boundary_condition = SetBoundaryCondition(self.element.functionspace, self.boundary)
        # Initialize solver parameters
        self.param['solver_type'] = 'newton_solver'
        self.param['newton_solver'] = {}

    def _form_SINS_equations(self):
        """
        Formulate the steady incompressible Navier-Stokes equations in their weak form.

        This method assembles the weak form of the steady Navier-Stokes equations, including any boundary traction
        terms for free boundaries.
        """
        # Steady Incompressible Navier-Stokes Equations
        self.SNS = self.eqn.SteadyNonlinear()

        ## dealing with free boundary/ zero boundary traction condition in bc_list
        for key, value in self.has_traction_bc.items():
            self.SNS += self.BoundaryTraction(self.eqn.p, self.eqn.u, self.eqn.nu, mark=value[0], mode=value[1])

    def _initialize_newton_solver(self):
        """
        Initialize the Newton solver for the nonlinear Navier-Stokes problem.

        This method sets up the nonlinear variational problem and initializes the Newton solver with the specified parameters.

        """
        J = derivative(self.SNS, self.element.w, self.element.tw)  # Jacobian matrix
        problem = NonlinearVariationalProblem(self.SNS, self.element.w, self.boundary_condition.bc_list,
                                              J)  # Nonlinear problem
        self.solver = NonlinearVariationalSolver(problem)  # Nonlinear solver
        self.solver.parameters.update({'newton_solver': self.param['newton_solver']})
        #self.solver.parameters.update(self.param['newton_solver'])

    def initial(self, ic=None, timestamp=0.0):
        """
        Set the initial condition for the simulation.

        Parameters
        ----------
        ic : str or dolfin.Function, optional
            The initial condition, either as a file path to stored data or as a FEniCS function. Default is None.
        timestamp : float, optional
            The timestamp for retrieving the initial condition from a time series. Default is 0.0.

        Notes
        -----
        - The initial condition is set using the `SetInitialCondition` utility.
        - This is particularly useful if a good initial guess is available to aid convergence of the Newton solver.
        """

        SetInitialCondition(flag=0, ic=ic, fw=self.eqn.fw[0], timestamp=timestamp)

    def update_parameters(self, param):
        """
        Update the solver parameters.

        Parameters
        ----------
        param : dict
            A dictionary containing solver parameters to update.

        Notes
        -----
        - The solver parameters control aspects of the Newton solver, such as tolerances and iteration limits.
        - Parameters are updated by merging the provided dictionary with the existing parameters.
        """
        self.param.update(param)

    def solve(self, Re=None, const_expr=None):
        """
        Solve the steady Navier-Stokes equations using the Newton method.

        Parameters
        ----------
        Re : float, optional
            The Reynolds number. If provided, updates the Reynolds number before solving. Default is None.
        const_expr : dolfin.Expression or dolfin.Function, optional
            The time-invariant source term for the flow field (e.g., body forces). If provided, updates the source term before solving. Default is None.

        Returns
        -------
        None

        Notes
        -----
        - Before solving, the method updates the equations with any new parameters, assembles the equations, and initializes the solver.
        - The solution is stored in `self.element.w`, which contains both the velocity and pressure fields.
        """

        if Re is not None:
            self.eqn.Re = Re
        if const_expr is not None:
            self.eqn.const_expr = const_expr

        self._form_SINS_equations()
        self._initialize_newton_solver()
        self.solver.solve()
        gc.collect()
