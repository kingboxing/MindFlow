#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the `NSolverBase` class, which serves as a base class for solving the Navier-Stokes equations using the FEniCS finite element library.

The class sets up the required finite element spaces, boundary conditions, and equations for solving incompressible fluid flow problems, including methods for evaluating forces (e.g., lift and drag) and vorticity in the flow field.

Classes
-------
- NSolverBase:
    Base class for solving Navier-Stokes equations in FEniCS.

Dependencies
------------
- FEniCS
- NumPy
- SciPy

Ensure that all dependencies are installed and properly configured.

Examples
--------
Typical usage involves subclassing `NSolverBase` to implement specific solvers for steady-state or time-dependent Navier-Stokes equations.

```python
from FERePack.NSolver.SolverBase import NSolverBase

# Define mesh, element, Reynolds number, etc.
mesh = ...
element = ...
Re = ...

# Initialize the solver base
solver = NSolverBase(mesh, element, Re, const_expr, time_expr)

# Set boundary conditions
solver.set_boundary(bc_list)
solver.set_boundarycondition(bc_list)

# Solve the Navier-Stokes equations (to be implemented in subclass)
solver.solve()

# Evaluate vorticity
vorticity = solver.eval_vorticity()

# Evaluate force (e.g., drag)
drag = solver.eval_force(mark=boundary_mark, dirc=0)
```
"""

from ..Deps import *
from ..BasicFunc.Boundary import SetBoundary, BoundaryCondition
from ..Eqns.NavierStokes import Incompressible


class NSolverBase:
    """
    Base class for solving Navier-Stokes equations in FEniCS.

    This class sets up the required elements, boundary conditions, and equations
    for solving the incompressible Navier-Stokes equations. It provides methods for
    evaluating forces (e.g., lift and drag) and vorticity in the flow field.

    Parameters
    ----------
    mesh : Mesh
        The mesh object used in the simulation.
    element : object
        The finite element object defining the function spaces (e.g., Taylor-Hood elements).
    Re : float
        The Reynolds number for the flow.
    const_expr : dolfin.Expression, dolfin.Function, or dolfin.Constant
        The time-invariant source term for the flow field (e.g., body forces).
    time_expr : dolfin.Expression, dolfin.Function, or dolfin.Constant
        The time-dependent source term for the flow field.

    Attributes
    ----------
    element : object
        The finite element object.
    flow : Function
        The solution function representing the flow field.
    boundary : SetBoundary
        Object for handling boundary definitions.
    eqn : Incompressible
        The incompressible Navier-Stokes equations object.
    has_traction_bc : dict
        Dictionary to track boundaries with traction boundary conditions.
    param : dict
        Dictionary of solver parameters.

    Methods
    -------
    set_boundary(bc_list=None)
        Define the boundary locations in the mesh.
    set_boundarycondition(bc_list=None)
        Apply Dirichlet boundary conditions to the problem.
    eval_vorticity(reuse=True)
        Evaluate the vorticity of the flow field.
    eval_force(mark=None, dirc=0, comp=None, reuse=True)
        Evaluate the force (e.g., lift or drag) on a boundary.
    solve()
        Placeholder method to solve the Navier-Stokes equations (to be implemented in subclasses).

    Notes
    -----
    - This class is intended to be subclassed, with the `solve` method implemented to perform the actual solution procedure.
    - The class assumes that the finite element spaces and elements are compatible with the incompressible Navier-Stokes equations.
    - Boundary conditions and source terms should be provided as per FEniCS conventions.
    """

    def __init__(self, mesh, element, Re, const_expr, time_expr):
        """
        Initialize the base solver for Navier-Stokes equations.

        Parameters
        ----------
        mesh : Mesh
            The mesh object used in the simulation.
        element : object
            The finite element object defining the function spaces (e.g., Taylor-Hood elements).
        Re : float
            The Reynolds number for the flow.
        const_expr : dolfin.Expression, dolfin.Function, or dolfin.Constant
            The time-invariant source term for the flow field (e.g., body forces).
        time_expr : dolfin.Expression, dolfin.Function, or dolfin.Constant
            The time-dependent source term for the flow field.
        """
        self.element = element
        self.flow = self.element.w  # store solution
        self.boundary = SetBoundary(mesh, element)  # boundary
        self.eqn = Incompressible(self.element, self.boundary, Re, const_expr=const_expr,
                                  time_expr=time_expr)  # NS equations
        self.has_traction_bc = {}  # Track boundary conditions with traction
        # pending
        self.param = {'solver_type': None,
                      'bc_reset': False}

    def set_boundary(self, bc_list=None):
        """
        Define the boundary locations in the mesh.


        Parameters
        ----------
        bc_list : dict, optional
            A dictionary containing boundary conditions.
            Each key represents a boundary identifier, and the value is a dictionary
            with boundary properties, including 'location' which specifies the boundary markers.
            If None, uses the boundary conditions defined in the `self.boundary.bc_list` object.

        """

        if bc_list is None:
            bc_list = self.boundary.bc_list

        for key in bc_list.keys():
            #self.has_traction_bc[key]=None
            self.boundary.set_boundary(bc_list[key]['location'], key)

    def set_boundarycondition(self, bc_list=None):
        """
        Apply Dirichlet boundary conditions to the problem.

        Parameters
        ----------
        bc_list : dict, optional
            A dictionary containing boundary conditions.
            Each key represents a boundary identifier, and the value is a dictionary
            with boundary properties, including 'BoundaryTraction' and 'BoundaryCondition'.
            If None, uses the boundary conditions defined in the `self.boundary.bc_list` object.

        Returns
        -------
        None.
        """

        if bc_list is None:
            bc_list = self.boundary.bc_list

        for key, bc in bc_list.items():
            self.boundary_condition.set_boundarycondition(bc, key)
            if bc.get('BoundaryTraction') is not None:  # default bc is zero Boundary Traction
                self.has_traction_bc[key] = bc['BoundaryTraction']

    def _vorticity_expr(self):
        """
        Initialize the vorticity expression solver.

        This method sets up the necessary function spaces and solvers to compute the vorticity
        of the flow field based on the current solution.

        Returns
        -------
        tuple
            A tuple containing:
            - u : dolfin.Function
                The vorticity function.
            - (solver, B) : tuple
                - solver : dolfin.PETScLUSolver
                    The solver used to compute the vorticity.
                - B : dolfin.PETScMatrix
                    The assembled right-hand side matrix.
        """

        vorticity = self.eqn.vorticity_expr()
        if self.eqn.dim == 2:
            W = FunctionSpace(self.element.mesh, 'P', self.element.order[0])
        elif self.eqn.dim == 3:
            W = VectorFunctionSpace(self.element.mesh, 'P', self.element.order[0])
        # set solve functions
        tu = TrialFunction(W)
        v = TestFunction(W)
        u = Function(W)
        # RHS matrix
        B = PETScMatrix()
        B = assemble(inner(vorticity, v) * dx, tensor=B)
        # LHS matrix
        A = PETScMatrix()
        assemble(inner(tu, v) * dx, tensor=A)
        # setup solver
        solver = PETScLUSolver(A, 'mumps')
        solver.parameters.add('reuse_factorization', True)

        return (u, (solver, B))

    def eval_vorticity(self, reuse=True):
        """
        Evaluate the vorticity of the flow field.

        Parameters
        ----------
        reuse : bool, optional
            If True (default), reuse the existing vorticity solver setup if available.
            If False, reassemble the vorticity expressions and solver.

        Returns
        -------
        u : dolfin.Function
            The computed vorticity function.
        """
        if reuse is False or not hasattr(self, 'vorticity'):
            self.vorticity = self._vorticity_expr()

        if self.element.type == 'TaylorHood':
            b = self.vorticity[1][1] * self.eqn.w.vector()
        elif self.element.type == 'Decoupled':
            b = self.vorticity[1][1] * self.eqn.u.vector()

        self.vorticity[1][0].solve(self.vorticity[0].vector(), b)

        return self.vorticity[0]

    def _force_expr(self, mark=None):
        """
        Initialize the force expression solver.

        This method assembles the force expressions (e.g., for lift and drag) over a specified boundary.

        Parameters
        ----------
        mark : int, optional
            The boundary marker where the force is evaluated.
            If None, integrates over all boundaries.

        Returns
        -------
        force : tuple
            A tuple containing assembled forces for each spatial dimension.
            Each element is a tuple of (pressure component, viscous stress component).
        """

        force_expr = self.eqn.force_expr()
        dim = self.element.dim

        # force = ()
        # for i in range(dim):
        #     temp = (assemble((force_expr[0][i]) * self.eqn.ds(mark)), assemble((force_expr[1][i]) * self.eqn.ds(mark)))
        #     force += (temp,) 

        force = tuple(
            (
                assemble(force_expr[0][i] * self.eqn.ds(mark)),
                assemble(force_expr[1][i] * self.eqn.ds(mark))
            ) for i in range(dim)
        )

        return force  # 1st index for direction, 2nd index for component

        # drag1 = assemble((force[0][0]) * self.eqn.ds(mark))
        # drag2 = assemble((force[1][0]) * self.eqn.ds(mark))
        # lift1 = assemble((force[0][1]) * self.eqn.ds(mark))
        # lift2 = assemble((force[1][1]) * self.eqn.ds(mark))
        # return ((drag1, drag2),(lift1, lift2))

    def _compute_force(self, vector, dirc, comp):
        """
        Helper function to compute force for given vector, direction, and component.

        Parameters
        ----------
        vector : tuple of dolfin.Vector
            The vectors containing the flow solution components.
        dirc : int
            Direction of the force (0 for X, 1 for Y, etc.).
        comp : int or None
            Component of the force:
            - 0 : pressure component
            - 1 : viscous stress component
            - None : sum of both components

        Returns
        -------
        float
            The computed force value in the specified direction.
        """
        if comp is None:
            return self.force[dirc][0].inner(vector[0]) + self.force[dirc][1].inner(vector[1])
        else:
            return self.force[dirc][comp].inner(vector[comp])

    def eval_force(self, mark=None, dirc=0, comp=None, reuse=True):
        """
        Evaluate the force (e.g., lift or drag) acting on a boundary.

        Parameters
        ----------
        mark : int, optional
            The boundary marker of the body on which the force is evaluated.
            If None, integrates over all boundaries.
        dirc : int, optional
            Direction of the force (0 for X, 1 for Y, etc.). Default is 0.
        comp : int or None, optional
            Component of the force:
            - 0 : pressure component
            - 1 : viscous stress component
            - None : sum of both components (default).
        reuse : bool, optional
            If True (default), reuse the existing force expressions if available.
            If False, reassemble the force expressions.

        Returns
        -------
        float
            The computed force acting on the body in the specified direction.
        """

        # force act on the body
        if reuse is False or not hasattr(self, 'force'):
            self.force = self._force_expr(mark)

        if self.element.type == 'TaylorHood':
            vec = (self.eqn.w.vector(), self.eqn.w.vector())
            return self._compute_force(vec, dirc, comp)
        elif self.element.type == 'Decoupled':
            vec = (self.eqn.p.vector(), self.eqn.u.vector())
            return self._compute_force(vec, dirc, comp)

        # previous version
        # if self.element.type == 'TaylorHood':
        #     if comp is None:
        #         return self.force[dirc][0].inner(self.eqn.w.vector())+self.force[dirc][1].inner(self.eqn.w.vector())
        #     else:
        #         return self.force[dirc][comp].inner(self.eqn.w.vector())     
        # elif self.element.type == 'Decoupled':
        #     if comp is None:
        #         return self.force[dirc][0].inner(self.eqn.p.vector())+self.force[dirc][1].inner(self.eqn.u.vector())
        #     else:
        #         sol=(self.eqn.p.vector(), self.eqn.u.vector())
        #         return self.force[dirc][comp].inner(sol[comp])

    def solve(self):
        """
        Placeholder method to solve the Navier-Stokes equations.

        This method should be implemented in subclasses to perform the actual solution
        procedure, whether steady-state or time-dependent.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.

        """
        raise NotImplementedError("The solve method must be implemented in subclasses.")
