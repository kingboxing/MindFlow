#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 22:36:04 2024

@author: bojin
"""

from ..Deps import *

from ..BasicFunc.Boundary import SetBoundary, BoundaryCondition
from ..Eqns.NavierStokes import Incompressible


class NSolverBase:
    """
    Base class for solving Navier-Stokes equations in FEniCS.

    This class sets up the required elements, boundary conditions, and equations
    for solving the Navier-Stokes equations, including methods for evaluating
    forces and vorticity.

    """

    def __init__(self, mesh, element, Re, const_expr, time_expr):
        """
        Initialize the base solver for Navier-Stokes equations.

        Parameters
        ----------
        mesh : Mesh
            The mesh object used in the simulation.
        element : object
            The finite element object (e.g., TaylorHood).
        Re : float
            The Reynolds number for the flow.
        const_expr : Expression, Function or Constant
            The time-invariant source term for the flow field.
        time_expr : Expression, Function or Constant
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
            A dictionary containing boundary conditions. The default is None, which uses the boundary conditions defined in the self.boundary object.
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
            A dictionary containing boundary conditions. The default is None, which uses the boundary conditions defined in the self.boundary object.
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

        Returns
        -------
        tuple
            A tuple containing the vorticity function, solver, and RHS matrix.
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
            Whether to reuse the existing vorticity solver setup. Default is True

        Returns
        -------
        Function
            The vorticity function.
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

        Parameters
        ----------
        mark : int, optional
            The boundary mark where the force is evaluated. Default is None.

        Returns
        -------
        tuple
            A tuple of assembled forces in different directions and components.
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
            The vectors containing the flow solution.
        dirc : int
            Direction of the force (0: X, 1: Y).
        comp : int or None
            Component of the force (0: pressure, 1: stress, None: both).

        Returns
        -------
        float
            The computed force value.
        """
        if comp is None:
            return self.force[dirc][0].inner(vector[0]) + self.force[dirc][1].inner(vector[1])
        else:
            return self.force[dirc][comp].inner(vector[comp])

    def eval_force(self, mark=None, dirc=0, comp=None, reuse=True):
        """
        Evaluate the force on a body (e.g., lift or drag).

        Parameters
        ----------
        mark : int
            The boundary mark of the body. Default is None.
        dirc : int
            Direction of the force (0: X, 1: Y). Default is 0.
        comp : int or None
            Component of the force (0: pressure, 1: stress, None: both). Default is None.
        reuse : bool
            Whether to reuse the existing force solver setup. Default is True.

        Returns
        -------
        float
            The computed force acting on the body.
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
        Placeholder for the solve method to be implemented by subclasses.

        Returns
        -------
        None.

        """
        pass
