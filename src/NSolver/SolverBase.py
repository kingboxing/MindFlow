#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 22:36:04 2024

@author: bojin
"""

from src.Deps import *

from src.BasicFunc.Boundary import SetBoundary,BoundaryCondition
from src.Eqns.NavierStokes import Incompressible


class NSolverBase:
    """
    Solver Base of Navier-Stokes equations


    Examples
    ----------------------------
    here is a snippet code shows how to use this class

    >>> see test 'CylinderBaseFlow.py'

    """
    
    def __init__(self, mesh, element, Re, const_expr, time_expr):
        """
        Solver Base for N-S equations

        Parameters
        ----------
        mesh : TYPE
            DESCRIPTION.
        element : TYPE
            DESCRIPTION.
        Re : TYPE
            DESCRIPTION.
        const_expr : TYPE
            DESCRIPTION.
        time_expr : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.element = element
        # store solution
        self.flow=self.element.w
        # boundary
        self.boundary=SetBoundary(mesh, element)
        # NS equations
        self.eqn=Incompressible(self.element, self.boundary, Re, const_expr=const_expr, time_expr=time_expr)
        # parameters
        self.has_traction_bc = {}
        # pending
        self.param={'solver_type':   None,
                    'bc_reset':      False}
    
        
    def set_boundary(self, bc_list=None):
        """
        

        Parameters
        ----------
        bc_list : dict, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if bc_list is None:
            bc_list=self.boundary.bc_list
        
        for key in bc_list.keys():
            #self.has_traction_bc[key]=None
            self.boundary.set_boundary(bc_list[key]['location'], key)
            
    def set_boundarycondition(self, bc_list=None):
        """
        

        Parameters
        ----------
        bc_list : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if bc_list is None:
            bc_list=self.boundary.bc_list
        
        for key in bc_list.keys():
            self.boundary_condition.set_boundarycondition(bc_list[key], key)
            #if key in self.has_traction_bc.keys():
            if bc_list[key]['BoundaryTraction'] is not None: # default bc is zero Boundary Traction
                self.has_traction_bc[key]=bc_list[key]['BoundaryTraction']
        
    def __force_expr(self, mark=None):
        """
        initialse force solver

        Parameters
        ----------
        mark : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        force=self.eqn.force_expr()
        
        drag1 = assemble((force[0][0]) * self.eqn.ds(mark))
        drag2 = assemble((force[1][0]) * self.eqn.ds(mark))
        lift1 = assemble((force[0][1]) * self.eqn.ds(mark))
        lift2 = assemble((force[1][1]) * self.eqn.ds(mark))
        
        return ((drag1, drag2),(lift1, lift2))
    
    def __vorticity_expr(self):
        """
        initialise vorticity solver

        Returns
        -------
        solver : TYPE
            DESCRIPTION.
        u : TYPE
            DESCRIPTION.
        b : TYPE
            DESCRIPTION.
        W : TYPE
            DESCRIPTION.

        """
        vorticity=self.eqn.vorticity_expr()
        if self.eqn.dim == 2:
            W = FunctionSpace(self.element.mesh, 'P', self.element.order[0])
        elif self.eqn.dim == 3:
            W = VectorFunctionSpace(self.element.mesh, 'P', self.element.order[0])
        # set solve functions
        tu = TrialFunction(W)
        v = TestFunction(W)
        u = Function(W)
        # RHS matrix
        B=PETScMatrix()
        B=assemble(inner(vorticity,v)*dx, tensor=B)
        # LHS matrix
        A=PETScMatrix()
        assemble(inner(tu,v)*dx, tensor=A)
        # setup solver
        solver=PETScLUSolver(A,'mumps')
        solver.parameters.add('reuse_factorization', True)
        
        return (u, (solver, B))
    
    def eval_vorticity(self, reuse=False):
        """
        evaluate vorticity of the flow field

        Parameters
        ----------
        reuse : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if reuse is False:
            self.vorticity = self.__vorticity_expr()
            
        if self.element.type == 'TaylorHood':
            b=self.vorticity[1][1]*self.eqn.w.vector()
        elif self.element.type == 'Decoupled':
            b=self.vorticity[1][1]*self.eqn.u.vector()
            
        self.vorticity[1][0].solve(self.vorticity[0].vector(), b)
        
        return self.vorticity[0]

        
    def eval_force(self, mark=None, dirc=None, comp=None,reuse=False):
        """
        evaluate the force on the body (lift or drag)

        Parameters
        ----------
        mark : int
            the boundary mark of the body. The default is None.
        dirc : int
            0 means X direction and 1 means Y direction. The default is None.
        comp : None or int
            0 means pressure part, 1 means stress part, None means the summation
            The default is None.
        reuse : bool
            if re-assemble the force expression. The default is False.

        Returns
        -------
        float
            force acting on the body.

        """

        # force act on the body
        if reuse is False:
            self.force = self.__force_expr(mark)
        
        if self.element.type == 'TaylorHood':
            if comp is None:
                return self.force[dirc][0].inner(self.eqn.w.vector())+self.force[dirc][1].inner(self.eqn.w.vector())
            else:
                return self.force[dirc][comp].inner(self.eqn.w.vector())
            
        elif self.element.type == 'Decoupled':
            if comp is None:
                return self.force[dirc][0].inner(self.eqn.p.vector())+self.force[dirc][1].inner(self.eqn.u.vector())
            else:
                sol=(self.eqn.p.vector(), self.eqn.u.vector())
                return self.force[dirc][comp].inner(sol[comp])

    def solve(self):
        pass
    

    
        
