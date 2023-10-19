#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:10:37 2023

@author: bojin
"""

from dolfin import *

#import copy
#import os,sys,inspect
#parentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
#sys.path.insert(0,parentdir) 

from src.BasicFunc.ElementFunc import TaylorHood
from src.BasicFunc.Boundary import SetBoundaryCondition



"""This module provides the classes that solve Navier-Stokes equations
"""

class Solver:
    
    def __init__(self):
        pass
    
    

"""
---------------- ABANDONED ----------------
"""
class NS_Newton_Solver:
    """
    Solver of steady Navier-Stokes equations using Newton method

    Parameters
    ----------------------------
    mesh : object created by FEniCS function Mesh
        mesh of the flow field

    boundary : object Boundary()
        the Boundary object with defined and marked boundaries

    nu : Constant()
        kinematic viscosity

    Attributes
    ----------------------------
    w : solution (u, p)

    u, p : velocity and pressure

    functionspace : a finite element function space

    boundaries : FacetFunction on given mesh

    bcs : a list with boundary conditions

    nu : Constant()
        kinematic viscosity

    solver_para: dict
        solver parameters

    Examples
    ----------------------------
    here is a snippet code shows how to use this class

    >>> from dolfin import *
    >>> from MindFlow.Boundary.Set_Boundary import Boundary
    >>> from MindFlow.FlowSolver.NS_Newton_Solver import NS_Newton_Solver
    >>> mesh = Mesh ('mesh.xml')
    >>> boundary=Boundary(mesh)
    >>> boundary.set_boundary(location=BoundaryLocations, mark=5)
    >>> nu = Constant(1.0/45)
    >>> solver = NS_Newton_Solver(mesh=mesh, boundary=boundary, nu=nu)
    >>> solver.solver_parameters(parameters={'newton_solver':{'linear_solver': 'mumps'}})
    >>> solver.set_boundarycondition(BoundaryConditions, BoundaryLocations)
    >>> solver.solve()
    >>> print('drag= %e lift= %e' % (solver.get_force(bodymark=5,direction=0), solver.get_force(bodymark=5,direction=1)))



    """
    def __init__(self, mesh, boundary, nu=None,sourceterm=None, bodyforce=None,order=(2,1),dim=2, constrained_domain=None):
        element = TaylorHood(mesh = mesh, order = order, dim = dim, constrained_domain = constrained_domain) # initialise finite element space
        self.element = element
        # inherit attributes : functionspace, boundries, bcs
        #SetBoundaryCondition.__init__(self, Functionspace=element.functionspace, boundary=boundary)
        self.BC = SetBoundaryCondition(Functionspace=element.functionspace, boundary=boundary)
        self.functionspace = element.functionspace
        self.subboundaries = boundary.get_subdomain()
        self.boundaries = boundary.get_domain() # Get the FacetFunction on given mesh
        # create public attributes
        self.nu = nu
        self.solver_para = {}
        # assign functions for the expression convenience
        self.mesh=mesh  
        self.submesh=boundary.submesh
        self.w=element.w
        self.u=element.u
        self.p=element.p
        self.v=element.v
        self.q=element.q
        self.dw=element.tw
        # get FacetNormal and boundary measurement
        self.n = FacetNormal(mesh)
        self.ds = boundary.get_measure()
        self.freeoutlet_mark=[]
        # source term, doesn't change outlet boundary conditions, not balanced by pressure
        if sourceterm is not None and sourceterm is not False:
            self.sourceterm = sourceterm
        else:
            self.sourceterm = Constant((0.0,0.0))
            
        # body force, may change utlet boundary conditions, balanced by pressure
        if bodyforce is not None and bodyforce is not False:
            self.bodyforce = bodyforce
        else:
            self.bodyforce = Constant((0.0,0.0))

    def set_boundarycondition(self, boucon, mark):
        try:
            test = boucon['FunctionSpace']
        except:
            boucon['FunctionSpace']=None
            info('No Dirichlet Boundary Condition at Boundary % g' % mark)
        else:
            pass
        
        try:
            test = boucon['Value']
        except:
            boucon['Value']=None
            info('No Dirichlet Boundary Condition at Boundary % g' % mark)
        else:
            pass
        
        if (boucon['FunctionSpace'] in ['Free Outlet','FreeOutlet','freeoutlet','free outlet'] or boucon['Value'] in ['Free Outlet','FreeOutlet','freeoutlet','free outlet']):
            self.freeoutlet=True
            self.freeoutlet_mark.append(mark)
            if len(self.freeoutlet_mark) > 1:
                raise ValueError('Only one free outlet boundary is allowed')
        else:
            self.BC.set_boundarycondition(boucon,mark)

    def __NS_expression(self):
        """UFL expression of steady Naviar-Stokes equations in the weak form

        """
        self.F = ((self.nu * inner(grad(self.u), grad(self.v)) +
                  inner(dot(self.u, nabla_grad(self.u)), self.v) -
                  self.p * div(self.v) - inner(self.sourceterm, self.v) - inner(self.bodyforce, self.v) + div(self.u) * self.q) * dx )
        
#        try:
#            test=self.freeoutlet
#        except:
#            ## hard to converge after adding this residual term
#            self.F = self.F+(dot(self.v,self.n)*self.p*self.ds-self.nu*inner(grad(self.u)*self.n,self.v)*self.ds)
#        else:
#            if self.freeoutlet is not True:
#                ## hard to converge after adding this residual term
#                self.F = self.F+dot(self.v,self.n)*self.p*self.ds-self.nu*inner(grad(self.u)*self.n,self.v)*self.ds
#            elif self.freeoutlet is True:
#                pass
            
#                dpdx=self.sourceterm.values()[0]
#                dpdy=self.sourceterm.values()[1]
#                self.pre_addition=Expression('dpdx*x[0]+dpdy*x[1]',degree=2,dpdx=dpdx,dpdy=dpdy)
#                self.F = self.F + dot(self.v,self.n)*self.pre_addition*self.ds(mark)

    def __BalancedPressure(self,mark):
        pass
#        submesh = SubMesh(self.submesh, self.subboundaries,mark)
#        element = TaylorHood(mesh = submesh)

    
    def __solver_init(self):
        """Initialise Newton solver

        """

        J = derivative(self.F, self.w, self.dw) # Jacobian matrix
        problem = NonlinearVariationalProblem(self.F, self.w, self.BC.bcs, J) # Nonlinear problem
        self.solver = NonlinearVariationalSolver(problem) # Nonlinear solver
        # update solver parameters
        self.prm = self.solver.parameters
        self.prm.update(self.solver_para)

    def __force_init(self):
        """Force expression (lift and drag)

        """
        I = Identity(self.u.geometric_dimension())
        D = sym(grad(self.u))
        T = -self.p * I + 2 * self.nu * D#
        self.force = - T * self.n

    def update_problem(self):
        """Update problem
        prepare everything to solve the N-S equation

        """
        self.__NS_expression()
        self.solver_parameters()
        self.__solver_init()
        self.__force_init()

    def solver_parameters(self, parameters={}):
        """Set parameters of the solver

        Parameters
        ----------------------------
        parameters : dict, optional
             For available choices for the solver parameters, look at:
             info(NonlinearVariationalSolver.default_parameters(), 1)

        """
        self.solver_para.update(parameters)

    def solve(self, nu=None,sourceterm=None):
        """Solve the problem

        Parameters
        ----------------------------
        nu : Constant(), optional
            kinematic viscosity
        """
        # set nu if it's given
        if nu is not None:
            self.nu = nu
        if sourceterm is not None:
            self.sourceterm = sourceterm

        # update problem
        self.update_problem()
        # call solve
        self.solver.solve()

    def get_force(self, bodymark=None, direction=None):
        """Get the force on the body (lift or drag)

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
        return assemble((self.force[direction]) * self.ds(bodymark))
