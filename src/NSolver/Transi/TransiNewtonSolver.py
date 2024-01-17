#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 20:08:46 2024

@author: bojin
"""

"""This module provides the classes that solve Navier-Stokes equations
"""

from dolfin import *
import numpy as np
from src.NSolver.Steady.SteadyNewtonSolver import NewtonSolver
from src.BasicFunc.Boundary import SetBoundary, SetBoundaryCondition

class DNS_Newton(NewtonSolver):
    """
    Solver of Transient Navier-Stokes equations using Newton method


    Examples
    ----------------------------
    here is a snippet code shows how to use this class

    >>> see test 'CylinderTransiFlow.py'
    """
    def __init__(self, mesh, dt=None, Re=None, sourceterm=None, bodyforce=None, order=(2,1), dim=2, constrained_domain=None, path=None, noise=False):
        """
        

        Parameters
        ----------
        mesh : TYPE
            DESCRIPTION.
        dt : TYPE, optional
            DESCRIPTION. The default is None.
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
        path : TYPE, optional
            DESCRIPTION. The default is None.
        noise : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        NewtonSolver.__init__(self, mesh, Re, sourceterm, bodyforce, order, dim, constrained_domain)
        
    def __funcs(self, num=1):
        """

        Parameters
        ----------
        num : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        self.wp=()
        self.up=()
        self.pp=()
        for x in range(num):
            self.wp+=(self.element.add_functions(),)
            (up, pp) = split(self.wp[x])
            self.up+=(up,)
            self.pp+=(pp,)
            
    def __InitialCondition(self, ic, noise=False):
        """

        Parameters
        ----------
        ic : TYPE
            DESCRIPTION.
        noise : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        if noise is not False:
            vec_pertub = (2 * np.random.rand(self.element.functionspace.dim()) - 1) * noise
            self.wp[0].vector()[:] += vec_pertub
        
        if type(ic) is type("path"):
            timeseries_flow = TimeSeries(ic)
            timeseries_flow.retrieve(self.wp[0].vector(), 0.0)
        elif type(ic) is type(self.wp[0]):
            assign(self.wp[0], ic)
        else:
            info("Wrong Format of Initial Conidtion (Please give a path or function)")
        
    
            
        
        



"""
Abandoned
"""
class DNS_Newton_Solver(SetBoundaryCondition):
    def __init__(self, mesh, boundary, nu=None, dt=None, path=None, noise=False, sourceterm=None):
        element = TaylorHood(mesh = mesh) # initialise finite element space
        element_pre = TaylorHood(mesh = mesh)
        # inherit attributes : functionspace, boundries, bcs
        SetBoundaryCondition.__init__(self, Functionspace=element.functionspace, boundary=boundary)
        # create public attributes
        self.stepcount = 0
        self.nu = nu
        self.dt = dt
        self.path = path
        self.sourceterm = sourceterm
        self.solver_para = {}
        # assign functions for the expression convenience
        # functions
        self.w=element.w
        self.u=element.u
        self.p=element.p
        # pre time step functions
        self.w_pre = element_pre.w
        self.u_pre = element_pre.u
        self.p_pre = element_pre.p
        # test functions
        self.v=element.v
        self.q=element.q
        # trial function
        self.dw=element.tw
        # get FacetNormal and boundary measurement
        self.n = FacetNormal(mesh)
        self.ds = boundary.get_measure()
        self.dim = element.functionspace.dim()
        # get the coordinates of the whole functionspace
        self.Coor_fun = self.functionspace.tabulate_dof_coordinates().reshape((-1, 2))
        # initial condition
        self.__InitialCondition(noise=noise)
        # initialise force
        if nu is not None:
            self.__force_init()

        if dt is not None and nu is not None:
            self.__NS_expression()

    def __InitialCondition(self, noise=False):
        """Assign base flow to the function w
        """
        if self.path is not None:
            timeseries_flow = TimeSeries(self.path)
            timeseries_flow.retrieve(self.w_pre.vector(), 0.0)
        if noise is True:
            vec_pertub = (2 * np.random.rand(self.dim) - 1) * 0.01
            self.w_pre.vector()[:] = self.w_pre.vector()[:] + vec_pertub


    def __NS_expression(self):
        """UFL expression of steady Naviar-Stokes equations in the weak form

        """
        self.F = (Constant(1.0/self.dt) * dot((self.u-self.u_pre),self.v) + self.nu * inner(grad(self.u), grad(self.v)) +
                  inner(dot(self.u, nabla_grad(self.u)), self.v) - self.p * div(self.v) + div(self.u) * self.q) * dx

        if self.sourceterm is not None:
            self.F = self.F - dot(self.sourceterm, self.v) * dx


    def __solver_init(self):
        """Initialise Newton solver

        """

        J = derivative(self.F, self.w, self.dw) # Jacobian matrix
        problem = NonlinearVariationalProblem(self.F, self.w, self.bcs, J) # Nonlinear problem
        self.solver = NonlinearVariationalSolver(problem) # Nonlinear solver
        # update solver parameters
        self.prm = self.solver.parameters
        self.prm.update(self.solver_para)

    def __force_init(self):
        """Force expression (lift and drag)

        """
        I = Identity(self.u.geometric_dimension())
        D = sym(grad(self.u))
        T = -self.p * I + 2 * self.nu * D
        self.force = T * self.n

    def solver_parameters(self, parameters={}):
        """Set parameters of the solver

        Parameters
        ----------------------------
        parameters : dict, optional
             For available choices for the solver parameters, look at:
             info(NonlinearVariationalSolver.default_parameters(), 1)

        """
        self.solver_para.update(parameters)

    def solve(self, nu=None, sourceterm=None, dt=None):
        """Solve the problem

        Parameters
        ----------------------------
        nu : Constant(), optional
            kinematic viscosity
        """
        mark = 0
        # set nu if it's given
        if nu is not None and nu != self.nu:
            self.nu = nu
            self.__force_init()
            self.__NS_expression()
            mark = 1

        if dt is not None and dt != self.dt:
            self.dt = dt
            self.__NS_expression()
            mark = 1

        if sourceterm != self.sourceterm:
            self.sourceterm = sourceterm
            self.__NS_expression()
            mark = 1

        # update problem
        if mark == 1 or self.stepcount == 0:
            self.__solver_init()


        # call solve
        self.solver.solve()
        self.w_pre.assign(self.w)
        self.stepcount += 1

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
