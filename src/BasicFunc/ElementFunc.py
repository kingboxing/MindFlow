#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 22:34:36 2023

@author: bojin
"""

from dolfin import *

"""
This module provides classes that define different finite elements
"""


class TaylorHood:
    """2D TaylorHood finite element
    Second order for the velocity vector space and first
    order for the pressure space

    Parameters
    ----------------------
    mesh : object created by FEniCS function Mesh
        mesh of the flow field

    Attributes
    ----------------------
    functionspace: the finite element function space

    v, q : TestFunctions of velocity vector and pressure

    tu, tp : TrialFunctions of velocity vector and pressure

    tw : TrialFunction vector (tu, tp)

    u, p : Functions of velocity vector and pressure

    w : Function vector (u, p)

    Examples
    ----------------------
    >>> from MindFlow.BasicFunc.ElementFunc import TaylorHood
    >>> from dolfin import *
    >>> mesh = Mesh("mesh.xml")
    >>> element = TaylorHood(mesh = mesh)
    >>> element.functionspace
    FunctionSpace(Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1),
     dim=2), 0), MixedElement(VectorElement(FiniteElement('Lagrange', triangle, 2),
      dim=2), FiniteElement('Lagrange', triangle, 1)))

    """
    def __init__(self, mesh=None,dim=2, order=(2,1), constrained_domain=None):
        self.type = 'TaylorHood'
        self.mesh = mesh
        self.dimension = dim
        self.order=order
        self.constrained_domain=constrained_domain
        self.set_functionspace()
        self.set_functions()
         

    def set_functionspace(self):
        """Create mixed function space
        Second order vector element space and first order finite element space

        """
        P2 = VectorElement("Lagrange", self.mesh.ufl_cell(), self.order[0],dim=self.dimension)
        P1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), self.order[1])
        TH = MixedElement([P2, P1])
        self.functionspace = FunctionSpace(self.mesh, TH, constrained_domain=self.constrained_domain)
        info("Dimension of the function space: %g" % self.functionspace.dim())

    def set_functions(self):
        """Create TestFunctions (v, q), TrialFunctions tw = (tu, tp)
         and Functions w = (u, p)

        """
        (self.v, self.q) = TestFunctions(self.functionspace)
        self.tw = TrialFunction(self.functionspace)
        (self.tu, self.tp) = split(self.tw)
        self.w = Function(self.functionspace)
        (self.u, self.p) = split(self.w)

    def add_functions(self):
        return Function(self.functionspace)

class SplitPV:
    def __init__(self, mesh=None,dim=2,order=(2,1),constrained_domain=[None,None]):
        self.type = 'Split'
        self.mesh = mesh
        self.dimension=dim
        self.order=order
        self.constrained_domain=constrained_domain
        self.set_functionspace()
        self.set_functions()


    def set_functionspace(self):
        """Create split function space
        Second order vector element space and first order finite element space

        """
        self.functionspace_V = VectorFunctionSpace(self.mesh, 'P', self.order[0],dim=self.dimension, constrained_domain=self.constrained_domain[0])
        self.functionspace_Q = FunctionSpace(self.mesh, 'P', self.order[1], constrained_domain=self.constrained_domain[1])
        info("Dimension of the function space: V: %g    Q: %g" % (self.functionspace_V.dim(),self.functionspace_Q.dim()))

    def set_functions(self):
        """Create TestFunctions (v, q), TrialFunctions (tu, tp)
         and Functions w = (u, p)

        """
        self.v = TestFunction(self.functionspace_V)
        self.q = TestFunction(self.functionspace_Q)

        self.tu = TrialFunction(self.functionspace_V)
        self.tp = TrialFunction(self.functionspace_Q)

        self.u = Function(self.functionspace_V)
        self.p = Function(self.functionspace_Q)

    def add_functions(self):
        return Function(self.functionspace_V), Function(self.functionspace_Q)
        
class PoissonPR:
    def __init__(self, mesh=None,dim=2,order=(1,0),constrained_domain=None):
        self.type = 'PoissonPR'
        self.mesh = mesh
        self.dimension=dim
        self.order=order
        self.constrained_domain=constrained_domain
        self.set_functionspace()
        self.set_functions()


    def set_functionspace(self):
        """Create split function space
        first order finite element space and zero order real finite element

        """
        P1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), self.order[0])
        R = FiniteElement("R", self.mesh.ufl_cell(), self.order[1])
        PR = MixedElement([P1, R])
        self.functionspace = FunctionSpace(self.mesh, PR, constrained_domain=self.constrained_domain)
        
        info("Dimension of the function space: %g" % (self.functionspace.dim()))

    def set_functions(self):
        """Create TestFunctions (q, d), TrialFunctions tw = (tp, tc)
         and Functions w = (p, c)

        """
        (self.q, self.d) = TestFunctions(self.functionspace)
        self.tw = TrialFunction(self.functionspace)
        (self.tp, self.tc) = split(self.tw)
        self.w = Function(self.functionspace)
        (self.p, self.c) = split(self.w)

    def add_functions(self):
        return Function(self.functionspace)
