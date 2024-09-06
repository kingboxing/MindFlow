#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 22:34:36 2023

@author: bojin
"""

from src.Deps import *

"""
This module provides classes that define different finite elements
"""
class FiniteElementBase:
    """
    Base class for different finite elements used in FEniCS.

    This class defines common attributes and methods shared by different finite elements.
    """

    def __init__(self, mesh=None, dim=2, order=(2, 1), constrained_domain=None):
        """
        Initialize the finite element base class.

        Parameters
        ----------
        mesh : Mesh, optional
            Object created by FEniCS function Mesh. The default is None.
        dim : int, float, optional
            Dimension of the flow field. The default is 2.
        order : tuple, optional
            Order/degree of the element. The default is (2, 1).
        constrained_domain : Sub_Domain, optional
            Constrained subdomain with map function in FEniCS (for periodic condition). The default is None.
        """
        self.mesh = mesh
        self.dim = int(dim)
        self.order = order
        self.constrained_domain = constrained_domain
        self.functionspace = None

    def set_functionspace(self):
        """
        Abstract method to create the function space.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def set_function(self):
        """
        Create TestFunctions, TrialFunctions, and Functions.
        """
        self.tew = TestFunction(self.functionspace)
        self.tw = TrialFunction(self.functionspace)
        self.w = Function(self.functionspace)


class combine_func:
    """
    Combine two separate function spaces.
    """
    def __init__(self, func_space1, func_space2):
        """
        combine two seperate function space

        Parameters
        ----------
        func_space1 : function space
            function space.
        func_space2 : function space
            function space.

        """
        self._func_space = (func_space1, func_space2)
    
    def sub(self, ind):
        """
        Access a subspace.

        Parameters
        ----------
        ind : int
            Index of the subspace.

        Returns
        -------
        FunctionSpace
            The selected sub-function space.

        """
        return self._func_space[ind]
    
    def num_sub_spaces(self):
        """
        Return the number of subspaces.

        Returns
        -------
        int
            Number of subspaces.
        """
        return len(self._func_space)
    
    def dim(self):
        """
        Return the dimensions of the combined function spaces.

        Returns
        -------
        tuple
            Dimensions of the two function spaces.
        """
        return (self._func_space[0].dim(), self._func_space[1].dim())

#%%
class TaylorHood(FiniteElementBase):
    """
    N-D TaylorHood finite element for velocity and pressure spaces.
    Default: Second order for the velocity vector space and first order for the pressure space.

    Attributes
    ----------------------
    functionspace: the finite element function space

    v, q : TestFunctions of velocity vector and pressure
    
    tew : TestFunction vector (v, q)

    tu, tp : TrialFunctions of velocity vector and pressure

    tw : TrialFunction vector (tu, tp)

    u, p : Functions of velocity vector and pressure

    w : Function vector (u, p)

    Examples
    ----------------------
    ...

    """
    def __init__(self, mesh=None, dim=2, order=(2,1), constrained_domain=None):
        """
        

        Parameters
        ----------
        mesh : Mesh, optional
            object created by FEniCS function Mesh. The default is None.
        dim : int, optional
            dimension of the mesh. The default is 2.
        order : tuple, optional
            Order/degree of the TaylorHood element. The default is (2,1).
        constrained_domain : Sub_Domain, optional
            constrained subdomain with map function in FEniCS (for periodic condition). The default is None.

        Returns
        -------
        None.

        """
        super().__init__(mesh, dim, order, constrained_domain)
        self.type = 'TaylorHood'
        self.set_functionspace()
        self.set_function()
        
    def new(self):
        return TaylorHood(self.mesh, self.dim, self.order, self.constrained_domain)

    def set_functionspace(self):
        """
        Create mixed function space: second order vector element space and first order finite element space.

        """
        P2 = VectorElement("Lagrange", self.mesh.ufl_cell(), self.order[0], dim=self.dim) # velocity space
        P1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), self.order[1]) # pressure space
        TH = MixedElement([P2, P1])
        self.functionspace = FunctionSpace(self.mesh, TH, constrained_domain=self.constrained_domain)
        info("Dimension of the function space: %g" % self.functionspace.dim())

    def set_function(self):
        """
        Create TestFunctions (v, q), TrialFunctions tw = (tu, tp) and Functions w = (u, p)

        """
        super().set_function()
        
        self.v, self.q = split(self.tew) # test functions
        self.tu, self.tp = split(self.tw) # trial functions
        self.u, self.p = split(self.w) # functions

    def add_functions(self):
        """
        Return an additional function from the function space.

        """
        return Function(self.functionspace)

#%%
class Decoupled(FiniteElementBase):
    """
    N-D finite element for decoupled velocity and pressure spaces
    Default: Second order for the velocity vector space and first order for the pressure space.

    Attributes
    ----------------------
    functionspace: the finite element function space

    v, q : TestFunctions of velocity vector and pressure
    
    tew : tuple with TestFunction (v, q)

    tu, tp : TrialFunctions of velocity vector and pressure
    
    tw : tuple with TrialFunction (tu, tp)

    u, p : Functions of velocity vector and pressure
    
    w : tuple of Function (u, p)

    Examples
    ----------------------
    ...

    """
    def __init__(self, mesh=None, dim=2, order=(2,1),constrained_domain=(None, None)):
        """
        

        Parameters
        ----------
        mesh : Mesh, optional
            object created by FEniCS function Mesh. The default is None.
        dim : int, optional
            dimension of the mesh. The default is 2.
        order : tuple, optional
            Order/degree of the Decoupled element. The default is (2,1).
        constrained_domain : tuple of Sub_Domain, optional
            constrained subdomain with map function in FEniCS (for periodic condition). The default is [None, None].
            
        Returns
        -------
        None.

        """
        super().__init__(mesh, dim, order, constrained_domain)
        self.type = 'Decoupled'
        self.set_functionspace()
        self.set_function()
        
    def new(self):
        return Decoupled(self.mesh, self.dim, self.order, self.constrained_domain)


    def set_functionspace(self):
        """
        Create separate function spaces for velocity and pressure.

        """
        self.functionspace_V = VectorFunctionSpace(self.mesh, 'P', self.order[0],dim=self.dim, constrained_domain=self.constrained_domain[0])
        self.functionspace_Q = FunctionSpace(self.mesh, 'P', self.order[1], constrained_domain=self.constrained_domain[1])
        self.functionspace = combine_func(self.functionspace_V, self.functionspace_Q)
        info("Dimension of the function space: Vel: %g    Pre: %g" % (self.functionspace_V.dim(),self.functionspace_Q.dim()))

    def set_function(self):
        """
        Create TestFunctions (v, q), TrialFunctions (tu, tp) and Functions w = (u, p)

        """
        self.v = TestFunction(self.functionspace_V)
        self.q = TestFunction(self.functionspace_Q)
        self.tew = (self.v, self.q)

        self.tu = TrialFunction(self.functionspace_V)
        self.tp = TrialFunction(self.functionspace_Q)
        self.tw = (self.tu, self.tp)

        self.u = Function(self.functionspace_V)
        self.p = Function(self.functionspace_Q)
        self.w = (self.u, self.p)

    def add_functions(self):
        """
        Return additional functions for velocity and pressure.
        """

        return (Function(self.functionspace_V), Function(self.functionspace_Q))
#%%

class PoissonPR(FiniteElementBase):
    """
    N-D finite element for solving the Poisson equation.

    Default: First order for one space and zero order for another space.

    Attributes
    ----------------------
    functionspace: the finite element function space

    q, d : TestFunctions 
    
    tew : TestFunction vector (q, d)

    tp, tc : TrialFunctions 
    
    tw : TrialFunction vector (tp, tc)

    p, c : Functions

    w : Function vector (p, c)

    Examples
    ----------------------
    ...

    """
    def __init__(self, mesh=None, order=(1,0),constrained_domain=None):
        """
        

        Parameters
        ----------
        mesh : Mesh, optional
            object created by FEniCS function Mesh. The default is None.
        order : tuple, optional
            Order/degree of the Decoupled element. The default is (1,0).
        constrained_domain : Sub_Domain, optional
            constrained subdomain with map function in FEniCS (for periodic condition). The default is None.
          
        Returns
        -------
        None.

        """
        super().__init__(mesh, dim=mesh.topology().dim(), order=order, constrained_domain=constrained_domain)
        self.type = 'PoissonPR'
        self.set_functionspace()
        self.set_function()
        
    def new(self):
        return PoissonPR(self.mesh, self.dim, self.order, self.constrained_domain)
    
    def set_functionspace(self):
        """
        Create mixed function space: first order Lagrange element and zero order real finite element.

        """
        P1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), self.order[0])
        R = FiniteElement("R", self.mesh.ufl_cell(), self.order[1])
        PR = MixedElement([P1, R])
        self.functionspace = FunctionSpace(self.mesh, PR, constrained_domain=self.constrained_domain)
        
        info("Dimension of the function space: %g" % (self.functionspace.dim()))

    def set_function(self):
        """
        Create TestFunctions (q, d), TrialFunctions tw = (tp, tc)
         and Functions w = (p, c)

        """
        super().set_function()
        self.q, self.d = split(self.tew)
        self.tp, self.tc = split(self.tw)
        self.p, self.c = split(self.w)
        

    def add_functions(self):
        """
        Return an additional function from the function space.
        """
        return Function(self.functionspace)
