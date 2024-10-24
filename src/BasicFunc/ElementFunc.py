#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides classes that define different finite elements based on FEniCS.
It includes base classes and specific implementations like TaylorHood, Decoupled, and PoissonPR
elements. These classes are designed to facilitate the creation of function spaces, test functions,
trial functions, and solution functions for various finite element methods.

Classes
-------
- FiniteElementBase: Base class for different finite elements.
- combine_func: Class to combine two separate function spaces.
- TaylorHood: Class for N-D Taylor-Hood finite elements.
- Decoupled: Class for N-D decoupled velocity and pressure spaces.
- PoissonPR: Class for N-D finite elements for solving the Poisson equation.

Examples
--------
To use the TaylorHood element:

    mesh = Mesh("mesh.xml")
    element = TaylorHood(mesh)
    v, q = element.v, element.q  # Test functions
    u, p = element.u, element.p  # Solution functions

"""

from ..Deps import *

class FiniteElementBase:
    """
    Base class for different finite elements used in FEniCS.

    This class defines common attributes and methods shared by different finite elements.
    It serves as a template for creating specific finite element classes by providing
    methods to set up function spaces and functions.

    Attributes
    ----------
    mesh : Mesh or None
        The mesh on which the finite element is defined.
    dim : int
        Dimension of the flow field.
    order : tuple of int
        Order or degree of the finite elements.
    constrained_domain : SubDomain or None
        Constrained subdomain for periodic boundary conditions.
    functionspace : FunctionSpace or None
        The finite element function space.
    """

    def __init__(self, mesh=None, dim=2, order=(2, 1), constrained_domain=None):
        """
        Initialize the finite element base class.

        Parameters
        ----------
        mesh : Mesh, optional
            The mesh on which the finite element is defined. Default is None.
        dim : int, float, optional
            Dimension of the flow field. The default is 2.
        order : tuple of int, optional
            Order or degree of the finite elements. Default is (2, 1).
        constrained_domain : Sub_Domain, optional
            Constrained subdomain with map function in FEniCS (for periodic condition). The default is None.
        """
        self.mesh = mesh
        self.dim = dim
        self.order = order
        self.constrained_domain = constrained_domain
        self.functionspace = None

    def set_functionspace(self):
        """
        Abstract method to create the function space.

        This method should be implemented by subclasses to define the specific
        function spaces for different finite elements.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def set_function(self):
        """
        Create test functions, trial functions, and solution functions.

        This method initializes the test functions (`tew`), trial functions (`tw`),
        and solution functions (`w`) for the finite element.
        """
        self.tew = TestFunction(self.functionspace)
        self.tw = TrialFunction(self.functionspace)
        self.w = Function(self.functionspace)


class combine_func:
    """
    Class to combine two separate function spaces.

    This class allows for the combination of two function spaces into a single
    object that can be used to access individual subspaces and their properties.

    Attributes
    ----------
    _func_space : tuple
        Tuple containing the two function spaces.
    """

    def __init__(self, func_space1, func_space2):
        """
        Initialize the combined function spaces.

        Parameters
        ----------
        func_space1 : FunctionSpace
            The first function space.
        func_space2 : FunctionSpace
            The second function space.
        """
        self._func_space = (func_space1, func_space2)

    def sub(self, ind):
        """
        Access a subspace by index.

        Parameters
        ----------
        ind : int
            Index of the subspace (0 or 1).

        Returns
        -------
        FunctionSpace
            The selected sub-function space.

        Raises
        ------
        IndexError
            If the index is out of bounds.
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
    N-D Taylor-Hood finite element for velocity and pressure spaces.

    This element uses a second-order Lagrange vector element for velocity and
    a first-order Lagrange element for pressure in default.

    Attributes
    ----------
    functionspace : FunctionSpace
        The mixed finite element function space.
    v : Function or tuple
        Test function for velocity.
    q : Function or tuple
        Test function for pressure.
    tew : tuple
        Tuple containing test functions (v, q).
    tu : Function or tuple
        Trial function for velocity.
    tp : Function or tuple
        Trial function for pressure.
    tw : tuple
        Tuple containing trial functions (tu, tp).
    u : Function
        Solution function for velocity.
    p : Function
        Solution function for pressure.
    w : tuple
        Tuple containing solution functions (u, p).

    Examples
    --------
        element = TaylorHood(mesh)
        v, q = element.v, element.q  # Test functions
        u, p = element.u, element.p  # Solution functions
    """

    def __init__(self, mesh=None, dim=2, order=(2, 1), constrained_domain=None):
        """
        Initialize the Taylor-Hood finite element.

        Parameters
        ----------
        mesh : Mesh, optional
            The mesh on which the finite element is defined. Default is None.
        dim : int, optional
            Dimension of the flow field. Default is 2.
        order : tuple of int, optional
            Order or degree of the Taylor-Hood element. Default is (2, 1).
        constrained_domain : SubDomain, optional
            Constrained subdomain for periodic boundary conditions. Default is None.

        Returns
        -------
        None.

        """
        super().__init__(mesh, dim, order, constrained_domain)
        self.type = 'TaylorHood'
        self.set_functionspace()
        self.set_function()

    def new(self):
        """
        Create a new instance of the TaylorHood class with the same parameters.

        Returns
        -------
        TaylorHood
            A new instance of the TaylorHood class.
        """
        return TaylorHood(self.mesh, self.dim, self.order, self.constrained_domain)

    def set_functionspace(self):
        """
        Create the mixed function space for velocity and pressure.

        This method defines a second-order vector element space for velocity and
        a first-order finite element space for pressure in default, then combines them into
        a mixed function space.
        """
        P2 = VectorElement("Lagrange", self.mesh.ufl_cell(), self.order[0], dim=self.dim)  # velocity space
        P1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), self.order[1])  # pressure space
        TH = MixedElement([P2, P1])
        self.functionspace = FunctionSpace(self.mesh, TH, constrained_domain=self.constrained_domain)
        info("Dimension of the function space: %g" % self.functionspace.dim())

    def set_function(self):
        """
        Create test functions, trial functions, and solution functions.

        This method initializes the test functions (`v`, `q`), trial functions (`tu`, `tp`),
        and solution functions (`u`, `p`) for the finite element.
        """
        super().set_function()

        self.v, self.q = split(self.tew)  # test functions
        self.tu, self.tp = split(self.tw)  # trial functions
        self.u, self.p = split(self.w)  # functions

    def add_functions(self):
        """
        Return an additional function from the function space.

        Returns
        -------
        Function
            A new function in the mixed function space.
        """
        return Function(self.functionspace)


#%%
class Decoupled(FiniteElementBase):
    """
    N-D finite element for decoupled velocity and pressure spaces.

    This element uses separate function spaces for velocity and pressure, allowing
    for decoupled computations.

    Attributes
    ----------
    functionspace : combine_func
        The combined function space object containing velocity and pressure spaces.
    v : Function
        Test function for velocity.
    q : Function
        Test function for pressure.
    tew : tuple
        Tuple containing test functions (v, q).
    tu : Function
        Trial function for velocity.
    tp : Function
        Trial function for pressure.
    tw : tuple
        Tuple containing trial functions (tu, tp).
    u : Function
        Solution function for velocity.
    p : Function
        Solution function for pressure.
    w : tuple
        Tuple containing solution functions (u, p).

    Examples
    --------
        element = Decoupled(mesh)
        v, q = element.v, element.q  # Test functions
        u, p = element.u, element.p  # Solution functions
    """

    def __init__(self, mesh=None, dim=2, order=(2, 1), constrained_domain=(None, None)):
        """
        Initialize the decoupled finite element.

        Parameters
        ----------
        mesh : Mesh, optional
            The mesh on which the finite element is defined. Default is None.
        dim : int, optional
            Dimension of the flow field. Default is 2.
        order : tuple of int, optional
            Order or degree of the finite elements. Default is (2, 1).
        constrained_domain : tuple of SubDomain, optional
            Tuple containing constrained subdomains for velocity and pressure.
            Default is (None, None).

        """
        super().__init__(mesh, dim, order, constrained_domain)
        self.type = 'Decoupled'
        self.set_functionspace()
        self.set_function()

    def new(self):
        """
        Create a new instance of the Decoupled class with the same parameters.

        Returns
        -------
        Decoupled
            A new instance of the Decoupled class.
        """
        return Decoupled(self.mesh, self.dim, self.order, self.constrained_domain)

    def set_functionspace(self):
        """
        Create decoupled function spaces for velocity and pressure.

        This method defines individual function spaces for velocity (`functionspace_V`)
        and pressure (`functionspace_Q`) and combines them using the `combine_func` class.
        """
        self.functionspace_V = VectorFunctionSpace(self.mesh, 'P', self.order[0], dim=self.dim,
                                                   constrained_domain=self.constrained_domain[0])
        self.functionspace_Q = FunctionSpace(self.mesh, 'P', self.order[1],
                                             constrained_domain=self.constrained_domain[1])
        self.functionspace = combine_func(self.functionspace_V, self.functionspace_Q)
        info("Dimension of the function space: Vel: %g    Pre: %g" % (
        self.functionspace_V.dim(), self.functionspace_Q.dim()))

    def set_function(self):
        """
        Create test functions, trial functions, and solution functions.

        This method initializes the test functions (`v`, `q`), trial functions (`tu`, `tp`),
        and solution functions (`u`, `p`) for the finite element.
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

        Returns
        -------
        tuple
            A tuple containing new functions for velocity and pressure.
        """

        return (Function(self.functionspace_V), Function(self.functionspace_Q))


#%%

class PoissonPR(FiniteElementBase):
    """
    N-D finite element for solving the Poisson equation.

    This element is designed for Poisson problems and uses a mixed function space
    consisting of a Lagrange element and a real finite element.

    Attributes
    ----------
    functionspace : FunctionSpace
        The mixed finite element function space.
    q : Function
        Test function for the primary variable.
    d : Function
        Test function for the secondary variable.
    tew : tuple
        Tuple containing test functions (q, d).
    tp : Function
        Trial function for the primary variable.
    tc : Function
        Trial function for the secondary variable.
    tw : tuple
        Tuple containing trial functions (tp, tc).
    p : Function
        Solution function for the primary variable.
    c : Function
        Solution function for the secondary variable.
    w : tuple
        Tuple containing solution functions (p, c).

    Examples
    --------
         element = PoissonPR(mesh)
         q, d = element.q, element.d  # Test functions
         p, c = element.p, element.c  # Solution functions

    """

    def __init__(self, mesh=None, order=(1, 0), constrained_domain=None):
        """
        Initialize the PoissonPR finite element.

        Parameters
        ----------
        mesh : Mesh, optional
            The mesh on which the finite element is defined. Default is None.
        order : tuple of int, optional
            Order or degree of the finite elements. Default is (1, 0).
        constrained_domain : SubDomain, optional
            Constrained subdomain for periodic boundary conditions. Default is None.

        """
        super().__init__(mesh, dim=mesh.topology().dim(), order=order, constrained_domain=constrained_domain)
        self.type = 'PoissonPR'
        self.set_functionspace()
        self.set_function()

    def new(self):
        """
        Create a new instance of the PoissonPR class with the same parameters.

        Returns
        -------
        PoissonPR
            A new instance of the PoissonPR class.
        """
        return PoissonPR(self.mesh, self.dim, self.order, self.constrained_domain)

    def set_functionspace(self):
        """
        Create the mixed function space for solving the Poisson equation.

        This method defines a first-order Lagrange element and a zero-order real finite element,
        then combines them into a mixed function space.
        """
        P1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), self.order[0])
        R = FiniteElement("R", self.mesh.ufl_cell(), self.order[1])
        PR = MixedElement([P1, R])
        self.functionspace = FunctionSpace(self.mesh, PR, constrained_domain=self.constrained_domain)

        info("Dimension of the function space: %g" % (self.functionspace.dim()))

    def set_function(self):
        """
        Create test functions, trial functions, and solution functions.

        This method initializes the test functions (`q`, `d`), trial functions (`tp`, `tc`),
        and solution functions (`p`, `c`) for the finite element.
        """
        super().set_function()
        self.q, self.d = split(self.tew)  # Test functions
        self.tp, self.tc = split(self.tw)  # Trial functions
        self.p, self.c = split(self.w)  # Solution functions

    def add_functions(self):
        """
        Return an additional function from the function space.

        Returns
        -------
        Function
            A new function in the mixed function space.
        """
        return Function(self.functionspace)
