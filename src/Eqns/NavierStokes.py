#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:07:48 2023

@author: bojin
"""

from dolfin import *

class Incompressible:
    """
    Expressions of different Navier-Stokes Equations
    """
    def __init__(self, element, boundary, nu=None,sourceterm=None, bodyforce=None):
        """
        

        Parameters
        ----------
        element : TYPE
            DESCRIPTION.
        boundary : TYPE
            DESCRIPTION.
        nu : TYPE, optional
            DESCRIPTION. The default is None.
        sourceterm : TYPE, optional
            DESCRIPTION. The default is None.
        bodyforce : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.rho = 1
        self.nu = nu
        # boundary 
        self.ds = boundary.get_measure()
        
        self.__funcalias(element)
        self.__constantterm(sourceterm, bodyforce)
    
    def __funcalias(self, element):
        """
        alias of function-space and other functions

        Parameters
        ----------
        element : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # solved function
        self.u=element.u
        self.p=element.p
        # test function
        self.v=element.v
        self.q=element.q
        # boundary normal vec 
        self.n = FacetNormal(element.mesh)
        
    def __constantterm(self, sourceterm, bodyforce):
        """
        theoretically sourceterm = bodyforce. 
        here may be used differently for flexibility

        Parameters
        ----------
        sourceterm : TYPE
            DESCRIPTION.
        bodyforce : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
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
        

    
    def SteadyNonlinear(self):
        """
        UFL expression of steady nonlinear Naviar-Stokes equations in the weak form
         
        au                  1  
        -- + u * grad(u) + --- grad(p) - nu * laplacian(u) - (s + f) = 0 
        at                 rho
        
        div(u)=0
        
        where u is a vector, p is scalar, 
        rho is density ( = 1 here), nu is kinematic viscosity, 
        s is the source term, f is the body force 
        
        au/at = 0 for steady flow
        
        Returns
        -------
        None.
        
        """
        
        F = (inner(dot(self.u, nabla_grad(self.u)), self.v) -
                  Constant(1.0 / self.rho) * self.p * div(self.v) +
                  self.nu * inner(grad(self.u), grad(self.v)) -
                  inner(self.sourceterm, self.v) - 
                  inner(self.bodyforce, self.v) +
                  div(self.u) * self.q * dx )
        
        return F
    
    def SteadyLinear(self):
        """
        UFL expression of steady linearised Naviar-Stokes equations in the weak form
        
        Returns
        -------
        None.
        """
        pass
    
    def Transient(self):
        """
        time relatived part 
        
        Returns
        -------
        None.
        """

        pass
    
    def Frequency(self):
        """
        time relatived part after fourier/laplace transformation

        Returns
        -------
        None.

        """
        pass
    
    def IPCS(self):
        
        pass
    
    def force_init(self):
        """
        Force expression (lift and drag)
        
        Returns
        -------
        None.
        """
        I = Identity(self.u.geometric_dimension())
        D = sym(grad(self.u))
        T = -self.p * I + 2 * self.nu * D#
        force = - T * self.n
        
        return force
        
        

class Compressible:
    """
    pending
    """
    def __init__(self):
        pass
    