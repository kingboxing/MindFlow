#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:07:48 2023

@author: bojin
"""

from src.Deps import *

class Incompressible:
    
    """
    Expressions of different Navier-Stokes Equations
    """
    def __init__(self, element, boundary, Re, sourceterm=None, bodyforce=None):
        """

        Parameters
        ----------
        element : TYPE
            DESCRIPTION.
        boundary : TYPE
            DESCRIPTION.
        Re : TYPE, optional
            DESCRIPTION. The default is None.
        sourceterm : TYPE, optional
            DESCRIPTION. The default is None.
        bodyforce : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.dim=element.dimension
        self.rho = 1.0 # fixed/ignored in incompressible flows
        self.Re=Re
        # boundary 
        self.ds = boundary.get_measure()
        # boundary normal vec 
        self.n = FacetNormal(element.mesh)
        
        self.__funcalias(element)
        self.__external(sourceterm, bodyforce)
    
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
        self.fw=(element.w,)
        self.fu=(self.u,)
        self.fp=(self.p,)
        # test function
        self.v=element.v
        self.q=element.q
        # trial function for linear problems
        self.tu=element.tu
        self.tp=element.tp

        
    def __tempfunc(self, num=1):
        """
        provide extra functions to strore results at previous steps
        self.fw[0]---> time step n+1 (element.w->cueenrt time step)
        
        Parameters
        ----------
        num : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.
        
        

        """
        for x in range(num-len(self.fw)+1):
            self.fw+=(self.element.add_functions(),)
            (up, pp) = split(self.fw[x+1])
            self.fu+=(up,)
            self.fp+=(pp,)
        
        
    def __external(self, sourceterm, bodyforce):
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
        # source term, doesn't change outlet boundary conditions, not balanced by pressure #ignore
        if type(sourceterm) is function.constant.Constant:
            self.sourceterm = sourceterm
        elif sourceterm is None:
            self.sourceterm = Constant(tuple([0.0]*self.dim))
        else:
            info('Please specify a Constant or Expression for sourceterm')
            
        # body force, may change outlet boundary conditions, balanced by pressure #ignore
        if type(bodyforce) is function.constant.Constant:
            self.bodyforce = bodyforce
        elif bodyforce is None:
            self.bodyforce = Constant(tuple([0.0]*self.dim))
        else:
            info('Please specify a Constant or Expression for bodyforce')
        

    
    def SteadyNonlinear(self): ## no trial functions
        """
        UFL expression of steady nonlinear Naviar-Stokes equations in the weak form
         
        au                  1  
        -- + u * grad(u) + --- grad(p) - nu * laplacian(u) - (s + f) = 0 
        at                 rho
        
        div(u)=0
        
        where u is a vector, p is scalar, 
        rho is density ( = 1 here), nu is kinematic viscosity, 
        s is the source term, f is the body force 
        
        au/at = 0 here for steady flows
        
        Returns
        -------
        F: UFL expression of steady Navier-Stokes Equations in the weak form
        
        """
        self.nu = Constant(1.0/self.Re)

        F = ((inner(dot(self.u, nabla_grad(self.u)), self.v) -
              Constant(1.0 / self.rho) * self.p * div(self.v) +
              self.nu * inner(grad(self.u), grad(self.v)) -
              inner(self.sourceterm, self.v) - inner(self.bodyforce, self.v) + 
              div(self.u) * self.q) * dx )
        
        return F
    
    def SteadyLinear(self):
        """
        UFL expression of steady linearised Naviar-Stokes equations in the weak form
        
        Returns
        -------
        None.
        """
        pass
    
    def Transient(self, dt, scheme="backward", order=1, implicit=True):
        """
        UFL expression of time discretization scheme in the weak form
        
        first order Backward scheme:
        au   u - u_p
        -- = ------
        at     dt

        Parameters
        ----------
        dt : TYPE
            DESCRIPTION.
        scheme : TYPE, optional
            DESCRIPTION. The default is "backward".
        order : TYPE, optional
            DESCRIPTION. The default is 1.
        implicit : TYPE, optional
            DESCRIPTION. The default is True.
            explicit for newton method expression

        Returns
        -------
        F : TYPE
            DESCRIPTION.

        """
       
        self.dt = dt
        if implicit is True:
            if order == 1 and scheme == "backward":
                self.__tempfunc(num=order) # add function at previous time step
                F = dot((self.tu - self.fu[order]) / Constant( self.dt ) , self.v) * dx  # self.fu[1] at previous time step
        
        if implicit is False:
            if order == 1 and scheme == "backward":
                self.__tempfunc(num=order)
                F =  dot((self.u - self.fu[order]) / Constant( self.dt ) , self.v) * dx # self.fu[1] at previous time step
            
        return F
    
    def Frequency(self):
        """
        time relatived part after fourier/laplace transformation

        Returns
        -------
        None.

        """
        pass
    
    def IPCS(self, dt):
        """
        Implicit Pressure Correction Scheme

        Parameters
        ----------
        dt : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Reynolds number
        self.nu = Constant(1.0/self.Re)
        # -------------------------------------------------
        # transient part
        order = 1
        transi = self.Transient(dt, order = order) # form transient part
        self.__tempfunc(num=order+1) # add temp function for inner iterations
        # align temp and pre functions
        self.fu[2].assign(self.fu[1])
        self.fu[2].assign(self.fu[1])
        # fw[2]:intermidate step; fw[1]:previous step
        
        
        # -------------------------------------------------
        # Define expressions used in variational forms
        U = 0.5 * (self.fu[2] + self.tu)
        # -------------------------------------------------
        # Define variational problem for step 1, Implicit-time integration
        F1 = transi + dot(dot(self.fu[2], nabla_grad(self.fu[2])), self.v) * dx \
             + inner(self.sigma(self.nu, U, self.fp[2]), self.epsilon(self.v)) * dx 
        
        #%% pending # if freeoutlet is False
        # if 'freeoutlet' in locals():
        F1 += dot(self.fp[2]*self.n, self.v)*self.ds - dot(self.nu*nabla_grad(U)*self.n, self.v)*self.ds
        #%%
        L1 = lhs(F1)
        R1_1 = rhs(F1)
        R1_2 = (dot(self.sourceterm, self.v) + dot(self.bodyforce, self.v))*dx
        
        # -------------------------------------------------
        # Define variational problem for step 2
        L2 = dot(nabla_grad(self.tp), nabla_grad(self.q)) * dx
             
        R2_1 = dot(nabla_grad(self.tp), nabla_grad(self.q)) * dx 
        R2_2 = - Constant(1.0 / dt) * div(self.tu) * self.q * dx
        
        # -------------------------------------------------
        # Define variational problem for step 3
        L3 = dot(self.tu, self.v) * dx
        
        R3_1 = dot(self.tu, self.v) * dx 
        R3_2 = - Constant(self.dt) * dot(nabla_grad(self.tp), self.v) * dx
        
        return (L1,L2,L3), ((R1_1, R1_2), (R2_1, R2_2), (R3_1,R3_2))
        
        
        
    def epsilon(self, u):
        """
        Define symmetric gradient

        Parameters
        ----------
        u : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return sym(nabla_grad(u))

    def sigma(self, nu, u, p):
        """
        

        Parameters
        ----------
        nu : TYPE
            DESCRIPTION.
        u : TYPE
            DESCRIPTION.
        p : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return 2 * nu * self.epsilon(u) - p * Identity(len(u))

    
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
    