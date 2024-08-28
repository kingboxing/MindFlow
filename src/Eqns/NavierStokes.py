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
    def __init__(self, element, boundary, Re, const_expr=None, time_expr=None):
        """
        
        Parameters
        ----------
        element : TYPE
            DESCRIPTION.
        boundary : TYPE
            DESCRIPTION.
        Re : TYPE, optional
            DESCRIPTION. The default is None.
        const_expr : TYPE, optional
            DESCRIPTION. The default is None.
        time_expr : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.element=element
        self.dim=element.dimension
        self.rho = 1.0 # fixed/ignored in incompressible flows
        # Reynolds number
        self.Re=Re
        # boundary 
        self.ds = boundary.get_measure()
        # boundary normal vec 
        self.n = FacetNormal(element.mesh)
        
        self.__funcalias(element)
        self.__sourceterm(const_expr, time_expr)
    
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
        self.w=element.w
        self.fw=(self.w,)
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
            if type(self.fw[x+1]) is tuple: # for decoupled element
                up = self.fw[x+1][0]
                pp = self.fw[x+1][1]
            else: # for taylorhood element
                (up, pp) = split(self.fw[x+1])
            self.fu+=(up,)
            self.fp+=(pp,)
        
        
    def __sourceterm(self, const_expr, time_expr):
        """
        external forcing

        Parameters
        ----------
        const_expr : TYPE
            for constant expressions
        time_expr : TYPE
            for time-varying expressions

        Returns
        -------
        None.

        """
        # 
        if type(const_expr) is function.constant.Constant:
            self.const_expr = const_expr
        elif const_expr is None:
            self.const_expr = Constant(tuple([0.0]*self.dim))
        else:
            info('Please specify a Constant or Expression as sourceterm')
            
        # 
        if type(time_expr) is function.constant.Constant:
            self.time_expr = time_expr
        elif time_expr is None:
            self.time_expr = Constant(tuple([0.0]*self.dim))
        else:
            info('Please specify a Constant or Expression as sourceterm')
            
    
    def SteadyNonlinear(self): ## no trial functions
        """
        UFL expression of steady nonlinear Naviar-Stokes equations in the weak form
         
        au                  1  
        -- + u * grad(u) + --- grad(p) - nu * laplacian(u) - F = 0 
        at                 rho
        
        div(u)=0
        
        where u is a vector, p is scalar, 
        rho is density ( = 1 here), nu is kinematic viscosity, 
        s is the source term, f is the body force 
        
        au/at = 0 here for steady flows
        
        Returns
        -------
        NS: UFL expression of steady Navier-Stokes Equations in the weak form
        
        F: RHS external force/source term
        
        """
        self.nu = Constant(1.0/self.Re)

        NS = ((inner(dot(self.u, nabla_grad(self.u)), self.v) -
              Constant(1.0 / self.rho) * self.p * div(self.v) +
              self.nu * inner(grad(self.u), grad(self.v)) + 
              div(self.u) * self.q) * dx )
        
        F = self.SourceTerm()
        
        return NS-F
    
    def SteadyLinear(self):
        """
        UFL expression of steady linearised Naviar-Stokes equations in the weak form
        
        Returns
        -------
        None.
        """
        # self.u represents base flow field
        self.nu = Constant(1.0/self.Re)
        LSNS = (inner(dot(self.u, nabla_grad(self.tu)), self.v) +
                inner(dot(self.tu, nabla_grad(self.u)), self.v) -
                Constant(1.0 / self.rho) * self.tp * div(self.v)+
                self.nu * inner(grad(self.tu), grad(self.v))
                + div(self.tu)* self.q) * dx
        
        return LSNS
        
    
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
        TNS : TYPE
            DESCRIPTION.

        """
       
        self.dt = dt
        if implicit is True:
            if order == 1 and scheme == "backward":
                self.__tempfunc(num=order) # add function at previous time step
                TNS = dot((self.tu - self.fu[order]) / Constant( self.dt ) , self.v) * dx  # self.fu[1] at previous time step
        
        if implicit is False:
            if order == 1 and scheme == "backward":
                self.__tempfunc(num=order)
                TNS =  dot((self.u - self.fu[order]) / Constant( self.dt ) , self.v) * dx # self.fu[1] at previous time step
            
        return TNS
    
    def Frequency(self, s):
        """
        time relatived part after fourier/laplace transformation

        Parameters
        ----------
        s : TYPE
            the Laplace variable s is also known as operator variable in the Laplace domain.

        Returns
        -------
        FNS_r : TYPE
            real part, growth rate.
        FNS_i : TYPE
            imag part, frequency.

        """
        omega = np.imag(s)
        sigma = np.real(s)
        
        FNS_i = Constant(omega) * inner(self.tu, self.v) * dx # imaginary part
        FNS_r = Constant(sigma) * inner(self.tu, self.v) * dx # real part
        
        return (FNS_r, FNS_i)
    
    def IPCS(self, dt):
        """
        Implicit Pressure Correction Scheme (seems compressible NS eqns are used)

        Parameters
        ----------
        dt : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.nu = Constant(1.0/self.Re)
        
        # -------------------------------------------------
        # transient part
        order = 1
        transi = self.Transient(dt, order = order) # form transient part
        self.__tempfunc(num=order+1) # add temp function for inner iterations
        # align temp and pre functions
        self.fu[2].assign(self.fu[1])
        self.fu[2].assign(self.fu[1])
        # fw[2]:intermidate step; fw[1]:previous step ; fw[0]: current step
        
        # -------------------------------------------------
        # Define expressions used in variational forms
        U = 0.5 * (self.fu[2] + self.tu)
        # -------------------------------------------------
        # Define variational problem for step 1, Implicit-time integration
        F1 = transi + dot(dot(self.fu[2], nabla_grad(self.fu[2])), self.v) * dx \
             + inner(self.sigma(self.nu, U, self.fp[2]), self.epsilon(self.v)) * dx 
             
        
        L1 = lhs(F1)
        R1_1 = rhs(F1)
        R1_2 = self.SourceTerm()#(dot(self.const_expr, self.v) + dot(self.time_expr, self.v))*dx
        
        # -------------------------------------------------
        # if bcs is not full and freeoutlet is False # gradient of [u, v] in boundary-normal direction is zero # fully developped
        # FBC = self.BoundaryTraction(self.fp[2], U, self.nu, mode = 1) #dot(self.fp[2]*self.n, self.v)*self.ds - dot(self.nu*nabla_grad(U)*self.n, self.v)*self.ds
   
        # LBC = lhs(FBC)
        # RBC = rhs(FBC)
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
        
    def SourceTerm(self):
        """
        

        Returns
        -------
        None.

        """
        F = (inner(self.const_expr, self.v)+ inner(self.time_expr, self.v))* dx
            #(dot(self.const_expr, self.v) + dot(self.time_expr, self.v))*dx
            
        return F
        
        
    def BoundaryTraction(self, p, u, nu, mark=None, mode=None):
        """
        residual term appearing in the standard NS variational problem
        A feature of variational formulations is that the test function v is 
        equired to vanish on the parts of the boundary where the solution u is known

        Parameters
        ----------
        p : TYPE
            DESCRIPTION.
        u : TYPE
            DESCRIPTION.
        nu : TYPE
            DESCRIPTION.
        mark : TYPE, optional
            DESCRIPTION. The default is None.
        mode : TYPE, optional
            DESCRIPTION. The default is None.
        Returns
        -------
        FBC : TYPE
            residual term due to variational form; cancelled by implicit(e.g. free outlet) boundary condition
            test: FBC cancelled by Dirchlet BC and Free BC at Outlet; convergence problem when no BC applied at Boundary. 
                    cause convergence problem if the number of BCs is not equal to that of boundraies

        """
        if mark is None:
            ds0= self.ds
        else:
            ds0= self.ds(mark)
        
        if mode is None or mode == 0:
            FBC = (dot(p*self.n,self.v)-dot(nu*grad(u)*self.n,self.v))*ds0 # from standard incompressible NS eqn
        elif mode == 1:
            FBC = (dot(p*self.n,self.v)-dot(nu*nabla_grad(u)*self.n,self.v))*ds0 # from compressible NS eqn with fully developed assumption + pressure bc
        elif mode == 2:
            FBC = (dot(p*self.n,self.v)-dot(nu*grad(u)*self.n,self.v)-dot(nu*nabla_grad(u)*self.n,self.v))*ds0 # from compressible NS eqn
        elif mode == 3:
            FBC = dot(p*self.n,self.v)*ds0 # from standard incompressible NS eqn with fully developed assumption + pressure bc

        return FBC
    
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

    
    def force_expr(self):
        """
        Force expression (lift and drag)

        Returns
        -------
        force1 : TYPE
            pressure part.
        force2 : TYPE
            stress part.

        """

        self.nu = Constant(1.0/self.Re)
        # I = Identity(self.u.geometric_dimension())
        # D = sym(grad(self.u))
        # T = -self.p * I + 2 * self.nu * D#
        # force = - T * self.n
        
        I = Identity(self.tu.geometric_dimension())
        D = sym(grad(self.tu))
        T1 = -self.tp * I
        T2 = 2.0 * self.nu * D
        force1 = -T1 * self.n
        force2 = -T2 * self.n
        
        return (force1, force2)
    
    def vorticity_expr(self):
        """
        vorticity exprression

        Returns
        -------
        vorticity : TYPE
            DESCRIPTION.

        """
        
        vorticity=curl(self.tu)
        return vorticity
        
        
        

class Compressible:
    """
    pending
    """
    def __init__(self):
        pass
    