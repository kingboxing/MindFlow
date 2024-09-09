#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:07:48 2023

@author: bojin
"""

from src.Deps import *

class Incompressible:
    
    """
    Class for defining various forms of the incompressible Navier-Stokes equations.
    """
    def __init__(self, element, boundary, Re, const_expr=None, time_expr=None):
        """
        Initialize the incompressible Navier-Stokes solver.

        Parameters
        ----------
        element : object
            Finite element object defining the solution space.
        boundary : object
            SetBoundary object.
        Re : float
            Reynolds number.
        const_expr : Expression, Function or Constant, optional
            The time-invariant source term for the flow field. Default is None.
        time_expr : Expression, Function or Constant, optional
            Time-dependent source term for the flow field. Default is None.
        """
        
        self.element=element
        self.dim=element.dim
        self.rho = 1.0 # Density is fixed at 1 for incompressible flows
        self.Re=Re # Reynolds number
        self.ds = boundary.get_measure() # boundary 
        self.n = FacetNormal(element.mesh) # boundary normal vec, outward direction
        
        
        self._initialize_functions(element)
        self._initialize_source_terms(const_expr, time_expr)

    
    def _initialize_functions(self, element):
        """
        Initialize function aliases for the finite element problem.

        Parameters
        ----------
        element : object
            Finite element object.
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

    def _add_temporary_functions(self, num=1):
        """
        Add temporary functions for storing results from previous steps.
        self.fw[0]---> time step n+1 (element.w -> current time step)

        Parameters
        ----------
        num : int, optional
            Number of temporary functions to add. Default is 1.
        """
        for _ in range(num - len(self.fw) + 1): # number of additional funcs: consider time discretization, intermediate state
            temp_func = self.element.add_functions()
            self.fw += (temp_func,)
            up, pp = split(temp_func) if not isinstance(temp_func, tuple) else temp_func
            self.fu += (up,)
            self.fp += (pp,)

    def _initialize_source_terms(self, const_expr, time_expr):
        """
        Initialize source terms for the Navier-Stokes equations.

        Parameters
        ----------
        const_expr : Expression, Function or Constant, optional
            The time-invariant source term for the flow field. Default is None.
        time_expr : Expression, Function or Constant, optional
            Time-dependent source term for the flow field. Default is None.
        """
        expr_type= (function.function.Function, function.expression.Expression, function.constant.Constant)
        self.const_expr = const_expr if isinstance(const_expr, expr_type) else Constant((0.0,) * self.dim)
        self.time_expr = time_expr if isinstance(time_expr, expr_type) else Constant((0.0,) * self.dim)
    
    def SteadyNonlinear(self): ## no trial functions
        """
        Define the steady nonlinear Navier-Stokes equations in weak form.
        
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
        NS : UFL expression
            The steady nonlinear Navier-Stokes equations in the weak form.
        F : UFL expression
            The external force/source term.
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
        Define the steady linearized Navier-Stokes equations in weak form.

        Returns
        -------
        SLNS : UFL expression
            The steady linearized Navier-Stokes equations.
        """

        # self.u represents base flow field
        self.nu = Constant(1.0/self.Re)
        SLNS = (inner(dot(self.u, nabla_grad(self.tu)), self.v) +
                inner(dot(self.tu, nabla_grad(self.u)), self.v) -
                Constant(1.0 / self.rho) * self.tp * div(self.v)+
                self.nu * inner(grad(self.tu), grad(self.v))
                + div(self.tu)* self.q) * dx
        
        return SLNS
    
    def QuasiSteadyLinear(self, sz):
        """
        Define the steady linearized Navier-Stokes equations of Quasi-dimension in weak form.
        
        U(x, y, z, t) = U(x, y) e^{iwt} e^{ikt}
        
        2.5D/1.5D N-S equation
        
        3D problem -> 2D problem
        2D problem -> 1D problem
        
        3D problem -> 1D problem

        Parameters
        ----------
        sz : complex or tuple/list of complex, optional
            Spatial frequency parameters for quasi-analysis of the flow field. Default is None.

        Returns
        -------
        None.

        """
        # self.nu = Constant(1.0/self.Re)
        # if isinstance(sz, (tuple, list)): # 3D -> 1D
        #     indz = self.element.dim - 2
        #     lambda_y = Constant(np.imag(sz[0]))
        #     lambda_z = Constant(np.imag(sz[1]))
            
        # elif isinstance(sz, (complex, np.complexfloating)): # 3D/2d -> 2D/1D
        #     indz = self.element.dim - 1 # mesh/baseflow is 2D, vel is 3D
        #     lambda_z = Constant(np.imag(sz)) # get imag part 
        #     QSLNS_i = Constant(1.0 / self.rho) * lambda_z*self.tp*self.v[indz]*dx + lambda_z*self.tu[indz]*self.q*dx # imag part
            
        #     QSLNS_r = (inner((self.u[0] * nabla_grad(self.tu)[0,:])+(self.u[1] * nabla_grad(self.tu)[1,:]), self.v) +
        #                inner((self.tu[0] * nabla_grad(self.u)[0,:])+(self.tu[1] * nabla_grad(self.u)[1,:]), self.v) -
        #                Constant(1.0 / self.rho) * (grad(self.v[0])[0]+grad(self.v[1])[1]) * self.tp + 
        #                self.nu * inner(grad(self.tu), grad(self.v)) + self.nu*lambda_z*lambda_z*inner(self.tu,self.v) +
        #                (grad(self.tu[0])[0]+grad(self.tu[1])[1]) * self.q) * dx # real part
        #     # QSLNS_r = (inner(dot(self.u[0:indz], nabla_grad(self.tu)[0:indz, :]), self.v) +
        #     #            inner(dot(self.tu[0:indz], nabla_grad(self.u)[0:indz, :]), self.v) -
        #     #            Constant(1.0 / self.rho)  * self.tp * div(self.v[0:indz]) + 
        #     #            self.nu * inner(grad(self.tu), grad(self.v)) + self.nu*lambda_z*lambda_z*inner(self.tu,self.v) +
        #     #            div(self.tu[0:indz]) * self.q) * dx # real part
            
        #     return QSLNS_r, QSLNS_i
        
        self.nu = Constant(1.0/self.Re)
        if isinstance(sz, (complex, np.complexfloating)): # 3D/2d -> 2D/1D
            sz = (sz, )
        elif isinstance(sz, (tuple, list)) and len(sz)>2: 
            raise ValueError('The maximum dimension reduction is limited to two.')
        
        indz = self.element.dim - len(sz)
        QSLNS_i = 0
        QSLNS_r = 0
        
        for lambda_ in sz:
            QSLNS_i += Constant(1.0 / self.rho) * lambda_*self.tp*self.v[indz]*dx + lambda_*self.tu[indz]*self.q*dx # imag part
            QSLNS_r += self.nu*lambda_*lambda_*inner(self.tu,self.v)
            
        QSLNS_r = (inner(dot(self.u[0:indz], nabla_grad(self.tu)[0:indz, :]), self.v) +
                    inner(dot(self.tu[0:indz], nabla_grad(self.u)[0:indz, :]), self.v) -
                    Constant(1.0 / self.rho)  * self.tp * div(self.v[0:indz]) + 
                    self.nu * inner(grad(self.tu), grad(self.v)) +
                    div(self.tu[0:indz]) * self.q) * dx # real part
            
        return QSLNS_r, QSLNS_i
            
            
        
    def Frequency(self, s):
        """
        Define the transient part of the frequency-domain Navier-Stokes equations after Fourier/Laplace transformation.

        Parameters
        ----------
        s : complex
            The Laplace variable s, representing the growth rate and frequency.

        Returns
        -------
        FNS_r : UFL expression
            The real part (growth rate) of the transient part.
        FNS_i : UFL expression
            The imaginary part (frequency) of the transient part.
        """
        self.s = s
        omega, sigma = np.imag(s), np.real(s)
        
        if omega != 0:
            FNS_i = Constant(omega) * inner(self.tu, self.v) * dx # imaginary part
        else:
            FNS_i = 0
            
        if sigma != 0:
            FNS_r = Constant(sigma) * inner(self.tu, self.v) * dx # real part
        else:
            FNS_r = 0
        
        return (FNS_r, FNS_i)
    
    def Transient(self, dt, scheme="backward", order=1, implicit=True):
        """
        Define the transient part using a time discretization scheme.

        Parameters
        ----------
        dt : float
            Time step size.
        scheme : str, optional
            Time discretization scheme. Default is "backward".
        order : int, optional
            Order of the scheme. Default is 1.
        implicit : bool, optional
            Whether to use an implicit scheme. Default is True.

        Returns
        -------
        TNS : UFL expression
            The transient part (first order time derivative) in the Navier-Stokes equations.
        """

       
        self.dt = dt
        self._add_temporary_functions(order) # add function at previous time step
        prev_u = self.fu[order]

        if implicit:
            if order == 1 and scheme == "backward":
                TNS = dot((self.tu - prev_u) / Constant( dt ) , self.v) * dx  # self.fu[1] at previous time step
        else:
            if order == 1 and scheme == "backward":
                TNS =  dot((self.u - prev_u) / Constant( dt ) , self.v) * dx # self.fu[1] at previous time step
            
        return TNS
    
    def IPCS(self, dt):
        """
        Define the Implicit Pressure Correction Scheme (IPCS) for transient Navier-Stokes equations.
        (seems compressible NS eqns are used)
        
        Parameters
        ----------
        dt : float
            Time step size.

        Returns
        -------
        tuple
            LHS and RHS expressions for the three-step IPCS method.
        """
        
        self.nu = Constant(1.0/self.Re)
        # -------------------------------------------------
        # transient part
        order = 1
        transi = self.Transient(dt, order = order) # form transient part
        self._add_temporary_functions(num=order+1) # add temp function for inner iterations
        # align temp and pre functions
        self.fu[2].assign(self.fu[1])
        self.fp[2].assign(self.fp[1])
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
             
        R2_1 = dot(nabla_grad(self.tp), nabla_grad(self.q)) * dx # for assembling RHS vec by mat
        R2_2 = - Constant(1.0 / dt) * div(self.tu) * self.q * dx
        
        # -------------------------------------------------
        # Define variational problem for step 3
        L3 = dot(self.tu, self.v) * dx
        
        R3_1 = dot(self.tu, self.v) * dx  # for assembling RHS vec by mat
        R3_2 = - Constant(self.dt) * dot(nabla_grad(self.tp), self.v) * dx
        
        return (L1,L2,L3), ((R1_1, R1_2), (R2_1, R2_2), (R3_1,R3_2))
        
    def SourceTerm(self):
        """
        Define the external source term for the Navier-Stokes equations.

        Returns
        -------
        F : UFL expression
            The external source term.
        """

        F = (inner(self.const_expr, self.v)+ inner(self.time_expr, self.v))* dx
            #(dot(self.const_expr, self.v) + dot(self.time_expr, self.v))*dx
            
        return F
        
        
    def BoundaryTraction(self, p, u, nu, mark=None, mode=None):
        """
        Define the boundary traction term for the Navier-Stokes equations.
        
        Note it is a residual term appearing in the standard NS variational problem
        A feature of variational formulations is that the test function v is equired to vanish on the parts of the boundary where the solution u is known
        
        Parameters
        ----------
        p : Function
            Pressure function.
        u : Function
            Velocity function.
        nu : Constant
            Kinematic viscosity.
        mark : int, optional
            Boundary marker. Default is None.
        mode : int, optional
            Mode for traction calculation. Default is None.

        Returns
        -------
        FBC : UFL expression
            The boundary traction term.
            a residual term due to variational form; 
            cancelled by implicit(e.g. free outlet) boundary condition
            test: FBC cancelled by Dirchlet BC and Free BC at Outlet; convergence problem when no BC applied at Boundary. 
                    cause convergence problem if the number of BCs is not equal to that of boundraies

        """
                    
        ds0 = self.ds if mark is None else self.ds(mark)
        n = self.n
        
        if mode is None or mode == 0:
            FBC = (dot(p*n, self.v)-dot(nu*grad(u)*n, self.v))*ds0 # from standard incompressible NS eqn
        elif mode == 1:
            FBC = (dot(p*n, self.v)-dot(nu*nabla_grad(u)*n, self.v))*ds0 # from compressible NS eqn with fully developed assumption + pressure bc
        elif mode == 2:
            FBC = (dot(p*n, self.v)-dot(nu*grad(u)*n, self.v)-dot(nu*nabla_grad(u)*n, self.v))*ds0 # from compressible NS eqn
        elif mode == 3:
            FBC = dot(p*n, self.v)*ds0 # from standard incompressible NS eqn with fully developed assumption + pressure bc
        else:
            raise ValueError("Invalid mode for boundary traction calculation.")

        return FBC
    
    def epsilon(self, u):
        """
        Compute the symmetric gradient of the velocity.

        Parameters
        ----------
        u : Function
            Velocity function.

        Returns
        -------
        epsilon_u : UFL expression
            Symmetric gradient of the velocity.
        """
        
        return sym(nabla_grad(u))

    def sigma(self, nu, u, p):
        """
        Compute the stress tensor for the Navier-Stokes equations.

        Parameters
        ----------
        u : Function
            Velocity function.

        Returns
        -------
        sigma_u : UFL expression
            Stress tensor.
        """

        return 2 * nu * self.epsilon(u) - p * Identity(len(u))

    
    def force_expr(self):
        """
        Define the force expressions for lift and drag calculations.

        Returns
        -------
        tuple
            Force components due to pressure and stress.
        """

        self.nu = Constant(1.0/self.Re)
        
        T1 = -self.tp * Identity(self.tu.geometric_dimension())
        T2 = 2.0 * self.nu * sym(grad(self.tu))
        force1 = -T1 * self.n
        force2 = -T2 * self.n
        
        return (force1, force2)
    
    def vorticity_expr(self):
        """
        Define the vorticity expression.

        Returns
        -------
        vorticity : UFL expression
            Vorticity of the velocity field.
        """
        
        return curl(self.tu)
        
        
        

class Compressible:
    """
    pending
    """
    def __init__(self):
        pass
    