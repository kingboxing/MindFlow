#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides classes for defining various forms of the incompressible Navier-Stokes equations,
suitable for finite element simulations using FEniCS. It includes support for steady and transient
problems, linearization, and various formulations such as frequency-domain and quasi-steady analyses.

Classes
-------
- Incompressible: Class for defining incompressible Navier-Stokes equations.
- Compressible: Placeholder class for future implementation of compressible Navier-Stokes equations.

Examples
--------
To use the `Incompressible` class in a simulation:

    from FERePack.Eqns import Incompressible
    from FERePack.BasicFunc import TaylorHood, SetBoundary
    from dolfin import Mesh

    # Define mesh and finite element
    mesh = Mesh("mesh.xml")
    element = TaylorHood(mesh)

    # Define boundary conditions
    boundary = SetBoundary(mesh, element)

    # Initialize the incompressible Navier-Stokes equations
    ns_eqn = Incompressible(element, boundary, Re=100)

    # Get the steady nonlinear Navier-Stokes form
    NS_form = ns_eqn.SteadyNonlinear()
"""

from ..Deps import *


class Incompressible:
    """
    Class for defining various forms of the incompressible Navier-Stokes equations for use in
    finite element simulations with FEniCS.

    Attributes
    ----------
    element : object
        Finite element object defining the solution space (e.g., TaylorHood element).
    dim : int
        Dimension of the flow field.
    rho : float
        Density of the fluid (fixed at 1.0 for incompressible flows).
    Re : float
        Reynolds number.
    ds : Measure
        Boundary measure for integration over boundaries.
    n : FacetNormal
        Outward pointing normal vector on the boundary.
    u : Function
        Velocity field function.
    p : Function
        Pressure field function.
    w : Function
        Combined function of velocity and pressure.
    fw : tuple of Functions
        Tuple containing functions at various time steps (e.g., current, previous).
    fu : tuple of Functions
        Tuple containing velocity functions at various time steps.
    fp : tuple of Functions
        Tuple containing pressure functions at various time steps.
    v : TestFunction
        Test function for velocity.
    q : TestFunction
        Test function for pressure.
    tu : TrialFunction
        Trial function for velocity.
    tp : TrialFunction
        Trial function for pressure.
    const_expr : Expression or Function
        Time-invariant source term for the flow field.
    time_expr : Expression or Function
        Time-dependent source term for the flow field.
    nu : Constant
        Kinematic viscosity (computed as 1/Re).

    Methods
    -------
    SteadyNonlinear()
        Define the steady nonlinear Navier-Stokes equations in weak form.
    SteadyLinear()
        Define the steady linearized Navier-Stokes equations in weak form.
    QuasiSteadyLinear(sz)
        Define the quasi-steady linearized Navier-Stokes equations for dimension reduction.
    Frequency(s)
        Define the transient part of the frequency-domain Navier-Stokes equations.
    Transient(dt, scheme="backward", order=1, implicit=True)
        Define the transient part using a time discretization scheme.
    IPCS(dt)
        Define the Implicit Pressure Correction Scheme (IPCS) for transient Navier-Stokes equations.
    SourceTerm()
        Define the external source term for the Navier-Stokes equations.
    BoundaryTraction(p, u, nu, mark=None, mode=None)
        Define the boundary traction term for the Navier-Stokes equations.
    epsilon(u)
        Compute the symmetric gradient of the velocity field.
    sigma(nu, u, p)
        Compute the stress tensor for the Navier-Stokes equations.
    force_expr()
        Define the expressions for computing forces (e.g., lift and drag).
    vorticity_expr()
        Define the expression for computing vorticity.
    """

    def __init__(self, element, boundary, Re, const_expr=None, time_expr=None):
        """
        Initialize the incompressible Navier-Stokes equations.

        Parameters
        ----------
        element : object
            Finite element object defining the solution space (e.g., instance of TaylorHood).
        boundary : SetBoundary
            Boundary object containing boundary definitions and measures.
        Re : float
            Reynolds number, representing the ratio of inertial forces to viscous forces.
        const_expr : Expression, Function, or Constant, optional
            Time-invariant source term (e.g., body force). Default is None.
        time_expr : Expression, Function, or Constant, optional
            Time-dependent source term (e.g., transient forcing). Default is None.
        """

        self.element = element
        self.dim = element.dim
        self.rho = 1.0  # Density is fixed at 1 for incompressible flows
        self.Re = Re  # Reynolds number
        self.ds = boundary.get_measure()  # boundary
        self.n = FacetNormal(element.mesh)  # boundary normal vec, outward direction

        self._initialize_functions(element)
        self._initialize_source_terms(const_expr, time_expr)

    def _initialize_functions(self, element):
        """
        Initialize function aliases and tuples for the finite element problem.

        Parameters
        ----------
        element : object
            Finite element object.
        """
        # Solution functions
        self.u = element.u  # Velocity function
        self.p = element.p  # Pressure function
        self.w = element.w  # Mixed/Combined function (velocity and pressure)
        self.fw = (self.w,)  # Tuple of solution functions at various time steps
        self.fu = (self.u,)  # Tuple of velocity functions at various time steps
        self.fp = (self.p,)  # Tuple of pressure functions at various time steps

        # Test functions
        self.v = element.v  # Test function for velocity
        self.q = element.q  # Test function for pressure

        # Trial functions (used in linearized problems)
        self.tu = element.tu  # Trial function for velocity
        self.tp = element.tp  # Trial function for pressure

    def _add_temporary_functions(self, num=1):
        """
        Add temporary functions for storing results from previous time steps.

        This method is used in transient simulations to keep track of solution
        variables at previous time levels.

        self.fw[0]---> time step n+1 (element.w -> current time step)

        Parameters
        ----------
        num : int, optional
            Number of temporary functions to add. Default is 1.
        """
        for _ in range(
                num - len(self.fw) + 1):  # number of additional funcs: consider time discretization, intermediate state
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
        expr_type = (function.function.Function, function.expression.Expression, function.constant.Constant)
        self.const_expr = const_expr if isinstance(const_expr, expr_type) else Constant((0.0,) * self.dim)
        self.time_expr = time_expr if isinstance(time_expr, expr_type) else Constant((0.0,) * self.dim)

    def SteadyNonlinear(self):  ## no trial functions
        """
        Define the steady nonlinear Navier-Stokes equations in weak form.

        The equations are defined as:

            (u ⋅ ∇)u + ∇p / ρ - ν Δu = F

            ∇ ⋅ u = 0

        where:
        - u is the velocity vector.
        - p is the pressure scalar.
        - ρ is the density (fixed at 1 for incompressible flows).
        - ν is the kinematic viscosity (ν = nu = 1/Re).
        - F is the external force (source term).
        
        Returns
        -------
        NS : UFL Form
            The weak form of the steady nonlinear Navier-Stokes equations.
        """

        self.nu = Constant(1.0 / self.Re)

        NS = ((inner(dot(self.u, nabla_grad(self.u)), self.v) -
               Constant(1.0 / self.rho) * self.p * div(self.v) +
               self.nu * inner(grad(self.u), grad(self.v)) +
               div(self.u) * self.q) * dx)

        F = self.SourceTerm()

        return NS - F

    def SteadyLinear(self):
        """
        Define the steady linearized Navier-Stokes equations in weak form.

        Linearization is performed around a base flow field u.

        Returns
        -------
        SLNS : UFL Form
            The weak form of the steady linearized Navier-Stokes equations.
        """

        # self.u represents base flow field
        self.nu = Constant(1.0 / self.Re)
        SLNS = (inner(dot(self.u, nabla_grad(self.tu)), self.v) +
                inner(dot(self.tu, nabla_grad(self.u)), self.v) -
                Constant(1.0 / self.rho) * self.tp * div(self.v) +
                self.nu * inner(grad(self.tu), grad(self.v))
                + div(self.tu) * self.q) * dx

        return SLNS

    def QuasiSteadyLinear(self, sz):
        """
        Define the quasi-steady linearized Navier-Stokes equations for dimension reduction.

        This method is used for analyzing flows where one spatial dimension can be
        represented using a harmonic function (e.g., for 2.5D or 1.5D flows).
        
        U(x, y, z, t) = U(x, y) e^{iwt} e^{ikt}
        
        2.5D/1.5D N-S equation
        
        3D problem -> 2D problem
        2D problem -> 1D problem
        
        3D problem -> 1D problem

        Parameters
        ----------
        sz : complex or tuple/list of complex
            Spatial frequency parameter(s) with only imaginary parts (e.g., sz = i * k) for quasi-analysis of the
            flow field. Default is None..

        Returns
        -------
        QSLNS_r : UFL Form
            The real part of the quasi-steady linearized Navier-Stokes equations.
        QSLNS_i : UFL Form
            The imaginary part of the quasi-steady linearized Navier-Stokes equations.

        Raises
        ------
        ValueError
            If the maximum dimension reduction is exceeded.
        TypeError
            If sz is not a complex number or a list/tuple of complex numbers.

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

        self.nu = Constant(1.0 / self.Re)
        if isinstance(sz, (complex, np.complexfloating)):  # 3D/2d -> 2D/1D
            sz = (sz,)
        elif isinstance(sz, (tuple, list)) and len(sz) > 1:  # pending for two
            raise ValueError('Maximum dimension reduction is limited to one.')
        else:
            raise TypeError('Invalid type for spatial frequency parameters')

        QSLNS_i = 0
        QSLNS_r = 0
        indz = self.element.dim - len(sz)
        for lamb in sz:
            lambda_ = Constant(np.imag(lamb))
            QSLNS_i += Constant(1.0 / self.rho) * lambda_ * self.tp * self.v[indz] * dx + lambda_ * self.tu[
                indz] * self.q * dx  # imag part
            QSLNS_r += self.nu * lambda_ * lambda_ * inner(self.tu, self.v) * dx
            indz += 1

        # pending for texting two-dimensional reduction
        indz = self.element.dim - len(sz)
        conv = 0  # convection term
        pres = 0  # pressure term
        cont = 0  # continuity term
        for i in range(indz):
            conv += (self.u[i] * nabla_grad(self.tu)[i, :]) + (self.tu[i] * nabla_grad(self.u)[i, :])
            pres += grad(self.v[i])[i]
            cont += grad(self.tu[i])[i]

        QSLNS_r += (inner(conv, self.v) -
                    Constant(1.0 / self.rho) * self.tp * pres +
                    self.nu * inner(grad(self.tu), grad(self.v)) +
                    cont * self.q) * dx  # real part

        # for 3D -> 2D
        # QSLNS_r += (inner((self.u[0] * nabla_grad(self.tu)[0,:])+(self.u[1] * nabla_grad(self.tu)[1,:]), self.v) +
        #             inner((self.tu[0] * nabla_grad(self.u)[0,:])+(self.tu[1] * nabla_grad(self.u)[1,:]), self.v) -
        #             Constant(1.0 / self.rho) * self.tp * (grad(self.v[0])[0]+grad(self.v[1])[1]) + 
        #             self.nu * inner(grad(self.tu), grad(self.v)) + 
        #             (grad(self.tu[0])[0]+grad(self.tu[1])[1]) * self.q) * dx # real part
        # for 3D -> 1D
        # QSLNS_r += (inner(dot(self.u[0:indz], nabla_grad(self.tu)[0:indz, :]), self.v) +
        #             inner(dot(self.tu[0:indz], nabla_grad(self.u)[0:indz, :]), self.v) -
        #             Constant(1.0 / self.rho)  * self.tp * div(self.v[0:indz]) + 
        #             self.nu * inner(grad(self.tu), grad(self.v)) +
        #             div(self.tu[0:indz]) * self.q) * dx # real part
        return QSLNS_r, QSLNS_i

    def Frequency(self, s):
        """
        Define the transient part of the frequency-domain Navier-Stokes equations.

        This method applies Fourier or Laplace transformations to the transient terms.

        Parameters
        ----------
        s : complex
            Laplace variable s = σ + iω, representing growth rate and frequency.

        Returns
        -------
        FNS_r : UFL Form
            The real part (growth rate term) of the frequency-domain equations.
        FNS_i : UFL Form
            The imaginary part (frequency term) of the frequency-domain equations.
        """
        self.s = s
        omega, sigma = np.imag(s), np.real(s)

        if omega != 0:
            FNS_i = Constant(omega) * inner(self.tu, self.v) * dx  # imaginary part
        else:
            FNS_i = 0

        if sigma != 0:
            FNS_r = Constant(sigma) * inner(self.tu, self.v) * dx  # real part
        else:
            FNS_r = 0

        return (FNS_r, FNS_i)

    def Transient(self, dt, scheme="backward", order=1, implicit=True):
        """
        Define the transient part using a time discretization scheme.

        Supports first-order backward difference schemes:
            At timestep n+1:
            if scheme == 'backward' and order == 1:
                du/dt = u_{n+1} - u_{n}

        Parameters
        ----------
        dt : float
            Time step size.
        scheme : str, optional
            Time discretization scheme (default is "backward").
        order : int, optional
            Order of the scheme (default is 1).
        implicit : bool, optional
            Whether to use an implicit scheme (default is True).

        Returns
        -------
        TNS : UFL Form
            The transient term in the Navier-Stokes equations.

        Raises
        ------
        ValueError
            If the scheme or order is not supported.
        """

        self.dt = dt
        self._add_temporary_functions(order)  # Add functions for previous time steps
        prev_u = self.fu[order]  # Velocity at previous time step

        if implicit:
            if order == 1 and scheme == "backward":
                TNS = dot((self.tu - prev_u) / Constant(dt), self.v) * dx  # self.fu[1] at previous time step
            else:
                raise ValueError("Unsupported scheme or order for implicit time discretization.")
        else:
            if order == 1 and scheme == "backward":
                TNS = dot((self.u - prev_u) / Constant(dt), self.v) * dx  # self.fu[1] at previous time step
            else:
                raise ValueError("Unsupported scheme or order for explicit time discretization.")

        return TNS

    def IPCS(self, dt):
        """
        Define the Implicit Pressure Correction Scheme (IPCS) for transient Navier-Stokes equations.

        This method implements a three-step IPCS algorithm for solving transient incompressible flows.
        (seems compressible NS eqns are used)
        
        Parameters
        ----------
        dt : float
            Time step size.

        Returns
        -------
        tuple
            Contains the left-hand side (LHS) and right-hand side (RHS) forms for the three IPCS steps.
        """

        self.nu = Constant(1.0 / self.Re)
        # -------------------------------------------------
        # transient part
        order = 1
        transi = self.Transient(dt, order=order)  # form transient part
        self._add_temporary_functions(num=order + 1)  # add temp function for inner iterations
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
        R1_2 = self.SourceTerm()  #(dot(self.const_expr, self.v) + dot(self.time_expr, self.v))*dx

        # -------------------------------------------------
        # if bcs is not full and freeoutlet is False # gradient of [u, v] in boundary-normal direction is zero # fully developped
        # FBC = self.BoundaryTraction(self.fp[2], U, self.nu, mode = 1) #dot(self.fp[2]*self.n, self.v)*self.ds - dot(self.nu*nabla_grad(U)*self.n, self.v)*self.ds

        # LBC = lhs(FBC)
        # RBC = rhs(FBC)
        # -------------------------------------------------
        # Define variational problem for step 2
        L2 = dot(nabla_grad(self.tp), nabla_grad(self.q)) * dx

        R2_1 = dot(nabla_grad(self.tp), nabla_grad(self.q)) * dx  # for assembling RHS vec by mat
        R2_2 = - Constant(1.0 / dt) * div(self.tu) * self.q * dx

        # -------------------------------------------------
        # Define variational problem for step 3
        L3 = dot(self.tu, self.v) * dx

        R3_1 = dot(self.tu, self.v) * dx  # for assembling RHS vec by mat
        R3_2 = - Constant(self.dt) * dot(nabla_grad(self.tp), self.v) * dx

        return (L1, L2, L3), ((R1_1, R1_2), (R2_1, R2_2), (R3_1, R3_2))

    def SourceTerm(self):
        """
        Define the external source term for the Navier-Stokes equations.

        Returns
        -------
        F : UFL Form
            The source term in the weak form.
        """

        F = (inner(self.const_expr, self.v) + inner(self.time_expr, self.v)) * dx
        #(dot(self.const_expr, self.v) + dot(self.time_expr, self.v))*dx

        return F

    def BoundaryTraction(self, p, u, nu, mark=None, mode=None):
        """
        Define the boundary traction term for the Navier-Stokes equations (usually used in the outlet boundary).

        This term appears in the weak formulation and accounts for boundary stresses.
        Note it is a residual term appearing in the standard NS variational problem
        A feature of variational formulations is that the test function v is equired to vanish on the parts of the
        boundary where the solution u is known.

        Parameters
        ----------
        p : Function
            Pressure function.
        u : Function
            Velocity function.
        nu : Constant
            Kinematic viscosity.
        mark : int, optional
            Boundary marker for applying the traction on specific boundaries. Default is None (all boundaries).
        mode : int, optional
            Mode for traction calculation:
            - 0: Standard incompressible Navier-Stokes equations.
            - 1: Compressible Navier-Stokes equations with fully developed assumption and pressure boundary condition.
            - 2: Compressible Navier-Stokes equations.
            - 3: Incompressible Navier-Stokes equations with fully developed assumption and pressure boundary condition.
            Default is 0.

        Returns
        -------
        FBC : UFL Form
            The boundary traction term.
            a residual term due to variational form;
            cancelled by implicit(e.g. free outlet) boundary condition
            test: FBC cancelled by Dirchlet BC and Free BC at Outlet; convergence problem when no BC applied at Boundary.
                    cause convergence problem if the number of BCs is not equal to that of boundraies

        Raises
        ------
        ValueError
            If an invalid mode is specified.
        """

        ds0 = self.ds if mark is None else self.ds(mark)
        n = self.n

        if mode is None or mode == 0:
            FBC = (dot(p * n, self.v) - dot(nu * grad(u) * n, self.v)) * ds0  # from standard incompressible NS eqn
        elif mode == 1:
            FBC = (dot(p * n, self.v) - dot(nu * nabla_grad(u) * n,
                                            self.v)) * ds0  # from compressible NS eqn with fully developed assumption + pressure bc
        elif mode == 2:
            FBC = (dot(p * n, self.v) - dot(nu * grad(u) * n, self.v) - dot(nu * nabla_grad(u) * n,
                                                                            self.v)) * ds0  # from compressible NS eqn
        elif mode == 3:
            FBC = dot(p * n,
                      self.v) * ds0  # from standard incompressible NS eqn with fully developed assumption + pressure bc
        else:
            raise ValueError("Invalid mode for boundary traction calculation.")

        return FBC

    def epsilon(self, u):
        """
        Compute the symmetric gradient (strain rate tensor) of the velocity field.

        Parameters
        ----------
        u : Function
            Velocity function.

        Returns
        -------
        epsilon_u : UFL Tensor
            The symmetric gradient of the velocity.
        """

        return sym(nabla_grad(u))

    def sigma(self, nu, u, p):
        """
        Compute the stress tensor for the Navier-Stokes equations.

        Parameters
        ----------
        nu : Constant
            Kinematic viscosity.
        u : Function
            Velocity function.
        p : Function
            Pressure function.

        Returns
        -------
        sigma_u : UFL Tensor
            The stress tensor.
        """

        return 2 * nu * self.epsilon(u) - p * Identity(len(u))

    def force_expr(self):
        """
        Define the expressions for computing force components due to pressure and viscous stress.

        Returns
        -------
        tuple
            Tuple containing force components:
            - force1: Force due to pressure.
            - force2: Force due to viscous stress.
        """

        self.nu = Constant(1.0 / self.Re)

        T1 = -self.tp * Identity(self.tu.geometric_dimension())
        T2 = 2.0 * self.nu * sym(grad(self.tu))
        force1 = -T1 * self.n
        force2 = -T2 * self.n

        return (force1, force2)

    def vorticity_expr(self):
        """
        Define the expression for computing the vorticity of the velocity field.

        Returns
        -------
        vorticity : UFL Expression
            The vorticity of the velocity field.
        """

        return curl(self.tu)


class Compressible:
    """
    Placeholder class for defining various forms of the compressible Navier-Stokes equations.

    This class is pending implementation and is included for future extensions.

    Methods
    -------
    (To be implemented)
    """

    def __init__(self):
        pass
