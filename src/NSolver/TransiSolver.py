#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides classes for solving transient (time-dependent) incompressible Navier-Stokes equations using various methods such as the Newton-Raphson method and the Implicit Pressure Correction Scheme (IPCS).

Classes
-------
- DNS_Newton:
    Solver for transient Navier-Stokes equations using the Newton-Raphson method.
- DNS_IPCS:
    Solver for transient incompressible Navier-Stokes equations using the Implicit Pressure Correction Scheme (IPCS).

Dependencies
------------
- FEniCS
- NumPy
- SciPy

Ensure that all dependencies are installed and properly configured.

Examples
--------
Typical usage involves creating an instance of `DNS_Newton` or `DNS_IPCS`, setting up the problem domain, boundary conditions, initial conditions, and solving the Navier-Stokes equations over time steps.

```python
from FERePack.NSolver.TransiSolver import DNS_Newton

# Define mesh and parameters
mesh = ...
Re = 100.0
dt = 0.01
const_expr = ...
time_expr = ...

# Initialize the solver
solver = DNS_Newton(mesh, Re=Re, dt=dt, const_expr=const_expr, time_expr=time_expr, order=(2, 1), dim=2)

# Set boundary conditions
solver.set_boundary(bc_list)
solver.set_boundarycondition(bc_list)

# Set initial condition
solver.initial(ic=initial_condition)

# Time-stepping loop
for t in time_steps:
    solver.solve()
    # Optionally, evaluate vorticity or forces
    vorticity = solver.eval_vorticity()
    force = solver.eval_force(mark=boundary_mark, dirc=0)
```

Notes
--------

- The DNS_Newton class uses the Newton-Raphson method for time-stepping and is suitable for transient simulations where implicit time integration is desired.
- The DNS_IPCS class implements the IPCS method, which is a commonly used fractional step method for solving incompressible Navier-Stokes equations.
"""

from ..Deps import *

from ..NSolver.SolverBase import NSolverBase
from ..BasicFunc.ElementFunc import TaylorHood, Decoupled
from ..NSolver.SteadySolver import NewtonSolver
from ..BasicFunc.Boundary import SetBoundary, SetBoundaryCondition
from ..BasicFunc.InitialCondition import SetInitialCondition


class DNS_Newton(NewtonSolver):
    """
    Solver for transient incompressible Navier-Stokes equations using the FEniCS build-in Newton method.

    This class extends the `NewtonSolver` class to handle time-dependent problems using implicit time integration schemes.

    Parameters
    ----------
    mesh : dolfin.Mesh
        The computational mesh.
    Re : float, optional
        The Reynolds number. Default is None.
    dt : float, optional
        Time step size. Default is None.
    const_expr : dolfin.Expression, dolfin.Function, or dolfin.Constant, optional
        The time-invariant source term for the flow field. Default is None.
    order : tuple of int, optional
        Order of the finite element spaces for velocity and pressure, respectively. Default is (2, 1).
    dim : int, optional
        Dimension of the flow field. Default is 2.
    constrained_domain : dolfin.SubDomain, optional
        Constrained subdomain for periodic boundary conditions. Default is None.

    Attributes
    ----------
    nstep : int
        The current time step number.
    Transient : dolfin.Form
        The transient term in the Navier-Stokes equations.

    Methods
    -------
    initial(ic=None, noise=False, timestamp=0.0)
        Set the initial condition for the simulation.
    solve(dt=None, Re=None, time_expr=None)
        Solve the transient Navier-Stokes equations using the Newton method.

    Notes
    -----
    - The solver uses backward Euler time integration by default.
    - Suitable for simulations where high accuracy in time is required.
    """

    def __init__(self, mesh, Re=None, dt=None, const_expr=None, order=(2, 1), dim=2,
                 constrained_domain=None):
        """
        Initialize the DNS_Newton solver.

        Parameters
        ----------
        mesh : dolfin.Mesh
            The computational mesh.
        Re : float, optional
            The Reynolds number. Default is None.
        dt : float, optional
            Time step size. Default is None.
        const_expr : dolfin.Expression, dolfin.Function, or dolfin.Constant, optional
            The time-invariant source term for the flow field. Default is None.
        order : tuple of int, optional
            Order of the finite element spaces for velocity and pressure, respectively. Default is (2, 1).
        dim : int, optional
            Dimension of the flow field. Default is 2.
        constrained_domain : dolfin.SubDomain, optional
            Constrained subdomain for periodic boundary conditions. Default is None.
        """
        super().__init__(mesh, Re, const_expr, order, dim, constrained_domain)
        self.nstep = 0
        self.bc_reset = False  # pending
        # note the order of the time discretization # use function (instead of trialfunction) as unknown for newton method expression
        self.Transient = self.eqn.Transient(dt, scheme="backward", order=1, implicit=False)

    def initial(self, ic=None, noise=False, timestamp=0.0):
        """
        Set the initial condition for the simulation.

        Parameters
        ----------
        ic : str or dolfin.Function, optional
            Initial condition as a file path or FEniCS function. Default is None.
        noise : bool, optional
            If True, add noise to the initial condition. Default is False.
        timestamp : float, optional
            Timestamp for retrieving the initial condition from a time series. Default is 0.0.
        """
        SetInitialCondition(1, ic=ic, fw=self.eqn.fw[1], noise=noise, timestamp=timestamp)
        assign(self.eqn.fw[0], self.eqn.fw[1])

    def solve(self, dt=None, Re=None, time_expr=None):
        """
        Solve the transient Navier-Stokes equations using the Newton method.

        Parameters
        ----------
        dt : float, optional
            Time step size. If provided, updates the time step size before solving. Default is None.
        Re : float, optional
            The Reynolds number. If provided, updates the Reynolds number before solving. Default is None.
        time_expr : dolfin.Expression, dolfin.Function, or dolfin.Constant, optional
            Time-dependent source term for the flow field. If provided, updates the source term before solving. Default is None.

        Notes
        -----
        - The method checks for changes in parameters and rebuilds the equations if necessary.
        - At each time step, the solution is stored in `self.eqn.fw[0]`.
        """
        rebuild = False

        if dt is not None and dt != self.eqn.dt:
            self.eqn.dt = dt
            rebuild = True

        if Re is not None and Re != self.eqn.Re:
            self.eqn.Re = Re
            rebuild = True

        if time_expr != self.eqn.time_expr:
            self.eqn.time_expr = time_expr
            rebuild = True

        if rebuild or self.nstep == 0:
            self._form_SINS_equations()
            self.SNS += self.Transient
            self._initialize_newton_solver()

        self.solver.solve()
        self.eqn.fw[1].assign(self.eqn.fw[0])  # note the order of the time discretization
        self.nstep += 1


#%%        
class DNS_IPCS(NSolverBase):
    """
    Solver for transient incompressible Navier-Stokes equations using the Implicit Pressure Correction Scheme (IPCS).

    This class implements the IPCS method, a fractional step method, to solve the time-dependent Navier-Stokes equations.

    Parameters
    ----------
    mesh : dolfin.Mesh
        The computational mesh.
    Re : float
        The Reynolds number.
    dt : float
        Time step size.
    const_expr : dolfin.Expression, dolfin.Function, or dolfin.Constant, optional
        The time-invariant source term for the flow field. Default is None.
    time_expr : dolfin.Expression, dolfin.Function, or dolfin.Constant, optional
        Time-dependent source term for the flow field. Default is None.
    order : tuple of int, optional
        Order of the finite element spaces for velocity and pressure, respectively. Default is (2, 1).
    dim : int, optional
        Dimension of the flow field. Default is 2.
    constrained_domain : list of dolfin.SubDomain, optional
        Constrained subdomains to apply constraints (e.g., for periodic boundary conditions) for velocity and pressure spaces, respectively.
        Default is [None, None].

    Attributes
    ----------
    nstep : int
        The current time step number.
    LHS : list of dolfin.Form
        Left-hand side forms for each step of the IPCS method.
    RHS : list of tuple
        Right-hand side forms for each step of the IPCS method.
    solver : list
        Solvers for each step of the IPCS method.
    has_free_bc : bool
        Indicates if free boundary conditions are present.
    bc_reset : bool or int
        Flag for resetting boundary conditions.
    FreeBoundary : dict
        Dictionary to store information for free boundary conditions.

    Methods
    -------
    initial(ic=None, noise=False, timestamp=0.0, element_init=None)
        Set the initial condition for the simulation.
    set_boundarycondition(bc_list=None, reset=True)
        Apply boundary conditions to the solver.
    parameters(param)
        Update solver parameters.
    solve(method='lu', lusolver='mumps', inner_iter_max=20, tol=1e-7, relax_factor=1.0, verbose=False)
        Solve the Navier-Stokes equations using the IPCS method.

    Notes
    -----
    - The IPCS method splits the Navier-Stokes equations into a series of steps to solve for velocity and pressure separately.
    - Suitable for simulations where computational efficiency is important.
    """

    def __init__(self, mesh, Re, dt, const_expr=None, time_expr=None, order=(2, 1), dim=2,
                 constrained_domain=[None, None]):
        """
        Initialize the DNS_IPCS solver.

        Parameters
        ----------
        mesh : Mesh
            The computational mesh.
        Re : float
            The Reynolds number.
        dt : float
            Time step size.
        const_expr : Expression, Function or Constant, optional
            The time-invariant source term for the flow field. Default is None.
        time_expr : Expression, Function or Constant, optional
            Time-dependent source term for the flow field. Default is None.
        order : tuple, optional
            Order of finite element method. Default is (2, 1).
        dim : int, optional
            Dimension of the flow field. Default is 2.
        constrained_domain : list of dolfin.SubDomain, optional
            Constrained subdomains to apply constraints (e.g., for periodic boundary conditions) for velocity and pressure spaces, respectively.
            Default is [None, None].
        """
        element = Decoupled(mesh=mesh, order=order, dim=dim,
                            constrained_domain=constrained_domain)  # initialise finite element space
        super().__init__(mesh, element, Re, const_expr, time_expr)
        self.mesh = mesh
        # boundary condition
        self.boundary_condition_V = SetBoundaryCondition(self.element.functionspace_V, self.boundary)  # velocity field
        self.boundary_condition_Q = SetBoundaryCondition(self.element.functionspace_Q, self.boundary)  # pressure field
        # NS equations
        self.LHS, self.RHS = self.eqn.IPCS(dt)
        #
        self.has_free_bc = False
        self.bc_reset = False
        self.nstep = 0
        #
        self.param['solver_type'] = 'IPCS_solver'
        self.param['IPCS_solver'] = {}

    def initial(self, ic=None, noise=False, timestamp=0.0, element_init=None):
        """
        Set the initial condition for the simulation.

        Parameters
        ----------
        ic : str or dolfin.Function, optional
            Initial condition as a file path or FEniCS function. Default is None.
        noise : bool, optional
            If True, add noise to the initial condition. Default is False.
        timestamp : float, optional
            Timestamp for retrieving the initial condition from a time series. Default is 0.0.
        element_init : object, optional
            Initial element (e.g., Taylor-Hood element). Default is None.

        Notes
        -----
        - The method can project an initial condition from a different finite element space if provided.
        """

        SetInitialCondition(2, ic=ic, fw=self.eqn.fw[1], noise=noise, timestamp=timestamp, mesh=self.mesh,
                            element_in=element_init, element_out=self.element)
        assign(self.eqn.fw[0][0], self.eqn.fw[1][0])
        assign(self.eqn.fw[0][1], self.eqn.fw[1][1])

    def set_boundarycondition(self, bc_list=None, reset=True):
        """
        Apply boundary conditions to the solver.

        Parameters
        ----------
        bc_list : dict, optional
            Dictionary of boundary conditions. If None, uses the boundary conditions defined in `self.boundary.bc_list`.
            Default is None.
        reset : int, optional
            Reset mode for boundary conditions:
            - 0: No reset
            - 1: Reset all (default)
            - 2: Reset values only
        """

        if bc_list is None:
            bc_list = self.boundary.bc_list

        for key, bc in bc_list.items():
            self._apply_boundarycondition(bc, key)

        self.bc_reset = reset  # reset boundary conditions mode 1 (reset everything) # due to matrix/vector method to employ boundary conditions

    def _apply_boundarycondition(self, bc_dict, mark):
        """
        Apply a specific boundary condition.

        Parameters
        ----------
        bc_dict : dict
            Dictionary containing boundary condition properties.
        mark : int
            Boundary identifier.

        Notes
        -----
        - Handles free boundary conditions by setting up appropriate pressure corrections.
        """

        # pending for dealing with free boundary/ zero boundary traction condition in bc_list

        if 'Free Boundary' in (bc_dict['FunctionSpace'], bc_dict['Value']):
            bc_dict['FunctionSpace'] = 'Q'
            bc_dict['Value'] = Constant(0.0)
            if self.has_free_bc is False:  # Create a dictionary (if it doesn't already exist)
                self.FreeBoundary = {}
            self.FreeBoundary['Boundary' + str(mark)] = self._initialize_free_boundary(mark=mark)
            self.has_free_bc += 1
            info(f'Free boundary condition (zero boundary traction) applied at Boundary {mark}')

        # setup all BCs(including free-outlet)
        if bc_dict['FunctionSpace'][0] == 'V':
            self.boundary_condition_V.set_boundarycondition(bc_dict, mark)
        elif bc_dict['FunctionSpace'][0] == 'Q':
            self.boundary_condition_Q.set_boundarycondition(bc_dict, mark)

    def parameters(self, param):
        """
        Update solver parameters.

        Parameters
        ----------
        param : dict
            Dictionary containing parameters to update.
        """
        # update solver parameters
        self.param.update(param)

    def _apply_matrix_bc(self):
        """
        Apply matrix-based boundary conditions.

        Notes
        -----
        - Constructs identity matrices with zeros at rows corresponding to Dirichlet boundary conditions.
        """
        self.Mat_vel = self.boundary_condition_V.MatrixBC_rhs()
        self.Mat_pre = self.boundary_condition_Q.MatrixBC_rhs()

    def _apply_vector_bc(self):
        """
        Apply vector-based boundary conditions.

        Notes
        -----
        - Constructs vectors containing boundary condition values at rows corresponding to Dirichlet boundary conditions.
        """
        self.Vec_vel = self.boundary_condition_V.VectorBC_rhs()
        self.Vec_pre = self.boundary_condition_Q.VectorBC_rhs()

    def _calculate_normal_vector(self, mark):
        """
        Calculate the normal vector at a specified boundary.

        Parameters
        ----------
        mark : int
            Boundary identifier.

        Returns
        -------
        normal : dolfin.Function
            Normal vector function defined over the boundary.
        """
        n = self.eqn.n
        ds = self.eqn.ds
        normal_vec = assemble(
            dot(n, sym(grad(self.eqn.tu)) * n) * ds(mark) + dot(self.eqn.tu, n) * ds(mark)).get_local()
        normal_vec[abs(normal_vec) < 1e-10] = 0
        normal = self.element.add_functions()[0]
        normal.vector()[:] = np.ascontiguousarray(normal_vec)

        normal_T = self.element.add_functions()[0]
        assign(normal_T.sub(0), normal.sub(1))
        assign(normal_T.sub(1), normal.sub(0))
        #% normalise? 
        weight_recip = np.sqrt((normal_T.vector() * normal_T.vector() + normal.vector() * normal.vector()).get_local())
        weight_recip[np.abs(weight_recip) == 0.0] = np.inf
        weight = 1.0 / weight_recip

        normal.vector()[:] = normal_vec * weight

        return interpolate(normal, VectorFunctionSpace(self.mesh, 'P', self.element.order[1]))

    def _initialize_free_boundary(self, mark=None, solver='mumps', func=None, BC_dict=None):
        """
        Initialize/evaluate pressure computations for a free boundary condition based on the velocity field.

        Parameters
        ----------
        mark : int, optional
            Boundary identifier. Default is None.
        solver : str, optional
            Solver type (e.g., 'mumps'). Default is 'mumps'.
        func : dolfin.Function, optional
            Velocity function. Default is None.
        BC_dict : dict, optional
            Boundary condition dictionary. Default is None.

        Returns
        -------
        BC_values : dolfin.Function or dict
            Pressure boundary condition function or updated boundary condition dictionary.

        Notes
        -----
        - If `mark` is provided, initializes the necessary matrices and solvers.
        - If `BC_dict` is provided, updates the pressure boundary condition values.
        """

        if isinstance(mark, (int, np.integer)):
            normal = self._calculate_normal_vector(mark)
            norm = interpolate(normal, self.element.functionspace_V)
            RHS_mat = assemble(Constant(1.0 / self.eqn.Re) * dot(norm, sym(grad(self.eqn.tu)) * norm) * self.eqn.q * dx)
            LHS_mat = assemble(self.eqn.tp * self.eqn.q * dx)
            solver_outlet = PETScLUSolver(LHS_mat.instance())
            solver_outlet.parameters.add('reuse_factorization', True)
            BC_outlet = DirichletBC(self.element.functionspace_Q, Constant(0.0), self.boundary.boundary, mark,
                                    method="geometric")
            MatBC_outlet = assemble(
                Constant(0.0) * dot(self.eqn.tp, self.eqn.q) * dx)  ## create a zero mat with pressure field size
            BC_outlet.apply(MatBC_outlet)

            return {
                'BC norm': norm,
                'Outlet Solver': solver_outlet,
                'Matrix for RHS': RHS_mat,
                'Matrix for BC': MatBC_outlet,
                'BC Values': self.element.add_functions()[1],
            }
        elif BC_dict:
            b = BC_dict['Matrix for RHS'] * func.vector()  # RHS
            BC_dict['Outlet Solver'].solve(BC_dict['BC Values'].vector(), b)  # solver for pressure
            BC_dict['BC Values'].vector()[:] = BC_dict['Matrix for BC'] * BC_dict[
                'BC Values'].vector()  # apply boundary conditions
            return BC_dict['BC Values']

    def _assemble_system(self):
        """
        Assemble the matrices and vectors for the IPCS solver.

        Notes
        -----
        - Assembles the left-hand side (A) and right-hand side (b) for each step of the IPCS method.
        - Applies boundary conditions to the matrices.
        """
        self.A = tuple(PETScMatrix() for _ in range(3))
        self.b = []
        for i in range(3):
            if i == 0: # no loop, only for first operation
                if not self.has_traction_bc:
                    assemble(self.LHS[i], tensor=self.A[i])
                    # no assemble
                    self.b.append([self.RHS[i][0], self.RHS[i][1]])
                else:
                    U = 0.5 * (self.enq.fu[2] + self.eqn.tu)
                    FBC = 0
                    j = 0
                    for key in self.has_traction_bc.keys():
                        # if j == 0:
                        #     FBC = self.eqn.BoundaryTraction(self.eqn.fp[2], U, self.eqn.nu, mark=self.has_traction_bc[key][0], mode=self.has_traction_bc[key][1])
                        # else:
                        #     FBC += self.eqn.BoundaryTraction(self.eqn.fp[2], U, self.eqn.nu, mark=self.has_traction_bc[key][0], mode=self.has_traction_bc[key][1])

                        FBC += self.eqn.BoundaryTraction(self.eqn.fp[2], U, self.eqn.nu, mark=self.has_traction_bc[key][0], mode=self.has_traction_bc[key][1])
                        j += 1
                    LBC = lhs(FBC)
                    RBC = rhs(FBC)
                    assemble(self.LHS[i] + LBC, tensor=self.A[i])
                    # no assemble
                    self.b.append((self.RHS[i][0] + RBC, self.RHS[i][1]))
            else:
                assemble(self.LHS[i], tensor=self.A[i])
                self.b.append((assemble(self.RHS[i][0]), assemble(self.RHS[i][1])))

        [bc.apply(self.A[0]) for bc in self.boundary_condition_V.bc_list]
        [bc.apply(self.A[1]) for bc in self.boundary_condition_Q.bc_list]

    def _initialize_solver(self, method='lu', lusolver='mumps'):
        """
        Initialize solvers for each step of the IPCS method.

        Parameters
        ----------
        method : str, optional
            Solver method ('lu' or 'krylov'). Default is 'lu'.
        lusolver : str, optional
            Type of LU solver. Default is 'mumps'.
        """

        self.solver = []
        for i in range(3):
            if method == 'lu':
                solver = PETScLUSolver(self.A[i], lusolver)
                solver.parameters.add('reuse_factorization', True)
            elif method == 'krylov':
                solver_type = 'cg' if i == 1 else 'gmres'
                solver = KrylovSolver(solver_type, 'jacobi' if i != 1 else 'hypre_euclid')
                solver.set_operator(self.A[i])
                solver.parameters['absolute_tolerance'] = 1E-10
                solver.parameters['relative_tolerance'] = 1E-8
            self.solver.append(solver)

    def apply_sourceterm(self, time_expr=None):
        """
        Apply a time-dependent source term.

        Parameters
        ----------
        time_expr : dolfin.Expression, dolfin.Function, or dolfin.Constant, optional
            Time-dependent source term for the flow field. Default is None.

        Notes
        -----
        - Updates the source term in the right-hand side of the equations.
        - Pending for test if it works
        """
        if time_expr is not None:
            self.eqn.time_expr = time_expr
        self.b[0][1] = self.eqn.SourceTerm()

    def _update_solution(self):
        """
        Update the solution after each time step.

        Notes
        -----
        - Updates the previous solutions for velocity and pressure for use in the next time step.
        """
        self.eqn.fu[1].assign(self.eqn.fu[0])
        self.eqn.fp[1].assign(self.eqn.fp[0])
        self.eqn.fu[2].assign(self.eqn.fu[1])
        self.eqn.fp[2].assign(self.eqn.fp[1])
        self.nstep += 1

    def solve(self, method='lu', lusolver='mumps', inner_iter_max=20, tol=1e-7, relax_factor=1.0, verbose=False):
        """
        Solve the Navier-Stokes equations using the IPCS method.

        Parameters
        ----------
        method : str, optional
            Solver method ('lu' or 'krylov'). Default is 'lu'.
        lusolver : str, optional
            Type of LU solver. Default is 'mumps'.
        inner_iter_max : int, optional
            Maximum number of inner iterations. Default is 20.
        tol : float, optional
            Convergence tolerance for the iterative solver. Default is 1e-7.
        relax_factor : float, optional
            Relaxation factor for updating the solution. Default is 1.0.
        verbose : bool, optional
            If True, print iteration information. Default is False.

        Notes
        -----
        - The method performs an iterative procedure to solve the equations at each time step.
        - Convergence is checked based on the infinity norm of the difference between successive velocity solutions.
        """

        if self.nstep == 0 or self.bc_reset:
            self._assemble_system()
            self._initialize_solver(method=method, lusolver=lusolver)
            self._apply_matrix_bc()  # if it is mode 1: reassembling matrices

        if self.bc_reset is not False:  # bc reset
            self._apply_vector_bc()
            self.bc_reset = False  # swith flag after assembling matrices and vectors

        niter, eps = 0, 1

        while eps > tol and niter < inner_iter_max:
            # Step 1: Tentative velocity step
            b1 = self.Mat_vel * assemble(self.b[0][0] + self.b[0][1]) + self.Vec_vel  # have assemble here
            self.solver[0].solve(self.eqn.fu[0].vector(), b1)

            # Step 2: Pressure correction step
            b2 = self.b[1][0] * self.eqn.fp[2].vector() + self.b[1][1] * self.eqn.fu[0].vector()
            b2 = self.Mat_pre * b2 + self.Vec_pre

            # 2nd step is poisson equation without dirichlet boundary condition
            if self.has_free_bc is not False:  # have free outlet then pressure BC is pre-set to zero
                for key in self.FreeBoundary.keys():
                    b2 += self._initialize_free_boundary(func=self.eqn.fu[0], BC_dict=self.FreeBoundary[key]).vector()
            self.solver[1].solve(self.eqn.fp[0].vector(), b2)

            # Step 3: Velocity correction step
            b3 = self.b[2][0] * self.eqn.fu[0].vector() + self.b[2][1] * (
                        self.eqn.fp[0].vector() - self.eqn.fp[2].vector())
            self.solver[2].solve(self.eqn.fu[0].vector(), b3)

            # eps
            eps = norm(self.eqn.fu[0].vector() - self.eqn.fu[2].vector(), 'linf')

            # update for the next iter
            self.eqn.fu[2].vector()[:] = relax_factor * self.eqn.fu[0].vector() + (1.0 - relax_factor) * self.eqn.fu[
                2].vector()
            self.eqn.fp[2].vector()[:] = relax_factor * self.eqn.fp[0].vector() + (1.0 - relax_factor) * self.eqn.fp[
                2].vector()

            # 
            niter += 1
            if verbose and comm_rank == 0:
                print(f'inner_iter={niter}: norm={eps}')

        # Update previous solution
        self._update_solution()
