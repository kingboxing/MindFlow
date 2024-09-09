#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 20:08:46 2024

@author: bojin
"""

"""This module provides the classes that solve Navier-Stokes equations
"""

from src.Deps import *

from src.NSolver.SolverBase import NSolverBase
from src.BasicFunc.ElementFunc import TaylorHood, Decoupled
from src.NSolver.SteadySolver import NewtonSolver
from src.BasicFunc.Boundary import SetBoundary, SetBoundaryCondition
from src.BasicFunc.InitialCondition import SetInitialCondition
from src.Eqns.NavierStokes import Incompressible

class DNS_Newton(NewtonSolver):
    """
    Solver for transient Navier-Stokes equations using the Newton method.
    """
    def __init__(self, mesh, Re=None, dt=None, const_expr=None, time_expr=None, order=(2,1), dim=2, constrained_domain=None):
        """
        Initialize the DNS_Newton solver.

        Parameters
        ----------
        mesh : Mesh
            The computational mesh.
        Re : float, optional
            The Reynolds number. Default is None.
        dt : float, optional
            Time step size. Default is None.
        const_expr : Expression, Function or Constant, optional
            The time-invariant source term for the flow field. Default is None.
        time_expr : Expression, Function or Constant, optional
            Time-dependent source term for the flow field. Default is None.
        order : tuple, optional
            Order of finite element method. Default is (2, 1).
        dim : int, optional
            Dimension of the problem (2D or 3D). Default is 2.
        constrained_domain : SubDomain, optional
            Constrained domain (e.g., for periodic boundary conditions). Default is None.
        """
        super().__init__(mesh, Re, const_expr, order, dim, constrained_domain)
        self.nstep = 0
        self.bc_reset = False# pending
        # note the order of the time discretization # use function (instead of trialfunction) as unknown for newton method expression
        self.Transient = self.eqn.Transient(dt, scheme="backward", order=1, implicit=False)
        
    def initial(self, ic=None, noise=False, timestamp=0.0):
        """
        Set the initial condition for the simulation.

        Parameters
        ----------
        ic : str or Function, optional
            Initial condition as a file path or FEniCS function. Default is None.
        noise : bool, optional
            Add noise to the initial condition. Default is False.
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
            Time step size. Default is None.
        Re : float, optional
            The Reynolds number. Default is None.
        time_expr : Expression, Function or Constant, optional
            Time-dependent source term for the flow field. Default is None.
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
        self.eqn.fw[1].assign(self.eqn.fw[0]) # note the order of the time discretization
        self.nstep += 1
            
#%%        
class DNS_IPCS(NSolverBase):
    """
    Solver for transient incompressible Navier-Stokes equations using the Implicit Pressure Correction Scheme (IPCS).
    """
    def __init__(self, mesh, Re, dt, const_expr=None, time_expr=None, order=(2,1), dim=2, constrained_domain=[None, None]):
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
            Dimension of the problem (2D or 3D). Default is 2.
        constrained_domain : list of SubDomain, optional
            Constrained domains (e.g., for periodic boundary conditions). Default is [None, None].
        """

        element = Decoupled(mesh = mesh, order = order, dim = dim, constrained_domain = constrained_domain) # initialise finite element space
        super().__init__(mesh, element, Re, const_expr, time_expr)
        self.mesh = mesh
        # boundary condition
        self.boundary_condition_V = SetBoundaryCondition(self.element.functionspace_V, self.boundary) # velocity field
        self.boundary_condition_Q = SetBoundaryCondition(self.element.functionspace_Q, self.boundary) # pressure field
        # NS equations
        self.LHS, self.RHS = self.eqn.IPCS(dt)
        #
        self.has_free_bc = False
        self.bc_reset=False
        self.nstep = 0
        #
        self.param['solver_type']='IPCS_solver'
        self.param['IPCS_solver']={}
        
        
    def initial(self, ic=None, noise=False, timestamp=0.0, element_init=None):
        """
        Set the initial condition for the IPCS solver.

        Parameters
        ----------
        ic : str or Function, optional
            Initial condition as a file path or FEniCS function. Default is None.
        noise : bool, optional
            Add noise to the initial condition. Default is False.
        timestamp : float, optional
            Timestamp for retrieving the initial condition from a time series. Default is 0.0.
        element_init : object, optional
            Initial element (e.g., TaylorHood). Default is None.
        """
        
        SetInitialCondition(2, ic=ic, fw=self.eqn.fw[1], noise=noise, timestamp=timestamp, mesh=self.mesh, element_in = element_init, element_out = self.element)
        assign(self.eqn.fw[0][0],self.eqn.fw[1][0])
        assign(self.eqn.fw[0][1],self.eqn.fw[1][1])
        
    def set_boundarycondition(self, bc_list=None, reset=True):
        """
        Apply boundary conditions to the IPCS solver.

        Parameters
        ----------
        bc_list : dict, optional
            Dictionary of boundary conditions. Default is None.
        reset : int, optional
            Reset mode for boundary conditions (0: no reset, 1: reset all, 2: reset values only). Default is 1.
        """
        
        if bc_list is None:
            bc_list=self.boundary.bc_list
        
        for key, bc in bc_list.items():
            self._apply_boundarycondition(bc, key)
        
        self.bc_reset=reset # reset boundary conditions mode 1 (reset everything) # due to matrix/vector method to employ boundary conditions
        
    
    def _apply_boundarycondition(self, bc_dict, mark):
        """
        Apply a specific boundary condition.

        Parameters
        ----------
        bc_dict : dict
            Dictionary of boundary condition properties.
        mark : int
            Boundary identifier.
        """
        
        # pending for dealing with free boundary/ zero boundary traction condition in bc_list
        
        if 'Free Boundary' in (bc_dict['FunctionSpace'],bc_dict['Value']):
            bc_dict['FunctionSpace']='Q'
            bc_dict['Value']=Constant(0.0)
            if self.has_free_bc is False:# Create a dictionary (if it doesn't already exist)
                self.FreeBoundary={}
            self.FreeBoundary['Boundary'+str(mark)]=self._initialize_free_boundary(mark=mark)
            self.has_free_bc+=1
            info(f'Free boundary condition (zero boundary traction) applied at Boundary {mark}')

        # setup all BCs(including free-outlet)
        if bc_dict['FunctionSpace'][0] == 'V':
            self.boundary_condition_V.set_boundarycondition(bc_dict, mark)
        elif bc_dict['FunctionSpace'][0] == 'Q':
            self.boundary_condition_Q.set_boundarycondition(bc_dict,mark)
            
    def parameters(self, param):
        """
        Update parameters

        Parameters
        ----------
        param : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # update solver parameters
        self.param.update(param)
    
    def _apply_matrix_bc(self):
        """
        Apply matrix-based boundary conditions.
        
        Identity Matrix that contains only zeros in the rows which have Dirichlet boundary conditions

        Returns
        -------
        None.

        """
        self.Mat_vel = self.boundary_condition_V.MatrixBC_rhs()
        self.Mat_pre = self.boundary_condition_Q.MatrixBC_rhs()

    def _apply_vector_bc(self):
        """
        Apply vector-based boundary conditions.
        
        Zero Vector that contains boundary condition values in the rows which have Dirichlet boundary conditions

        Returns
        -------
        None.

        """
        self.Vec_vel = self.boundary_condition_V.VectorBC_rhs()
        self.Vec_pre = self.boundary_condition_Q.VectorBC_rhs()    
        
    def _calculate_normal_vector(self, mark):
        """
        Calculate the normal vector at a boundary.

        Parameters
        ----------
        mark : int
            Boundary identifier.

        Returns
        -------
        Function
            Normal vector function.
        """
        n = self.eqn.n
        ds = self.eqn.ds
        normal_vec = assemble(dot(n, sym(grad(self.eqn.tu)) * n) * ds(mark) + dot(self.eqn.tu, n) * ds(mark)).get_local()
        normal_vec[abs(normal_vec) < 1e-10] = 0
        normal = self.element.add_functions()[0]
        normal.vector()[:] = np.ascontiguousarray(normal_vec)
        
        normal_T=self.element.add_functions()[0]
        assign(normal_T.sub(0),normal.sub(1))
        assign(normal_T.sub(1),normal.sub(0))
        #% normalise? 
        weight_recip=np.sqrt((normal_T.vector()*normal_T.vector()+normal.vector()*normal.vector()).get_local())
        weight_recip[np.abs(weight_recip)==0.0]=np.inf
        weight=1.0/weight_recip
        
        normal.vector()[:]=normal_vec*weight
        
        return interpolate(normal, VectorFunctionSpace(self.mesh, 'P', self.element.order[1]))

    def _initialize_free_boundary(self, mark=None, solver='mumps', func=None, BC_dict=None):
        """
        Compute pressure on a free boundary based on the velocity field.

        Parameters
        ----------
        mark : int, optional
            Boundary identifier. Default is None.
        solver : str, optional
            Solver type (e.g., 'mumps'). Default is 'mumps'.
        func : Function, optional
            Velocity function. Default is None.
        BC_dict : dict, optional
            Boundary condition dictionary. Default is None.

        Returns
        -------
        Function or dict
            Pressure boundary condition function or updated BC dictionary.
        """
        
        if isinstance(mark, (int, np.integer)):
            normal = self._calculate_normal_vector(mark)
            norm = interpolate(normal, self.element.functionspace_V)
            RHS_mat = assemble(Constant(1.0 / self.eqn.Re) * dot(norm, sym(grad(self.eqn.tu)) * norm) * self.eqn.q * dx)
            LHS_mat = assemble(self.eqn.tp * self.eqn.q * dx)
            solver_outlet = PETScLUSolver(LHS_mat.instance())
            solver_outlet.parameters.add('reuse_factorization', True)
            BC_outlet = DirichletBC(self.element.functionspace_Q, Constant(0.0), self.boundary.boundary, mark, method="geometric")
            MatBC_outlet = assemble(Constant(0.0) * dot(self.eqn.tp, self.eqn.q) * dx) ## create a zero mat with pressure field size
            BC_outlet.apply(MatBC_outlet)

            return {
                'BC norm': norm,
                'Outlet Solver': solver_outlet,
                'Matrix for RHS': RHS_mat,
                'Matrix for BC': MatBC_outlet,
                'BC Values': self.element.add_functions()[1],
            }
        elif BC_dict:
            b = BC_dict['Matrix for RHS'] * func.vector() # RHS
            BC_dict['Outlet Solver'].solve(BC_dict['BC Values'].vector(), b) # solver for pressure
            BC_dict['BC Values'].vector()[:] = BC_dict['Matrix for BC'] * BC_dict['BC Values'].vector() # apply boundary conditions
            return BC_dict['BC Values']
        
    def _assemble_system(self):
        """
        Assemble the matrices and vectors for the IPCS solver.

        Returns
        -------
        None.

        """
        self.A = tuple(PETScMatrix() for _ in range(3))
        self.b=[]
        
        for i in range(3):
            if i==0:
                if not self.has_traction_bc:
                    assemble(self.LHS[i], tensor=self.A[i])
                    # no assemble
                    self.b.append([self.RHS[i][0], self.RHS[i][1]])
                else:
                    j=0
                    for key in self.has_traction_bc.keys():
                        if j == 0:
                            FBC = self.BoundaryTraction(self.fp[2], U, self.nu, mark=self.has_traction_bc[key][0], mode=self.has_traction_bc[key][1])
                        else:
                            FBC += self.BoundaryTraction(self.fp[2], U, self.nu, mark=self.has_traction_bc[key][0], mode=self.has_traction_bc[key][1])
                        j+=1
                    LBC = lhs(FBC)
                    RBC = rhs(FBC)
                    assemble(self.LHS[i]+LBC, tensor=self.A[i])
                    # no assemble
                    self.b.append((self.RHS[i][0]+RBC, self.RHS[i][1]))
            else:
                assemble(self.LHS[i], tensor=self.A[i])
                self.b.append((assemble(self.RHS[i][0]), assemble(self.RHS[i][1])))

        [bc.apply(self.A[0]) for bc in self.boundary_condition_V.bc_list]
        [bc.apply(self.A[1]) for bc in self.boundary_condition_Q.bc_list]   

    def _initialize_solver(self, method = 'lu', lusolver='mumps'):
        """
        Initialize the solver for the IPCS method.

        Parameters
        ----------
        method : str, optional
            Solver method ('lu' or 'krylov'). Default is 'lu'.
        lusolver : str, optional
            Type of LU solver. Default is 'mumps'.
        """
        
        self.solver=[]
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
        pending for test if it works

        Parameters
        ----------
        time_expr : Expression, Function or Constant, optional
            Time-dependent source term for the flow field. Default is None.

        Returns
        -------
        None.

        """
        if time_expr is not None:
            self.eqn.time_expr=time_expr
        self.b[0][1]=self.eqn.SourceTerm()
    
    def _update_solution(self):
        """
        Update the solution after each time step.
        """
        self.eqn.fu[1].assign(self.eqn.fu[0])
        self.eqn.fp[1].assign(self.eqn.fp[0])
        self.eqn.fu[2].assign(self.eqn.fu[1])
        self.eqn.fp[2].assign(self.eqn.fp[1])
        self.nstep += 1
    
    def solve(self, method='lu', lusolver='mumps',inner_iter_max=20, tol=1e-7,relax_factor=1.0,verbose=False):
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
            Convergence tolerance. Default is 1e-7.
        relax_factor : float, optional
            Relaxation factor for updating the solution. Default is 1.0.
        verbose : bool, optional
            If True, print iteration information. Default is False.
        """
        
        
        if self.nstep == 0 or self.bc_reset:
            self._assemble_system()
            self._initialize_solver(method=method, lusolver=lusolver)
            self._apply_matrix_bc() # if it is mode 1: reassembling matrices
            
        if self.bc_reset is not False:# bc reset
            self._apply_vector_bc() 
            self.bc_reset=False # swith flag after assembling matrices and vectors
            
        
        niter, eps = 0, 1
        
        while eps > tol and niter < inner_iter_max:
            # Step 1: Tentative velocity step
            b1 = self.Mat_vel * assemble(self.b[0][0]+self.b[0][1]) + self.Vec_vel # have assemble here
            self.solver[0].solve(self.eqn.fu[0].vector(), b1)
            
            # Step 2: Pressure correction step
            b2 = self.b[1][0] * self.eqn.fp[2].vector() + self.b[1][1] * self.eqn.fu[0].vector()
            b2 = self.Mat_pre * b2 + self.Vec_pre
            
            # 2nd step is poisson equation without dirichlet boundary condition
            if self.has_free_bc is not False: # have free outlet then pressure BC is pre-set to zero
                for key in self.FreeBoundary.keys():
                    b2 += self._initialize_free_boundary(func=self.eqn.fu[0],BC_dict=self.FreeBoundary[key]).vector()
            self.solver[1].solve(self.eqn.fp[0].vector(), b2)
            
            # Step 3: Velocity correction step
            b3 = self.b[2][0] * self.eqn.fu[0].vector() + self.b[2][1] * (self.eqn.fp[0].vector() - self.eqn.fp[2].vector())
            self.solver[2].solve(self.eqn.fu[0].vector(), b3)
            
            # eps
            eps = norm(self.eqn.fu[0].vector() - self.eqn.fu[2].vector(),'linf')
            
            # update for the next iter
            self.eqn.fu[2].vector()[:]=relax_factor*self.eqn.fu[0].vector()+(1.0-relax_factor)*self.eqn.fu[2].vector()
            self.eqn.fp[2].vector()[:]=relax_factor*self.eqn.fp[0].vector()+(1.0-relax_factor)*self.eqn.fp[2].vector()
            
            # 
            niter+=1
            if verbose and comm_rank == 0:
                print(f'inner_iter={niter}: norm={eps}')
                
        # Update previous solution
        self._update_solution()
        
            