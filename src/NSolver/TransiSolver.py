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

parameters["std_out_all_processes"] = False
comm_mpi4py = MPI.comm_world
comm_rank = comm_mpi4py.Get_rank()
comm_size = comm_mpi4py.Get_size()

    

class DNS_Newton(NewtonSolver):
    """
    Solver of Transient Navier-Stokes equations using Newton method


    Examples
    ----------------------------
    here is a snippet code shows how to use this class

    >>> see test 'CylinderTransiFlow.py'
    """
    def __init__(self, mesh, Re=None, dt=None, sourceterm=None, bodyforce=None, order=(2,1), dim=2, constrained_domain=None):
        """
        

        Parameters
        ----------
        mesh : TYPE
            DESCRIPTION.
        dt : TYPE, optional
            DESCRIPTION. The default is None.
        Re : TYPE, optional
            DESCRIPTION. The default is None.
        sourceterm : TYPE, optional
            DESCRIPTION. The default is None.
        bodyforce : TYPE, optional
            DESCRIPTION. The default is None.
        order : TYPE, optional
            DESCRIPTION. The default is (2,1).
        dim : TYPE, optional
            DESCRIPTION. The default is 2.
        constrained_domain : TYPE, optional
            DESCRIPTION. The default is None.
            
        Returns
        -------
        None.

        """
        NewtonSolver.__init__(self, mesh, Re, sourceterm, bodyforce, order, dim, constrained_domain)
        self.nstep = 0
        # note the order of the time discretization # explicit for newton method expression
        self.Transient=self.eqn.Transient(dt, scheme="backward", order=1, implicit=False)
        
            
    def initial(self, ic=None, noise=False, timestamp=0.0):
        """

        Parameters
        ----------
        ic : TYPE
            DESCRIPTION.
        noise : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        
        SetInitialCondition(1, ic=ic, fw=self.eqn.fw[1], noise=noise, timestamp=timestamp)


    def solve(self, dt=None, Re=None, sourceterm=None):
        """
        

        Parameters
        ----------
        dt : TYPE, optional
            DESCRIPTION. The default is None.
        Re : TYPE, optional
            DESCRIPTION. The default is None.
        sourceterm : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        mark=0
        
        if dt is not None and dt != self.eqn.dt:
            self.eqn.dt = dt
            mark = 1
        
        if Re is not None and Re != self.eqn.Re:
            self.eqn.Re = Re
            mark = 1
            
        if sourceterm != self.eqn.sourceterm:
            self.eqn.sourceterm = sourceterm
            mark = 1

    
        if mark == 1 or self.nstep == 0:
            self.__SNSEqn()
            self.SNS+=self.Transient
            self.__NewtonMethod()
        
        self.solver.solve()
        self.eqn.fw[1].assign(self.eqn.fw[0]) # note the order of the time discretization
        self.nstep += 1
            
#%%        
class DNS_IPCS(NSolverBase):
    """
    Implicit Pressure Correction Scheme
    """
    def __init__(self, mesh, Re, dt, sourceterm=None, bodyforce=None, order=(2,1), dim=2, constrained_domain=[None, None]):
        """
        

        Parameters
        ----------
        mesh : TYPE
            DESCRIPTION.
        dt : TYPE, optional
            DESCRIPTION. The default is None.
        Re : TYPE, optional
            DESCRIPTION. The default is None.
        sourceterm : TYPE, optional
            DESCRIPTION. The default is None.
        bodyforce : TYPE, optional
            DESCRIPTION. The default is None.
        order : TYPE, optional
            DESCRIPTION. The default is (2,1).
        dim : TYPE, optional
            DESCRIPTION. The default is 2.
        constrained_domain : TYPE, optional
            DESCRIPTION. The default is [None, None].

        Returns
        -------
        None.

        """
        self.mesh = mesh
        element = Decoupled(mesh = mesh, order = order, dim = dim, constrained_domain = constrained_domain) # initialise finite element space
        NSolverBase.__init__(self, mesh, element, Re, sourceterm, bodyforce)
        # boundary condition
        self.BCs_vel = SetBoundaryCondition(self.element.functionspace_V, self.boundary)
        self.BCs_pre = SetBoundaryCondition(self.element.functionspace_Q, self.boundary)
        # NS equations
        self.LHS, self.RHS = self.eqn.IPCS(dt)
        self.force_exp=self.eqn.force_init() 
  
        #
        self.nstep = 0
        # 
        self.freeoutlet=False
        
        
    def initial(self, ic=None, noise=False, timestamp=0.0, element_init=None):
        """
        

        Parameters
        ----------
        ic : TYPE, optional
            DESCRIPTION. The default is None.
        noise : TYPE, optional
            DESCRIPTION. The default is False.
        timestamp : TYPE, optional
            DESCRIPTION. The default is 0.0.
        element_init : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        SetInitialCondition(2, ic=ic, fw=self.eqn.fw[1], noise=noise, timestamp=timestamp, mesh=self.mesh, element_in = element_init, element_out = self.element)
        
    
    def set_boundarycondition(self, bc_dict, mark):
        """
        

        Parameters
        ----------
        bc_dict : TYPE
            DESCRIPTION.
        mark : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if bc_dict['FunctionSpace'] in ['FreeOutlet']:
            bc_dict['FunctionSpace']='Q'
            bc_dict['Value']=Constant(0.0)
            if self.freeoutlet is False:# Create a dictionary (if it doesn't already exist)
                self.FreeOutlet={}
            self.FreeOutlet['Boundary'+str(mark)]=self.__FreeOutlet(mark=mark)
            self.freeoutlet+=1
        
        # setup all BCs(including free-outlet)
        if bc_dict['FunctionSpace'][0] == 'V':
            self.BCs_vel.set_boundarycondition(bc_dict, mark)
        elif bc_dict['FunctionSpace'][0] == 'Q':
            self.BCs_pre.set_boundarycondition(bc_dict,mark)
            
    def parameters(self, param):
        """
        Set solve parameters

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
    
    def __BCviaMatrix_Mat(self):
        """

        Identity Matrix that contains only zeros in the rows which have Dirichlet boundary conditions

        Returns
        -------
        None.

        """
        self.Mat_vel = self.BC_vel.MatrixBC_rhs()
        self.Mat_pre = self.BC_pre.MatrixBC_rhs()

    def __BCviaMatrix_Vec(self):
        """
        
        Zero Vector that contains boundary condition values in the rows which have Dirichlet boundary conditions

        Returns
        -------
        None.

        """
        self.Vec_vel = self.BC_vel.VectorBC_rhs()
        self.Vec_pre = self.BC_pre.VectorBC_rhs()    
        
        
    def __FreeOutlet(self, mark=False, solver='mumps', func=None, BC_dict=None):
        """
        compute pressure on BC based on velocity field and free-outlet condition

        Parameters
        ----------
        mark : TYPE, optional
            DESCRIPTION. The default is False.
        solver : TYPE, optional
            DESCRIPTION. The default is 'mumps'.
        func : TYPE, optional
            DESCRIPTION. The default is None.
        BC_dict : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if type(mark) is int:# to initialise freeoutlet BCs
            n=self.eqn.n
            ds=self.eqn.ds
            u=self.eqn.tu
            p=self.eqn.tp
            q=self.eqn.q
            nu=Constant(1.0/self.eqn.Re)
            normal=self.element.add_functions()[0]
            normal_T=self.element.add_functions()[0]
            # norm vector?
            normal_vec=assemble(dot(n,sym(grad(u))*n)*ds(mark)+dot(u,n)*ds(mark)).get_local()
            normal_vec[abs(normal_vec) < 1e-10] = 0
            normal.vector()[:]=np.ascontiguousarray(normal_vec)
            assign(normal_T.sub(0),normal.sub(1))
            assign(normal_T.sub(1),normal.sub(0))
            # normalise?
            weight=1.0/np.sqrt((normal_T.vector()*normal_T.vector()+normal.vector()*normal.vector()).get_local())
            weight[np.abs(weight)==np.inf]=0.0
            normal.vector()[:]=normal_vec*weight
            # interpolate to velocity filed with 1st order?
            normal=interpolate(normal,VectorFunctionSpace(self.mesh, 'P', self.element.order[1]))
            norm=interpolate(normal,self.element.functionspace_V)
            # solver for pressure condition?
            RHS_mat=assemble(nu*dot(norm,sym(grad(u))*norm)*q*dx)
            LHS_mat=assemble(p*q*dx)
            solver_outlet = PETScLUSolver(solver)
            solver_outlet.set_operator(LHS_mat)
            solver_outlet.parameters.add('reuse_factorization', True)
            # mat for set BC values
            BC_outlet=DirichletBC(self.element.functionspace_Q, Constant(0.0) , self.boundary.boundaries, mark,method="geometric")
            MatBC_outlet = assemble(Constant(0.0)*dot(p, q) * dx) ## create a zero mat with pressure field size
            BC_outlet.apply(MatBC_outlet)
            
            return {'BC norm': norm, 'Outlet Solver': solver_outlet, 'Matrix for RHS': RHS_mat, 'Matrix for BC': MatBC_outlet, 'BC Values': self.element.add_functions()[1]}
        elif BC_dict is not None:# to compute exact BC values
            # RHS
            b=BC_dict['Matrix for RHS']*func.vector()
            # solver for pressure
            BC_dict['Outlet Solver'].solve(BC_dict['BC Values'].vector(), b)
            # apply boundary conditions
            BC_dict['BC Values'].vector()[:] = BC_dict['Matrix for BC']*BC_dict['BC Values'].vector()
            
            return BC_dict['BC Values']
        
    def __Assemble(self):
        """
        

        Returns
        -------
        None.

        """
        
        self.A=()
        self.b=[]
        self.solver=[]
        for i in range(3):
            self.A+=(PETScMatrix(),)
            assemble(self.LHS[i], tensor=self.A[i])
            
            self.b+=((assemble(self.RHS[i][0]), assemble(self.RHS[i][1])),)
            
        [bc.apply(self.LHS[0]) for bc in self.BCs_vel.bcs]
        [bc.apply(self.LHS[1]) for bc in self.BCs_pre.bcs]   

    def __Solve_Init(self, method = 'direct', lusolver='mumps'):
        """
        

        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is 'direct'.
        lusolver : TYPE, optional
            DESCRIPTION. The default is 'mumps'.

        Returns
        -------
        None.

        """
        self.solver=()
        if method == 'lu':
            for i in range(3):
                self.solver+=(PETScLUSolver(self.A[i],lusolver),)
                self.solver[i].parameters.add('reuse_factorization', True)
                
        elif method == 'krylov':
            for i in range(3):
                if i==1:
                    self.solver+=(KrylovSolver('cg', 'hypre_euclid'),)
                else:
                    self.solver+=(KrylovSolver('gmres', 'jacobi'),)
                    
                self.solver[i].set_operator(self.A[i])
                self.solver[i].parameters['absolute_tolerance'] = 1E-10
                self.solver[i].parameters['relative_tolerance'] = 1E-8
    
    def force(self, mark=None, direction=None):
        """
        Get the force on the body (lift or drag)

        Parameters
        ----------------------------
        bodymark : int
            the boundary mark of the body

        direction: int
            0 means X direction and 1 means Y direction

        Returns
        ----------------------------
        force : Fx or Fy

        """
        return assemble((self.force_exp[direction]) * self.eqn.ds(mark))
        
            
    def solve(self, method='lu', lusolver='mumps',iter_max=20, tol=1e-7,relax_factor=1.0):
        """
        

        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is 'direct'.
        lusolver : TYPE, optional
            DESCRIPTION. The default is 'mumps'.
        iterative_num : TYPE, optional
            DESCRIPTION. The default is 20.
        iterative_tol : TYPE, optional
            DESCRIPTION. The default is 1e-7.
        relax_factor : TYPE, optional
            DESCRIPTION. The default is 1.0.

        Returns
        -------
        None.

        """
        if self.nstep == 0:
            self.__Assemble()
            self.__Solver_Init(method=method, lusolver=lusolver)
            self.__BCviaMatrix_Mat()
            self.__BCviaMatrix_Vec()
            
        niter = 0
        eps = 1
        
        while eps > tol and niter < iter_max:
            # Step 1: Tentative velocity step
            b1 = self.Mat_vel * (self.b[0][0]+self.b[0][1]) + self.Vec_vel
            self.solver[0].solve(self.eqn.fu[0].vector(), b1)
            
            # Step 2: Pressure correction step
            b2 = self.b[1][0] * self.eqn.fp[2].vector() + self.b[1][1] * self.eqn.fu[0].vector()
            b2 = self.Mat_pre * b2 + self.Vec_pre
            
            if self.freeoutlet is not False:
                for key in self.FreeOutlet.keys():
                    b2 += self.__FreeOutlet(func=self.eqn.fu[0],BC_dict=self.FreeOutlet[key])
            self.solver[1].solve(self.eqn.fp[0].vector(), b2)
            
            # Step 3: Velocity correction step
            b3 = self.b[2][0] * self.eqn.fu[0].vector() + self.b[2][1] * (self.eqn.fp[0].vector() - self.eqn.fp[2].vector())
            self.solver[2].solve(self.eqn.fu[0].vector(), b3)
            
            # eps
            eps = norm(self.eqn.fu[0].vector() - self.eqn.fu[2].vector(),'linf')
            
            # update for the next iter
            self.eqn.fu[2].vector()[:]=relax_factor*self.eqn.fu[0].vector()+(1.0-relax_factor)*self.eqn.fu[2].vector()
            self.eqn.fp[2].vector()[:]=relax_factor*self.eqn.fp[0].vector()+(1.0-relax_factor)*self.eqn.fp[2].vector()
            
            # assemble RHS1, pending modify expression
            self.b[0]=(assemble(self.RHS[i][0]), assemble(self.RHS[i][1]))
            # 
            niter+=1
            if comm_rank == 0:
                info('iter=%d: norm=%g' % (niter, eps))
                
                
        # Update previous solution
        self.eqn.fu[1].assign(self.eqn.fu[0])
        self.eqn.fp[1].assign(self.eqn.fp[0])
        self.eqn.fu[2].assign(self.eqn.fu[0])
        self.eqn.fp[2].assign(self.eqn.fp[0])
        self.nstep += 1
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        