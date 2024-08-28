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
    Solver of Transient Navier-Stokes equations using Newton method


    Examples
    ----------------------------
    here is a snippet code shows how to use this class

    >>> see test 'CylinderTransiFlow.py'
    """
    def __init__(self, mesh, Re=None, dt=None, const_expr=None, time_expr=None, order=(2,1), dim=2, constrained_domain=None):
        """
        

        Parameters
        ----------
        mesh : TYPE
            DESCRIPTION.
        dt : TYPE, optional
            DESCRIPTION. The default is None.
        Re : TYPE, optional
            DESCRIPTION. The default is None.
        const_expr : TYPE, optional
            DESCRIPTION. The default is None.
        time_expr : TYPE, optional
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
        NewtonSolver.__init__(self, mesh, Re, const_expr, order, dim, constrained_domain)
        self.nstep = 0
        self.bc_reset=False# pending
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
        assign(self.eqn.fw[0],self.eqn.fw[1])
        assign(self.eqn.fw[0],self.eqn.fw[1])

    def solve(self, dt=None, Re=None, time_expr=None):
        """
        

        Parameters
        ----------
        dt : TYPE, optional
            DESCRIPTION. The default is None.
        Re : TYPE, optional
            DESCRIPTION. The default is None.
        const_expr : TYPE, optional
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
            
        if time_expr != self.eqn.time_expr:
            self.eqn.time_expr = time_expr
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
    def __init__(self, mesh, Re, dt, const_expr=None, time_expr=None, order=(2,1), dim=2, constrained_domain=[None, None]):
        """
        

        Parameters
        ----------
        mesh : TYPE
            DESCRIPTION.
        dt : TYPE, optional
            DESCRIPTION. The default is None.
        Re : TYPE, optional
            DESCRIPTION. The default is None.
        const_expr : TYPE, optional
            DESCRIPTION. The default is None.
        time_expr : TYPE, optional
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
        NSolverBase.__init__(self, mesh, element, Re, const_expr, time_expr)
        # boundary condition
        self.boundary_condition_V = SetBoundaryCondition(self.element.functionspace_V, self.boundary) # velocity field
        self.boundary_condition_Q = SetBoundaryCondition(self.element.functionspace_Q, self.boundary) # pressure field
        # NS equations
        self.LHS, self.RHS = self.eqn.IPCS(dt)
  
        #
        self.has_free_bc = False
        self.bc_reset=False
        self.nstep = 0
        
        
        
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
        assign(self.eqn.fw[0][0],self.eqn.fw[1][0])
        assign(self.eqn.fw[0][1],self.eqn.fw[1][1])
        
    def set_boundarycondition(self, bc_list=None, reset=True):
        """
        

        Parameters
        ----------
        bc_list : dict, optional
            DESCRIPTION. The default is None.
        reset : TYPE, optional
            DESCRIPTION. The default is True.
            False: no reset
            True or 1 : mode 1, reset BC locations and values
            2: mode 2, only reset values

        Returns
        -------
        None.

        """
        
        if bc_list is None:
            bc_list=self.boundary.bc_list
        
        for key in bc_list.keys():
            self.__set_boundarycondition(bc_list[key], key)
        
        self.bc_reset=reset # reset boundary conditions mode 1 (reset everything) # due to matrix/vector method to employ boundary conditions
        
    
    def __set_boundarycondition(self, bc_dict, mark):
        """
        

        Parameters
        ----------
        bc_dict : dict
            DESCRIPTION.
        mark : TYPE
            DESCRIPTION.
        Returns
        -------
        None.

        """
        
        " pending for dealing with free boundary/ zero boundary traction condition in bc_list"
        
        if 'Free Boundary' in (bc_dict['FunctionSpace'],bc_dict['Value']):
            bc_dict['FunctionSpace']='Q'
            bc_dict['Value']=Constant(0.0)
            if self.has_free_bc is False:# Create a dictionary (if it doesn't already exist)
                self.FreeBoundary={}
            self.FreeBoundary['Boundary'+str(mark)]=self.__FreeBoundary(mark=mark)
            self.has_free_bc+=1
            info('Free boundary condition (zero boundary traction) applied at Boundary % g' % mark)
        
        # setup all BCs(including free-outlet)
        if bc_dict['FunctionSpace'][0] == 'V':
            self.boundary_condition_V.set_boundarycondition(bc_dict, mark)
        elif bc_dict['FunctionSpace'][0] == 'Q':
            self.boundary_condition_Q.set_boundarycondition(bc_dict,mark)
            
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
        self.Mat_vel = self.boundary_condition_V.MatrixBC_rhs()
        self.Mat_pre = self.boundary_condition_Q.MatrixBC_rhs()

    def __BCviaMatrix_Vec(self):
        """
        
        Zero Vector that contains boundary condition values in the rows which have Dirichlet boundary conditions

        Returns
        -------
        None.

        """
        self.Vec_vel = self.boundary_condition_V.VectorBC_rhs()
        self.Vec_pre = self.boundary_condition_Q.VectorBC_rhs()    
        
        
    def __FreeBoundary(self, mark=False, solver='mumps', func=None, BC_dict=None):
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
            #%% normalise? 
            weight_recip=np.sqrt((normal_T.vector()*normal_T.vector()+normal.vector()*normal.vector()).get_local())
            weight_recip[np.abs(weight_recip)==0.0]=np.inf
            weight=1.0/weight_recip
            
            # weight=1.0/np.sqrt((normal_T.vector()*normal_T.vector()+normal.vector()*normal.vector()).get_local())
            # # divide by zero encountered in divide
            # weight[np.abs(weight)==np.inf]=0.0
            #%%
            normal.vector()[:]=normal_vec*weight
            # interpolate to velocity filed with 1st order?
            normal=interpolate(normal,VectorFunctionSpace(self.mesh, 'P', self.element.order[1]))
            norm=interpolate(normal,self.element.functionspace_V)
            # solver for pressure condition?
            RHS_mat=assemble(nu*dot(norm,sym(grad(u))*norm)*q*dx)
            LHS_mat=assemble(p*q*dx)
            
            solver_outlet = PETScLUSolver(LHS_mat.instance())
            #solver_outlet.set_operator(LHS_mat)
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
        for i in range(3):
            self.A+=(PETScMatrix(),)
            if i==0:
                # if self.has_free_bc is False:# add residual term # should give pressure outlet BC 
                #     assemble(self.LHS[i]+self.LHS[3], tensor=self.A[i])
                #     # no assemble
                #     self.b+=((self.RHS[i][0]+self.RHS[3][0], self.RHS[i][1]),)
                # else:
                #     assemble(self.LHS[i], tensor=self.A[i])
                #     # no assemble
                #     self.b+=((self.RHS[i][0], self.RHS[i][1]),)
                if not self.has_traction_bc:
                    assemble(self.LHS[i], tensor=self.A[i])
                    # no assemble
                    self.b+=((self.RHS[i][0], self.RHS[i][1]),)
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
                    self.b+=((self.RHS[i][0]+RBC, self.RHS[i][1]),)
            else:
                assemble(self.LHS[i], tensor=self.A[i])
                self.b+=((assemble(self.RHS[i][0]), assemble(self.RHS[i][1])),)

        [bc.apply(self.A[0]) for bc in self.boundary_condition_V.bc_list]
        [bc.apply(self.A[1]) for bc in self.boundary_condition_Q.bc_list]   

    def __Solver_Init(self, method = 'lu', lusolver='mumps'):
        """
        

        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is 'lu'.
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
            
    def sourceterm(self, const_expr=None, time_expr=None):
        """
        

        Parameters
        ----------
        const_expr : TYPE
            DESCRIPTION.
        time_expr : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if const_expr is not None:
            self.eqn.const_expr=const_expr
        if time_expr is not None:
            self.eqn.time_expr=time_expr
        
    
    def solve(self, method='lu', lusolver='mumps',inner_iter_max=20, tol=1e-7,relax_factor=1.0,verbose=False):
        """
        

        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is 'lu'.
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
        if self.nstep == 0 or self.bc_reset is True:
            self.__Assemble()
            self.__Solver_Init(method=method, lusolver=lusolver)
            self.__BCviaMatrix_Mat() # if it is mode 1: reassembling matrices
            
        if self.bc_reset is not False:# bc reset
            self.__BCviaMatrix_Vec() 
            self.bc_reset=False # swith flag after assembling matrices and vectors
            
        
        niter = 0
        eps = 1
        
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
                    b2 += self.__FreeBoundary(func=self.eqn.fu[0],BC_dict=self.FreeBoundary[key]).vector()
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
            if verbose is True:
                if comm_rank == 0:
                    print('inner_iter=%d: norm=%g' % (niter, eps))
                
                
        # Update previous solution
        self.eqn.fu[1].assign(self.eqn.fu[0])
        self.eqn.fp[1].assign(self.eqn.fp[0])
        self.eqn.fu[2].assign(self.eqn.fu[1])
        self.eqn.fp[2].assign(self.eqn.fp[1])
        self.nstep += 1
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        