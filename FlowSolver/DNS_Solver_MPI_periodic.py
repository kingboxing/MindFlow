from __future__ import print_function
from fenics import *
import mpi4py as pympi
import numpy as np
import copy
import os,sys,inspect
from .FiniteElement import TaylorHood, SplitPV, PoissonPR
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0,parentdir)
from Boundary.Set_BoundaryCondition import BoundaryCondition


# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False
comm_mpi4py = mpi_comm_world().tompi4py()
comm_rank = comm_mpi4py.Get_rank()
comm_size = comm_mpi4py.Get_size()

class DNS_IPCS_Solver:
    def __init__(self, mesh, boundary, nu=None, dt=None, path=None, noise=False, sourceterm=None,mass_rate=None, restart=0.0,order=(2,1),dim=2, constrained_domain=[None,None]):
        """
        :param mesh:
        :param boundary:
        :param nu:
        :param dt:
        :param path:
        :param noise:
        :param sourceterm:
        :param restart: choose solution at specified restart time stored in 'path' as
                        initial condition to continue the calculation
        """
        element = SplitPV(mesh = mesh,order=order,dim=dim, constrained_domain=constrained_domain) # initialise finite element space
        # element for poisson equation
        self.element_poi=PoissonPR(mesh=mesh, order=(order[1],0), constrained_domain=constrained_domain[1])
        # inherit attributes : functionspace, boundries, bcs
        self.BC_vel = BoundaryCondition(Functionspace=element.functionspace_V, boundary=boundary)
        self.BC_pre = BoundaryCondition(Functionspace=element.functionspace_Q, boundary=boundary)
        self.clean_BC=False
        # create public attributes
        self.stepcount = 0
        self.nu = nu
        self.dt = dt
        self.path = path
        if sourceterm is not None and sourceterm is not False:
            self.sourceterm = sourceterm
        else:
            self.sourceterm = Constant((0.0,0.0))
            
        self.mass_rate=mass_rate
        self.solver_para = {}
        # assign functions for the expression convenience
        # trial functions
        self.u=element.tu
        self.p=element.tp
        # test functions
        self.v=element.v
        self.q=element.q
        # current time step functions
        self.u_ = element.u
        self.p_ = element.p
        # pre time step functions
        self.u_pre, self.p_pre = element.add_functions()
        # get FacetNormal and boundary measurement
        self.boundaries = boundary.boundaries        
        self.n = FacetNormal(mesh)
        self.ds = boundary.get_measure()
        self.element = element
        self.mesh=mesh
        
        # test 
        self.pq = self.element_poi.q
        self.pd = self.element_poi.d
        # trial
        self.pp = self.element_poi.tp
        self.pc = self.element_poi.tc
        # function
        self.pw_ = self.element_poi.w
        self.pp_ = self.element_poi.p
        self.pc_ = self.element_poi.c
        # initial condition
        self.__InitialCondition(noise=noise, restart=restart)

    def __InitialCondition(self, noise=False, restart=0.0):
        if self.path is not None:
            time_bool = False
            hdf = HDF5File(self.mesh.mpi_comm(), self.path, 'r')
            if hdf.has_dataset('Velocity') and hdf.has_dataset('Pressure'):
                for ts in range(hdf.attributes('Velocity')['count']):
                    time_bool = (hdf.attributes('Velocity/vector_' + str(ts))['timestamp']== restart)
                    if time_bool:
                        restart = ts
                if time_bool:
                    print('Set vector_'+str(restart)+'as the initial condition')
                else:
                    raise Exception('Timestamp not found')
                hdf.read(self.u_pre, 'Velocity/vector_' + str(restart))
                hdf.read(self.p_pre, 'Pressure/vector_' + str(restart))
            elif hdf.has_dataset('Coupled Field'):
                for ts in range(hdf.attributes('Coupled Field')['count']):
                    time_bool = (hdf.attributes('Coupled Field/vector_' + str(ts))['timestamp'] == restart)
                    if time_bool:
                        restart = ts
                if time_bool:
                    print('Set vector_' + str(restart) + ' as the initial condition')
                else:
                    raise Exception('Timestamp not found')
                element_base = TaylorHood(mesh=self.mesh)
                w = element_base.w
                (vel, pre) = split(w)
                hdf.read(w, 'Coupled Field/vector_' + str(restart))
                self.p_pre = project(pre, self.BC_pre.functionspace, solver_type='gmres')
                self.u_pre = project(vel, self.BC_vel.functionspace, solver_type='gmres')
        if noise is True:
            Coor_fun = np.sign(self.element.functionspace_V.tabulate_dof_coordinates().reshape((-1, 2))[:, 1])
            force_pertub = np.multiply((np.random.rand(self.element.functionspace_V.dim())) * self.nu.values()[0],
                                     Coor_fun)
            self.noise = self.element.add_functions()[0] #Constant((0, force_pertub))
            self.noise.vector()[:] = np.ascontiguousarray(force_pertub)
        elif noise is False or noise is None:
            self.noise = Constant((0.0, 0.0))
        elif noise is not None:
            self.noise = noise

    def epsilon(self, u):
        return sym(nabla_grad(u))

    def sigma(self, u, p):
        # Define stress tensor
        return 2 * self.nu * self.epsilon(u) - p * Identity(len(u))


    def set_boundarycondition(self, bounarycondition, mark, reset=True):
        boucon=bounarycondition.copy()
        ## number of boundary conditions        
        try:
            self.bc_num
        except:
            self.bc_num=1
            ## bc resetting flag
            if reset==True:
                self.bcreset = 1
            elif reset==False:
                self.bcreset = 0 
        else:
            if reset is True and self.clean_BC is True:
                self.bc_num = 1
                self.bcreset = 1
            else:
                self.bc_num +=1
       
        ## clean relvent bc information if clean BC
        if reset is True and self.clean_BC is True:
            self.BC_pre.bcs=[]
            self.BC_vel.bcs=[]            
            ## if there is freeoutlet bc, clean it
            try: 
                test=self.FreeOutlet
            except:
                pass
            else:
#                if self.freeoutlet is True:
#                    self.bcreset=-1*self.bcreset
                del self.FreeOutlet
            try: 
                test=self.freeoutlet
            except:
                pass 
            else:
                del self.freeoutlet
            self.clean_BC=False
            
        ## if it is a Dirichlet boundary condition
        try:
            test = boucon['FunctionSpace']
        except:
            boucon['FunctionSpace']=None
            info('No Dirichlet Boundary Condition at Boundary % g' % mark)
        else:
            pass
        
        try:
            test = boucon['Value']
        except:
            boucon['Value']=None
            info('No Dirichlet Boundary Condition at Boundary % g' % mark)
        else:
            pass
        ## set BC
        if (boucon['FunctionSpace'] in ['Free Outlet','FreeOutlet','freeoutlet','free outlet'] or boucon['Value'] in ['Free Outlet','FreeOutlet','freeoutlet','free outlet']):
            boucon['FunctionSpace']='Q'
            boucon['Value']=Constant(0.0)
            self.BC_pre.set_boundarycondition(boucon,mark)
            try: 
                test=self.FreeOutlet
            except:
                self.FreeOutlet={}
            else:
                pass
            key='Boundary'+str(mark)
            self.FreeOutlet[key]=self.__FreeOutlet(mark=mark,init=True)
            self.freeoutlet=True
        elif boucon['FunctionSpace'][0] == 'V':
            self.BC_vel.set_boundarycondition(boucon, mark)
        elif boucon['FunctionSpace'][0] == 'Q':
            self.BC_pre.set_boundarycondition(boucon,mark)

            
    def __FreeOutlet(self, mark=None, func=None, init=False,solver='mumps',BC_dict=None):
        if init is True and BC_dict is None:
            normal_vec=assemble(dot(self.n,sym(grad(self.u))*self.n)*self.ds(mark)+dot((self.u),self.n)*self.ds(mark))
            normal_numpy=normal_vec.get_local()
            normal_numpy[abs(normal_numpy) < 1e-10] = 0
            normal_vec[:]=normal_numpy        
            normal=self.element.add_functions()[0]
            normal.vector()[:]=normal_vec
            
            vec_x=normal.sub(0)
            vec_y=normal.sub(1)
            
            normal_T=self.element.add_functions()[0]
            assign(normal_T.sub(0),vec_y)
            assign(normal_T.sub(1),vec_x)
            
            weight=1.0/np.sqrt((normal_T.vector()*normal_T.vector()+normal.vector()*normal.vector()).get_local())
            weight[np.abs(weight)==np.inf]=0.0
            normal.vector()[:]=normal_vec*weight
        
            normal=interpolate(normal,VectorFunctionSpace(self.mesh, 'P', 1))
            self.norm=interpolate(normal,self.element.functionspace_V)

            A=assemble(self.p*self.q*dx)
            self.Mat_outlet=assemble(self.nu*dot(self.norm,sym(grad(self.u))*self.norm)*self.q*dx)
            self.solver_outlet = PETScLUSolver(solver)
            self.solver_outlet.set_operator(A)
            self.solver_outlet.parameters['reuse_factorization'] = True
            self.p_outlet = self.element.add_functions()[1]
            
            BC_outlet=DirichletBC(self.element.functionspace_Q, Constant(0.0) , self.boundaries, mark, method="geometric")
            self.MatBC_outlet = assemble(Constant(0.0)*dot(self.p, self.q) * dx)
            BC_outlet.apply(self.MatBC_outlet)
            return {'BC norm': self.norm, 'Outlet Solver': self.solver_outlet, 'Matrix for RHS': self.Mat_outlet, 'Matrix for BC': self.MatBC_outlet, 'BC Values': self.p_outlet}
        
        if func is not None and BC_dict is not None:
            Mat_outlet=BC_dict['Matrix for RHS']
            solver_outlet=BC_dict['Outlet Solver']
            p_outlet=BC_dict['BC Values']
            MatBC_outlet=BC_dict['Matrix for BC']
            b=Mat_outlet*func.vector()
            solver_outlet.solve(p_outlet.vector(), b)
            p_outlet.vector()[:] = MatBC_outlet*p_outlet.vector()
            return p_outlet
        
    def __PressureOutlet(self,mark):
        pass
        
        

    def __NS_expression(self):
        self.u_mid, self.p_mid = self.element.add_functions()
        self.u_mid.assign(self.u_pre)
        self.p_mid.assign(self.p_pre)

        U = 0.5 * (self.u_mid + self.u)
        # Define variational problem for step 1, Implicit-time integration
        F1 = dot((self.u - self.u_pre) / Constant(self.dt), self.v) * dx \
             + dot(dot(self.u_mid, nabla_grad(self.u_mid)), self.v) * dx \
             + inner(self.sigma(U, self.p_mid), self.epsilon(self.v)) * dx
        
#        try:
#            test=self.freeoutlet
#        except:
#            F1 = F1+dot(self.p_mid*self.n, self.v)*self.ds - dot(self.nu*nabla_grad(U)*self.n, self.v)*self.ds 
#        else:
#            if self.freeoutlet is not True:
#                F1 = F1+dot(self.p_mid*self.n, self.v)*self.ds - dot(self.nu*nabla_grad(U)*self.n, self.v)*self.ds
            
        a1 = lhs(F1)
        self.A1=PETScMatrix()
        assemble(a1, tensor=self.A1)
        self.L1 = rhs(F1)
#%%
        # Define variational problem for step 2
#        a2 = dot(nabla_grad(self.p), nabla_grad(self.q)) * dx
#        self.A2=PETScMatrix()
#        assemble(a2, tensor=self.A2)
#        # L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx
#        L2_1 = dot(nabla_grad(self.p), nabla_grad(self.q)) * dx  ##for p_n
#        L2_2 = Constant(-1.0 / self.dt) * div(self.u) * self.q * dx  ## u_
#        self.b2_1 = assemble(L2_1)
#        self.b2_2 = assemble(L2_2)


        a2 = (dot(nabla_grad(self.pp), nabla_grad(self.pq)) + self.pc*self.pq + self.pp*self.pd) * dx
        self.A2=PETScMatrix()
        assemble(a2, tensor=self.A2)
        
        L2_1 = dot(nabla_grad(self.p), nabla_grad(self.pq)) * dx  ##for p_n
        L2_2 = Constant(-1.0 / self.dt) * div(self.u) * self.pq * dx  ## u_
        self.b2_1 = assemble(L2_1)
        self.b2_2 = assemble(L2_2)
#%%        
        # Define variational problem for step 3
        a3 = dot(self.u, self.v) * dx
        self.A3=PETScMatrix()
        assemble(a3, tensor=self.A3)
        # L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx
        L3_1 = dot(self.u, self.v) * dx  ##u_
        L3_2 = -Constant(self.dt) * dot(nabla_grad(self.p), self.v) * dx  ##p_-p_n
        self.b3_1 = assemble(L3_1)
        self.b3_2 = assemble(L3_2)

        [bc.apply(self.A1) for bc in self.BC_vel.bcs]
        [bc.apply(self.A2) for bc in self.BC_pre.bcs]

#    def __BCviaMatrix(self):
#        I = assemble(Constant(0.0)*dot(self.u, self.v) * dx)
#        I.ident_zeros()
#        bcvel = assemble(Constant(0.0)*dot(self.u, self.v) * dx)
#        [bc.apply(bcvel) for bc in self.BC_vel.bcs]
#        self.Mat_vel = I - bcvel
#        bcvel = assemble(dot(Constant((0.0, 0.0)), self.v) * dx)
#        [bc.apply(bcvel) for bc in self.BC_vel.bcs]
#        self.Vec_vel = bcvel
#
#        I = assemble(Constant(0.0) * dot(self.p, self.q) * dx)
#        I.ident_zeros()
#        bcpre = assemble(Constant(0.0)*dot(self.p, self.q) * dx)
#        [bc.apply(bcpre) for bc in self.BC_pre.bcs]
#        self.Mat_pre = I - bcpre
#        bcpre = assemble(dot(Constant(0.0), self.q) * dx)
#        [bc.apply(bcpre) for bc in self.BC_pre.bcs]
#        self.Vec_pre = bcpre

    def __BCviaMatrix_Mat(self):
        self.Mat_vel = self.BC_vel.MatrixBC_rhs()
        self.Mat_pre = self.BC_pre.MatrixBC_rhs()

    def __BCviaMatrix_Vec(self):
        self.Vec_vel = self.BC_vel.VectorBC_rhs()
        self.Vec_pre = self.BC_pre.VectorBC_rhs()        

    def __solver_init(self, method = 'iterative', lusolver='mumps'):
        """Initialise IPCS solver
        Iterative solver using FEniCS build-in functions
        """
        if method == 'iterative':
            # solver parameters
            self.solver1 = KrylovSolver('gmres', 'jacobi')
            self.solver1.set_operator(self.A1)
            prm1 = self.solver1.parameters
            prm1.absolute_tolerance = 1E-10
            prm1.relative_tolerance = 1E-8

            self.solver2 = KrylovSolver('cg', 'hypre_euclid')#'cg', 'ilu')
            self.solver2.set_operator(self.A2)
            prm2 = self.solver2.parameters
            prm2.absolute_tolerance = 1E-10
            prm2.relative_tolerance = 1E-8

            self.solver3 = KrylovSolver('gmres', 'jacobi')
            self.solver3.set_operator(self.A3)
            prm3 = self.solver3.parameters
            prm3.absolute_tolerance = 1E-10
            prm3.relative_tolerance = 1E-8
        elif method == 'direct':
            self.solver1 = PETScLUSolver(self.A1,lusolver)
            #self.solver1.parameters['reuse_factorization'] = True

            self.solver2 = PETScLUSolver(self.A2,lusolver)
            #self.solver2.parameters['reuse_factorization'] = True

            self.solver3 = PETScLUSolver(self.A3,lusolver)
            #self.solver3.parameters['reuse_factorization'] = True
            try:
                dolver=dolfin.__version__
            except:
                pytver=dolfin.__pythonversion__
                if float(pytver[0:3])>2.7:
                    self.solver1.parameters.add('reuse_factorization', True)
                    self.solver2.parameters.add('reuse_factorization', True)
                    self.solver3.parameters.add('reuse_factorization', True)
                elif float(dolver[0:4])==2.7:
                    self.solver1.parameters['reuse_factorization'] = True
                    self.solver2.parameters['reuse_factorization'] = True
                    self.solver3.parameters['reuse_factorization'] = True
            else:
                if int(dolver[0:4])>2017:
                    self.solver1.parameters.add('reuse_factorization', True)
                    self.solver2.parameters.add('reuse_factorization', True)
                    self.solver3.parameters.add('reuse_factorization', True)
                elif int(dolver[0:4])==2017:
                    self.solver1.parameters['reuse_factorization'] = True
                    self.solver2.parameters['reuse_factorization'] = True
                    self.solver3.parameters['reuse_factorization'] = True

    def solve(self, method='iterative', lusolver='mumps',iterative_num=20,iterative_tol=1e-7,relax_factor=1.0):
        
        if self.bcreset not in [-1,0,1,2]:
            raise ValueError('Please specify bcreset value from the list [-1,0,1,2]')  
       
        if self.stepcount == 0 or self.bcreset==-1:
            """
            if it's the first step (stepcount=0) or NS expression changed with boundary condition (bcreset=-1)
            """ 
            self.__NS_expression()
            self.__solver_init(method=method, lusolver=lusolver)
            self.__BCviaMatrix_Mat()
            self.__BCviaMatrix_Vec()
            self.bcreset = 0
            
         
        if self.bcreset == 1:         
            """
            if resetting boundary condition with different values and locations, then bcreset=1. 
            """   
            self.__BCviaMatrix_Mat()
            self.__BCviaMatrix_Vec()
            self.bcreset = 0
        
        if self.bcreset == 2:
            """
            if only changing values in boundary conditions, then bcreset=2
            """
            self.__BCviaMatrix_Vec()
            self.bcreset = 0            
        
        iter_max = iterative_num
        iter_num = 0
        eps = 1
        mass_delta=0
        tol = iterative_tol
        timer=Timer()
        while eps > tol and iter_num < iter_max:
            iter_num += 1
#            while np.abs(mass_delta-1)>1e-4:
            # Step 1: Tentative velocity step
            b1 = assemble(self.L1 + self.sourceterm[0]* self.v[0]*dx + self.sourceterm[1]* self.v[1]*dx + dot(self.noise, self.v)*dx)
            b1 = self.Mat_vel * b1  + self.Vec_vel
            #[bc.apply(b1) for bc in self.BC_vel.bcs]
            self.solver1.solve(self.u_.vector(), b1)
                
                
#                if self.mass_rate is not None:
#                    mass_delta=assemble(inner(self.u_,self.n)*self.ds(self.mass_rate[1]))/self.mass_rate[0]
#                    self.sourceterm = Constant((self.sourceterm.values()[0]/(100*mass_delta-99),0.0))
#                else:
#                    mass_delta=1.0          
#                info('mass-delta: %g' %mass_delta)
            #%%
            # Step 2: Pressure correction step
#            b2 = self.b2_1 * self.p_mid.vector() + self.b2_2 * self.u_.vector()
#            b2 = self.Mat_pre * b2 + self.Vec_pre
#
#            try:
#                if self.freeoutlet==True:
#                    for key in self.FreeOutlet.keys():
#                        self.p_outlet=self.__FreeOutlet(func=self.u_,BC_dict=self.FreeOutlet[key])
#                        b2 = b2 + self.p_outlet.vector()
#            except:
#                pass
#            else:
#                pass#b2 = b2 + p_outlet.vector()
#            
#            #[bc.apply(b2) for bc in self.BC_pre.bcs]
#            self.solver2.solve(self.p_.vector(), b2)
            
            b2 = self.b2_1 * self.p_mid.vector() + self.b2_2 * self.u_.vector()
            #b2 = self.Mat_pre * b2 + self.Vec_pre #size mismatch here, BC_pre,BC_vel
            
            self.solver2.solve(self.pw_.vector(), b2)
            assign(self.p_, self.pw_.sub(0))
            #%%
            # Step 3: Velocity correction step
            b3 = self.b3_1 * self.u_.vector() + self.b3_2 * (self.p_.vector() - self.p_mid.vector())
            self.solver3.solve(self.u_.vector(), b3)

            diff = self.u_.vector() - self.u_mid.vector()
            eps = norm(diff,'linf')
            # diff = self.u_.vector().array() - self.u_mid.vector().array()
            # eps = np.linalg.norm(diff, ord=np.Inf)
            # eps = comm_mpi4py.reduce(eps, root=0, op=pympi.MPI.MAX)
            # eps = comm_mpi4py.bcast(eps if comm_rank == 0 else None, root=0)
            self.mass_rate[0]=assemble(inner(self.u_,self.n)*self.ds(self.mass_rate[1]))
            if comm_rank == 0:
                info('iter=%d: norm=%g, mass:%e' % (iter_num, eps, self.mass_rate[0]))
            
            self.u_mid.vector()[:]=relax_factor*self.u_.vector()+(1.0-relax_factor)*self.u_mid.vector()
            self.p_mid.vector()[:]=relax_factor*self.p_.vector()+(1.0-relax_factor)*self.p_mid.vector()
#            self.u_mid.assign(self.u_)  # update for next iteration
#            self.p_mid.assign(self.p_)
            
        # Update previous solution
        self.u_pre.assign(self.u_)
        self.p_pre.assign(self.p_)
        self.stepcount += 1

    def __force_init(self, bodymark=None):
        """Force expression (lift and drag)

        """

        I = Identity(self.u.geometric_dimension())
        D = sym(grad(self.u))
        T1 = -self.p * I
        T2 = 2.0 * self.nu * D
        force1 = T1 * self.n
        force2 = T2 * self.n
        self.drag1 = assemble((force1[0]) * self.ds(bodymark))
        self.drag2 = assemble((force2[0]) * self.ds(bodymark))
        self.lift1 = assemble((force1[1]) * self.ds(bodymark))
        self.lift2 = assemble((force2[1]) * self.ds(bodymark))

    def get_force(self, bodymark=None, direction=None):
        if self.stepcount == 0 or self.stepcount == 1:
            self.__force_init(bodymark=bodymark)
        if direction == 0:
            return (self.drag1.inner(self.p_pre.vector())+self.drag2.inner(self.u_pre.vector()))
        elif direction == 1:
            return (self.lift1.inner(self.p_pre.vector())+self.lift2.inner(self.u_pre.vector()))

class DNS_Newton_Solver(BoundaryCondition):
    def __init__(self, mesh, boundary, nu=None, dt=None, path=None, noise=False, sourceterm=None):
        element = TaylorHood(mesh = mesh) # initialise finite element space
        element_pre = TaylorHood(mesh = mesh)
        # inherit attributes : functionspace, boundries, bcs
        BoundaryCondition.__init__(self, Functionspace=element.functionspace, boundary=boundary)
        # create public attributes
        self.stepcount = 0
        self.nu = nu
        self.dt = dt
        self.path = path
        self.sourceterm = sourceterm
        self.solver_para = {}
        # assign functions for the expression convenience
        # functions
        self.w=element.w
        self.u=element.u
        self.p=element.p
        # pre time step functions
        self.w_pre = element_pre.w
        self.u_pre = element_pre.u
        self.p_pre = element_pre.p
        # test functions
        self.v=element.v
        self.q=element.q
        # trial function
        self.dw=element.tw
        # get FacetNormal and boundary measurement
        self.n = FacetNormal(mesh)
        self.ds = boundary.get_measure()
        self.dim = element.functionspace.dim()
        # get the coordinates of the whole functionspace
        self.Coor_fun = self.functionspace.tabulate_dof_coordinates().reshape((-1, 2))
        # initial condition
        self.__InitialCondition(noise=noise)
        # initialise force
        if nu is not None:
            self.__force_init()

        if dt is not None and nu is not None:
            self.__NS_expression()

    def __InitialCondition(self, noise=False):
        pass

    def __NS_expression(self):
        pass

    def __force_init(self):
        pass