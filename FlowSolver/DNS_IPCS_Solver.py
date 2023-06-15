from __future__ import print_function
from fenics import *
import numpy as np
from scipy.sparse import dia_matrix
import scipy.sparse.linalg as spla
from .FiniteElement import *
import os,sys,inspect
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0,parentdir) 
from Boundary.Set_BoundaryCondition import BoundaryCondition
from FrequencyAnalysis.MatrixAssemble import MatrixAssemble

"""This module provides the classes that solve Navier-Stokes equations
"""

class DNS_IPCS_Solver(MatrixAssemble):
    def __init__(self, mesh, boundary, nu=None, dt=None, path=None, noise=False, sourceterm=None, restart=0.0):
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
        MatrixAssemble.__init__(self)
        element = SplitPV(mesh = mesh) # initialise finite element space
        # inherit attributes : functionspace, boundries, bcs
        self.BC_vel = BoundaryCondition(Functionspace=element.functionspace_V, boundary=boundary)
        self.BC_pre = BoundaryCondition(Functionspace=element.functionspace_Q, boundary=boundary)
        # create public attributes
        self.stepcount = 0
        self.nu = nu
        self.dt = dt
        self.path = path
        self.sourceterm = sourceterm
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
        self.n = FacetNormal(mesh)
        self.ds = boundary.get_measure()
        self.element = element
        # initial condition
        self.__InitialCondition(noise=noise,mesh=mesh, restart=restart)

    def epsilon(self, u):
        return sym(nabla_grad(u))

    def sigma(self, u, p):
        # Define stress tensor
        return 2 * self.nu * self.epsilon(u) - p * Identity(len(u))

    def __InitialCondition(self, noise=False, mesh=mesh, restart=0.0):
        """Assign base flow to the function w
        """
        if self.path is not None:
            if type(self.path) is not dict:
                element_base = TaylorHood(mesh=mesh)
                w = element_base.w
                (vel, pre) = split(w)
                timeseries_flow = TimeSeries(self.path)
                timeseries_flow.retrieve(w.vector(), restart)
                if noise is True:
                    Coor_fun = np.sign(element_base.functionspace.tabulate_dof_coordinates().reshape((-1, 2))[:,1])
                    vec_pertub = np.multiply((np.random.rand(element_base.functionspace.dim())) * self.nu.values()[0],Coor_fun)
                    w.vector()[:] = w.vector()[:] + vec_pertub
                self.p_pre = project(pre, self.BC_pre.functionspace, solver_type='gmres')
                self.u_pre = project(vel, self.BC_vel.functionspace, solver_type='gmres')
            elif type(self.path) is dict:
                timeseries_v = TimeSeries(self.path['Velocity'])
                timeseries_p = TimeSeries(self.path['Pressure'])
                timeseries_v.retrieve(self.u_pre.vector(), restart)
                timeseries_p.retrieve(self.p_pre.vector(), restart)
                if noise is True:
                    Coor_fun = np.sign(self.element.functionspace_V.tabulate_dof_coordinates().reshape((-1, 2))[:,1])
                    vec_pertub = np.multiply((np.random.rand(self.element.functionspace_V.dim())) * self.nu.values()[0],Coor_fun)
                    self.u_pre.vector()[:] = self.u_pre.vector()[:] + vec_pertub
                    Coor_fun = np.sign(self.element.functionspace_Q.tabulate_dof_coordinates().reshape((-1, 2))[:, 1])
                    pre_pertub = np.multiply((np.random.rand(self.element.functionspace_Q.dim())) * self.nu.values()[0],Coor_fun)
                    self.p_pre.vector()[:] = self.p_pre.vector()[:] + pre_pertub

        else:
            self.u_pre.vector()[:] = np.zeros(self.element.functionspace_V.dim())

    def set_boundarycondition(self, boucon, mark):
        if boucon['FunctionSpace'][0] == 'V':
            self.BC_vel.set_boundarycondition(boucon, mark)
        elif boucon['FunctionSpace'][0] == 'Q':
            self.BC_pre.set_boundarycondition(boucon,mark)

    def __NS_expression(self):
        self.u_mid, self.p_mid = self.element.add_functions()
        self.u_mid.assign(self.u_pre)
        self.p_mid.assign(self.p_pre)

        U = 0.5 * (self.u_mid + self.u)
        # Define variational problem for step 1
        F1 = dot((self.u - self.u_pre) / Constant(self.dt), self.v) * dx \
             + dot(dot(self.u_mid, nabla_grad(self.u_mid)), self.v) * dx \
             + inner(self.sigma(U, self.p_mid), self.epsilon(self.v)) * dx#\
        #   + dot(self.p_pre*self.n, self.v)*self.ds - dot(self.nu*nabla_grad(U)*self.n, self.v)*self.ds \
        #   - dot(self.f, self.v)*dx
        a1 = lhs(F1)
        self.A1 = assemble(a1)
        self.L1 = rhs(F1)

        # Define variational problem for step 2
        a2 = dot(nabla_grad(self.p), nabla_grad(self.q)) * dx
        self.A2 = assemble(a2)
        # L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx
        L2_1 = dot(nabla_grad(self.p), nabla_grad(self.q)) * dx  ##for p_n
        L2_2 = Constant(-1.0 / self.dt) * div(self.u) * self.q * dx  ## u_
        self.b2_1 = assemble(L2_1)
        self.b2_2 = assemble(L2_2)

        # Define variational problem for step 3
        a3 = dot(self.u, self.v) * dx
        self.A3 = assemble(a3)
        # L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx
        L3_1 = dot(self.u, self.v) * dx  ##u_
        L3_2 = -Constant(self.dt) * dot(nabla_grad(self.p), self.v) * dx  ##p_-p_n
        self.b3_1 = assemble(L3_1)
        self.b3_2 = assemble(L3_2)

        [bc.apply(self.A1) for bc in self.BC_vel.bcs]
        [bc.apply(self.A2) for bc in self.BC_pre.bcs]

    def __solver_init(self, method = 'iterative'):
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

            self.solver2 = KrylovSolver('cg', 'ilu')
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
            self.solver1 = PETScLUSolver('mumps')
            self.solver1.set_operator(self.A1)
            self.solver1.parameters['reuse_factorization'] = True

            self.solver2 = PETScLUSolver('mumps')
            self.solver2.set_operator(self.A2)
            self.solver2.parameters['reuse_factorization'] = True

            self.solver3 = PETScLUSolver('mumps')
            self.solver3.set_operator(self.A3)
            self.solver3.parameters['reuse_factorization'] = True

    def solve(self, method='iterative'):
        if self.stepcount == 0:
            self.__NS_expression()
            self.__solver_init(method=method)
        iter_max = 20
        iter_num = 0
        eps = 1
        tol = 1e-7
        while eps > tol and iter_num < iter_max:
            iter_num += 1
            # Step 1: Tentative velocity step
            b1 = assemble(self.L1)
            [bc.apply(b1) for bc in self.BC_vel.bcs]
            self.solver1.solve(self.u_.vector(), b1)

            # Step 2: Pressure correction step
            b2 = self.b2_1 * self.p_mid.vector() + self.b2_2 * self.u_.vector()
            [bc.apply(b2) for bc in self.BC_pre.bcs]
            self.solver2.solve(self.p_.vector(), b2)

            # Step 3: Velocity correction step
            b3 = self.b3_1 * self.u_.vector() + self.b3_2 * (self.p_.vector() - self.p_mid.vector())
            self.solver3.solve(self.u_.vector(), b3)

            diff = self.u_.vector().array() - self.u_mid.vector().array()
            eps = np.linalg.norm(diff, ord=np.Inf)
            info('iter=%d: norm=%g' % (iter_num, eps))
            self.u_mid.assign(self.u_)  # update for next iteration
            self.p_mid.assign(self.p_)

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


    """
    solver using scipy functions, need further development for efficient implementation
    """
    def __direct_solver_init(self):
        """Initialise IPCS solver
        Direct solver using scipy functions
        """
        self.A_step1 = self.convertmatrix(self.A1).tocsc()
        self.A_step2 = self.convertmatrix(self.A2).tocsc()
        self.A_step3 = self.convertmatrix(self.A3).tocsc()

        self.solver1 = spla.splu(self.A_step1)
        print('Step 1 completed ... ')
        self.solver2 = spla.splu(self.A_step2)
        print('Step 2 completed ... ')
        self.solver3 = spla.splu(self.A_step3)
        print('Step 3 completed ... ')

    def __iterative_solver_init(self):
        """Initialise IPCS solver
        Iterative solver using scipy functions
        """
        self.A_step1 = self.convertmatrix(self.A1).tocsc()
        self.A_step2 = self.convertmatrix(self.A2).tocsc()
        self.A_step3 = self.convertmatrix(self.A3).tocsc()

        self.M_step1 = precondition_jacobi(self.A_step1)
        self.M_step2 = precondition_lu(self.A_step2)
        self.M_step3 = precondition_jacobi(self.A_step3)

    def direct_solve(self):
        if self.stepcount == 0:
            self.__NS_expression()
            self.__direct_solver_init()
            print('Start Iterations ... ')

        # Step 1: Tentative velocity step
        b1 = self.assemblevector(self.L1, self.BC_vel.bcs).transpose()
        self.u_.vector()[:] = np.ascontiguousarray(self.solver1.solve(b1))

        # Step 2: Pressure correction step
        b2 = self.b2_1 * self.p_pre.vector() + self.b2_2 * self.u_.vector()
        [bc.apply(b2) for bc in self.BC_pre.bcs]
        b2 = self.convertvector(b2).transpose()
        self.p_.vector()[:] = np.ascontiguousarray(self.solver2.solve(b2))

        # Step 3: Velocity correction step
        b3 = self.b3_1 * self.u_.vector() + self.b3_2 * (self.p_.vector() - self.p_pre.vector())
        b3 = self.convertvector(b3).transpose()
        self.u_.vector()[:] = np.ascontiguousarray(self.solver3.solve(b3))
        # Update previous solution
        self.u_pre.assign(self.u_)
        self.p_pre.assign(self.p_)
        self.stepcount += 1

    def iterative_solve(self):
        if self.stepcount == 0:
            self.__NS_expression()
            self.__iterative_solver_init()
            print('Start Iterations ... ')

        # Step 1: Tentative velocity step
        b1 = self.assemblevector(self.L1, self.BC_vel.bcs).transpose()
        self.u_.vector()[:] = spla.gmres(self.A_step1, b1,tol=1e-08, M=self.M_step1)[0]

        # Step 2: Pressure correction step
        b2 = self.b2_1 * self.p_pre.vector() + self.b2_2 * self.u_.vector()
        [bc.apply(b2) for bc in self.BC_pre.bcs]
        b_2 = self.convertvector(b2).transpose()
        self.p_.vector()[:] = spla.gmres(self.A_step2, b_2,tol=1e-08, M=self.M_step2)[0]
        # Step 3: Velocity correction step
        b3 = self.b3_1 * self.u_.vector() + self.b3_2 * (self.p_.vector() - self.p_pre.vector())
        b_3 = self.convertvector(b3).transpose()
        self.u_.vector()[:] = spla.gmres(self.A_step3, b_3,tol=1e-08,M=self.M_step3)[0]
        # Update previous solution
        self.u_pre.assign(self.u_)
        self.p_pre.assign(self.p_)
        self.stepcount += 1



"""
Define some useful functions
"""
def precondition_ilu(A, useUmfpack=False):
    """
    :param A: matrix A
    :param useUmfpack:
    :return: inverse of the preconditioner
    """
    spla.use_solver(useUmfpack=useUmfpack)
    if useUmfpack is True:
        A = to_int64(A.astype('D'))
    M2 = spla.spilu(A)
    M_x = lambda x: M2.solve(x)
    M = spla.LinearOperator(A.shape, M_x)
    return M

def precondition_lu(A, useUmfpack=False):
    """
    :param A: matrix A
    :param useUmfpack:
    :return: inverse of the preconditioner
    """
    spla.use_solver(useUmfpack=useUmfpack)
    if useUmfpack is True:
        A = to_int64(A.astype('D'))
    M2 = spla.splu(A)
    M_x = lambda x: M2.solve(x)
    M = spla.LinearOperator(A.shape, M_x)
    return M

def precondition_jacobi(A):
    """
    :param A: matrix A
    :param useUmfpack:
    :return: inverse of the preconditioner
    """
    data = np.reciprocal(A.diagonal())
    offsets = 0
    M = dia_matrix((data, offsets), shape=A.shape)
    return M.tocsc()

def to_int64(x):
    y = csc_matrix(x).copy()
    y.indptr = y.indptr.astype(np.int64)
    y.indices = y.indices.astype(np.int64)
    return y

def DataConvert_mesh(mesh1,mesh2, path1, path2, etype = 'TaylorHood', time = 0.0, time_store = 0.0):
    """
    Data convert between different meshes: from mesh1 to mesh2
    :param mesh1:
    :param mesh2:
    :param path1:
    :param path2:
    :param etype:
    :return:
    """

    parameters["allow_extrapolation"] = True
    if etype is 'TaylorHood' and type(path1) is not dict and type(path2) is not dict:
        element_1 = TaylorHood(mesh=mesh1)
        w1 = element_1.w
        element_2 = TaylorHood(mesh=mesh2)
        timeseries_1 = TimeSeries(path1)
        timeseries_1.retrieve(w1.vector(), time)
        w2 = project(w1, element_2.functionspace, solver_type='gmres')
        timeseries_2 = TimeSeries(path2)
        timeseries_2.store(w2.vector(), time_store)
    if etype is 'SplitPV' and type(path1) is dict and type(path2) is dict:
        element_1 = SplitPV(mesh=mesh1)
        u1 = element_1.u
        p1 = element_1.p
        timeseries_v1 = TimeSeries(path1['Velocity'])
        timeseries_v1.retrieve(u1.vector(), time)
        timeseries_p1 = TimeSeries(path1['Pressure'])
        timeseries_p1.retrieve(p1.vector(), time)

        element_2 = SplitPV(mesh=mesh2)
        u2 = project(u1, element_2.functionspace_V, solver_type='gmres')
        p2 = project(p1, element_2.functionspace_Q, solver_type='gmres')
        timeseries_v2 = TimeSeries(path2['Velocity'])
        timeseries_v2.store(u2.vector(), time_store)
        timeseries_p2 = TimeSeries(path2['Pressure'])
        timeseries_p2.store(p2.vector(), time_store)

def DataConvert_element(mesh1,mesh2, path1, path2, etype = 'TaylorHood2SplitPV', time = 0.0, time_store = 0.0):
    parameters["allow_extrapolation"] = True
    if etype is 'TaylorHood2SplitPV' and type(path1) is not dict and type(path2) is dict:
        element_1 = TaylorHood(mesh=mesh1)
        element_2 = SplitPV(mesh=mesh2)
        timeseries_1 = TimeSeries(path1)
        timeseries_1.retrieve(element_1.w.vector(), time)
        u2 = project(element_1.u, element_2.functionspace_V, solver_type='gmres')
        p2 = project(element_1.p, element_2.functionspace_Q, solver_type='gmres')
        timeseries_v2 = TimeSeries(path2['Velocity'])
        timeseries_v2.store(u2.vector(), time_store)
        timeseries_p2 = TimeSeries(path2['Pressure'])
        timeseries_p2.store(p2.vector(), time_store)
    elif etype is 'SplitPV2TaylorHood' and type(path2) is not dict and type(path1) is dict:
        element_1 = SplitPV(mesh=mesh1)
        element_2 = TaylorHood(mesh=mesh2)
        timeseries_v1 = TimeSeries(path1['Velocity'])
        timeseries_v1.retrieve(element_1.u.vector(), time)
        timeseries_p1 = TimeSeries(path1['Pressure'])
        timeseries_p1.retrieve(element_1.p.vector(), time)
        u2 = project(element_1.u, element_2.functionspace.sub(0), solver_type='gmres')
        p2 = project(element_1.p, element_2.functionspace.sub(1), solver_type='gmres')
        element_2.u.vector()[:] = u2.vector()[:]
        element_2.p.vector()[:] = p2.vector()[:]
        timeseries_2 = TimeSeries(path2)
        timeseries_2.store(element_2.w.vector(), time_store)
