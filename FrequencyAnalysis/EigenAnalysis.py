from __future__ import print_function
from fenics import *
import numpy as np
import gc
from scipy.sparse import csr_matrix, csc_matrix,identity,isspmatrix_csc
import scipy.sparse.linalg as spla
from .MatrixAssemble import MatrixAssemble
import os,sys,inspect
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0,parentdir)
from SVD.MatrixOperator import MatInv
from Boundary.Set_BoundaryCondition import BoundaryCondition
from FlowSolver.FiniteElement import TaylorHood


class EigenAnalysis(MatrixAssemble):
    def __init__(self,mesh, boundary, nu=None, path=None,baseflow=None,dim='2D', lambda_z=None,element_order=(2,1),constrained_domain = None):
        self.dimension=dim
        self.element_order=element_order
        if self.dimension == '2D':
            element = TaylorHood(mesh=mesh,order=element_order,constrained_domain = constrained_domain) # initialise finite element space
        elif self.dimension == '2.5D':
            element = TaylorHood(mesh=mesh,dim=3,order=element_order,constrained_domain = constrained_domain) # initialise finite element space
        # inherit attributes : functionspace, boundries, bcs; methods: assemblematrix, assemblevector
        #BoundaryCondition.__init__(self, Functionspace=element.functionspace, boundary=boundary)
        self.BC = BoundaryCondition(Functionspace=element.functionspace, boundary=boundary)
        self.functionspace = element.functionspace
        self.boundaries = boundary.get_domain() # Get the FacetFunction on given mesh
        MatrixAssemble.__init__(self)
        # create public attributes
        self.mesh = mesh
        self.path = path
        self.baseflow=baseflow
        self.nu = nu
        self.lambda_z = lambda_z
        self.n = FacetNormal(self.mesh)
        self.ds = boundary.get_measure()
        # assign functions for the expression convenience
        self.u=element.tu
        self.p=element.tp

        self.v=element.v
        self.q=element.q

        self.w=element.w
        self.u_base=element.u
        self.p_base=element.p
        ##
        self.extra_function=element.add_functions()
    
    def set_boundarycondition(self, boucon, mark):
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
        
        if (boucon['FunctionSpace'] in ['Free Outlet','FreeOutlet','freeoutlet','free outlet'] or boucon['Value'] in ['Free Outlet','FreeOutlet','freeoutlet','free outlet']):
            self.freeoutlet=True
        else:
            self.BC.set_boundarycondition(boucon,mark)

    def __set_baseflow(self):
        """Assign base flow to the function w
        """
        if self.dimension == '2D':
            if self.path is not None and self.baseflow is None:
                timeseries_flow = TimeSeries(self.path)
                timeseries_flow.retrieve(self.w.vector(), 0.0)
            if self.baseflow is not None and self.path is None:
                try:
                    assign(self.w,self.baseflow)
                except:
                    assign(self.w.sub(0),self.baseflow)
                else:
                    pass
        elif self.dimension == '2.5D':
            ## pseudo 3D analysis using 2D base flow
            element_2d=TaylorHood(mesh=self.mesh,order=self.element_order)
            if self.path is not None and self.baseflow is None:
                timeseries_flow = TimeSeries(self.path)
                timeseries_flow.retrieve(element_2d.w.vector(), 0.0)
            if self.baseflow is not None and self.path is None:
                try:
                    assign(element_2d.w,self.baseflow)
                except:
                    assign(element_2d.w.sub(0),self.baseflow)
                else:
                    pass
            assign(self.w.sub(0).sub(0),element_2d.w.sub(0).sub(0))
            assign(self.w.sub(0).sub(1),element_2d.w.sub(0).sub(1))
            assign(self.w.sub(1),element_2d.w.sub(1))
                
    def __NS_expression(self):
        """Create the weak form of the N-S equations in the frequency domain
        wj*u+U*grad(u)+u*grad(U)=-grad(p)+nu*div(grad(u))
        div(u)=0
        2D   : (u,p)=(\hat(u),\hat(p))e^(i*omega*t)
        2.5D : (u,p)=(\hat(u),\hat(p))e^(i*omega*t-i*k_z*z)
        """
        if self.dimension == '2D':
            self.ai = inner(self.u, self.v) * dx # imaginary part
            self.ar = (inner(dot(self.u_base, nabla_grad(self.u)), self.v) +
                       inner(dot(self.u, nabla_grad(self.u_base)), self.v) -
                       div(self.v) * self.p + self.nu * inner(grad(self.u), grad(self.v))
                       + self.q * div(self.u)) * dx
            
#            try:
#                self.freeoutlet
#            except:
#                self.ar = self.ar+dot(self.v,self.n)*self.p*self.ds-self.nu*inner(grad(self.u)*self.n,self.v)*self.ds
#            else:
#                if self.freeoutlet is not True:
#                    self.ar = self.ar+dot(self.v,self.n)*self.p*self.ds-self.nu*inner(grad(self.u)*self.n,self.v)*self.ds
        
        elif self.dimension == '2.5D':   
            self.ai = inner(self.u, self.v) * dx 
            
            self.ai_1 = (-self.lambda_z*self.p*self.v[2])*dx + (-self.lambda_z*self.u[2]*self.q)*dx# + (-self.lambda_z*self.u_base[2]*inner(self.u,self.v)) * dx + (-self.lambda_z*self.u[2]*inner(self.u_base,self.v))*dx# imaginary part
            
            self.ar = (inner((self.u_base[0] * nabla_grad(self.u)[0,:])+(self.u_base[1] * nabla_grad(self.u)[1,:]), self.v) +
                       inner((self.u[0] * nabla_grad(self.u_base)[0,:])+(self.u[1] * nabla_grad(self.u_base)[1,:]), self.v) -
                       (grad(self.v[0])[0]+grad(self.v[1])[1]) * self.p + self.nu * inner(grad(self.u), grad(self.v))
                       + self.nu*self.lambda_z*self.lambda_z*inner(self.u,self.v) 
                       + self.q * (grad(self.u[0])[0]+grad(self.u[1])[1])) * dx
                       
#            try:
#                self.freeoutlet
#            except:
#                self.ar = self.ar+(self.v[0]*self.n[0]+self.v[1]*self.n[1])*self.p*self.ds-self.nu*inner(grad(self.u)*self.n,self.v)*self.ds
#            else:
#                if self.freeoutlet is not True:
#                    self.ar = self.ar+(self.v[0]*self.n[0]+self.v[1]*self.n[1])*self.p*self.ds-self.nu*inner(grad(self.u)*self.n,self.v)*self.ds

    def __assemble_NSfunction(self):
        """Assemble the matrix that represents N-S equations as A*x = \lambda*M*x

        """
        symmetric = False # apply boundary condition symmetrically or not
        if symmetric is False:
            Ar_sparray = self.assemblematrix(self.ar,self.BC.bcs) # assemble the real part
            Ai_sparray = self.assemblematrix(self.ai,self.BC.bcs) # assemble the imag part
            # matrix has only ones in diagonal for rows specified by the boundary condition
            I_sparray = self.assemblematrix(Constant(0.0)*self.ai,self.BC.bcs)
        elif symmetric is True:
            Ar_sparray,vec_r=self.assemblesystem(self.ar,inner(Constant((0.0,0.0)),self.v)*dx,self.BC.bcs)
            Ai_sparray,vec_i=self.assemblesystem(self.ai,inner(Constant((0.0,0.0)),self.v)*dx,self.BC.bcs)
            I_sparray = self.assemblesystem(Constant(0.0)*self.ai,inner(Constant((0.0,0.0)),self.v)*dx,self.BC.bcs)[0]
#        self.error=np.max(np.abs(vec_r-vec_i))
        # Complex matrix L with boundary conditions
        if self.dimension == '2D':
            A = -(Ar_sparray)
        elif self.dimension == '2.5D':
            Ai_1_sparray = self.assemblematrix(self.ai_1,self.BC.bcs)
            A = -(Ar_sparray) - (Ai_1_sparray.multiply(1j)-I_sparray.multiply(1j))
        M = Ai_sparray- I_sparray
        # Change the format to CSC, it will be more efficient while LU decomposition
        self.A = A.tocsc()
        self.M = M.tocsc()

    def update_problem(self):
        self.__set_baseflow()
        self.__NS_expression()
        self.__assemble_NSfunction()    

    def solve(self, k=None,sigma=None, which=None,v0=None,path=None,baseflow=None,lambda_z=None,solver='Implicit',inverse=False, ReuseLU=False,tol=0):
        """
        inverse: if calculate the inverse of the right hand eigenvectors
        """
        if path is not None and baseflow is None:
            self.path=path
        if baseflow is not None and path is None:
            self.baseflow=baseflow
        if lambda_z is not None:
            self.lambda_z = lambda_z
        self.update_problem()
        
        if sigma is not None and sigma != 0.0:
            OP=(self.A-sigma*self.M)
        elif sigma is None or sigma == 0.0:
            OP=self.A
            if solver=='Explicit':
                sigma=0.0
        if ReuseLU is False and sigma is not None:
            if inverse is True:
                OP=OP.T
            info('LU decomposition...')
            gc.collect()
            if not isspmatrix_csc(OP):
                OP=OP.tocsc()
            try:
                self.OPinv=MatInv(OP,lusolver='mumps',echo=True)
            except:
                info('switch to SuperLU solver')
                try:
                    self.OPinv=MatInv(OP,lusolver='superlu',echo=True)
                except:
                    info('switch to useUmfpack solver')
                    self.OPinv=MatInv(OP,lusolver='umfpack',echo=True)
                    info('Using Umfpack solver')
                else:
                    info('Using SuperLU solver')
            else:
                info('Using Mumps solver')
            info('Done.')
        
        if solver=='Implicit' and inverse is False:
            if sigma is not None:
                info('Eigen decomposition...')
                self.vals, self.vecs = spla.eigs(self.A, k=k, M=self.M, OPinv=self.OPinv, sigma=sigma, which=which,v0=v0,tol=tol)
                info('Done.')
            elif sigma is None:
                info('Eigen decomposition...')
                self.vals, self.vecs = spla.eigs(self.A, k=k, M=self.M, which=which,v0=v0,tol=tol)
                info('Done.')
        elif solver=='Explicit':
            self.explicitsolve(k=k,sigma=sigma, which=which,OPinv=self.OPinv,inverse=inverse, ReuseLU=ReuseLU,tol=tol)
        else:
            raise ValueError('Please specify solver type: Implicit/Explicit while using Shift-Invert Mode (Explicit solver for inverse=True)')
        
    def explicitsolve(self,k=None,sigma=None, which=None,OPinv=None,inverse=False, ReuseLU=False,v0=None, tol=0):
        """
        explicit expression of Shift-Invert Mode
        
        inverse: if calculate the inverse of the right hand eigenvectors
        """   
        if inverse is True:
            exp = (spla.aslinearoperator(self.M.transpose())*OPinv)
        elif inverse is False:
            exp = (OPinv*spla.aslinearoperator(self.M))
        info('Eigen decomposition...')
        vals, self.vecs = spla.eigs(exp, k=k, which=which,v0=v0,tol=tol)
        self.vals=1.0/vals+sigma
        info('Done.')
