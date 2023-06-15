from __future__ import print_function
import gc
import numpy as np
import time as tm
import pymess as mess
from fenics import *
import scipy.linalg as scila
import scipy.sparse.linalg as spla
from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix,csr_matrix, identity, bmat,isspmatrix,isspmatrix_csc,isspmatrix_csr,isspmatrix_coo
from joblib import Parallel, delayed
import multiprocessing
#%%
import os,sys,inspect
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0,parentdir)
#%%
from Boundary.Set_BoundaryCondition import BoundaryCondition
from FlowSolver.Tools import sort_complex
from FlowSolver.FiniteElement import TaylorHood
from FrequencyAnalysis.FrequencyResponse import InputOutput
from FrequencyAnalysis.MatrixAssemble import MatrixAssemble
from SVD.MatrixOperator import MatInv
from SVD.MatrixCreator import MatrixCreator
#%%
class StateSpaceDAE2(MatrixAssemble):
    """
    | M   0 | d(|vel|     = | A    G  | |vel| + | B | u
    | 0   0 |   |pre|)/dt   | GT  Z=0 | |pre|   | 0 |
    
    y = | C   0 | |vel|
                  |pre|
    """
    def __init__(self,mesh=None,boundary=None,nu=None,Re=None,path=None,baseflow=None,boundaryconditions=None,IO=None):
        MatrixAssemble.__init__(self)
        ## 
        if mesh is not None:
            self.element = TaylorHood(mesh=mesh)
            self.u=self.element.tu
            self.p=self.element.tp
            self.v=self.element.v
            self.q=self.element.q
            self.w=self.element.w
            self.u_base=self.element.u
            self.p_base=self.element.p
            self.mesh=mesh
            self.n = FacetNormal(mesh)
            self.functionspace = self.element.functionspace
            self.Coor_fun = self.functionspace.tabulate_dof_coordinates().reshape((-1, 2))
        else:
            raise ValueError('Please provide mesh')
        ##
        if boundary is not None:
            self.boundary=boundary
            self.BC = BoundaryCondition(Functionspace=self.functionspace, boundary=self.boundary)
            self.boundaries = self.boundary.get_domain()
            self.ds = self.boundary.get_measure()
            
        else:
            raise ValueError('Please define boundary')
        ##
        if nu is not None and Re is not None:
            raise ValueError('Only nu or Re required')
        elif nu is None and Re is None:
            raise ValueError('nu or Re required')
        elif nu is None:
            if 'Constant' in str(type(Re)):
                self.Re=Re
                self.nu=Constant(1.0/Re.values()[0])
            elif 'int' in str(type(Re)) or 'float' in str(type(Re)):
                self.Re=Constant(Re)
                self.nu=Constant(1.0/Re)
        elif Re is None:
            if 'Constant' in str(type(nu)):
                self.Re=Constant(1.0/nu.values()[0])
                self.nu=nu
            elif 'int' in str(type(nu)) or 'float' in str(type(nu)):
                self.Re=Constant(1.0/nu)
                self.nu=Constant(nu)
                
        ##
        if path is None and baseflow is None:
            raise ValueError('Base flow required')
        elif path is not None and baseflow is not None:
            raise ValueError('Only Base flow function or its path required')
        else:
            self.baseflow=baseflow
            self.path=path
            self.__set_baseflow()
            
        ##
        if boundaryconditions is not None:
            if type(boundaryconditions) is dict:
                if 'Mark'in boundaryconditions[list(boundaryconditions.keys())[0]].keys():
                    for key in boundaryconditions.keys():
                        self.set_boundarycondition(boundaryconditions[key], boundaryconditions[key]['Mark'])
                else:
                    raise ValueError('Boundary Marks required')
            else:
                raise ValueError('boundaryconditions should be a dict')
            self.update()
        else:
            info('PLease manually apply boundary conditions later')
        
        if self.BC.bcs is not None and self.BC.bcs !=[]:
            if IO is not None:
                if type(IO) is dict:
                    self.assembleIOvectors(IO=IO)
                else:
                    raise ValueError('Input/Output information should be stored in a dict')
            else:
                self.B=0
                self.C=0
        else:
            raise ValueError('PLease specify boundary conditions')
            
    def __set_baseflow(self):
        """Assign base flow to the function w
        """
        if self.path is not None and self.baseflow is None:
            timeseries_flow = TimeSeries(self.path)
            timeseries_flow.retrieve(self.w.vector(), 0.0)
            self.baseflow=self.w
        if self.baseflow is not None and self.path is None:
            try:
                assign(self.w,self.baseflow)
            except:
                assign(self.w.sub(0),self.baseflow)
            else:
                pass
            
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



            
    def __NS_expression(self):
        """Create the weak form of the N-S equations
        du/dt+U*grad(u)+u*grad(U)=-grad(p)+nu*div(grad(u))
        div(u)=0

        """
        self.mass = inner(self.u, self.v) * dx #
        self.state = -(inner(dot(self.u_base, nabla_grad(self.u)), self.v) +
                   inner(dot(self.u, nabla_grad(self.u_base)), self.v) -
                   div(self.v) * self.p + self.nu * inner(grad(self.u), grad(self.v))
                   - self.q * div(self.u)) * dx
        """  
        # do not need these if Dirichlet or zero Neumann boundary conditions applied on all boundaries
        try:
            self.freeoutlet
        except:
            self.state = self.state+dot(self.v,self.n)*self.p*self.ds-self.nu*inner(grad(self.u)*self.n,self.v)*self.ds
        else:
            if self.freeoutlet is not True:
                self.state = self.state+dot(self.v,self.n)*self.p*self.ds-self.nu*inner(grad(self.u)*self.n,self.v)*self.ds
        """

    def __assemble_NSfunction(self):
        """Assemble the matrices of linearised N-S equations

        """
        Mat_assitant=MatrixCreator(mesh=self.mesh, functionspace=self.functionspace, boundarycondition=self.BC.bcs, boundary=self.boundary)
        # matrix has only ones in diagonal for rows specified by the boundary condition
        BC_diag = self.assemblematrix(Constant(0.0)*self.mass,self.BC.bcs)
        # identity matrix-BC_diag
        NBC_diag = identity(BC_diag.shape[0], dtype=BC_diag.dtype, format='csr')-BC_diag
        indices = NBC_diag.nonzero()
        # prolongation matrices
        self.P_del_bcs=NBC_diag[:, indices[1]]
        self.P_del_pre=Mat_assitant.MatP()
        self.P_del_vel=Mat_assitant.MatP(delete='velocity')
        # matrix has only zeros in diagonal for rows specified by the boundary condition
        State = self.assemblematrix(self.state,self.BC.bcs)-BC_diag # assemble the state part with zero at boundary row
        Mass = self.assemblematrix(self.mass,self.BC.bcs)-BC_diag # assemble the mass part with zero at boundary row
        # matrices without bcs rows
        self.State=self.P_del_bcs.transpose() * State * self.P_del_bcs
        self.Mass=self.P_del_bcs.transpose() * Mass * self.P_del_bcs
        
    def __extractblocks(self):
        """
        Mass=| M   0 |      State=| A   G  |
             | 0   0 |            | GT  Z=0| 
        """
        # blocks in matrices
        P=self.P_del_bcs.transpose() *self.P_del_pre*self.P_del_pre.transpose()* self.P_del_bcs
        P_p=P[:, P.nonzero()[1]]
        self.M=P_p.transpose()*self.Mass*P_p
        self.A=P_p.transpose()*self.State*P_p
        
        P=self.P_del_bcs.transpose() *self.P_del_vel*self.P_del_vel.transpose()* self.P_del_bcs
        P_v=P[:, P.nonzero()[1]]
        
        self.G =P_p.transpose()*self.State*P_v
        self.GT=P_v.transpose()*self.State*P_p
        if (self.G!=self.GT.transpose()).nnz==0:
            pass
        elif (self.G!=-self.GT.transpose()).nnz==0:
            raise ValueError('Please change the sign of the continous equation and try again')
        else:
            raise ValueError('G != G^T (G is a block matrix of the State Matrix)')
        ## if G is of rank np
        
        self.Z=P_v.transpose()*self.State*P_v
        
        if self.Z.nnz == 0:
            pass
        else:
            raise ValueError('Z block is non-zero!!')
        
    def assembleblocks(self):
        self.Mass_b=bmat([[self.M, None], [None, self.Z]],format='csr')
        self.State_b=bmat([[self.A, self.G], [self.GT, self.Z]],format='csr')
    
    def assembleIOvectors(self,IO=None):
        if IO is not None:
            if type(IO) is dict:
                keys=list(IO.keys())
                IO_vec=InputOutput(mesh=self.mesh, boundary=self.boundary, nu=self.nu, element=self.element,bcs=self.BC.bcs)
                P=self.P_del_bcs.transpose() *self.P_del_pre*self.P_del_pre.transpose()* self.P_del_bcs
                P_p=P[:, P.nonzero()[1]]
            else:
                raise ValueError('Input/Output information should be stored in a dict')
        else:
            raise ValueError('Please specify Input/Output information in a dict')
        matrices = MatrixCreator(mesh=self.mesh,functionspace=self.element.functionspace) 
        if np.size(keys)==2:
            if 'Input' in keys and 'Output'in keys:
                options=IO['Input']
                input_vec = eval("IO_vec."+IO_vec.dict_inputs[options['Variable']])#+"[0]")
                try:
                    options['Range']
                except:
                    pass
                else:
                    if options['Range'] is not None:
                        I_partial=matrices.Identity(condition=options['Range'])
                        input_vec=input_vec*I_partial              
                self.B=P_p.transpose()*(self.P_del_bcs.transpose() *input_vec.transpose())
                options=IO['Output']
                output_vec =eval("IO_vec."+IO_vec.dict_outputs[options['Variable']])#+"[0]")
                try:
                    options['Range']
                except:
                    pass
                else:
                    if options['Range'] is not None:
                        I_partial=matrices.Identity(condition=options['Range'])
                        output_vec=output_vec*I_partial  
                self.C=(output_vec*self.P_del_bcs)*P_p
            else:
                raise ValueError('Unidentified I/O dict provided')
        elif np.size(keys)==1:
            if keys[0]=='Input':
                options=IO['Input']
                input_vec = eval("IO_vec."+IO_vec.dict_inputs[options['Variable']]+"[0]")
                try:
                    options['Range']
                except:
                    pass
                else:
                    if options['Range'] is not None:
                        I_partial=matrices.Identity(condition=options['Range'])
                        input_vec=input_vec*I_partial   
                self.B=P_p.transpose()*(self.P_del_bcs.transpose() *input_vec.transpose())
                self.C=0
            elif keys[0]=='Output':
                options=IO['Output']
                output_vec = np.matrix(eval("IO_vec."+IO_vec.dict_outputs[options['Variable']]+"[0]"))
                try:
                    options['Range']
                except:
                    pass
                else:
                    if options['Range'] is not None:
                        I_partial=matrices.Identity(condition=options['Range'])
                        output_vec=output_vec*I_partial  
                self.C=(output_vec*self.P_del_bcs)*P_p
                self.B=0
            else:
                raise ValueError('Unidentified I/O dict provided')
        else:
            raise ValueError('Only support single-input single output systems')
    
        
    def update(self,IO=None):
        self.__NS_expression()
        self.__assemble_NSfunction()
        self.__extractblocks()
        self.assembleblocks()
        if IO is not None:
            self.assembleIOvectors(IO=IO)
        
    def check_eigs(self, E=None, A=None, k=3,sigma=None,v0=None,which=None,echo=False):
        """
        eigenvalues for \lambda*E*x=A*x
        """
        if E is None and A is None:
            E=self.Mass_b
            A=self.State_b
        elif E is None:
            E=identity(self.Mass_b.shape[0], dtype=self.Mass_b.dtype, format='csr')
        elif A is None:
            A=identity(self.Mass_b.shape[0], dtype=self.Mass_b.dtype, format='csr')
            
        if sigma is not None and sigma != 0.0:
            OP=(A-sigma*E)
        elif sigma is None or sigma == 0.0:
            OP=A

        if sigma is not None:
            info('LU decomposition...')
            gc.collect()
            if not isspmatrix_csc(OP):
                OP=OP.tocsc()
            try:
                OPinv=MatInv(OP,lusolver='mumps',echo=echo)
            except:
                info('switch to SuperLU solver')
                try:
                    OPinv=MatInv(OP,lusolver='superlu',echo=echo)
                except:
                    info('switch to useUmfpack solver')
                    OPinv=MatInv(OP,lusolver='umfpack',echo=echo)
                    info('Using Umfpack solver')
                else:
                    info('Using SuperLU solver')
            else:
                info('Using Mumps solver')
            info('Done.')
        
        if sigma is not None:
            info('Eigen decomposition...')
            self.vals, self.vecs = spla.eigs(A, k=k, M=E, OPinv=OPinv, sigma=sigma, which=which,v0=v0)
            info('Done.')
        elif sigma is None:
            info('Eigen decomposition...')
            self.vals, self.vecs = spla.eigs(A, k=k, M=E, which=which,v0=v0)
            info('Done.')
            
        return self.vals,self.vecs

class H2OptimalControl:
    """
    for LQR, LQE, LQG problems
    
    MESS_DIRECT_SUPERLU_LU : 4, check error occured, kernel died
    
    MESS_DIRECT_UMFPACK_LU: 3, normal
    
    MESS_DIRECT_SPARSE_LU 1, too long time to check
    
    MESS_DIRECT_CSPARSE_LU 5, too long time to check
    
    MESS_DIRECT_DEFAULT_LU 0, the same as UMFPACK

    """
    def __init__(self,model=None,Bd=None,Cs=None,Du=None,Dn=None,option=None):
        """
        | model.M   0 | d(|vel|     = | model.A   model.G  | |vel| + | model.B | u + | Bd | d
        |    0      0 |   |pre|)/dt   | model.GT model.Z=0 | |pre|   |    0    |     |  0 |
        
        y = | model.C   0 | |vel| + | Dn | n
                            |pre|            
        
        z = | Cs 0 | |vel| + | 0 | u
            | 0  0 | |pre|   | Du|
        
        
        model: state space model of linearised N-S equations
               model.A:  a block of state matrix
               model.G:  a block of state matrix
               model.GT: a block of state matrix
               model.Z:  a block of state matrix
               model.B:  input vector
               model.C:  output vector
               model.M:  full rank block of mass matrix
    
        k0: initial feedback that stabilises the system    
        
        Bd: weight matrix of disturbance d
        
        Cs: weight matrix of measured state for the cost function
        
        Du: weight matrix of control signal for the cost function
        
        Dn: weight matrix of sensor noise n
        
        option: options for computing initial feedback
        """
        self.model=model
        self.options=option
        #assume all in matrix format rather than array or int/float
            
        if Bd is None or Bd==0:
            self.Bd=np.matrix(np.zeros((self.model.B.shape[0],1)))
        else:
            self.Bd=np.matrix(Bd)
        if Cs is None or Cs==0:
            self.Cs=np.matrix(np.zeros((1,self.model.M.shape[1])))
        else:
            self.Cs=np.matrix(Cs)
        if Du is None or Du==0:
            self.Du=np.matrix(np.zeros((self.model.B.shape[1],self.model.B.shape[1])))
        else:
            self.Du=np.matrix(Du)
        if Dn is None or Dn==0:
            self.Dn=np.matrix(np.zeros((self.model.C.shape[0],1)))
        else:
            self.Dn=np.matrix(Dn)
        # check dimensions
        if self.Dn.shape[0]!=self.model.C.shape[0]:
            raise ValueError('Number of the rows in Dn should be equal to the number of sensors')
        if self.Bd.shape[0]!=self.model.B.shape[0]:
            raise ValueError('Number of the rows in Bd should be equal to the number of vel states')
        if self.Du.shape[0]!=self.Du.shape[1] or self.Du.shape[0]!=self.model.B.shape[1]:
            raise ValueError('Shape of Du should be equal to the number of actuators')
        if self.Cs.shape[1]!=self.model.M.shape[1]:
            raise ValueError('Number of the columns in Cs should be equal to the number of vel states')
        
    def __LREigs(self,E=None, A=None, k=2,sigma=0.0,v0=None,which='LR',gamma=0.0,transpose=False):
        """
        form right and left eigenvectors corresponding to unstable eigenvalues
        and return real-valued projection basis
        """
        if E is None and A is None:
            E=self.model.Mass_b
            A=self.model.State_b-gamma*E
        
        if transpose is True:
            A=A.transpose()
            M=E.transpose()
        elif transpose is False:
            pass
        else:
            raise ValueError('Please give a bool value for transpose')
        #right eigenvectors
        vals_r,vecs_r=self.model.check_eigs(E=E, A=A, k=k,sigma=sigma,which=which,v0=v0)
        #left eigenvectors
        vals_l,vecs_l=self.model.check_eigs(E=E.transpose(), A=A.transpose(), k=k,sigma=sigma,which=which,v0=v0)
        #align left/right eigenvalues
        if np.min(np.real(vals_r))<=0 or np.min(np.real(vals_l))<=0:
            raise ValueError('Contains stable eigenvectors, try a smaller k')
        if np.sum(np.abs(vals_r-vals_l))>1e-10:
            vals_rs,ind_r=sort_complex(vals_r)
            vals_ls,ind_l=sort_complex(vals_l)            
        else:
            vals_rs=vals_r
            vals_ls=vals_l
            ind_r=range(np.size(vals_rs))
            ind_l=range(np.size(vals_ls))
        #if not aligned, raise error, just for check
        if np.sum(np.abs(vals_rs-vals_ls))>1e-10:
            raise ValueError('Fail to align complex left/right eigenvalues')
        #form real-valued projection basis        
        self.W=self.__RPB(vals_ls,vecs_l[:,ind_l]) # left
        self.H=self.__RPB(vals_rs,vecs_r[:,ind_r]) #right
        return self.W, self.H
        
    def __RPB(self,vals,vecs):
        """
        form real-valued projection basis
        
        vals: sorted complex array
        vecs: corresponding complex array
        """
        n=np.size(vals)
        Basis=np.zeros(np.shape(vecs))
        flag=False
        for i in range(n):
            if np.imag(vals[i])==0:
                Basis[:,i]=np.real(vecs[i])
            elif i+1<n:
                if vals[i]==np.conj(vals[i+1]): 
                    Basis[:,i]=np.real(vecs[:,i])
                    Basis[:,i+1]=np.imag(vecs[:,i])
            else:
                pass
        if np.shape(Basis)==np.shape(vecs):
            return np.matrix(Basis)
        else:
            raise ValueError('Failed to construct real-valued projection basis')
                
                
    def __GABE(self, k=2,sigma=0.0,v0=None,which='LR',B=None, gamma=0.0,transpose=False):
        """
        form generalised algebraic Bernoulli equation matrices
        """
        if B is None:
            B=self.model.B
        elif np.shape(B)[0]!=np.shape(self.model.B)[0]:
            raise ValueError('Initial feedback controller should be the same rows as mass matrix')
        if transpose is True:
            A=(self.model.State_b-gamma*self.model.Mass_b).transpose()
            M=self.model.Mass_b.transpose()
        elif transpose is False:
            A=(self.model.State_b-gamma*self.model.Mass_b)
            M=self.model.Mass_b
        else:
            raise ValueError('Please give a bool value for transpose')
        
        W,H=self.__LREigs(k=k,sigma=sigma,v0=v0,which=which,gamma=gamma,transpose=transpose)
        M_tilda=np.matrix(W).transpose().conj()*M*np.matrix(H)
        A_tilda=np.matrix(W).transpose().conj()*A*np.matrix(H)
        B_tilda=np.matrix(W).transpose().conj()*np.matrix(np.pad(B,((0,M.shape[0]-B.shape[0]),(0,0)),'constant',constant_values=0))
        
        return M_tilda, A_tilda, B_tilda, W, H
        
    def __GARE(self,M=None,A=None,G=None,B=None,C=None,delta=-0.02,option=None):
        """
        solving generalised algebraic Riccati equation
        """
        if option is None:
            opt=mess.Options()
        else:
            opt=option
        mess.direct_select(mess.MESS_DIRECT_UMFPACK_LU)
        eqn = mess.EquationGRiccatiDAE2(opt, M, A, G, B, C, delta)
        z, status = mess.lrnm(eqn, opt)   
        return z,status
        
    def __GALE(self,M=None,A=None,G=None,B=None,delta=-0.02,k0=None,option=None):
        """
        solving generalised algebraic Lyapunov equation 
        """
        if option is None:
            opt=mess.Options()
        else:
            opt=option
        mess.direct_select(mess.MESS_DIRECT_UMFPACK_LU)
        eqn = mess.EquationGLyapDAE2(opt, M, A, G, B, delta,k0)
        z, status = mess.lradi(eqn, opt)   
        return z,status
        
    def InitFeedback(self,option=None,B=None,gamma=0.0,transpose=False):
        # set up options
        options={'k': 2,'shift':0.0,'v0':None,'which':'LR'}
        if option is not None and type(option) is dict:
            for key in option.keys():
                options[key]=option[key]
        if B is None:
            B=self.model.B
            
        if transpose is True:
            A=(self.model.State_b-gamma*self.model.Mass_b).transpose()
            M=self.model.Mass_b.transpose()
        elif transpose is False:
            A=(self.model.State_b-gamma*self.model.Mass_b)
            M=self.model.Mass_b
        else:
            raise ValueError('Please give a bool value for transpose')
        M_tilda, A_tilda, B_tilda, W, H=self.__GABE(B=B,k=options['k'],sigma=options['shift'],v0=options['v0'],which=options['which'],gamma=gamma,transpose=transpose)
        X0=np.matrix(scila.solve_continuous_are(A_tilda, B_tilda, np.zeros(np.shape(A_tilda)), np.identity(np.shape(B_tilda)[1]), e=M_tilda, s=None, balanced=True))
        B_b=np.matrix(np.pad(B,((0,M.shape[0]-B.shape[0]),(0,0)),'constant',constant_values=0))
        K0=B_b.transpose()*W*X0*W.transpose()*M
        return np.asarray(K0[:,0:B.shape[0]])
 
    def sensornoise(self,Vn=None):
        """
        square root of noise weight matrix: V^{-1/2}
        ----------------------
        Vn: int or float or 1D array
        """
        size=np.shape(self.model.C)[0]
        if np.asarray((Vn==None)).any() or np.asarray((Vn==0)).any():
            raise ValueError('Please specify non-zero sensor noise')
        else:
            if np.size(Vn)==size:
                V_half=np.matrix(np.diag(1.0/np.asarray(Vn).flatten()))
            else:
                raise ValueError('Number of sensor noise should be the same as the number of sensors(outputs)')
        return V_half

    def disturbance(self,Wd=None):
        """
        square root of noise weight matrix: W^{-1/2}
        -----------------------
        Wd: 1d array or ndarray or matrix
        """
        size=np.shape(self.model.B)[0]
        if np.shape(Wd)[0]!=size:
            raise ValueError('Number of rows in Wd should be the same as the states')
        elif isspmatrix(Wd):
            W_half=Wd
        else:
            W_half=np.matrix(Wd)
        return W_half
    
    def costfunction(self,Ru=1,beta=1,Qq=None):
        """
        J=(beta*q)^T*(Qq^TQq)*(beta*q)+(Ru*u)^T*(Ru*u)
        return R^(-1/2), beta*Qq=(beta^2*Q)^(1/2)
        -------------------
        beta: int or float
        Ru: int or float 1D array
        Q: matrix or ndarray
        """
        ## for R^(-1/2)
        size=np.shape(self.model.B)[1]
        if Ru==None or Ru==0:
            raise ValueError('Please specify non-zero control signal weights')
        else:
            if np.size(Ru)==size:
                R_half=np.matrix(np.diag(1.0/np.asarray(Ru).flatten()))
            else:
                raise ValueError('Number of control signal weights should be the same as the number of actuators(inputs)')
        ## beta*Qq
        size=np.shape(self.model.B)[0]
        if np.shape(Qq)[1]!=size:
            raise ValueError('Number of columns in Qq should be the same as the states')
        else:
            Q_half=beta*Qq
            
        if isspmatrix(Q_half):
            return Q_half, R_half
        elif 'numpy.ndarray' in str(type(Q_half)):
            return np.matrix(Q_half), R_half
        else:
            return Q_half, R_half
            
    def __h2norm(self,MatA, MatZ, divide=None, form='direct'):
        n=np.shape(MatZ)[1]
        print('dimension=%d' %(n))
        if form == 'direct':
            Sigma2=np.sum(np.square(MatA*MatZ))
        elif form =='split':
            dn=int(np.ceil(1.0*n/divide))
            s=0
            for i in range(divide):
                if dn*i<n:
                    s=s+np.sum(np.square(MatA*MatZ[:,dn*i:dn*i+dn]))
            Sigma2=s
        return Sigma2
        
    def __h2norm_parallel(self, MatA, MatZ, pid):
        num_cores = multiprocessing.cpu_count()
        n=np.shape(MatZ)[1]
        print('dimension=%d' %(n))
        inds=range(n)
        if pid > num_cores:
            pid=num_cores-1
            info('Limited cores available')
        pids=range(pid)
        dn=int(np.ceil(1.0*n/pid))
        ss=np.zeros(pid)
        
        def h2norm_inds(j):
            s=np.sum(np.square(MatA*MatZ[:,j]))
            return s
            
        def h2norm(i):
            s=0
            ind=inds[dn*i:dn*i+dn]
            ddn=int(np.ceil(5000.0/pid))
            divide=int(np.ceil(1.0*np.size(ind)/ddn))
            for j in range(divide):
                s=s+h2norm_inds(ind[ddn*j:ddn*(j+1)])
            ss[i]=s
        
        Parallel(n_jobs=pid,require='sharedmem')(delayed(h2norm)(i) for i in pids)
        Sigma2=np.sum(ss)
        return Sigma2
                    
    def __LQE(self,k0=None,noise=None,disturbance=None,opts=None,Ru=1,beta=1,Qq=None,delta=-0.02,gamma=0.0):
        """
        LQE problem (optimal estimation)
        --------------------
        noise: int or float or 1D array, V^{-1/2} in the paper
        disturbance: 1d array or ndarray or matrix, W^{-1/2} in the paper
        """
        """
        default options???????
        """
        V_half=self.sensornoise(Vn=noise)
        W_half=self.disturbance(Wd=disturbance)    
        Q_half, R_half=self.costfunction(Ru=Ru,beta=beta,Qq=Qq)
        if isspmatrix(W_half):
            W_half=W_half.todense()

        # setup solver
        if opts is None:
            opts=mess.Options()
            opts.adi.memory_usage = mess.MESS_MEMORY_MID
            opts.type=mess.MESS_OP_NONE
            opts.adi.output = 1
            opts.nm.output = 1
            opts.adi.shifts.paratype = mess.MESS_LRCFADI_PARA_MINMAX
            opts.adi.res2_tol = 1e-5
            opts.nm.res2_tol = 1e-2
        elif 'pymess.options.Options' in str(type(opts)):
            if opts.type!=mess.MESS_OP_NONE:
                raise ValueError('Please change LQE option type to pymess.MESS_OP_NONE')
        else:
            raise ValueError('opts must be a pymess.options.Options object')
        
        if k0 is 'Bernoulli': ## if it's 'Bernoulli', then form k0 by solving a Bernoulli equation
            info('Initial feedback...')            
            try:
                self.options['k0']
            except:
                self.k0=self.InitFeedback(option=None,B=np.asarray(V_half*self.model.C).transpose(),gamma=gamma,transpose=True)
            else:
                self.k0=self.InitFeedback(option=self.options['k0'],B=np.asarray(V_half*self.model.C).transpose(),gamma=gamma,transpose=True)
        elif np.any(k0)==None:
            self.k0=None        
        elif np.size(list(k0))==1: ## if constant then constant matrix
            self.k0=np.matrix(np.diag(1.0/np.diag(V_half)))*k0*np.matrix(np.ones(self.model.C.shape))
        elif np.shape(np.matrix(k0))==self.model.C.shape:
            self.k0=np.matrix(np.diag(1.0/np.diag(V_half)))*np.matrix(k0)
        elif np.shape(np.matrix(k0))==np.transpose(self.model.C).shape:
            self.k0=np.matrix(k0)*np.matrix(np.diag(1.0/np.diag(V_half)))
        else:
            raise ValueError('Invalid initial feedback')
            
        if self.k0 is not None:
            if np.shape(self.k0)==np.transpose(self.model.C).shape:
                opts.nm.k0=self.k0
            elif np.shape(self.k0)==self.model.C.shape:
                opts.nm.k0=np.transpose(self.k0)
        
        Z, Status=self.__GARE(M=self.model.M,A=self.model.A,G=self.model.G,B=np.asarray(W_half),C=np.asarray(V_half*self.model.C),delta=delta,option=opts)
        return Z, Status           

    def LQE(self,k0=None,noise=None,disturbance=None,opts=None,Ru=1,beta=1,Qq=None,delta=-0.02,gamma=0.0,Kr=None):
        """
        LQE problem (optimal estimation)
        --------------------
        noise: int or float or 1D array, V^{-1/2} in the paper
        disturbance: 1d array or ndarray or matrix, W^{-1/2} in the paper
        """
        """
        default options???????
        """
        Q_half=self.costfunction(Ru=Ru,beta=beta,Qq=Qq)[0]
        V_half=self.sensornoise(Vn=noise)
        Z, Status=self.__LQE(k0=k0,noise=noise,disturbance=disturbance,opts=opts,Ru=Ru,beta=Ru,Qq=Qq,delta=delta,gamma=gamma)
        ## kalman filter
        ## multiply M ???  yes, when perform simulation
        Kf=Z*(Z.transpose()*self.model.C.transpose()*V_half.transpose()*V_half)
        ##squared H2 norm
        #Sigma2=np.sum(np.square(Q_half*Z))
#        tt=tm.time()
#        Sigma2=self.__h2norm(Q_half, Z, divide=10, form='split')
#        tt=tm.time()-tt
#        info('Time: %e, sigma= %e' %(tt/3600.0, Sigma2))     
                
        tt=tm.time()
        Sigma2=self.__h2norm_parallel(Q_half, Z, pid=24)
        tt=tm.time()-tt
        info('Time: %e, sigma= %e' %(tt/3600.0, Sigma2))  
        if Kr is None:
            return Status, Kf, Sigma2, Z
        else:
            norm_lqr=Kr*Z
            norm_lqr=np.diag(norm_lqr*norm_lqr.transpose())
            return Status, Kf, Sigma2, norm_lqr, Z
        
    def LQE_split(self,k0=None,noise=None,disturbance=None,opts=None,Ru=1,beta=1,Qq=None,delta=-0.02,gamma=0.0,divide=10,sort_type='seq',dist_alloc=None,Kr=None):
        Q_half=self.costfunction(Ru=Ru,beta=beta,Qq=Qq)[0]
        V_half=self.sensornoise(Vn=noise)
        n=np.shape(disturbance)[1]
        dn=int(np.ceil(1.0*n/divide))
        Kf=np.matrix(np.zeros(np.shape(self.model.C.transpose())))
        Status=[]
        Sigma2=0
        A=self.model.A.copy()
        
        num_iter=range(divide)
        ind_dist=np.asarray(range(0,n))
        if sort_type=='random':
            np.random.shuffle(ind_dist)
        elif sort_type=='seq' and dist_alloc==None:
            dist_alloc=[dn]*divide
            if dn*divide>n:
                for i in range(dn*divide-n):
                    dist_alloc[-1-i]=dist_alloc[-1-i]-1
            
        iter_inds=[]   
        if dist_alloc==None:
            for i in num_iter:
                iter_inds.append(list(ind_dist[range(i,n,divide)]))
        elif np.sum(dist_alloc)==n and np.size(dist_alloc)==divide:
            for i in num_iter:
                iter_inds.append(list(ind_dist[int(np.sum(dist_alloc[0:i])):int(np.sum(dist_alloc[0:i+1]))]))
        else:
            raise ValueError('Wrong disturbance allocation list')

        for i in num_iter:
            self.model.A=A-csr_matrix(self.model.M*Kf)*csr_matrix(self.model.C)
            self.model.A.sort_indices() # sort data indices, prevent unmfpack error -8
            self.model.assembleblocks()

            print(*iter_inds[i], sep=' ')
            Z, status=self.__LQE(k0=k0,noise=noise,disturbance=disturbance[:,iter_inds[i]],opts=opts,Ru=Ru,beta=Ru,Qq=Qq,delta=delta,gamma=gamma)#disturbance=disturbance[:,dn*i:dn*i+dn]
            
            ksub=(Z*(Z.transpose()*self.model.C.transpose()*V_half.transpose()*V_half))#mmwrite('subkf2nd_'+str(i)+'.mtx',self.model.M*ksub)

            tt=tm.time()
            ssub=self.__h2norm_parallel(Q_half, Z, pid=24)
            tt=tm.time()-tt
            info('Time: %e, sigma= %e' %(tt/3600.0, ssub))

            Kf=Kf+ksub
            k0=None#self.model.M*Kf
            Status.append(status)
            Sigma2=Sigma2+ssub
            
            if Kr is not None:
                mat_lqr=Kr*Z
                if i==0:  
                    norm_lqr=np.diag(mat_lqr*mat_lqr.transpose())
                else:
                    norm_lqr=norm_lqr+np.diag(mat_lqr*mat_lqr.transpose())
            del Z
            info('progress=%d in %d' %(i+1,np.size(num_iter)))
#        for i in range(divide):
#            if dn*i<n:
#                self.model.A=A-csr_matrix(self.model.M*Kf)*csr_matrix(self.model.C)
#                self.model.A.sort_indices() # sort data indices, prevent unmfpack error -8
#                self.model.assembleblocks()
#                Z, status=self.__LQE(k0=k0,noise=noise,disturbance=disturbance[:,ind_dist[range(i,n,divide)]],opts=opts,Ru=Ru,beta=Ru,Qq=Qq,delta=delta,gamma=gamma)#disturbance=disturbance[:,dn*i:dn*i+dn]
#                ksub=(Z*(Z.transpose()*self.model.C.transpose()*V_half.transpose()*V_half))#mmwrite('subkf2nd_'+str(i)+'.mtx',self.model.M*ksub)
#                Kf=Kf+ksub
#                k0=self.model.M*Kf
#                Status.append(status)
#                try:
#                    ssub=self.__h2norm(Q_half, Z, divide=20, form='split')#mmwrite('subnorm2nd_'+str(i)+'.mtx',np.matrix(ssub))
#                except:
#                    info('Warning: out of memoty while computing H2 norms')
#                    ssub=0
#                else:
#                    pass
#                Sigma2=Sigma2+ssub
#                del Z
#                info('progress=%d in %d' %(i+1,divide))
        num_pods=[]
        [num_pods.append(np.size(iter_inds[i])) for i in num_iter]
        Status.append({'num_pods':num_pods})
        info('Total number of disturbances=%d' %(np.sum(num_pods)))
        if Kr is None:
            return Status, Kf, Sigma2
        else:
            return Status, Kf, Sigma2, norm_lqr
    
    def __LQR(self,k0=None,Ru=1,beta=1,Qq=None,opts=None,disturbance=None,delta=-0.02,gamma=0.0):
        """
        LQR problem (full information control)
        use MESS_OP_TRANSPOSE
        ----------------------
        beta: int or float
        Ru: int or float 1D array
        Q: matrix or ndarray
        """
        """
        default options?????????
        """
        W_half=self.disturbance(Wd=disturbance)
        Q_half, R_half=self.costfunction(Ru=Ru,beta=beta,Qq=Qq)
        if isspmatrix(Q_half):
            Q_half=Q_half.todense()

        
        if opts is None:
            opts=mess.Options()
            opts.type = mess.MESS_OP_TRANSPOSE
            opts.adi.memory_usage = mess.MESS_MEMORY_MID
            opts.adi.output = 1
            opts.nm.output = 1
            opts.adi.shifts.paratype = mess.MESS_LRCFADI_PARA_MINMAX
            opts.adi.res2_tol = 1e-5
            opts.nm.res2_tol = 1e-5
        elif 'pymess.options.Options' in str(type(opts)):
            if opts.type!=mess.MESS_OP_TRANSPOSE:
                raise ValueError('Please change LQR option type to pymess.MESS_OP_TRANSPOSE')
        else:
            raise ValueError('opts must be a pymess.options.Options object')
        
        if k0 is 'Bernoulli': ## if it's 'Bernoulli', then form k0 by solving a Bernoulli equation
            info('Initial feedback...')            
            try:
                self.options['k0']
            except:
                self.k0=self.InitFeedback(option=None,B=np.asarray(self.model.B*R_half),gamma=gamma)
            else:
                self.k0=self.InitFeedback(option=self.options['k0'],B=np.asarray(self.model.B*R_half),gamma=gamma)  
        
        elif np.any(k0)==None:
            self.k0=None        
        elif np.size(list(k0))==1: ## if constant then constant matrix
            self.k0=k0*np.matrix(np.ones(self.model.B.shape))*np.matrix(np.diag(1.0/np.diag(R_half)))
        elif np.shape(np.matrix(k0))==self.model.B.shape:
            self.k0=np.matrix(k0)*np.matrix(np.diag(1.0/np.diag(R_half)))
        elif np.shape(np.matrix(k0))==np.transpose(self.model.B).shape:
            self.k0=np.matrix(np.diag(1.0/np.diag(R_half)))*np.matrix(k0)
        else:
            raise ValueError('Invalid initial feedback')

        if self.k0 is not None:
            if np.shape(self.k0)==np.transpose(self.model.B).shape:
                opts.nm.k0=self.k0
            elif np.shape(self.k0)==self.model.B.shape:
                opts.nm.k0=np.transpose(self.k0)

        Z, Status=self.__GARE(M=self.model.M,A=self.model.A,G=self.model.G,B=np.asarray(self.model.B*R_half),C=np.asarray(Q_half),delta=delta,option=opts)
        ## optimal controller
        ## multiply M ???  yes, when perform simulation
#        Kr=R_half*R_half.transpose()*self.model.B.transpose()*Z*Z.transpose()
        ##norm
        #Sigma2=np.sum((csr_matrix(W_half.transpose()*Z).power(2)).data)
#        Sigma2=np.sum(np.square(W_half.transpose()*Z))
#        Sigma2_2=self.__h2norm_parallel(W_half.transpose(), Z, pid=24)
#        info('sigma2_para: %e' %Sigma2_2)
        #Sigma2=np.trace((W_half.transpose()*Z*Z.transpose()*W_half))
        return Z, Status
    
    def LQR(self,k0=None,Ru=1,beta=1,Qq=None,opts=None,disturbance=None,delta=-0.02,gamma=0.0, Kf=None):
        """
        LQR problem (full information control)
        use MESS_OP_TRANSPOSE
        ----------------------
        beta: int or float
        Ru: int or float 1D array
        Q: matrix or ndarray
        """
        """
        default options?????????
        """
        W_half=self.disturbance(Wd=disturbance)
        R_half=self.costfunction(Ru=Ru,beta=beta,Qq=Qq)[1]
        
        Z, Status=self.__LQR(k0=k0,Ru=Ru,beta=beta,Qq=Qq,opts=opts,disturbance=disturbance,delta=delta,gamma=gamma)
        
        Kr=R_half*R_half.transpose()*self.model.B.transpose()*Z*Z.transpose()
        
        tt=tm.time()
        Sigma2=self.__h2norm_parallel(W_half.transpose(), Z, pid=24)
        tt=tm.time()-tt
        info('Time: %e, sigma= %e' %(tt/3600.0, Sigma2))  
        if Kf is None:
            return Status, Kr, Sigma2, Z
        else:
            norm_lqe=Z.transpose()*Kf
            norm_lqe=np.diag(norm_lqe.transpose()*norm_lqe)
            return Status, Kr, Sigma2, norm_lqe, Z
        
    def LQR_split(self,k0=None,Ru=1,beta=1,Qq=None,opts=None,disturbance=None,delta=-0.02,gamma=0.0,divide=10,sort_type='seq',dist_alloc=None,Kf=None):
        
        W_half=self.disturbance(Wd=disturbance)
        R_half=self.costfunction(Ru=Ru,beta=beta,Qq=Qq)[1]
        
        n=np.shape(Qq)[0]
        dn=int(np.ceil(1.0*n/divide))
        Kr=np.matrix(np.zeros(np.shape(self.model.B.transpose())))
        Status=[]
        Sigma2=0
        
        A=self.model.A.copy() 
        
        num_iter=range(divide)
        ind_dist=np.asarray(range(0,n))
        if sort_type=='random':
            np.random.shuffle(ind_dist)
        elif sort_type=='seq' and dist_alloc==None:
            dist_alloc=[dn]*divide
            if dn*divide>n:
                for i in range(dn*divide-n):
                    dist_alloc[-1-i]=dist_alloc[-1-i]-1
            
        iter_inds=[]   
        if dist_alloc==None:
            for i in num_iter:
                iter_inds.append(list(ind_dist[range(i,n,divide)]))
        elif np.sum(dist_alloc)==n and np.size(dist_alloc)==divide:
            for i in num_iter:
                iter_inds.append(list(ind_dist[int(np.sum(dist_alloc[0:i])):int(np.sum(dist_alloc[0:i+1]))]))
        else:
            raise ValueError('Wrong disturbance allocation list')
## ...
        for i in num_iter:
            self.model.A=A-csr_matrix(self.model.B)*csr_matrix(Kr*self.model.M)
            self.model.A.sort_indices() # sort data indices, prevent unmfpack error -8
            self.model.assembleblocks()

            print(*iter_inds[i], sep=' ')
            Z, status=self.__LQR(k0=k0,Ru=Ru,beta=beta,Qq=Qq[iter_inds[i],:],opts=opts,disturbance=disturbance,delta=delta,gamma=gamma)
            
            ksub=R_half*R_half.transpose()*self.model.B.transpose()*Z*Z.transpose()
            
            tt=tm.time()
            ssub=self.__h2norm_parallel(W_half.transpose(), Z, pid=24)
            tt=tm.time()-tt
            info('Time: %e, sigma= %e' %(tt/3600.0, ssub))  
            

            Kr=Kr+ksub
            k0=None#Kr*self.model.M
            Status.append(status)
            Sigma2=Sigma2+ssub
            
            if Kf is not None:
                mat_lqe=Z.transpose()*Kf
                if i==0:  
                    norm_lqe=np.diag(mat_lqe.transpose()*mat_lqe)
                else:
                    norm_lqe=norm_lqe+np.diag(mat_lqe.transpose()*mat_lqe)
                    
            del Z
            info('progress=%d in %d' %(i+1,np.size(num_iter)))
        num_pods=[]
        [num_pods.append(np.size(iter_inds[i])) for i in num_iter]
        Status.append({'num_pods':num_pods})
        info('Total number of measurements=%d' %(np.sum(num_pods)))
        if Kf is None:
            return Status, Kr, Sigma2
        else:
            return Status, Kf, Sigma2, norm_lqe
        
    
    def LQG(self,Ru=1,beta=1,Qq=None,noise=None,disturbance=None,opts=None,delta=-0.02):
        """
        LQG problem (input/output problem)
        """
        """
        default options?????????
        """
        if opts is 'default':
            opts={}
            opts['LQE']=mess.Options()
            opts['LQE'].type = mess.MESS_OP_NONE
            opts['LQE'].nm.k0=self.k0
            
            opts['LQR']=mess.Options()
            opts['LQR'].type = mess.MESS_OP_TRANSPOSE
            opts['LQR'].nm.k0=self.k0
        elif opts==None:
            opts={}
            opts['LQE']=None
            opts['LQR']=None
        else:
            try:
                opts['LQE']
                opts['LQR']
            except:
                raise ValueError('Please contain LQE/LQR options in a dict')
            else:
                if opts['LQR'].type !=mess.MESS_OP_TRANSPOSE:
                    raise ValueError('Please change LQR option type to pymess.MESS_OP_TRANSPOSE')
                if opts['LQE'].type !=mess.MESS_OP_NONE:
                    raise ValueError('Please change LQE option type to pymess.MESS_OP_NONE')
        if self.k0 is not None:
            opts['LQE'].nm.k0=self.k0
            opts['LQR'].nm.k0=self.k0
            
        Z_x,S_x,Kr=self.LQR(Ru=Ru,beta=beta,Qq=Qq,opts=opts['LQR'],disturbance=disturbance,delta=delta)[0:2]
        Z_y,S_y,Kf=self.LQE(noise=noise,disturbance=disturbance,opts=opts['LQE'],Ru=Ru,beta=beta,Qq=Qq,delta=delta)[0:2]
        ## LQG plant....
        """
        LQG plant....?multiply M ??? yes, when perform simulation
        """
        Klqg=None
        ##norm
        V_half=self.sensornoise(Vn=noise)
        W_half=self.disturbance(Wd=disturbance) 
        Q_half, R_half=self.costfunction(Ru=Ru,beta=beta,Qq=Qq)
        ## if is sparse
        Sigma2=np.trace((Q_half*Z_y*Z_y.transpose()*Q_half.transpose()).todense())+np.trace((V_half.transpose()*V_half*self.model.C*Z_y*Z_y.transpose()*Z_x*Z_x.transpose()*Z_y*Z_y.transpose()*self.model.C.transpose()))
        Sigma21=np.trace((W_half.transpose()*Z_x*Z_x.transpose()*W_half))+np.trace((R_half*R_half.transpose()*self.model.B.transpose()*Z_x*Z_x.transpose()*Z_y*Z_y.transpose()*Z_x*Z_x.transpose()*self.model.B))
        print('Residual of two H2norm: ' %(Sigma2-Sigma21))
        return Klqg,Sigma2
            
    def H2Norm(self,mytype=None):
        pass