from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as spla
import os,sys,inspect
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0,parentdir) 
from SVD.MatrixOperator import MatInv, Mumpslu, Superlu, Umfpacklu,PETSclu
from SVD.MatrixCreator import MatrixCreator
from Boundary.Set_BoundaryCondition import BoundaryCondition
from FlowSolver.FiniteElement import TaylorHood
from Plot.Matplot import contourf_cylinder


from scipy.sparse.linalg.interface import aslinearoperator, LinearOperator
from scipy.sparse import eye, issparse, isspmatrix, isspmatrix_csr
from scipy.linalg import eig, eigh, lu_factor, lu_solve
from scipy.sparse.sputils import isdense
from scipy.sparse.linalg import splu
"""This module provides the class that calculates singular modes of a flow field
"""


class SingularMode:
    """Compute global singular modes of the 2D incompressible flow
    For the flow system u = C(w) f, where C(w)=P^T(iwB-A)^-1PM, the SVD of the matrix
    Qu^(1/2)C(w)Qf^(-1/2) = U \Sigma F gives the input vector(force) fi and output vector (response) 
    ui and the corresponding singular value(energy ratio) \lambda^2 = (ui^H Qu ui)/(fi^H Qf fi).

    Parameters
    ----------------------------
    mesh : object created by FEniCS function Mesh
        mesh of the flow field

    boundary : object Boundary()
        the Boundary object with defined and marked boundaries

    omega : Constant()
        frequency

    nu : Constant()
        kinematic viscosity

    path : string
        path to the base flow

    Attributes
    ----------------------------
    mesh : object created by FEniCS function Mesh
        mesh of the flow field

    functionspace : a finite element function space

    boundaries : FacetFunction on given mesh

    bcs : a list with boundary conditions

    matrices : object MatrixCreator()
        assemble useful matrices, e.g. the prolongation matrix P,
        mass matrix M, the weight function Q

    omega : Constant()
        frequency
        
    lambda_z : Constant()
        wavenumber in z direction (periodic assumption in spanwise direction)

    nu : Constant()
        kinematic viscosity

    path : string
        path to the base flow

    Qu_bound : float
        X coordinate of the vertical boundary that defines subdomains
        Qu is used to compute the response energy of the left subdomain

    Qf_bound : float
        X coordinate of the vertical boundary that defines subdomains
        Qf is used to compute the force energy of the left subdomain



    Examples
    ----------------------------
    here is a snippet code shows how to use this class

    >>> from fenics import *
    >>> from RAPACK.Boundary.Set_Boundary import Boundary
    >>> from RAPACK.SVD.SingularMode import SingularMode
    >>> mesh = Mesh ('mesh.xml')
    >>> boundary=Boundary(mesh)
    >>> nu = Constant(1.0/45)
    >>> omega=Constant(1.0)
    >>> path = 'path/to/baseflow'
    >>> singularmodes=SingularMode(mesh=mesh, boundary=boundary, omega=omega, nu=nu, path=path)
    >>> singularmodes.set_boundarycondition(BoundaryConditions, BoundaryLocations)
    >>> singularmodes.solve_SVD()
    >>> singularmodes.plot_mode() # plot the first mode
    >>> force, response, ratio = singularmodes.get_mode() # get the first mode

    """

    def __init__(self, mesh, boundary, omega=None, nu=None, path=None,baseflow=None,order=(2,1),dim='2D', lambda_z=None, constrained_domain=None):
        self.dimension=dim
        if self.dimension == '2D':
            element = TaylorHood(mesh=mesh,order=order,constrained_domain = constrained_domain) # initialise finite element space
        elif self.dimension == '2.5D':
            element = TaylorHood(mesh=mesh,order=order,constrained_domain = constrained_domain,dim=3) # initialise finite element space
        # inherit attributes : functionspace, boundries, bcs
        #BoundaryCondition.__init__(self, Functionspace=element.functionspace, boundary=boundary)
        self.BC = BoundaryCondition(Functionspace=element.functionspace, boundary=boundary)
        self.functionspace = element.functionspace
        self.boundaries = boundary.get_domain() # Get the FacetFunction on given mesh
        self.matrices = MatrixCreator(mesh=mesh, functionspace=self.BC.functionspace) # initialise matrix creator
        # create public attributes
        self.mesh = mesh
        self.path = path
        self.baseflow = baseflow
        self.omega = omega
        self.lambda_z = lambda_z
        self.n = FacetNormal(self.mesh)
        self.ds = boundary.get_measure()
        self.nu = nu
        self.Qu_bound = None
        self.Qf_bound = None
        # assign functions for the expression convenience
        self.u=element.tu
        self.p=element.tp
        self.v=element.v
        self.q=element.q
        self.w=element.w
        self.u_base=element.u
        self.p_base=element.p
        # get the coordinates of the whole functionspace
        self.Coor_fun = self.functionspace.tabulate_dof_coordinates().reshape((-1, 2))

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
            element_2d=TaylorHood(mesh=self.mesh)
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
            self.ai = self.omega * inner(self.u, self.v) * dx # imaginary part
            self.ar = (inner(dot(self.u_base, nabla_grad(self.u)), self.v) +
                       inner(dot(self.u, nabla_grad(self.u_base)), self.v) -
                       div(self.v) * self.p + self.nu * inner(grad(self.u), grad(self.v))
                       + self.q * div(self.u)) * dx
                
            try:
                self.freeoutlet
            except:
                self.ar = self.ar+dot(self.v,self.n)*self.p*self.ds-self.nu*inner(grad(self.u)*self.n,self.v)*self.ds
            else:
                if self.freeoutlet is not True:
                    self.ar = self.ar+dot(self.v,self.n)*self.p*self.ds-self.nu*inner(grad(self.u)*self.n,self.v)*self.ds
                    
        elif self.dimension == '2.5D':   
            self.ai = self.omega * inner(self.u, self.v) * dx + (-self.lambda_z*self.p*self.v[2])*dx + (-self.lambda_z*self.u[2]*self.q*dx)# imaginary part
            
            self.ar = (inner((self.u_base[0] * nabla_grad(self.u)[0,:])+(self.u_base[1] * nabla_grad(self.u)[1,:]), self.v) +
                       inner((self.u[0] * nabla_grad(self.u_base)[0,:])+(self.u[1] * nabla_grad(self.u_base)[1,:]), self.v) -
                       (grad(self.v[0])[0]+grad(self.v[1])[1]) * self.p + self.nu * inner(grad(self.u), grad(self.v))
                       + self.nu*self.lambda_z*self.lambda_z*inner(self.u,self.v) 
                       + self.q * (grad(self.u[0])[0]+grad(self.u[1])[1])) * dx
                       
            try:
                self.freeoutlet
            except:
                self.ar = self.ar+(self.v[0]*self.n[0]+self.v[1]*self.n[1])*self.p*self.ds-self.nu*inner(grad(self.u)*self.n,self.v)*self.ds
            else:
                if self.freeoutlet is not True:
                    self.ar = self.ar+(self.v[0]*self.n[0]+self.v[1]*self.n[1])*self.p*self.ds-self.nu*inner(grad(self.u)*self.n,self.v)*self.ds
        

    def __assemblematrix(self,expression,bcs=[]):
        """General function to assemble the matrix and convert it to CSR format

        Parameters
        ----------------------------
        expression : UFL expression

        bcs : list
            boundary conditions that change values in the specified rows

        Returns
        ----------------------------
        A_sparray : sparse matrix in CSR format

        """
        A = PETScMatrix() # FEniCS using PETSc for matrix operation
        assemble(expression, tensor=A) # store assembled matrix in A
        [bc.apply(A) for bc in bcs]
        # convert the format to CSR
        A_mat = as_backend_type(A).mat()
        A_sparray = csr_matrix(A_mat.getValuesCSR()[::-1], shape=A_mat.size)
        return A_sparray

    def __assemble_NSfunction(self,MatK=None):
        """Assemble the matrix L that represents N-S equations as L*[u, p]^T = f
           MatK: feedback matrix

        """
        Ar_sparray = self.__assemblematrix(self.ar,self.BC.bcs) # assemble the imag part
        if MatK is not None:
            Ar_sparray=Ar_sparray+MatK
            
        Ai_sparray = self.__assemblematrix(self.ai,self.BC.bcs) # assemble the real part
        # matrix has only ones in diagonal for rows specified by the boundary condition
        I_sparray = self.__assemblematrix(Constant(0.0)*self.ai,self.BC.bcs)

        # Complex matrix L with boundary conditions
        L = Ar_sparray + Ai_sparray.multiply(1j)-I_sparray.multiply(1j)
        # Change the format to CSC, it will be more efficient while LU decomposition
        self.L = L.tocsc()
        stype='mumps'
        echo=False
        if stype=='old':
            # LU decomposition
            self.L_lu = spla.splu(self.L).solve
            # Linear operator : A^-1 and (A^H)^-1
            self.Linv = MatInv(self.L, A_lu=self.L_lu, trans='N',lusolver='superlu',echo=echo)
            self.LinvH = MatInv(self.L, A_lu=self.L_lu, trans='H',lusolver='superlu')             
        elif stype=='mumps':
            self.L_lu = Mumpslu(self.L).solve
            self.Linv = MatInv(self.L, A_lu=self.L_lu, trans='N',lusolver='mumps',echo=echo)
            self.LinvH = MatInv(self.L,A_lu=self.L_lu,  trans='H',lusolver='mumps')
        elif stype=='superlu':
            self.L_lu = Superlu(self.L).solve
            self.Linv = MatInv(self.L, A_lu=self.L_lu, trans='N',lusolver='superlu',echo=echo)
            self.LinvH = MatInv(self.L,A_lu=self.L_lu,  trans='H',lusolver='superlu')
        elif stype=='umfpack':
            self.L_lu = Umfpacklu(self.L).solve
            self.Linv = MatInv(self.L, A_lu=self.L_lu, trans='N',lusolver='umfpack',echo=echo)
            self.LinvH = MatInv(self.L,A_lu=self.L_lu,  trans='H',lusolver='umfpack')
        elif stype=='petsc':
            self.L_lu = PETSclu(self.L).solve
            self.Linv = MatInv(self.L, A_lu=self.L_lu, trans='N',lusolver='petsc',echo=echo)
            self.LinvH = MatInv(self.L,A_lu=self.L_lu,  trans='H',lusolver='petsc')



    def __setSVDexpression(self):
        """Problem expression
        the SVD problem should be convert to an eigen problem D*f = \lambda^2*Qf*f,
        where D = M^T*P^T*(L^H)^-1*P*Qu*P^T*(L)^-1*P*M with L = (iwB-A)

        """
        self.matrices.bcs = self.BC.bcs # set boundary conditions that necessary to construct mass matrix M
        self.P = self.matrices.MatP() # prolongation operator
        self.M = self.matrices.MatM() # mass matrix
        self.Ibcs = self.matrices.MatBcs() # boundary condition matrix
        self.Qf = self.matrices.MatQf(self.Qf_bound) # weight matrix for force
        self.Qu = self.matrices.MatQu(self.Qu_bound) # weight matrix for response
        # Linear operator D = M^T*P^T*(L^H)^-1*P*Qu*P^T*(L)^-1*P*M with L = (iwB-A)
        self.SVDexp = (spla.aslinearoperator(self.M.transpose()) * spla.aslinearoperator(self.P.transpose()) *
                       self.LinvH * spla.aslinearoperator(self.P) * spla.aslinearoperator(self.Qu) *
                       spla.aslinearoperator(self.P.transpose()) * self.Linv * spla.aslinearoperator(self.P) *
                       spla.aslinearoperator(self.M))

    def update_problem(self,MatK=None):
        """Update problem
        update values or setting after changing baseflow, omega, nu or something else

        """
        self.__set_baseflow()
        self.__NS_expression()
        self.__assemble_NSfunction(MatK=MatK)
        self.__setSVDexpression()

    def solve_SVD(self, k=3, omega=None,lambda_z=None, normalize=True,baseflow=None, path=None,MatK=None):
        """Solve the problem

        Parameters
        ----------------------------
        k : int
            number of modes needed to compute, default : 3

        omega : Constant, optional
            frequency, default : set while initialize the class
            
        lambda_z : Constant()
            wavenumber in z direction (periodic assumption in spanwise direction)

        normalize : bool, optional
            normalize the eigenvector, default : True
            
        Which k eigenvectors and eigenvalues to find:

        LM : largest magnitude
        
        SM : smallest magnitude
        
        LR : largest real part
        
        SR : smallest real part
    
        LI : largest imaginary part
        
        SI : smallest imaginary part
        """
        if omega is not None:
            self.omega = omega
        if lambda_z is not None:
            self.lambda_z = lambda_z
        if path is not None and baseflow is None:
            self.path=path
        if baseflow is not None and path is None:
            self.baseflow=baseflow
        # update problem
        self.update_problem(MatK=MatK)
        # start vector for iterative solver
        vec_start = np.ones(self.functionspace.sub(0).dim())
        #
        self.Qf_lu = Mumpslu(self.Qf).solve
        self.Qfinv = MatInv(self.Qf, A_lu=self.Qf_lu, trans='N',lusolver='mumps',echo=False)
        self.vecstart=vec_start
        # solve eigen problem, returns first k eigenvalues and vectors with first k largest real part
        #vals, vecs = spla.eigs(self.SVDexp, k=k, M=self.Qf, which='LR')#, v0=vec_start)
        vals, vecs = spla.eigs(self.SVDexp, k=k, M=self.Qf, Minv=self.Qfinv, which='LR', v0=vec_start)
        # sort eigenvalues and vectors in descending order
        index=vals.argsort()[::-1]
        self.vals=vals[index]
        self.vecs=vecs[:,index]
        # add warning
        imags=np.max(np.abs(np.imag(self.vals/np.real(self.vals))))
        if imags>1e-9:
            print('Large imaginary part at fre = %e with I = %e' %(self.omega.values()[0], imags))
        # normalize the eigenvectors with energy
        if normalize is True:
            for ind in index:
                vecs_energy = np.dot(self.vecs[:,ind].T.conj(), self.Qf.dot(self.vecs[:,ind]))
                self.vecs[:, ind] = self.vecs[:,ind] / np.sqrt(np.real(vecs_energy))
                self.vals[ind] = self.vals[ind]*np.sqrt(np.real(vecs_energy))

    def get_mode(self, k=0):
        """Get the k-th singular mode

        Parameters
        ----------------------------
        k : int, list or array
            default : 0

        Returns
        ----------------------------
        force : force(input) vector

        response : response(output) vector

        ratio : singular value (ratio between the output energy and input energy)

        """
        force = self.P.dot(self.vecs[:, k]) # prolong the vector
        response = self.L_lu(self.P.dot(self.M.dot(self.vecs[:, k]))) # calculate the response [u, p]^T = L^-1*f
        ratio = np.real(self.vals[k]) # singular value (ratio between the output energy and input energy)
        
        #self.resp_energy = np.dot(self.P.T.dot(response).T.conj(), self.Qu.dot(self.P.T.dot(response)))
        return force, response/np.sqrt(ratio), ratio
        
    def save_mode(self, k=0, path = None):
        """Save k-th singular mode as a time series
        time 0 is the real part of the force, time 1 is the imag part of the force,
        time 2 is the real part of the response, time 3 is the imag part of the response

        Parameters
        ----------------------------
        k : int
            default : 0
            
        path : string
            absolute path to save the mode

        """
        mode = Function(self.functionspace) # create function
        force, response = self.get_mode(k)[0:2] # get mode vectors
        # savepath = 'path/cylinder_mode(k)_Re(nu)_Omerga(omega)'
        savepath=path+'/cylinder_mode'+str(k)+'_Re'+str(int(1/float(self.nu))).zfill(3)+'_Omega'+str(float(self.omega))
        if self.lambda_z is not None:
            savepath=savepath+'_Lambdaz'+str(float(self.lambda_z))
        timeseries_r = TimeSeries(savepath)
        # store the mode
        mode.vector()[:]=np.ascontiguousarray(np.real(force), dtype=np.float64)
        timeseries_r.store(mode.vector(), 0.0)
        mode.vector()[:]=np.ascontiguousarray(np.imag(force), dtype=np.float64)
        timeseries_r.store(mode.vector(), 1.0)
        mode.vector()[:]=np.ascontiguousarray(np.real(response), dtype=np.float64)
        timeseries_r.store(mode.vector(), 2.0)
        mode.vector()[:]=np.ascontiguousarray(np.imag(response), dtype=np.float64)
        timeseries_r.store(mode.vector(), 3.0)        

    def EnergyDensityFun(self, k=0, alpha=None):
        """Energy density function
        energy densities of k-th force and response vector changes along the X-axis

        Parameters
        ----------------------------
        k : int

        alpha : array or list, optional
            X coordinates to evaluate the energy densities, default : X coordinates
            of nodes on the X-axis

        Returns
        ----------------------------
        EDF_force : energy density function of the force

        EDF_response : energy density function of the response

        alpha : X coordinates to evaluate the energy densities

        """
        # if alpha is not specified, then use X coordinates of nodes on the X-axis
        if alpha is None:
            Dofs_fun = self.functionspace.sub(0).sub(0).dofmap().dofs()
            alpha = self.Coor_fun[Dofs_fun, :] # array of coordinates
            ind = np.where(abs(alpha[:,1]) < 1e-10)[0] # find nodes on the X-axis
            alpha = alpha[ind,0] # assign X coordinates
            alpha.sort() # sort in ascending order

        edf_f = []
        edf_r = []
        # get k-th singular mode
        force, response = self.get_mode(k)[0:2]
        # delete the pressure part
        force_short = self.P.transpose().dot(force)
        response_short = self.P.transpose().dot(response)
        # compute energy
        for coor in alpha:
            Q_vec = self.matrices.MatQu(coor) # weight matrix for energy computing
            e_f = np.dot(force_short.T.conj(), Q_vec.dot(force_short)) # energy of the force
            e_r = np.dot(response_short.T.conj(), Q_vec.dot(response_short)) # energy of the response
            edf_f.append(e_f)
            edf_r.append(e_r)
        # return the gradients of the energy array
        return np.gradient(np.real(edf_f),alpha), np.gradient(np.real(edf_r),alpha), alpha

    def plot_mode(self, k = 0, direction=0, part='real', xlim=None, ylim=None):
        """Contour plot of the k-th mode

        Parameters
        ----------------------------
        k : int, optional
            default : 0

        direction : int, optional
            0 means stream-wise part and 1 means vertical part, default : 0

        part : string, optional
            plot 'real' or 'imag' part of the vectors, default : 'real'

        xlim : list, optional
            x range of the plot

        ylim : list, optional
            y range of the plot

        """
        # get the coordinates to plot
        Dofs_fun = self.functionspace.sub(0).sub(direction).dofmap().dofs()
        Coor_plot = self.Coor_fun[Dofs_fun, :]

        # get k-th mode
        force, response = self.get_mode(k)[0:2]
        f = eval('np.'+part+'(force[Dofs_fun])') # get the specified force part
        r = eval('np.'+part+'(response[Dofs_fun])') # get the specified response part

        # default range of the plot is the whole domain
        if xlim is None:
            xlim = [np.min(self.Coor_fun[:, 0]), np.max(self.Coor_fun[:, 0])]
        if ylim is None:
            ylim = [np.min(self.Coor_fun[:, 1]), np.max(self.Coor_fun[:, 1])]

        # contour plot of vectors
        plt.figure(0)
        plt.tricontourf(Coor_plot[:, 0], Coor_plot[:, 1], f, 100)
        plt.colorbar()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.figure(1)
        plt.tricontourf(Coor_plot[:, 0], Coor_plot[:, 1], r, 100)
        plt.colorbar()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()


class SpLuInv(LinearOperator):
    """
    SpLuInv:
       helper class to repeatedly solve M*x=b
       using a sparse LU-decopposition of M
    """
    def __init__(self, M):
        self.M_lu = splu(M)
        self.shape = M.shape
        self.dtype = M.dtype
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)

    def _matvec(self, x):
        # careful here: splu.solve will throw away imaginary
        # part of x if M is real
        x = np.asarray(x)
        if self.isreal and np.issubdtype(x.dtype, np.complexfloating):
            return (self.M_lu.solve(np.real(x).astype(self.dtype))
                    + 1j * self.M_lu.solve(np.imag(x).astype(self.dtype)))
        else:
            return self.M_lu.solve(x.astype(self.dtype))