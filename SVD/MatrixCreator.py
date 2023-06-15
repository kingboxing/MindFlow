from __future__ import print_function
from fenics import *
import numpy as np
from scipy.sparse import csr_matrix, identity,diags
import os,sys,inspect
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0,parentdir)
from FrequencyAnalysis.MatrixAssemble import MatrixAssemble
from FlowSolver.Tools import *
"""This module provides classes that define assistant matrices using assemble feature in FEniCS
"""


class MatrixCreator:
    """Assemble useful matrices using FEniCS
    The class helps to construct matrices for resolvent analysis.
    e.g. N-S equations can be written as u = C(w)f in the frequency
    domain, where C is known as the resolvent operator, and obtained as
    C = P^T*(iwB - A)^-1*P*M. (iwB - A) could be the matrix obtained by the
    finite element discretization. P is the prolongation operator and P^T is
    the restriction operator, which extracts the velocity components from a
    velocity-pressure vector. M designates the mass-matrix. Some weight matrices
    Qu, Qf may also need to calculate the total energy of the velocity and force.
    All theses could be constructed using this class.

    Parameters
    ----------------------------
    mesh : object created by FEniCS function Mesh
        mesh of the flow field

    functionspace : finite element function space
        Mixed function space with velocity subspace and pressure subspace (2D flow field)

    boundarycondition : list, optional except for matrix MatM
        boundary conditions applied via FEniCS function DirichletBC()

    boundary : object Boundary()
        the Boundary object with defined and marked boundaries

    Attributes
    ----------------------------
    bcs : list, optional except for matrix MatM
        boundary conditions applied via FEniCS function DirichletBC()

    v, q : TestFunctions of velocity vector and pressure

    u, p : rialFunctions of velocity vector and pressure

    tol : float
        tolerance used in the class, default 1e-8

    Examples
    ----------------------------
    >>> from fenics import *
    >>> from RAPACK.SVD.MatrixCreator import MatrixCreator
    >>> from RAPACK.FlowSolver.FiniteElement import TaylorHood
    >>> mesh = Mesh("mesh.xml")
    >>> element = TaylorHood(mesh = mesh)
    >>> matrices = MatrixCreator(mesh=mesh, functionspace=element.functionspace)
    >>> P = matrices.MatP() # prolongation operator

    """
    def __init__(self, mesh=None, functionspace=None, boundarycondition=None, boundary=None):
        self.__mesh=mesh
        self.__functionspace=functionspace
        self.bcs=boundarycondition
        self.__boundary=boundary
        self.__functions()
        self.tol = 1E-8

    def __functions(self):
        """Create TestFunctions and TrialFunctions
        """
        fun_test = TestFunction(self.__functionspace)
        (self.v, self.q) = split(fun_test)
        fun_trial = TrialFunction(self.__functionspace)
        (self.u, self.p) = split(fun_trial)

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

    def __Multidomains(self,Q_bound=None,condition=None):
        """Vertically split the domain into two domains 0 and 1
        the left one is marked as 1 and the right one is marked as 0

        Parameters
        ----------------------------
        Q_bound : float (optional)
            coordinate of the vertical boundary

        condition : string 
            condition of defining subdomains
            default value: 'x[0] <= Q_bound +- tol'

        Returns
        ----------------------------
        domains : CellFunction
            the object represents two subdomains

        """
        tol = self.tol
        if condition is None:
            condition='x[0] <= '+str(Q_bound)+'+ '+str(tol) +' or '+'x[0] <= '+str(Q_bound)+'- '+str(tol)
        class Omega_1(SubDomain):
            def inside(self, x, on_boundary):
                return eval(condition)

        domains = MeshFunction('size_t', self.__mesh, self.__mesh.topology().dim())
        domains.set_all(0) # mark the whole domain as 0
        subdomain1 = Omega_1()
        subdomain1.mark(domains, 1) # mark the left subdomain as 1
        
        boundaries = MeshFunction('size_t', self.__mesh, self.__mesh.topology().dim() - 1)
        boundaries.set_all(0)
        subboundary1=Omega_1()
        subboundary1.mark(boundaries,1)
        return domains, boundaries

    def MatP(self,delete=None):
        """Construct the prolongation matrix P
        P is the prolongation operator and P^T is the restriction operator,
        which extracts the velocity components from a velocity-pressure vector.

        Returns
        ----------------------------
        sparse matrix P in CSR format

        """
        
        #boundaries = MeshFunction('size_t', self.__mesh, self.__mesh.topology().dim() - 1)#FacetFunction("size_t", self.__mesh)
        # set all facets as 0
        #boundaries.set_all(0)
        if delete is None or delete=='pressure':
            # Dirichlet boundary condition for velocity space
            subspace=0
            flag=True
            bc=[]        
            while flag is True:
                try: 
                    self.__functionspace.sub(0).sub(subspace)
                except:
                    flag=False
                else:
                    bc.append(DirichletBC(self.__functionspace.sub(0).sub(subspace), Constant(0.0),"", method="pointwise"))
                    subspace +=1
            # sparse matrix : expression u*v*dx without p ; Dirichlet boundary conditions for velocity
            #               In the velocity subspace, the matrix only has ones in diagonal
            #               In the pressure subspace, the matrix only has zeros
            P_sparray = self.__assemblematrix(inner(self.u,self.v)*dx, bc)
            # find columns with only zeros and delete them
            indices = P_sparray.nonzero()
            self.ins=indices
            return P_sparray[:, indices[1]]
        elif delete=='velocity':
            bc=[]
            bc.append(DirichletBC(self.__functionspace.sub(1), Constant(0.0),"", method="pointwise"))
            P_sparray = self.__assemblematrix(self.p*self.q*dx, bc)
            indices = P_sparray.nonzero()
            return P_sparray[:, indices[1]]
        elif delete=='velocity_y+pressure':
            bc=[]
            bc.append(DirichletBC(self.__functionspace.sub(0).sub(0), Constant(0.0),"", method="pointwise"))
            P_sparray = self.__assemblematrix(self.u[0]*self.v[0]*dx, bc)
            indices = P_sparray.nonzero()
            return P_sparray[:, indices[1]]
        elif delete=='velocity_x+pressure':
            bc=[]
            bc.append(DirichletBC(self.__functionspace.sub(0).sub(1), Constant(0.0),"", method="pointwise"))
            P_sparray = self.__assemblematrix(self.u[1]*self.v[1]*dx, bc)
            indices = P_sparray.nonzero()
            return P_sparray[:, indices[1]]

    def MatM(self):
        """Construct the mass matrix M
        boundary conditions of N-S equations are required, need to set the value of the attribute bcs.

        Returns
        ----------------------------
        sparse matrix M in CSR format

        """

        # sparse matrix : expression u*v*dx without p, M_sparray is the mass matrix without boundary conditions
        #               In the pressure subspace, the matrix M_sparray only has zeros
        M_sparray = self.__assemblematrix(inner(self.u,self.v)*dx)
        # sparse matrix : expression 0*u*v*dx; Dirichlet boundary conditions for velocity and pressure
        #               For the entries with Dirichlet boundary conditions, the matrix only has ones in diagonal
        #               In the remaining subspace, the matrix only has zeros
        Z_sparray = self.__assemblematrix(Constant(0.0)*inner(self.u,self.v)*dx, self.bcs)
        # Identity matrix
        I = identity(Z_sparray.shape[0], dtype=Z_sparray.dtype, format='csr')
        # For the entries without Dirichlet boundary conditions, the matrix only has ones in diagonal
        # Z_sparray helps to apply the boundary conditions (assign zeros) to the full mass matrix M_sparray
        Z_sparray = (I - Z_sparray)
        # the prolongation matrix P
        P = self.MatP()

        # delete the pressure part and return the mass matrix
        return P.transpose() * (Z_sparray * M_sparray) * P

    def MatBcs(self):
        """Construct the mass matrix Bcc
        boundary conditions of N-S equations are required, need to set the value of the attribute bcs.

        Returns
        ----------------------------
        sparse matrix Bcc in CSR format

        """
        # sparse matrix : expression 0*u*v*dx; Dirichlet boundary conditions for velocity and pressure
        #               For the entries with Dirichlet boundary conditions, the matrix only has ones in diagonal
        #               In the remaining subspace, the matrix only has zeros
        Z_sparray = self.__assemblematrix(Constant(0.0)*inner(self.u,self.v)*dx, self.bcs)
        # Identity matrix
        I = identity(Z_sparray.shape[0], dtype=Z_sparray.dtype, format='csr')
        # For the entries without Dirichlet boundary conditions, the matrix only has ones in diagonal
        # Z_sparray helps to apply the boundary conditions (assign zeros) to the full mass matrix M_sparray
        Z_sparray = (I - Z_sparray)
        # the prolongation matrix P
        P = self.MatP()

        # delete the pressure part and return the boundary condition matrix (no pressure condition)
        return P.transpose() * Z_sparray * P

    def MatQf(self, Qf_bound=None,condition=None):
        """Construct the weight function Qf for the subdomain 1
        Qf evaluates the weight of each node in the mesh and could helps
        to calculate the subtotal energy of the force as E = f^H*Qf*f

        Parameters
        ----------------------------
        Qf_bound : float
            coordinate of the vertical boundary
            used in the subdomain 1 condition: 'x[0] <= Qf_bound + tol'

        Returns
        ----------------------------
        sparse matrix Qf in CSR format

        """
        if Qf_bound!=None or condition!=None:
            info('Subdomain evaluation may result in errors in the current version')
        # Of should be positive definite in SVD and for periodic flow, coord is not true ?
        # if Qf_bound is not given, then set it as the maximum X coordinates
        if Qf_bound is None:
            Coor_fun = self.__functionspace.tabulate_dof_coordinates().reshape((-1, 2))
            Qf_bound = 1.5*np.max(Coor_fun[:, 0])
        # vertically split into two subdomains
        if condition is None:
            domains = self.__Multidomains(Q_bound=Qf_bound)[0]
        elif type(condition)is str:
            domains = self.__Multidomains(condition=condition)[0]
            
        dx = Measure("dx", domain=self.__mesh, subdomain_data=domains)
#        # the weight function for the left subdomain 1
#        Q_sparray = self.__assemblematrix(inner(self.u,self.v)*dx(subdomain_id=1))
        
        
        Q_sparray = self.__assemblematrix(inner(self.u,self.v)*dx(1))
        # the prolongation matrix P
        P = self.MatP()
        # return the weight function Qf without pressure part
        return P.transpose() * Q_sparray * P

    def MatQu(self, Qu_bound=None,condition=None): # the same as Qf, could be deleted
        """Construct the weight function Qu for the subdomain 1
        Qu evaluates the weight of each node in the mesh and could helps
        to calculate the subtotal energy of the velocity as E = u^H*Qf*u

        Parameters
        ----------------------------
        Qu_bound : float
            coordinate of the vertical boundary
            used in the subdomain 1 condition: 'x[0] <= Qu_bound + tol'

        Returns
        ----------------------------
        sparse matrix Qu in CSR format

        """
        if Qu_bound!=None or condition!=None:
            info('Subdomain evaluation may result in errors in the current version')
        
        # if Qu_bound is not given, then set it as the maximum X coordinates
        if Qu_bound is None:
            Coor_fun = self.__functionspace.tabulate_dof_coordinates().reshape((-1, 2))
            Qu_bound = 1.5*np.max(Coor_fun[:, 0])
        # vertically split into two subdomains
        if condition is None:
            domains = self.__Multidomains(Q_bound=Qu_bound)[0]
        elif type(condition)is str:
            domains = self.__Multidomains(condition=condition)[0]

        dx = Measure('dx', domain=self.__mesh, subdomain_data=domains)
#        # the weight function for the left subdomain 1
#        Q_sparray = self.__assemblematrix(inner(self.u,self.v)*dx(1))


        Q_sparray = self.__assemblematrix(inner(self.u,self.v)*dx(1))
        # the prolongation matrix P
        P = self.MatP()
        # return the weight function Qu without pressure part
        return P.transpose()*Q_sparray*P
        
    def Identity(self,condition=None):
        if condition is None:
            return identity(self.__functionspace.dim())
        elif type(condition) is str:
            matrices=MatrixAssemble()
            domains, boundaries = self.__Multidomains(condition=condition)
            #dx = Measure('dx', domain=self.__mesh, subdomain_data=domains)
            bcs=[]
            if self.__functionspace.num_sub_spaces()==0:
                bcs.append(DirichletBC(self.__functionspace, Constant(1.0),domains, method="pointwise"))# remains to check
            else:
                for i in range(self.__functionspace.num_sub_spaces()):
                    if self.__functionspace.sub(i).num_sub_spaces()==0:
                        bcs.append(DirichletBC(self.__functionspace.sub(i), Constant(1.0),domains, method="pointwise"))
                    else:
                        for j in range(self.__functionspace.sub(i).num_sub_spaces()):
                            bcs.append(DirichletBC(self.__functionspace.sub(i).sub(j), Constant(1.0),domains, method="pointwise"))
            I = matrices.assemblevector(inner(Constant((0.0,0.0)),self.v)*dx,bcs=bcs)
            return diags(np.asarray(I).flatten())
            