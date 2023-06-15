from __future__ import print_function
from fenics import *
import numpy as np
from scipy.sparse.linalg.interface import LinearOperator
import scipy.sparse.linalg as spla
from scipy.sparse import isspmatrix_csc,isspmatrix
import os,sys,inspect
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0,parentdir)
from FrequencyAnalysis.MatrixAssemble import MatrixAssemble
"""This module provides classes that define linear operators
"""
class MatInv(LinearOperator):
    """Linear operator for performing matrix inverse
    Many iterative methods (e.g. Arnoldi iteration for eigenvalue problems)
    do not need to inverse a matrix to perform a multiplication between the
    inverse matrix and a vector A^-1*b. Instead, a linear system A*x=b is
    solve via a LU solver. This class helps to repeatedly solve A*x=b using
    a sparse LU-decomposition of A in iterative solvers

    Parameters
    ----------------------------
    A : sparse matrix
        An array or sparse matrix representing the operation A^-1*v, where
        A is a real or complex square matrix.

    A_lu : solve instance, optional
        Object, which is a solve instance

    lusolver: string, indicate solver type
        'mumps': using FEniCS build-in solver
        'superlu' using scipy.sparse.splu
        'umfpack' using scipy.sparse.factorize with Umfpack=True

    trans : {'N', 'T', 'H'}, optional
        'N':   A   * x == b  (default)
        'T':   A^T * x == b
        'H':   A^H * x == b
        i.e., normal, transposed, and hermitian conjugate

    Attributes
    ----------------------------
    shape : tuple
        Matrix dimensions (M,N)

    dtype : dtype
        Data type of the matrix.

    A_lu : solve instance, optional
        Object, which is a solve instance

    trans : {'N', 'T', 'H'}, optional
        'N':   A   * x == b  (default)
        'T':   A^T * x == b
        'H':   A^H * x == b
        i.e., normal, transposed, and hermitian conjugate

    Returns
    ----------------------------
    Linear operator represents a inverse matrix

    Examples
    ----------------------------
    compute first three largest eigenvalues and eigenvectors of A^-1
    >>> import scipy.sparse.linalg as spla
    >>> from scipy.sparse import identity
    >>> from RAPACK.SVD.MatrixOperator import Matinv
    >>> A = identity(30)
    >>> A_lu = spla.splu(A).solve
    >>> Ainv = Matinv(A, A_lu = A_lu, trans = 'N')
    >>> vals, vecs = spla.eigs(Ainv, k=3, which='LM')

    """
    def __init__(self,A,A_lu=None,lusolver='mumps',trans='N'):
        self.shape = A.shape
        self.dtype = A.dtype
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)
        if A_lu is None:
            if not isspmatrix(A):
                raise ValueError('Matrix must be a sparse matrix. Try to use scipy.sparse lib.')
            self.trans={'N':'A.copy()','T':'A.T.copy()','H':'A.H.copy()'}
            A=eval(self.trans[trans])
            if lusolver == 'mumps':
                if not self.isreal:
                    raise ValueError('Currently only support real matrices')
                self.mat_op=MatrixAssemble()
                self.solver = PETScLUSolver(lusolver)
                self.solver.set_operator(self.mat_op.convertmatrix(A,flag='Mat2PETSc'))
                self.solver.parameters['reuse_factorization'] = True
                self.A_lu = self.solver.solve
                self.u=self.mat_op.convertvector(np.asarray(np.zeros(self.shape[0])).astype(self.dtype),flag='Vec2PETSc')
            elif lusolver =='superlu':
                if not isspmatrix_csc(A):
                    A=A.tocsc()
                self.A_lu = spla.splu(A).solve
            elif lusolver == 'umfpack':
                if not isspmatrix_csc(A):
                    A=A.tocsc()
                spla.use_solver(useUmfpack=True)
                A.indptr=A.indptr.astype(np.int64)
                A.indices=A.indices.astype(np.int64)
                self.A_lu = spla.factorized(A)
        else:
            if lusolver == 'superlu' or trans == 'N':
                self.A_lu=A_lu
                self.tans=trans
            elif lusolver !='superlu' and trans!='N':
                raise ValueError("Please provide A_lu using SuperLU because trans!='N'")
        self.lusolver=lusolver
        
    def _matvec(self,b):
        b = np.asarray(b)
        if self.lusolver == 'mumps':
            if np.issubdtype(b.dtype, np.complexfloating):
                B=self.mat_op.convertvector(np.real(b).astype(self.dtype),flag='Vec2PETSc')
                self.A_lu(self.u, B)
                x=self.u.vector().get_local()
                B=self.mat_op.convertvector(np.imag(b).astype(self.dtype),flag='Vec2PETSc')
                self.A_lu(self.u, B)
                x+=self.u.get_local()*1j
                return x
            else:
                B=self.mat_op.convertvector(b.astype(self.dtype),flag='Vec2PETSc')
                self.A_lu(self.u, B)
                x=self.u.get_local()
                return x
        elif type(self.trans) is dict or self.trans == 'N':
            if self.isreal and np.issubdtype(b.dtype, np.complexfloating):
                return (self.A_lu(np.real(b).astype(self.dtype)) + 1j * self.A_lu(np.imag(b).astype(self.dtype)))
            else:
                return self.A_lu(b.astype(self.dtype))
                
        elif type(self.trans) is str and self.trans !='N':
            if self.isreal and np.issubdtype(b.dtype, np.complexfloating):
                return (self.A_lu(np.real(b).astype(self.dtype),self.trans) + 1j * self.A_lu(np.imag(b).astype(self.dtype),self.trans))
            else:
                return self.A_lu(b.astype(self.dtype),self.trans)


class SuperluInv(LinearOperator):
    """Linear operator for performing matrix inverse
    Many iterative methods (e.g. Arnoldi iteration for eigenvalue problems)
    do not need to inverse a matrix to perform a multiplication between the
    inverse matrix and a vector A^-1*b. Instead, a linear system A*x=b is
    solve via a LU solver. This class helps to repeatedly solve A*x=b using
    a sparse LU-decomposition of A in iterative solvers

    Parameters
    ----------------------------
    A : ndarray, sparse matrix
        An array or sparse matrix representing the operation A^-1*v, where
        A is a real or complex square matrix.

    A_lu : scipy.sparse.linalg.SuperLU, optional
        Object, which has a solve method

    trans : {'N', 'T', 'H'}, optional
        'N':   A   * x == b  (default)
        'T':   A^T * x == b
        'H':   A^H * x == b
        i.e., normal, transposed, and hermitian conjugate

    Attributes
    ----------------------------
    shape : tuple
        Matrix dimensions (M,N)

    dtype : dtype
        Data type of the matrix.

    A_lu : scipy.sparse.linalg.SuperLU
        Object, which has a solve method

    trans : {'N', 'T', 'H'}, optional
        'N':   A   * x == b  (default)
        'T':   A^T * x == b
        'H':   A^H * x == b
        i.e., normal, transposed, and hermitian conjugate

    Returns
    ----------------------------
    Linear operator represents a inverse matrix

    Examples
    ----------------------------
    compute first three largest eigenvalues and eigenvectors of A^-1
    >>> import scipy.sparse.linalg as spla
    >>> from scipy.sparse import identity
    >>> from RAPACK.SVD.MatrixOperator import Matinv
    >>> A = identity(30)
    >>> A_lu = spla.splu(A)
    >>> Ainv = Matinv(A, A_lu = A_lu, trans = 'N')
    >>> vals, vecs = spla.eigs(Ainv, k=3, which='LM')

    """
    def __init__(self, A, A_lu=None, trans='N'):

        if A_lu is not None:
            self.A_lu = A_lu
        else:
            self.A_lu = spla.splu(A).solve #
        self.trans=trans
        self.shape = A.shape
        self.dtype = A.dtype
        self.__isreal = not np.issubdtype(self.dtype, np.complexfloating) # complex matrix or not

    def _matvec(self, b):
        """Define the operation A^-1*b

        Parameters
        ----------------------------
        b : 1D array

        Returns
        ----------------------------
        x = A^-1*b

        """
        b = np.asarray(b)

        # careful here: splu.solve will throw away imaginary part of x if A is real
        if self.__isreal and np.issubdtype(b.dtype, np.complexfloating):
            return (self.A_lu(np.real(b).astype(self.dtype),self.trans)
                    + 1j * self.A_lu(np.imag(b).astype(self.dtype),self.trans))
        else:
            return self.A_lu(b.astype(self.dtype),self.trans)


class UmfpackInv(LinearOperator):
    """
    SpLuInv:
       helper class to repeatedly solve M*x=b
       using a sparse LU-decopposition of M
    """
    def __init__(self, M, M_lu=None):
        if M_lu is not None:
            self.M_lu = M_lu
        else:
            if not isspmatrix_csc(M):
                M=M.tocsc()
            spla.use_solver(useUmfpack=True)
            M.indptr=M.indptr.astype(np.int64)
            M.indices=M.indices.astype(np.int64)
            self.M_lu = spla.factorized(M)
        self.shape = M.shape
        self.dtype = M.dtype
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)

    def _matvec(self, x):
        # careful here: splu.solve will throw away imaginary
        # part of x if M is real
        x = np.asarray(x)
        if self.isreal and np.issubdtype(x.dtype, np.complexfloating):
            return (self.M_lu(np.real(x).astype(self.dtype))
                    + 1j * self.M_lu(np.imag(x).astype(self.dtype)))
        else:
            return self.M_lu(x.astype(self.dtype))
            
class MumpsInv(LinearOperator):
    def __init__(self,A,x=None,lusolver='mumps'):
        if x is None:        
            raise ValueError('x should be a finite element function')
        elif x.thisown:
            self.u=x
        else:
            raise ValueError('x should be a finite element function')
        self.mat_op=MatrixAssemble()
        self.solver = PETScLUSolver(lusolver)
        self.solver.set_operator(self.mat_op.convertmatrix(A,flag='Mat2PETSc'))
        self.solver.parameters['reuse_factorization'] = True
        self.shape = A.shape
        self.dtype = A.dtype
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)
        
    
    def _matvec(self,b):
        
        b = np.asarray(b)
        if np.issubdtype(b.dtype, np.complexfloating):
            B=self.mat_op.convertvector(np.real(b).astype(self.dtype),flag='Vec2PETSc')
            self.solver.solve(self.u.vector(), B)
            x=self.u.vector().get_local()
            B=self.mat_op.convertvector(np.imag(b).astype(self.dtype),flag='Vec2PETSc')
            self.solver.solve(self.u.vector(), B)
            x+=self.u.vector().get_local()*1j
            return x
        else:
            B=self.mat_op.convertvector(b.astype(self.dtype),flag='Vec2PETSc')
            self.solver.solve(self.u.vector(), B)
            x=self.u.vector().get_local()
            return x
 