from __future__ import print_function
from fenics import *
import numpy as np
import os,sys,inspect
from petsc4py import PETSc
from scipy.io import mmwrite
import scipy.sparse.linalg as spla
from scipy.sparse.linalg.interface import LinearOperator
#from interface import LinearOperator
from scipy.sparse import isspmatrix_csc,isspmatrix_csr, bmat, csc_matrix, csr_matrix
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0,parentdir)
from FrequencyAnalysis.MatrixAssemble import MatrixAssemble
noScikit = False
try:
    import scikits.umfpack as umfpack
except ImportError:
    noScikit = True
#%%
"""This module provides classes that define linear operators for solving linear equations
"""
class MatInv(LinearOperator):
    def __init__(self, A, A_lu=None,lusolver='mumps',trans='N',echo=False):
        self.shape = A.shape
        self.dtype = A.dtype
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)
        self.count=0
        self.echo=echo
        self.trans=trans        
        self.lusolver=lusolver
        self.__solver(A, A_lu,lusolver)

    def __solver(self,A, A_lu,lusolver):
        opts={'mumps':  "Mumpslu(A,lusolver='mumps')",
              'superlu':'Superlu(A)',
              'umfpack':'Umfpacklu(A)',
              'petsc':"PETSclu(A,lusolver='mumps')"}        
        if A_lu is None:
            self.solve=eval(opts[lusolver]).solve
        else:
            try:
                A_lu.stype
            except:          
                def solve(b, trans='N'):
                    if trans=='N':
                        if self.isreal and np.issubdtype(b.dtype, np.complexfloating):
                            return (A_lu(np.real(b).astype(self.dtype)) + 1j * A_lu(np.imag(b).astype(self.dtype)))
                        else:
                            return A_lu(b.astype(self.dtype))
                    elif trans=='T' or trans=='H':
                        try:
                            if self.isreal and np.issubdtype(b.dtype, np.complexfloating):
                                return (A_lu(np.real(b).astype(self.dtype),trans) + 1j * A_lu(np.imag(b).astype(self.dtype),trans))
                            else:
                                return A_lu(b.astype(self.dtype),trans)
                        except:
                            raise ValueError("Factorized object 'A_lu' doesn't receive system parameter 'trans'")
                    else:
                        raise ValueError("Undefined system parameter 'trans'")
                self.solve=solve
            else:
                if A_lu.stype in opts.keys():
                    try:
                        self.solve=A_lu.solve
                    except:
                        self.solve=A_lu
                else:
                    raise ValueError("Undefined solver type 'stype'")

    def _matvec(self,b):
        b = np.asarray(b)
        self.count+=1
        if self.echo is True:
            info('Iteration: %d' %self.count)
        return self.solve(b,self.trans)
    
    def _rmatvec(self,b):
        """
        
        """
        self.count+=1
        if self.echo is True:
            info('Number of the linear system solved: %d' %self.count)
            
        if self.trans=='N':
            return self.solve(b,'H')
        elif self.trans=='H':
            return self.solve(b,'N')
        elif self.trans=='T':
            return np.conj(self.solve(np.conj(b),'N'))
    
class Mumpslu:
    stype='mumps'
    def __init__(self,A,lusolver='mumps'):
        self.shape = A.shape
        self.dtype = A.dtype
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)
        self.mat_op=MatrixAssemble()
        self.__solver(A, lusolver)
    
    def __solver(self, A, lusolver):
        if not isspmatrix_csc(A):
            A = csc_matrix(A)        
        A = A.asfptype()  # upcast to a floating point format
        if A.dtype.char not in 'dD':
            raise ValueError("convert matrix data to double, please, using"
                  " .astype(), or set linsolve.useUmfpack = False")
        self.solver = PETScLUSolver(lusolver)
        self.solver.parameters['reuse_factorization'] = True
        if not self.isreal:
            self.solver.set_operator(self.mat_op.convertmatrix(bmat([[np.real(A), -np.imag(A)], [np.imag(A), np.real(A)]],format='csc'),flag='Mat2PETSc'))
            self.u=self.mat_op.convertvector(np.asarray(np.zeros(self.shape[0]*2)),flag='Vec2PETSc')
        else:
            self.solver.set_operator(self.mat_op.convertmatrix(A,flag='Mat2PETSc'))
            self.u=self.mat_op.convertvector(np.asarray(np.zeros(self.shape[0])).astype(self.dtype),flag='Vec2PETSc')
    
    def __solve(self,A_lu, b, transpose=False):
        if np.size(b.shape)==1:
            b = np.asarray(b)
        elif np.min(b.shape)==1:
            b = np.asarray(b).flatten()
        else:
            raise ValueError('Please give RHS in 1D array')
        if not self.isreal:
            b_r = np.block([np.real(b.astype(self.dtype).flatten()),np.imag(b.astype(self.dtype).flatten())])
            B=self.mat_op.convertvector(b_r,flag='Vec2PETSc')
            A_lu(B.vec(), self.u.vec())
            x=self.u.get_local()[0:self.shape[0]]+self.u.get_local()[self.shape[0]:self.shape[0]*2]*1j
        else:
            if np.issubdtype(b.dtype, np.complexfloating):
                B=self.mat_op.convertvector(np.real(b).astype(self.dtype),flag='Vec2PETSc')
                A_lu(B.vec(), self.u.vec())
                x=self.u.get_local()
                B=self.mat_op.convertvector(np.imag(b).astype(self.dtype),flag='Vec2PETSc')
                A_lu(B.vec(), self.u.vec())
                x=x+self.u.get_local()*1j
            else:
                B=self.mat_op.convertvector(b.astype(self.dtype),flag='Vec2PETSc')
                A_lu(B.vec(), self.u.vec())
                x=self.u.get_local()
        return x
    def solve(self, b, trans='N'):
        b = np.asarray(b)
        if trans=='N':
            A_lu = self.solver.ksp().solve
            x=self.__solve(A_lu, b,transpose=False)
        elif trans=='H':
            A_lu = self.solver.ksp().solveTranspose
            x=self.__solve(A_lu, b,transpose=True)
        elif trans=='T':
            A_lu = self.solver.ksp().solveTranspose
            if not self.isreal:
                x=np.conjugate(self.__solve(A_lu, np.conjugate(b),True))
            else:
                x=self.__solve(A_lu, b, transpose=True)
        return x
    solve.stype=stype                
            
class Superlu:
    stype='superlu'
    def __init__(self,A):
        self.shape = A.shape
        self.dtype = A.dtype
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)
        self.__solver(A)

    def __solver(self,A):
        if not isspmatrix_csc(A):
            A = csc_matrix(A)        
        A = A.asfptype()  # upcast to a floating point format
        if A.dtype.char not in 'dD':
            raise ValueError("convert matrix data to double, please, using"
                  " .astype(), or set linsolve.useUmfpack = False")
        self.solver = spla.splu(A)

    def __solve(self,A_lu, b, trans):
        if np.size(b.shape)==1:
            b = np.asarray(b)
        elif np.min(b.shape)==1:
            b = np.asarray(b).flatten()
        else:
            raise ValueError('Please give RHS in 1D array')
        if self.isreal and np.issubdtype(b.dtype, np.complexfloating):
            x = (A_lu(np.real(b).astype(self.dtype),trans) + 1j * A_lu(np.imag(b).astype(self.dtype),trans))
        else:
            x = A_lu(b.astype(self.dtype),trans)
        return x

    def solve(self, b, trans='N'):
        b = np.asarray(b)
        A_lu = self.solver.solve
        x = self.__solve(A_lu, b, trans)
        return x
    solve.stype=stype
        
class Umfpacklu:
    stype='umfpack'
    def __init__(self,A):
        self.shape = A.shape
        self.dtype = A.dtype
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)
        if noScikit:
            raise RuntimeError('Scikits.umfpack not installed.')        
        self.__solver(A)
    
    def __solver(self, A):
        if not isspmatrix_csc(A):
            A = csc_matrix(A)        
        A = A.asfptype()  # upcast to a floating point format
        if A.dtype.char not in 'dD':
            raise ValueError("convert matrix data to double, please, using"
                  " .astype(), or set linsolve.useUmfpack = False")
        A.indptr=A.indptr.astype(np.int64)
        A.indices=A.indices.astype(np.int64)
        umf = umfpack.UmfpackContext(self._get_umf_family(A))
        # Make LU decomposition.
        umf.numeric(A)
        self.solver=umf
        self.operator=A
        
    def _get_umf_family(self, A):
        """Get umfpack family string given the sparse matrix dtype."""
        _families = {
            (np.float64, np.int32): 'di',
            (np.complex128, np.int32): 'zi',
            (np.float64, np.int64): 'dl',
            (np.complex128, np.int64): 'zl'
        }
        f_type = np.sctypeDict[A.dtype.name]
        i_type = np.sctypeDict[A.indices.dtype.name]
        try:
            family = _families[(f_type, i_type)]
        except KeyError:
            msg = 'only float64 or complex128 matrices with int32 or int64' \
                ' indices are supported! (got: matrix: %s, indices: %s)' \
                % (f_type, i_type)
            raise ValueError(msg)
        return family
    
    def __solve(self,A_lu,b,trans):
        if np.size(b.shape)==1:
            b = np.asarray(b)
        elif np.min(b.shape)==1:
            b = np.asarray(b).flatten()
        else:
            raise ValueError('Please give RHS in 1D array')
        A=self.operator
        b = np.asarray(b)
        sys_para={'N':umfpack.UMFPACK_A, 'T':umfpack.UMFPACK_Aat,'H':umfpack.UMFPACK_At}
        
        if self.isreal and np.issubdtype(b.dtype, np.complexfloating):
            x = (A_lu(sys_para[trans], A, np.real(b).astype(self.dtype),autoTranspose=True) + 1j * A_lu(sys_para[trans], A, np.imag(b).astype(self.dtype),autoTranspose=True))
        else:
            x = A_lu(sys_para[trans], A, b.astype(self.dtype),autoTranspose=True)
        return x
    
    def solve(self,b,trans='N'):
        b = np.asarray(b)
        A_lu = self.solver.solve
        x = self.__solve(A_lu, b, trans)
        return x
    solve.stype=stype
        
class PETSclu:
    stype='petsc'
    def __init__(self,A, lusolver='mumps'):
        self.shape = A.shape
        self.dtype = A.dtype
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)
        self.__solver(A,lusolver)
        
    def __solver(self,A, lusolver):
        if not isspmatrix_csc(A):
            A = csc_matrix(A)        
        A = A.asfptype()  # upcast to a floating point format
        if A.dtype.char not in 'dD':
            raise ValueError("convert matrix data to double, please, using"
                  " .astype(), or set linsolve.useUmfpack = False")
        if not self.isreal:
            operator=bmat([[np.real(A), -np.imag(A)], [np.imag(A), np.real(A)]],format='csc')
        else:
            operator=A
        A_op = self.__convert(operator)
        self.u = A_op.getVecs()[0]
        self.u.set(0)
        
        self.solver = PETSc.KSP()
        self.solver.create(PETSc.COMM_WORLD)
        self.solver.setType('preonly')
        self.solver.getPC().setType('lu')
        self.solver.getPC().setFactorSolverPackage(lusolver)
        self.solver.getPC().setReusePreconditioner(True)
        self.solver.setOperators(A_op)
        self.solver.setFromOptions()
    
    def __convert(self,A):
        if not isspmatrix_csr(A):
            A = csr_matrix(A)
        p1=A.indptr
        p2=A.indices
        p3=A.data
        petsc_mat = PETSc.Mat().createAIJ(size=A.shape,csr=(p1,p2,p3))# creating petsc matrix from csr
        return petsc_mat
   
    def __solve(self,A_lu, b):
        if np.size(b.shape)==1:
            b = np.asarray(b)
        elif np.min(b.shape)==1:
            b = np.asarray(b).flatten()
        else:
            raise ValueError('Please give RHS in 1D array')
            
        if not self.isreal:
            b_r = np.block([np.real(b.astype(self.dtype).flatten()),np.imag(b.astype(self.dtype).flatten())])
            B=PETSc.Vec().createWithArray(b_r)
            A_lu(B, self.u)
            x=self.u.getArray()[0:self.shape[0]]+self.u.getArray()[self.shape[0]:self.shape[0]*2]*1j
        else:
            if np.issubdtype(b.dtype, np.complexfloating):
                B=PETSc.Vec().createWithArray(np.real(b).astype(self.dtype))
                A_lu(B, self.u)
                x=self.u.getArray()
                B=PETSc.Vec().createWithArray(np.imag(b).astype(self.dtype))
                A_lu(B, self.u)
                x=x+self.u.getArray()*1j
            else:
                B=PETSc.Vec().createWithArray(b.astype(self.dtype))
                A_lu(B, self.u)
                x=self.u.getArray()
        return x
    def solve(self, b, trans='N'):
        b = np.asarray(b)
        if trans=='N':
            A_lu = self.solver.solve
            x=self.__solve(A_lu, b)
        elif trans=='H':
            A_lu = self.solver.solveTranspose
            x=self.__solve(A_lu, b)
        elif trans=='T':
            A_lu = self.solver.solveTranspose
            if not self.isreal:
                x=np.conjugate(self.__solve(A_lu, np.conjugate(b)))
            else:
                x=self.__solve(A_lu, b)
        return x
    solve.stype=stype
        