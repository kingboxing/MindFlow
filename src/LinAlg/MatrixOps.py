#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 23:53:10 2024

@author: bojin
"""

"""
This module provides functions for matrix/vector conversion, assembling linear operators, and solving linear equations using various LU solvers.
"""

from ..Deps import *

#%%
"""
This class provides functions that assemble FEniCS expression into CSR matrix/ numpy vector
"""


def ConvertMatrix(matrix, flag='PETSc2Mat'):
    """
    convert a matrix between Scipy.sparse.csr_matrix and FEniCS PETScMatrix

    Parameters
    ----------
    matrix : FEniCS PETScMatrix or scipy.sparse matrix
        The matrix to be converted.
    flag : str, optional
        'PETSc2Mat' (default) converts from FEniCS PETScMatrix to scipy.sparse.csr_matrix.
        'Mat2PETSc' converts from scipy.sparse.csr_matrix to FEniCS PETScMatrix.

    Returns
    -------
    Converted matrix in the desired format.

    """
    if flag == 'PETSc2Mat':
        A_mat = as_backend_type(matrix).mat()
        A_sp = sp.csr_matrix(A_mat.getValuesCSR()[::-1], shape=A_mat.size)
        return A_sp
    elif flag == 'Mat2PETSc':
        A_mat = matrix.tocsr()
        p1, p2, p3 = A_mat.indptr, A_mat.indices, A_mat.data
        petsc_mat = PETSc.Mat().createAIJ(size=A_mat.shape, csr=(p1, p2, p3))  # creating petsc matrix from csr
        return PETScMatrix(petsc_mat)
    else:
        raise ValueError(f"Invalid flag: {flag}. Use 'PETSc2Mat' or 'Mat2PETSc'.")


def ConvertVector(vector, flag='PETSc2Vec'):
    """
    Convert a vector between 1D numpy array and FEniCS PETSc Vector.

    Parameters
    ----------
    vector : FEniCS PETScVector or numpy array
        The vector to be converted.
    flag : str, optional
        'PETSc2Vec' (default) converts from PETScVector to numpy array.
        'Vec2PETSc' converts from numpy array to PETScVector.

    Returns
    -------
    Converted vector in the desired format.

    """
    if flag == 'PETSc2Vec':
        return np.matrix(vector.get_local())
    elif flag == 'Vec2PETSc':
        return PETScVector(PETSc.Vec().createWithArray(np.asarray(vector)))
    else:
        raise ValueError(f"Invalid flag: {flag}. Use 'PETSc2Vec' or 'Vec2PETSc'.")


def AssembleSystem(mat_expr, vec_expr, bcs=None):
    """
    Assemble a linear system from FEniCS expressions.

    Parameters
    ----------
    mat_expr : UFL form
        The left-hand side expression.
    vec_expr : UFL form
        The right-hand side expression.
    bcs : list of DirichletBC, optional
        Boundary conditions to apply. Default is None.

    Returns
    -------
    A_sparry : scipy.sparse.csr_matrix
        Assembled LHS matrix in CSR format.
    b_vec : numpy array
        Assembled vector as a numpy matrix.
    """
    A, b = PETScMatrix(), PETScVector()
    assemble_system(mat_expr, vec_expr, bcs, A_tensor=A, b_tensor=b)
    return ConvertMatrix(A), ConvertVector(b)


def AssembleMatrix(expr, bcs=[]):
    """
    Assemble a matrix from a UFL expression.
    
    Parameters
    ----------
    expr : UFL form
        The expression to assemble.
    bcs : list of DirichletBC, optional
        Boundary conditions to apply. Default is [].

    Returns
    -------
    scipy.sparse.csr_matrix
    Assembled matrix in the format of Scipy.sparse.csr_matrix.

    """
    A = PETScMatrix()  # FEniCS using PETSc for matrix operation
    assemble(expr, tensor=A, keep_diagonal=True)  # store assembled matrix in A
    [bc.apply(A) for bc in bcs]
    # convert the format to CSR
    return ConvertMatrix(A)


def AssembleVector(expr, bcs=[]):
    """
    Assemble a vector from a UFL expression.

    Parameters
    ----------
    expr : UFL form
        The expression to assemble.
    bcs : list of DirichletBC, optional
        Boundary conditions to apply. Default is [].

    Returns
    -------
    numpy matrix
    Assembled vector as a numpy matrix.

    """
    b = PETScVector()  # FEniCS using PETSc for matrix operation
    assemble(expr, tensor=b)  # store assembled matrix in A
    [bc.apply(b) for bc in bcs]
    # convert to the numpy matrix
    return ConvertVector(b)


def TransposePETScMat(A):
    """
    Transpose a FEniCS PETScMatrix and return the transposed matrix.

    Parameters:
    - A: A FEniCS PETScMatrix object.

    Returns:
    - A new PETScMatrix that is the transpose of the input matrix.
    """
    # Transpose the PETSc matrix
    petsc_transposed = A.mat().transpose()
    # Convert back to a FEniCS PETScMatrix
    transposed_matrix = PETScMatrix(petsc_transposed)
    return transposed_matrix


#%%
# Helper function for complex matrix handling

def handle_complex_matrix(A, flag):
    """
    Convert a complex matrix into a real block matrix format if needed.

    Parameters
    ----------
    A : scipy.sparse matrix
        The matrix to be converted.
    flag : str, optional
        'PETSc2Mat' converts from FEniCS PETScMatrix to scipy.sparse.csr_matrix.
        'Mat2PETSc' converts from scipy.sparse.csr_matrix to FEniCS PETScMatrix.

    Returns
    -------
    TYPE
        scipy.sparse.csr_matrix or FEniCS PETScMatrix

    """
    isreal = not np.issubdtype(A.dtype, np.complexfloating)
    if not isreal:  # If complex
        A_real_imag = sp.bmat([[np.real(A), -np.imag(A)], [np.imag(A), np.real(A)]], format='csc')
        return ConvertMatrix(A_real_imag, flag=flag)
    else:
        return ConvertMatrix(A, flag=flag)


class SparseLUSolver:
    """
    Base class for sparse LU solvers to solve A*x = b.
    """
    stype = ''

    def __init__(self, A, lusolver):
        """
        Base class for sparse LU solvers to solve A*x = b.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix to be solved.
        lusolver : str
            Type of LU solver to use ('mumps', 'superlu', 'umfpack', 'petsc'). Default is 'mumps'.

        Returns
        -------
        None.

        """
        self.shape = A.shape
        self.dtype = A.dtype
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)
        self.operator = self._check_mtx(A)

    def _check_mtx(self, A):
        """
        Validate and prepare the matrix for LU factorization.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix to be checked.

        Raises
        ------
        ValueError
            Data type of the matrix.

        Returns
        -------
        A : scipy.sparse.csc_matrix
            Matrix in CSC format and floating-point type.

        """
        if not sp.isspmatrix_csc(A):
            A = sp.csc_matrix(A)
        A = A.asfptype()  # upcast to a floating point format
        if A.dtype.char not in 'dD':
            raise TypeError(
                "Matrix data should be double precision. Use .astype() if necessary.")  #or set linsolve.useUmfpack = False

        return A

    def _check_vec(self, b):
        """
        Validate and prepare the vector for solving.

        Parameters
        ----------
        b : numpy array
            The vector to be checked. 1-D numpy array.

        Raises
        ------
        ValueError
            The shape of the array.

        Returns
        -------
        b : numpy array
            Flattened 1D array.
            
        """

        if np.size(b.shape) == 1 or np.min(b.shape) == 1:
            b = np.asarray(b).flatten()
        else:
            raise ValueError('Please give 1D RHS array')

        return b

    def solve(self, b, trans='N'):
        """
        Abstract solve method to be implemented by subclasses.

        """
        pass

    solve.stype = stype
#%%


class FEniCSLU(SparseLUSolver):
    """
    LU solver using FEniCS functions with PETSc backend.
    """
    stype = 'mumps'

    def __init__(self, A, lusolver='mumps'):
        """
        LU solver using FEniCS functions with PETSc backend.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix to be solved.
        lusolver : str, optional
            Type of LU solver to use. The default is 'mumps'. pending ...

        Returns
        -------
        None.

        """
        super().__init__(A, lusolver)
        self._initialize_solver(self.operator, lusolver)

    def _initialize_solver(self, A, lusolver):
        """
        initialize solver object

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix to be solved.
            
        lusolver : str
            Type of LU solver to use. pending ...

        Returns
        -------
        None.

        """
        A_ops = handle_complex_matrix(A, flag='Mat2PETSc')  # assemble complex matrix into real block matrix

        self.u = PETScVector()
        A_ops.init_vector(self.u, 0)  # initialise vector using row dim
        self.u.zero()

        sol = PETScLUSolver(A_ops, lusolver)
        sol.parameters.add('reuse_factorization', True)
        self.solver = sol.ksp()

    def __solve(self, A_lu, b):
        """
        Detailed solve operations for different data types

        Parameters
        ----------
        A_lu : solve method in LU decomposition object
            Precomputed LU decomposition of A.
        b : numpy array
            The RHS vector to solve.

        Returns
        -------
        x : numpy array
            The solution of the linear problem.

        """
        b = self._check_vec(b)

        if not self.isreal:  # if A is complex
            b_r = np.block([np.real(b), np.imag(b)])
            B = ConvertVector(b_r, flag='Vec2PETSc')
            A_lu(B.vec(), self.u.vec())
            return self.u.get_local()[:self.shape[0]] + 1j * self.u.get_local()[self.shape[0]:]  # pending for parallel
        else:
            B = ConvertVector(np.real(b), flag='Vec2PETSc')
            A_lu(B.vec(), self.u.vec())
            x = self.u.get_local()

            if np.issubdtype(b.dtype, np.complexfloating):  # if b is complex
                B = ConvertVector(np.imag(b), flag='Vec2PETSc')
                A_lu(B.vec(), self.u.vec())
                x = x + 1j * self.u.get_local()
            return x

    def solve(self, b, trans='N'):
        """
        Solve method considering trans argumnet

        Parameters
        ----------
        b : numpy array
            The RHS vector to solve.
        trans : str, optional
            Transpose operation ('N' for none, 'T' for transpose, 'H' for conjugate transpose). Default is 'N'.

        Returns
        -------
        x : numpy array
            The solution of the linear problem.

        """
        b = np.asarray(b)
        # low level PETSc interface that accepts PETSc vectors 
        A_lu = self.solver.solve if trans == 'N' else self.solver.solveTranspose

        if trans == 'N' or trans == 'H' or (trans == 'T' and self.isreal):  #A*u=b or A^H*u=b
            # note the feature of a real block version of complex matrix: its direct transpose = conjugate transpose of the original matrix
            x = self.__solve(A_lu, b)
        else:  # trans=='T' and A is complex: A^T*u=b
            x = np.conjugate(self.__solve(A_lu, np.conjugate(b)))
        return x

    solve.stype = stype


class PETScLU(SparseLUSolver):
    """
    LU solver using PETSc in petsc4py.
    """
    stype = 'petsc'

    def __init__(self, A, lusolver='mumps'):
        """
        LU solver using PETSc in petsc4py.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix to be solved.
        lusolver : str, optional
            Type of LU solver to use. The default is 'mumps'. pending ...

        Returns
        -------
        None.

        """
        super().__init__(A, lusolver)
        self._initialize_solver(self.operator, lusolver)

    def _initialize_solver(self, A, lusolver):
        """
        initialize solver object

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix to be solved.
        lusolver : str
            Type of LU solver to use. pending ...

        Returns
        -------
        None.

        """
        A_ops = handle_complex_matrix(A, flag='Mat2PETSc')
        self.u = PETSc.Vec().createSeq(A_ops.mat().getSize()[0])
        self.u.set(0)

        self.solver = PETSc.KSP().create()
        self.solver.setOperators(A_ops.mat())

        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver.getPC().setFactorSolverType(lusolver)
        self.solver.getPC().setReusePreconditioner(True)
        self.solver.setFromOptions()

    def __solve(self, A_lu, b):
        """
        Detailed solve operations for different data types

        Parameters
        ----------
        A_lu : solve method in LU decomposition object
            Precomputed LU decomposition of A.
        b : numpy array
            The RHS vector to solve.

        Returns
        -------
        x : numpy array
            The solution of the linear problem.

        """
        b = self._check_vec(b)

        if not self.isreal:
            b_r = np.block([np.real(b), np.imag(b)])
            B = PETSc.Vec().createWithArray(b_r)
            A_lu(B, self.u)
            return self.u.getArray()[:self.shape[0]] + 1j * self.u.getArray()[self.shape[0]:]
        else:
            B = PETSc.Vec().createWithArray(np.real(b))
            A_lu(B, self.u)
            x = self.u.getArray().copy()

            if np.issubdtype(b.dtype, np.complexfloating):
                B = PETSc.Vec().createWithArray(np.imag(b))
                A_lu(B, self.u)
                x = x + 1j * self.u.getArray()
            return x

    def solve(self, b, trans='N'):
        """
        Solve method considering trans argumnet

        Parameters
        ----------
        b : numpy array
            The RHS vector to solve.
        trans : str, optional
            Transpose operation ('N' for none, 'T' for transpose, 'H' for conjugate transpose). Default is 'N'.

        Returns
        -------
        x : numpy array
            The solution of the linear problem.

        """

        b = np.asarray(b)
        # low level PETSc interface that accepts PETSc vectors 
        A_lu = self.solver.solve if trans == 'N' else self.solver.solveTranspose

        if trans == 'N' or trans == 'H' or (trans == 'T' and self.isreal):  #A*u=b or A^H*u=b
            # note the feature of a real block version of complex matrix: its direct transpose = conjugate transpose of the original matrix
            x = self.__solve(A_lu, b)
        else:  # trans=='T' and A is complex: A^T*u=b
            x = np.conjugate(self.__solve(A_lu, np.conjugate(b)))
        return x

    solve.stype = stype


class SuperLU(SparseLUSolver):
    """
    LU solver using splu in scipy.sparse.linalg
    """
    stype = 'superlu'

    def __init__(self, A, lusolver='superlu'):
        """
        LU solver using splu in scipy.sparse.linalg

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix to be solved.
        lusolver : str, optional
            Type of LU solver to use. The default is 'superlu'.

        Returns
        -------
        None.

        """
        super().__init__(A, lusolver)
        self.solver = spla.splu(self.operator)

    def __solve(self, A_lu, b, trans):
        """
        Detailed solve operations for different data types and trans argumnet

        Parameters
        ----------
        A_lu : solve method in LU decomposition object
            Precomputed LU decomposition of A.
        b : numpy array
            The RHS vector to solve.
        trans : str, optional
            Transpose operation ('N' for none, 'T' for transpose, 'H' for conjugate transpose). Default is 'N'.

        Returns
        -------
        x : numpy array
            The solution of the linear problem.

        """
        b = self._check_vec(b)

        if self.isreal and np.issubdtype(b.dtype, np.complexfloating):
            x = (A_lu(np.real(b), trans) + 1j * A_lu(np.imag(b), trans))
        else:
            x = A_lu(b, trans)
        return x

    def solve(self, b, trans='N'):
        """
        Solve method 

        Parameters
        ----------
        b : numpy array
            The RHS vector to solve.
        trans : str, optional
                Transpose operation ('N' for none, 'T' for transpose, 'H' for conjugate transpose). Default is 'N'.

        Returns
        -------
        x : numpy array
            The solution of the linear problem.

        """
        b = np.asarray(b)
        return self.__solve(self.solver.solve, b, trans)

    solve.stype = stype


class UmfpackLU(SparseLUSolver):
    """
    LU solver using UMFPACK in scikits.umfpack.
    """
    stype = 'umfpack'

    def __init__(self, A, lusolver='umfpack'):
        """
        LU solver using UMFPACK in scikits.umfpack.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix to be solved.
        lusolver : str, optional
            Type of LU solver to use. The default is 'umfpack'.

        Returns
        -------
        None.

        """
        super().__init__(A, lusolver)
        self.operator.indptr, self.operator.indices = self.operator.indptr.astype(
            np.int64), self.operator.indices.astype(np.int64)
        self._initialize_solver(self.operator, lusolver)

    def _initialize_solver(self, A, lusolver):
        """
        initialize solver object

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix to be solved.
        lusolver : str
            Type of LU solver to use.

        Returns
        -------
        None.

        """
        umf = umfpack.UmfpackContext(self._get_umf_family(A))
        # Make LU decomposition.
        umf.numeric(A)
        self.solver = umf
        self.operator = A

    def _get_umf_family(self, A):
        """
        Get UMFPACK family string given the sparse matrix dtype.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix to determine the family for.

        Raises
        ------
        ValueError
            Matrix data type.

        Returns
        -------
        str
            UMFPACK family string.

        """

        _families = {
            (np.float64, np.int32): 'di',
            (np.complex128, np.int32): 'zi',
            (np.float64, np.int64): 'dl',
            (np.complex128, np.int64): 'zl'
        }
        f_type = np.sctypeDict[A.dtype.name]
        i_type = np.sctypeDict[A.indices.dtype.name]

        if (f_type, i_type) in _families.keys():
            return _families[(f_type, i_type)]
        else:
            msg = 'Unsupported matrix type! only float64 or complex128 matrices with int32 or int64' \
                  ' indices are supported! (got: matrix: %s, indices: %s)' \
                  % (f_type, i_type)
            raise ValueError(msg)

    def __solve(self, A_lu, b, trans):
        """
        Detailed solve operations for different data types and trans argumnet

        Parameters
        ----------
        A_lu : solve method in LU decomposition object 
            Precomputed LU decomposition of A.
        b : numpy array
            The RHS vector to solve.
        trans : str, optional
            Transpose operation ('N' for none, 'T' for transpose, 'H' for conjugate transpose). Default is 'N'.

        Returns
        -------
        x : numpy array
            The solution of the linear problem.

        """
        b = self._check_vec(b)
        sys_para = {'N': umfpack.UMFPACK_A, 'T': umfpack.UMFPACK_At, 'H': umfpack.UMFPACK_At}

        if not self.isreal and trans == 'T':  # if complex and A^T*x=b; umfpack.UMFPACK_At now means conjugate transpose
            return np.conjugate(A_lu(sys_para[trans], self.operator, np.conjugate(b), autoTranspose=True))
        elif self.isreal and np.issubdtype(b.dtype, np.complexfloating):
            return A_lu(sys_para[trans], self.operator, np.real(b), autoTranspose=True) + 1j * A_lu(sys_para[trans],
                                                                                                    self.operator,
                                                                                                    np.imag(b),
                                                                                                    autoTranspose=True)
        else:
            return A_lu(sys_para[trans], self.operator, b, autoTranspose=True)

        return x

    def solve(self, b, trans='N'):
        """
        Solve method

        Parameters
        ----------
        b : numpy array
            The RHS vector to solve.
        trans : str, optional
            Transpose operation ('N' for none, 'T' for transpose, 'H' for conjugate transpose). Default is 'N'.

        Returns
        -------
        x : numpy array
            The solution of the linear problem.

        """
        b = np.asarray(b)
        return self.__solve(self.solver.solve, b, trans)

    solve.stype = stype


#%%
"""
This class provides classes that define linear operators for solving linear equations
"""


class InverseMatrixOperator(spla.LinearOperator):
    """
    Inverse operator for iterative solvers requiring matrix-vector products A^-1*b 
    where A is a sparse matrix and b is a dense vector. This class serves as an 
    abstract interface between iterative solvers and matrix-like objects.
    Supports various LU solvers.
    """

    def __init__(self, A, A_lu=None, lusolver='mumps', trans='N', echo=False):
        """
        Initialize the InverseMatrixOperator object.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix to invert or solve.
        A_lu : LU decomposition object, optional
            Precomputed LU decomposition of A. Default is None.
        lusolver : str, optional
            Type of LU solver to use ('mumps', 'superlu', 'umfpack', 'petsc'). Default is 'mumps'.
        trans : str, optional
            Transpose operation ('N' for none, 'T' for transpose, 'H' for conjugate transpose). Default is 'N'.
        echo : bool, optional
            If True, prints the number of times the linear system has been solved. Default is False.

        Returns
        -------
        None.

        """
        self.shape = A.shape
        self.dtype = A.dtype
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)
        self.count = 0
        self.echo = echo
        self.trans = trans
        self.solve = self._initialize_solver(A, A_lu, lusolver)

    def _initialize_solver(self, A, A_lu, lusolver):
        """
        Initialize the LU solver based on the chosen solver type.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix to invert or solve.
        A_lu : LU decomposition object, optional
            Precomputed LU decomposition of A.
        lusolver : str
            Type of LU solver to use ('mumps', 'superlu', 'umfpack', 'petsc').

        Raises
        ------
        ValueError
            Undefined solver type

        Returns
        -------
        solve method
            solve method in LU decomposition object of A.

        """
        solver_map = {
            'mumps': FEniCSLU,
            'superlu': SuperLU,
            'umfpack': UmfpackLU,
            'petsc': PETScLU
        }

        if A_lu is not None:
            self.A_lu = A_lu  # if A_lu not None, create attribute

            if hasattr(A_lu, 'stype'):  # if A_lu has stype
                if A_lu.stype in solver_map:
                    if hasattr(A_lu, 'solve'):
                        return A_lu.solve
                    else:  # else assume A_lu is a solve method
                        return A_lu
                else:
                    raise ValueError(f"Undefined LU object solver type: {A_lu.stype}")
            else:  # A_lu from other packages, need to define solve behavior
                return self._solve

        else:
            if lusolver not in solver_map:
                raise ValueError(f"Undefined solver type: {lusolver}")
            solver_instance = solver_map[lusolver](A)
            self.A_lu = solver_instance  # if A_lu is None, return solve method
            return solver_instance.solve

        # opts={'mumps':  FEniCSLU(A,lusolver=lusolver),
        #       'superlu':SuperLU(A),
        #       'umfpack':UmfpackLU(A),
        #       'petsc':PETScLU(A,lusolver=lusolver)}       

        # if A_lu is None:
        #     self.solve=opts[lusolver].solve
        # else:
        #     if hasattr(A_lu, 'stype'):
        #         if A_lu.stype in opts.keys():
        #             if hasattr(A_lu, 'solve'):
        #                 self.solve=A_lu.solve
        #             else:
        #                 self.solve=A_lu
        #         else:
        #             raise ValueError("Undefined solver type (stype = " + stype +")")
        #     else:
        #         self.A_lu=A_lu
        #         self.solve=self.__solve

    def _solve(self, b, trans='N'):
        """
        Internal solve method for precomputed LU decomposition object from public packages

        Parameters
        ----------
        b : numpy array
            The RHS vector to solve.
        trans : str, optional
            Transpose operation ('N' for none, 'T' for transpose, 'H' for conjugate transpose). Default is 'N'.


        Raises
        ------
        ValueError
            Argument (trans) conflicts.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if hasattr(self.A_lu, 'solve'):
            A_lu = self.A_lu.solve  # else assume A_lu is a solve method
        else:
            A_lu = self.A_lu

        # assume A_lu can solve complex problems
        if trans == 'N':
            if self.isreal and np.issubdtype(b.dtype, np.complexfloating):
                return (A_lu(np.real(b)) + 1j * A_lu(np.imag(b)))
            else:
                return A_lu(b)
        elif trans == 'T' or trans == 'H':
            try:
                if self.isreal and np.issubdtype(b.dtype, np.complexfloating):
                    return (A_lu(np.real(b), trans) + 1j * A_lu(np.imag(b), trans))
                else:
                    return A_lu(b, trans)
            except:
                raise ValueError("LU factorized object 'A_lu' doesn't have parameter 'trans'")
        else:
            ValueError(f"Incompatible trans parameter: {trans}")

    def _matvec(self, b):
        """
        Perform the matrix-vector product x = A^-1*b.

        Parameters
        ----------
        b : numpy array
            The vector to multiply.

        Returns
        -------
        numpy array
            The result of the multiplication.

        """
        b = np.asarray(b)
        self.count += 1
        if self.echo:
            print(f"Number of the linear system solved: {self.count}")

        return self.solve(b, self.trans)

    def _rmatvec(self, b):
        """
        Perform the adjoint matrix-vector product  (A^trans)^H*x = b.
        
        Parameters
        ----------
        b : numpy array
            The vector to multiply.

        Returns
        -------
        numpy array
            The result of the adjoint multiplication.
            
        """
        b = np.asarray(b)
        self.count += 1
        if self.echo:
            print(f"Number of the linear system solved: {self.count}")

        if self.trans == 'N':
            return self.solve(b, 'H')
        elif self.trans == 'H':
            return self.solve(b, 'N')
        elif self.trans == 'T':
            return np.conj(self.solve(np.conj(b), 'N'))
