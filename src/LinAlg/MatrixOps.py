#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides classes and functions for matrix and vector conversions, assembling linear systems,
and solving linear equations using various LU solvers. It is intended to facilitate the handling of matrices
and vectors in finite element simulations and to provide efficient solvers for large sparse systems.

Classes
-------
- SparseLUSolver:
    Base class for sparse LU solvers.
- FEniCSLU:
    LU solver using FEniCS functions with PETSc backend.
- PETScLU:
    LU solver using PETSc in petsc4py.
- SuperLU:
    LU solver using SuperLU in scipy.sparse.linalg.
- UmfpackLU:
    LU solver using UMFPACK in scikits.umfpack.
- InverseMatrixOperator:
    Inverse operator for iterative solvers requiring matrix-vector products A^{-1} * b.

Functions
---------
- ConvertMatrix(matrix, flag='PETSc2Mat'):
    Convert between FEniCS PETScMatrix and scipy.sparse.csr_matrix.
- ConvertVector(vector, flag='PETSc2Vec'):
    Convert between FEniCS PETScVector and NumPy array.
- AssembleSystem(mat_expr, vec_expr, bcs=None):
    Assemble a linear system from FEniCS expressions.
- AssembleMatrix(expr, bcs=[]):
    Assemble a matrix from a UFL expression.
- AssembleVector(expr, bcs=[]):
    Assemble a vector from a UFL expression.
- TransposePETScMat(A):
    Transpose a FEniCS PETScMatrix.

"""

from ..Deps import *
from ..LinAlg.Utils import woodbury_solver

#%%
"""
Helper function for matrix/vector assembling and convertion
"""


def ConvertMatrix(matrix, flag='PETSc2Mat'):
    """
    Convert a matrix between FEniCS PETScMatrix and scipy.sparse.csr_matrix formats.

    Parameters
    ----------
    matrix : FEniCS PETScMatrix or scipy.sparse.csr_matrix
        The matrix to be converted.
    flag : str, optional
        Specifies the conversion direction. Options are:
        - 'PETSc2Mat' (default): Convert from FEniCS PETScMatrix to scipy.sparse.csr_matrix.
        - 'Mat2PETSc': Convert from scipy.sparse.csr_matrix to FEniCS PETScMatrix.

    Returns
    -------
    Converted matrix in the desired format.

    Raises
    ------
    ValueError
        If an invalid flag is provided.

    Examples
    --------
        A_petsc = assemble(a)
        A_csr = ConvertMatrix(A_petsc, flag='PETSc2Mat')
        A_petsc_again = ConvertMatrix(A_csr, flag='Mat2PETSc')
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
    Convert a vector between FEniCS PETScVector and NumPy array formats.

    Parameters
    ----------
    vector : PETScVector or numpy.ndarray
        The vector to be converted.
    flag : str, optional
        Specifies the conversion direction. Options are:
        - 'PETSc2Vec' (default): Convert from FEniCS PETScVector to NumPy array.
        - 'Vec2PETSc': Convert from NumPy array to FEniCS PETScVector.

    Returns
    -------
    Converted vector in the desired format.

    Raises
    ------
    ValueError
        If an invalid flag is provided.

    Examples
    --------
        b_petsc = assemble(L)
        b_array = ConvertVector(b_petsc, flag='PETSc2Vec')
        b_petsc_again = ConvertVector(b_array, flag='Vec2PETSc')
    """
    if flag == 'PETSc2Vec':
        return np.matrix(vector.get_local())
    elif flag == 'Vec2PETSc':
        return PETScVector(PETSc.Vec().createWithArray(np.asarray(vector)))
    else:
        raise ValueError(f"Invalid flag: {flag}. Use 'PETSc2Vec' or 'Vec2PETSc'.")


def AssembleSystem(mat_expr, vec_expr, bcs=None):
    """
    Assemble a linear system from FEniCS expressions and return the matrix and vector in scipy and NumPy formats.

    Parameters
    ----------
    mat_expr : UFL Form
        The left-hand side (LHS) UFL form to assemble.
    vec_expr : UFL Form
        The right-hand side (RHS) UFL form to assemble.
    bcs : list of DirichletBC, optional
        List of Dirichlet boundary conditions to apply. Default is None.

    Returns
    -------
    A_sparse : scipy.sparse.csr_matrix
        Assembled LHS matrix in CSR format.
    b_vec : numpy.ndarray
        Assembled RHS vector as a NumPy array.

    Examples
    --------
        A_sparse, b_vec = AssembleSystem(a, L, bcs)
    """
    A, b = PETScMatrix(), PETScVector()
    assemble_system(mat_expr, vec_expr, bcs, A_tensor=A, b_tensor=b)
    return ConvertMatrix(A), ConvertVector(b)


def AssembleMatrix(expr, bcs=[]):
    """
    Assemble a matrix from a UFL expression and apply boundary conditions.

    Parameters
    ----------
    expr : UFL Form
        The UFL form to assemble into a matrix.
    bcs : list of DirichletBC, optional
        List of Dirichlet boundary conditions to apply. Default is an empty list.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        The assembled matrix in CSR format.

    Examples
    --------
        A = AssembleMatrix(a, bcs)
    """
    A = PETScMatrix()  # FEniCS using PETSc for matrix operation
    assemble(expr, tensor=A, keep_diagonal=True)  # store assembled matrix in A
    [bc.apply(A) for bc in bcs]
    # convert the format to CSR
    return ConvertMatrix(A)


def AssembleVector(expr, bcs=[]):
    """
    Assemble a vector from a UFL expression and apply boundary conditions.

    Parameters
    ----------
    expr : UFL Form
        The linear form to assemble into a vector.
    bcs : list of DirichletBC, optional
        List of Dirichlet boundary conditions to apply. Default is an empty list.

    Returns
    -------
    b : numpy.ndarray
        The assembled vector as a NumPy array.

    Examples
    --------
        b = AssembleVector(L, bcs)
    """
    b = PETScVector()  # FEniCS using PETSc for matrix operation
    assemble(expr, tensor=b)  # store assembled matrix in A
    [bc.apply(b) for bc in bcs]
    # convert to the numpy matrix
    return ConvertVector(b)


def TransposePETScMat(A):
    """
    Transpose a FEniCS PETScMatrix.

    Parameters
    ----------
    A : PETScMatrix
        The PETScMatrix to transpose.

    Returns
    -------
    transposed_matrix : PETScMatrix
        The transposed PETScMatrix.

    Examples
    --------
        A_T = TransposePETScMat(A)
    """
    # Transpose the PETSc matrix
    petsc_transposed = A.mat().transpose()
    # Convert back to a FEniCS PETScMatrix
    transposed_matrix = PETScMatrix(petsc_transposed)
    return transposed_matrix


#%%
"""
Helper function for complex matrix handling
"""


def handle_complex_matrix(A, flag):
    """
    Convert a complex matrix into a real block matrix format suitable for real solvers.

    Parameters
    ----------
    A : scipy.sparse matrix
        The matrix to be converted. Can be real or complex.
    flag : str
        Specifies the conversion direction. Options are:
        - 'PETSc2Mat': Convert from PETScMatrix to scipy.sparse.csr_matrix.
        - 'Mat2PETSc': Convert from scipy.sparse.csr_matrix to PETScMatrix.

    Returns
    -------
    Converted matrix in the desired format, possibly converted to real block matrix form.

    Notes
    -----
    If the input matrix is complex, it is converted into a real block matrix of the form:

        [ Re(A)  -Im(A) ]
        [ Im(A)   Re(A) ]

    This allows the use of real-valued solvers to handle complex matrices.

    Examples
    --------
        A_real_block = handle_complex_matrix(A_complex, flag='Mat2PETSc')
    """
    isreal = not np.issubdtype(A.dtype, np.complexfloating)
    if not isreal:  # If complex
        A_real_imag = sp.bmat([[np.real(A), -np.imag(A)], [np.imag(A), np.real(A)]], format='csc')
        return ConvertMatrix(A_real_imag, flag=flag)
    else:
        return ConvertMatrix(A, flag=flag)


#%%
"""
Provide linear solver class for solving linear problems
"""


class SparseLUSolver:
    """
    Base class for sparse LU solvers to solve linear systems of the form A * x = b.

    This class provides a template for implementing different sparse LU solvers. Subclasses should implement the
    `_initialize_solver` and `solve` methods for specific solver implementations.

    Attributes
    ----------
    shape : tuple
        Shape of the matrix A.
    dtype : data-type
        Data type of the matrix A.
    isreal : bool
        True if the matrix A is real-valued, False if complex.
    operator : scipy.sparse.csc_matrix
        The matrix A converted to CSC format and as floating-point type.
    stype: str
        Type of the linear solver.

    Methods
    -------
    solve(b, trans='N')
        Solve the linear system A * x = b or its variants.
    """
    stype = ''

    def __init__(self, A, lusolver):
        """
        Initialize the sparse LU solver.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix A in the linear system A * x = b.
        lusolver : str
            Type of LU solver to use. Should be specified in subclasses (e.g. 'mumps', 'superlu', 'umfpack', 'petsc')..

        Notes
        -----
        This is an abstract base class and should not be instantiated directly.
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
        TypeError
            If the matrix data type is not double precision.

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
        b : numpy.ndarray
            The vector to be checked.

        Raises
        ------
        ValueError
            If the input is not a 1D array.

        Returns
        -------
        b : numpy.ndarray
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

        Parameters
        ----------
        b : numpy.ndarray
            Right-hand side vector.
        trans : str, optional
            Transpose operation ('N' for none, 'T' for transpose, 'H' for conjugate transpose).

        Returns
        -------
        x : numpy.ndarray
            Solution vector.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this!")

    solve.stype = stype


class FEniCSLU(SparseLUSolver):
    """
    LU solver using FEniCS PETSc backend to solve A * x = b.

    This class utilizes FEniCS PETSc functions to perform LU decomposition and solve linear systems.

    Parameters
    ----------
    A : scipy.sparse matrix
        The matrix A in the linear system A * x = b.
    lusolver : str, optional
        Type of LU solver to use within PETSc. Default is 'mumps'.

    Attributes
    ----------
    solver : PETScLUSolver.kso
        The KSP (linear solver) object from PETScLUSolver.
    u : PETScVector
        The solution vector.

    Methods
    -------
    solve(b, trans='N')
        Solve the linear system A * x = b or its variants.

    Examples
    --------
        solver = FEniCSLU(A)
        x = solver.solve(b)
    """
    stype = 'mumps'

    def __init__(self, A, lusolver='mumps'):
        """
        Initialize the sparse LU solver FEniCS functions with PETSc backend.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix A in the linear system A * x = b.
        lusolver : str
            Type of LU solver to use. The default is 'mumps'. pending ...
        """
        super().__init__(A, lusolver)
        self._initialize_solver(self.operator, lusolver)

    def _initialize_solver(self, A, lusolver):
        """
        Initialize the solver object.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix to be solved.
        lusolver : str
            Type of LU solver to use within PETSc.

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
        Internal method to solve the linear system.

        Parameters
        ----------
        A_lu : function
            Solve method in the LU decomposition object. Precomputed LU decomposition of A.
        b : numpy.ndarray
            Right-hand side vector.

        Returns
        -------
        x : numpy.ndarray
            Solution vector.
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
        Solve the linear system A * x = b or its variants.

        Parameters
        ----------
        b : numpy.ndarray
            Right-hand side vector.
        trans : str, optional
            Transpose operation ('N' for none, 'T' for transpose, 'H' for conjugate transpose). Default is 'N'.

        Returns
        -------
        x : numpy.ndarray
            Solution vector.
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
    LU solver using PETSc in petsc4py to solve A * x = b.

    Parameters
    ----------
    A : scipy.sparse matrix
        The matrix A in the linear system A * x = b.
    lusolver : str, optional
        Type of LU solver to use within PETSc. Default is 'mumps'.

    Attributes
    ----------
    solver : PETSc.KSP
        The KSP (linear solver) object from PETSc.
    u : PETSc.Vec
        The solution vector.

    Methods
    -------
    solve(b, trans='N')
        Solve the linear system A * x = b or its variants.

    Examples
    --------
        solver = PETScLU(A)
        x = solver.solve(b)
    """
    stype = 'petsc'

    def __init__(self, A, lusolver='mumps'):
        """
        LU solver using PETSc in petsc4py to solve A * x = b.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix A in the linear system A * x = b.
        lusolver : str, optional
            Type of LU solver to use within PETSc. Default is 'mumps'.
        """
        super().__init__(A, lusolver)
        self._initialize_solver(self.operator, lusolver)

    def _initialize_solver(self, A, lusolver):
        """
        Initialize the solver object.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix to be solved.
        lusolver : str
            Type of LU solver to use within PETSc.

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
        Internal method to solve the linear system.

        Parameters
        ----------
        A_lu : function
            Solve method in the LU decomposition object. Precomputed LU decomposition of A.
        b : numpy.ndarray
            Right-hand side vector.

        Returns
        -------
        x : numpy.ndarray
            Solution vector.

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
        Solve the linear system A * x = b or its variants.

        Parameters
        ----------
        b : numpy.ndarray
            Right-hand side vector.
        trans : str, optional
            Transpose operation ('N' for none, 'T' for transpose, 'H' for conjugate transpose). Default is 'N'.

        Returns
        -------
        x : numpy.ndarray
            Solution vector.
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
    LU solver using SuperLU in scipy.sparse.linalg to solve A * x = b.

    Parameters
    ----------
    A : scipy.sparse matrix
        The matrix A in the linear system A * x = b.

    Attributes
    ----------
    solver : scipy.sparse.linalg.SuperLU
        SuperLU from scipy.sparse.linalg.

    Methods
    -------
    solve(b, trans='N')
        Solve the linear system A * x = b or its variants.

    Examples
    --------
        solver = SuperLU(A)
        x = solver.solve(b)

    """
    stype = 'superlu'

    def __init__(self, A):
        """
        LU solver using SuperLU in scipy.sparse.linalg to solve A * x = b.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix A in the linear system A * x = b.
        """
        super().__init__(A, 'superlu')
        self.solver = spla.splu(self.operator)

    def __solve(self, A_lu, b, trans):
        """
        Internal method to solve the linear system.

        Parameters
        ----------
        A_lu : function
            Solve method in the LU decomposition object. Precomputed LU decomposition of A.
        b : numpy.ndarray
            Right-hand side vector.
        trans : str, optional
            Transpose operation ('N' for none, 'T' for transpose, 'H' for conjugate transpose). Default is 'N'.

        Returns
        -------
        x : numpy.ndarray
            Solution vector.
        """
        b = self._check_vec(b)

        if self.isreal and np.issubdtype(b.dtype, np.complexfloating):
            x = (A_lu(np.real(b), trans) + 1j * A_lu(np.imag(b), trans))
        else:
            x = A_lu(b, trans)
        return x

    def solve(self, b, trans='N'):
        """
        Solve the linear system A * x = b or its variants.

        Parameters
        ----------
        b : numpy.ndarray
            Right-hand side vector.
        trans : str, optional
            Transpose operation ('N' for none, 'T' for transpose, 'H' for conjugate transpose). Default is 'N'.

        Returns
        -------
        x : numpy.ndarray
            Solution vector.
        """
        b = np.asarray(b)
        return self.__solve(self.solver.solve, b, trans)

    solve.stype = stype


class UmfpackLU(SparseLUSolver):
    """
    LU solver using UMFPACK in scikits.umfpack to solve A * x = b.

    Parameters
    ----------
    A : scipy.sparse matrix
        The matrix A in the linear system A * x = b.

    Attributes
    ----------
    solver : umfpack.UmfpackContext
        UMFPACK solver context.

    Methods
    -------
    solve(b, trans='N')
        Solve the linear system A * x = b or its variants.

    Examples
    --------
        solver = UmfpackLU(A)
        x = solver.solve(b)
    """
    stype = 'umfpack'

    def __init__(self, A):
        """
        LU solver using UMFPACK in scikits.umfpack to solve A * x = b.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix A in the linear system A * x = b.
        """
        super().__init__(A, 'umfpack')
        self.operator.indptr, self.operator.indices = self.operator.indptr.astype(
            np.int64), self.operator.indices.astype(np.int64)
        self._initialize_solver(self.operator)

    def _initialize_solver(self, A):
        """
        Initialize the solver object.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix to be solved.

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
        Internal method to solve the linear system.

        Parameters
        ----------
        A_lu : function
            Solve method in the LU decomposition object. Precomputed LU decomposition of A.
        b : numpy.ndarray
            Right-hand side vector.
        trans : str, optional
            Transpose operation ('N' for none, 'T' for transpose, 'H' for conjugate transpose). Default is 'N'.

        Returns
        -------
        x : numpy.ndarray
            Solution vector.
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
        Solve the linear system A * x = b or its variants.

        Parameters
        ----------
        b : numpy.ndarray
            Right-hand side vector.
        trans : str, optional
            Transpose operation ('N' for none, 'T' for transpose, 'H' for conjugate transpose). Default is 'N'.

        Returns
        -------
        x : numpy.ndarray
            Solution vector.
        """
        b = np.asarray(b)
        return self.__solve(self.solver.solve, b, trans)

    solve.stype = stype


#%%
"""
Provide linear operator class for solving linear equations
"""


class InverseMatrixOperator(spla.LinearOperator):
    """
    Inverse operator for iterative solvers requiring matrix-vector products A^{-1} * b.

    This class provides an interface to represent the inverse of a matrix A as a linear operator,
    allowing its use in iterative solvers that require only matrix-vector products.

    Parameters
    ----------
    A : scipy.sparse matrix
        The matrix A whose inverse is to be represented.
    A_lu : optional
        Precomputed LU decomposition of A. If not provided, it will be computed.
    Mat : dict, optional
        Additional matrix to include in the form A^{trans} + Mat, where Mat = U * V^T.
    lusolver : str, optional
        Type of LU solver to use ('mumps', 'superlu', 'umfpack', 'petsc'). Default is 'mumps'.
    trans : str, optional
        Specifies whether to solve with A or its transpose/conjugate transpose. Options are:
        - 'N': No transpose (default).
        - 'T': Transpose.
        - 'H': Conjugate transpose.
    echo : bool, optional
        If True, prints the number of times the linear system has been solved. Default is False.

    Methods
    -------
    _matvec(b)
        Compute the matrix-vector product A^{-1} * b.
    _rmatvec(b)
        Compute the adjoint matrix-vector product (A^H)^{-1} * b.

    Examples
    --------
        invA = InverseMatrixOperator(A)
        x = invA @ b  # Equivalent to solving A * x = b
    """

    def __init__(self, A, A_lu=None, Mat=None, lusolver='mumps', trans='N', echo=False):
        """
        Initialize the InverseMatrixOperator object.

        Parameters
        ----------
        A : scipy.sparse matrix
            The matrix A whose inverse is to be represented.
        A_lu : optional
            Precomputed LU decomposition of A. If not provided, it will be computed.
        Mat : dict, optional
            Additional matrix to include in the form A^{trans} + Mat, where Mat = U * V^T.
        lusolver : str, optional
            Type of LU solver to use ('mumps', 'superlu', 'umfpack', 'petsc'). Default is 'mumps'.
        trans : str, optional
            Specifies whether to solve with A or its transpose/conjugate transpose. Options are:
            - 'N': No transpose (default).
            - 'T': Transpose.
            - 'H': Conjugate transpose.
        echo : bool, optional
            If True, prints the number of times the linear system has been solved. Default is False.

        """
        self.shape = A.shape
        self.dtype = A.dtype
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)
        self.count = 0
        self.echo = echo
        self.trans = trans
        self.Mat = Mat
        self.solve = self._initialize_solver(A, A_lu, lusolver)
        super().__init__(dtype=self.dtype, shape=self.shape)  # remains testing
        # if self.Mat is not None:
        #     self._initialize_woodbury()

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

    def _initialize_woodbury(self):
        """
        Initialize the Sherman-Morrison-Woodbury formula based on the chosen solver type.
        """
        k = self.Mat['U'].shape[1]
        m = self.shape[0]
        Uk = np.zeros((m, k))
        for i in range(k):
            Uk[:, i] = self.solve(self.Mat['U'][:, i], self.trans)
        Wk = np.linalg.inv(np.identity(k) + self.Mat['V'].T @ Uk)
        Vk = Wk @ self.Mat['V'].T
        self.Mat.update({'U': Uk, 'V': Vk})

    def _woodbury_solve(self, bk, trans):
        k = self.Mat['U'].shape[1]
        m = self.shape[0]
        Uk = np.zeros((m, k)).astype(self.dtype)
        for i in range(k):
            Uk[:, i] = self.solve(self.Mat['U'][:, i], trans)

        return woodbury_solver(Uk, self.Mat['V'], bk)

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
                raise ValueError("LU factorized object 'A_lu' doesn't support 'trans' parameter.")
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
        bk = self.solve(b, self.trans)

        return bk if self.Mat is None else self._woodbury_solve(bk,
                                                                self.trans)  # bk - self.Mat['U'] @ (self.Mat['V'] @ bk)

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
            bk = self.solve(b, 'H')
            return bk if self.Mat is None else self._woodbury_solve(bk, 'H')

        elif self.trans == 'H':
            bk = self.solve(b, 'N')
            return bk if self.Mat is None else self._woodbury_solve(bk, 'N')
        elif self.trans == 'T':
            bk = np.conj(self.solve(np.conj(b), 'N'))
            if self.Mat is None:
                return bk
            else:
                Uk = np.zeros(self.Mat['U'].shape).astype(self.dtype)
                for i in range(self.Mat['U'].shape[1]):
                    Uk[:, i] = np.conj(self.solve(np.conj(self.Mat['U'][:, i]), 'N'))
                return woodbury_solver(Uk, self.Mat['V'], bk)
        else:
            raise ValueError(f"Incompatible trans parameter: {self.trans}")
